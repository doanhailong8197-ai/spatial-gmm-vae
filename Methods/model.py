import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2 as chi2_dist


# =====================================================================
# 1. KIẾN TRÚC MẠNG: ENCODER, DECODER & GMM-VAE
# =====================================================================

class Encoder(nn.Module):
    """
    Encoder của GMM-VAE: x -> (mu, log_var).

    Args:
        input_dim:   số gene đầu vào (~2000-3000 HVGs).
        hidden_dims: danh sách kích thước các lớp ẩn, giảm dần.
        latent_dim:  số chiều không gian ẩn.
        dropout:     tỉ lệ dropout trong các khối ẩn.
    """

    def __init__(self, input_dim, hidden_dims=(1024, 512, 256), latent_dim=32, dropout=0.1):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_log_var = nn.Linear(current_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Khởi tạo log_var ≈ 0 -> sigma ≈ 1 ở đầu huấn luyện, tránh KL bùng nổ
        nn.init.zeros_(self.fc_log_var.weight)
        nn.init.zeros_(self.fc_log_var.bias)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)  # ổn định số học
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """
    Decoder của GMM-VAE: z -> x_hat, với x_hat nằm trong [0, 1].

    Args:
        latent_dim:  số chiều không gian ẩn (bằng latent_dim của Encoder).
        hidden_dims: kích thước các lớp ẩn, TĂNG dần (ví dụ (256, 512, 1024)).
        output_dim:  số gene đầu ra (bằng input_dim của Encoder).
        dropout:     tỉ lệ dropout trong các khối ẩn (mặc định 0).
    """

    def __init__(self, latent_dim, hidden_dims=(256, 512, 1024), output_dim=2000, dropout=0.0):
        super().__init__()
        layers = []
        current_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.out_layer = nn.Linear(current_dim, output_dim)
        self.out_act = nn.Sigmoid()  # dữ liệu nằm trong [0, 1]

        self._init_weights()

    @classmethod
    def from_encoder_dims(cls, encoder_hidden_dims, latent_dim, output_dim, dropout=0.0):
        """Tạo Decoder đối xứng với Encoder bằng cách đảo ngược hidden_dims."""
        return cls(latent_dim, tuple(reversed(encoder_hidden_dims)), output_dim, dropout)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Dữ liệu thưa: khởi tạo bias âm để sigmoid(bias) ≈ 0.12, gần mức biểu hiện thấp
        nn.init.constant_(self.out_layer.bias, -2.0)

    def forward(self, z):
        h = self.backbone(z)
        return self.out_act(self.out_layer(h))


class GMM_VAE(nn.Module):
    """
    GMM-VAE với mạng nhỏ h_beta dự đoán tâm (mu_c) và phương sai (var_c) 
    từ chữ ký biểu hiện gene trung bình của K loại tế bào scRNA-seq.
    """
    SIGMA_MIN, SIGMA_MAX = 0.1, 1.5

    def __init__(self, encoder, decoder_src, decoder_tgt,
                 n_clusters, latent_dim, input_dim=None, prior_hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.decoder_src = decoder_src
        self.decoder_tgt = decoder_tgt
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Tự động lấy số chiều gene (input_dim) từ lớp Linear đầu tiên của Encoder nếu không truyền vào
        if input_dim is None:
            input_dim = encoder.backbone[0].in_features
        self.input_dim = input_dim

        # Trọng số tiên nghiệm pi_c học được
        self.pi_logits = nn.Parameter(torch.zeros(n_clusters))

        # Buffer lưu K vector biểu hiện gene trung bình (K, input_dim) của scRNA-seq
        self.register_buffer("cluster_signatures", torch.zeros(n_clusters, input_dim))

        # Mạng nhỏ h_beta dự đoán tâm mu_c và phương sai var_c trong không gian Euclid tự do
        self.h_beta = nn.Sequential(
            nn.Linear(input_dim, prior_hidden_dim),
            nn.LayerNorm(prior_hidden_dim),
            nn.GELU(),
            nn.Linear(prior_hidden_dim, prior_hidden_dim),
            nn.LayerNorm(prior_hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(prior_hidden_dim, latent_dim)
        self.sigma_head = nn.Linear(prior_hidden_dim, latent_dim)

        # Khởi tạo trực giao cho mu_head để K tâm cụm ban đầu tách rời nhau
        nn.init.orthogonal_(self.mu_head.weight, gain=2.0)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

    @property
    def pi(self):
        return torch.softmax(self.pi_logits, dim=0)  # (K,)

    def compute_gmm_params(self):
        """Chạy mạng h_beta một lần để lấy (mu_c, var_c) có kèm gradient cho backprop."""
        h = self.h_beta(self.cluster_signatures)
        mu_c = self.mu_head(h)  # (K, D) - Tự do trong không gian Euclid
        
        min_sq, max_sq = self.SIGMA_MIN ** 2, self.SIGMA_MAX ** 2
        var_c = min_sq + (max_sq - min_sq) * torch.sigmoid(self.sigma_head(h))  # (K, D)
        return mu_c, var_c

    @property
    def mu_c(self):
        mu_c, _ = self.compute_gmm_params()
        return mu_c

    @property
    def var_c(self):
        _, var_c = self.compute_gmm_params()
        return var_c

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def get_log_posterior(self, z, pi=None, temperature=1.0, mu_c=None, var_c=None):
        """
        Tính log p(c | z) theo Công thức (16) của VaDE dùng cho hàm phạt phân loại L_sc^cls.
        """
        if mu_c is None or var_c is None:
            mu_c, var_c = self.compute_gmm_params()

        z_exp = z.unsqueeze(1)          # (B, 1, D)
        mu_exp = mu_c.unsqueeze(0)      # (1, K, D)
        var_exp = var_c.unsqueeze(0)    # (1, K, D)

        log_gauss = -0.5 * (
            math.log(2 * math.pi) + torch.log(var_exp + 1e-8) + (z_exp - mu_exp).pow(2) / (var_exp + 1e-8)
        ).sum(dim=2)  # (B, K)

        if pi is None:
            pi = self.pi
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)

        log_joint = torch.log(pi + 1e-10) + log_gauss
        return F.log_softmax(log_joint / temperature, dim=1)

    def get_gamma(self, z, pi=None, temperature=1.0):
        """Tính xác suất hậu nghiệm p(c | z) = exp(log p(c | z))."""
        return self.get_log_posterior(z, pi=pi, temperature=temperature).exp()

    def mahalanobis_sq(self, z):
        """Khoảng cách Mahalanobis bình phương từ z (B, D) tới K cụm -> (B, K)."""
        mu_c, var_c = self.compute_gmm_params()
        diff = z.unsqueeze(1) - mu_c.unsqueeze(0)
        return (diff.pow(2) / (var_c.unsqueeze(0) + 1e-8)).sum(dim=2)

    @torch.no_grad()
    def sample_augmented_latent(self, cluster_idx, num_samples, truncation=2.0):
        mu_c, var_c = self.compute_gmm_params()
        if isinstance(cluster_idx, int):
            cluster_idx = torch.full(
                (num_samples,), cluster_idx, dtype=torch.long, device=mu_c.device
            )
        else:
            cluster_idx = cluster_idx.to(mu_c.device)

        mu = mu_c[cluster_idx]
        std = var_c[cluster_idx].sqrt()

        eps = torch.empty_like(mu)
        nn.init.trunc_normal_(eps, mean=0.0, std=1.0, a=-truncation, b=truncation)
        return mu + eps * std

    def forward(self, x, domain):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        decoder = self.decoder_src if domain == "src" else self.decoder_tgt
        x_hat = decoder(z)
        mu_c, var_c = self.compute_gmm_params()
        gamma = self.get_log_posterior(z, mu_c=mu_c, var_c=var_c).exp()

        return {
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "gamma": gamma,
            "mu_c": mu_c,
            "var_c": var_c,
        }


# =====================================================================
# 2. CÁC HÀM MẤT MÁT: RECONSTRUCTION, GMM KL & OPTIMAL TRANSPORT
# =====================================================================

def reconstruction_loss(x_hat, x):
    return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)


def gmm_kl_loss(mu, log_var, gamma, model, eps=1e-10):
    """KL divergence không giám sát giữa q(z, c | x) và GMM prior p(z, c)."""
    mu = mu.unsqueeze(1)
    log_var = log_var.unsqueeze(1)
    mu_c = model.mu_c.unsqueeze(0)
    var_c = model.var_c.unsqueeze(0)

    term = 0.5 * (
        torch.log(var_c) + log_var.exp() / var_c + (mu - mu_c).pow(2) / var_c
    ).sum(dim=2)
    expected = (gamma * term).sum(dim=1)
    kl_cat = (
        gamma * (torch.log(model.pi + eps).unsqueeze(0) - torch.log(gamma + eps))
    ).sum(dim=1)
    entropy_q = 0.5 * (1.0 + log_var.squeeze(1)).sum(dim=1)

    return (expected - kl_cat - entropy_q).mean()


def supervised_gmm_loss_and_cls(mu, log_var, z_sample, y_rna, model, lambda_cls=1.0, eps=1e-8):
    """
    Tính đồng thời:
    1. L_align (Supervised KL): Kéo cả Encoder (mu, log_var) và mạng dự đoán tâm cụm h_beta (mu_c, var_c)
       khớp với nhau theo nhãn thật y_rna.
    2. L_sc^cls (Posterior Cross-Entropy): Tối đa hóa log p(c = y_rna | z_sc) theo Công thức (16) của VaDE
       để đẩy K phân phối Gaussian tách rời nhau, chống gộp cụm.
    """
    valid = (y_rna >= 0) & (y_rna < model.n_clusters)
    if not valid.any():
        zero = mu.new_zeros(())
        return zero, {"align_kl": 0.0, "cls_ce": 0.0}

    mu = mu[valid]
    log_var = log_var[valid]
    z_sample = z_sample[valid]
    y_true = y_rna[valid].long()

    # Lấy tâm cụm và phương sai được dự đoán từ mạng nhỏ h_beta (có gradient truyền về h_beta)
    mu_c, var_c = model.compute_gmm_params()

    # ---- 1. L_align: KL( q(z|x_sc) || N(mu_{y_i}, sigma^2_{y_i}) ) ----
    target_mu = mu_c[y_true]      # (N, J)
    target_var = var_c[y_true]    # (N, J)
    tilde_var = log_var.exp()     # \tilde{\sigma}^2_{i,j}

    loss_align = 0.5 * (
        torch.log(target_var + eps)
        + (tilde_var + (mu - target_mu).pow(2)) / (target_var + eps)
        - 1.0
        - log_var
    ).sum(dim=1).mean()

    # ---- 2. L_sc^cls: - log p(c = y_sc | z_sc) theo Công thức (16) của VaDE ----
    log_p_c_given_z = model.get_log_posterior(z_sample, mu_c=mu_c, var_c=var_c)  # (N, K)
    loss_cls = F.nll_loss(log_p_c_given_z, y_true)

    total_sup_gmm = loss_align + lambda_cls * loss_cls
    return total_sup_gmm, {"align_kl": loss_align.item(), "cls_ce": loss_cls.item()}


def _pseudo_huber(x, delta):
    """Gradient bị chặn trên bởi delta, trơn tại 0."""
    return delta ** 2 * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


@torch.no_grad()
def class_var_stats(mu, y, n_clusters, min_n=3):
    """Phương sai nội cụm (trung bình theo chiều) của từng lớp -> (var[K], ok[K])."""
    var = mu.new_zeros(n_clusters)
    ok = torch.zeros(n_clusters, dtype=torch.bool, device=mu.device)
    for k in range(n_clusters):
        m = mu[y == k]
        if m.size(0) >= min_n:
            var[k] = m.var(dim=0, unbiased=True).mean()
            ok[k] = True
    return var, ok


def gmm_contrastive_alignment_loss(z_src, z_tgt, model, temperature=0.1,
                                   radial_weight=1.0, eps=1e-8,
                                   n_src_real=None,            # số điểm scRNA THẬT ở đầu z_src (phần sau là z_aug)
                                   pull_weight=1.0, ce_weight=0.5, pull_quantile=0.9,
                                   src_class_var=None, src_class_var_ok=None,
                                   var_weight=0.5, var_min_n=5, return_parts=False):
    D = z_tgt.size(1)
    z_src_d = z_src.detach()

    # ---- 1. Deep CORAL (giữ nguyên) ----
    mean_src = z_src_d.mean(dim=0, keepdim=True)
    mean_tgt = z_tgt.mean(dim=0, keepdim=True)
    loss_mean = F.mse_loss(mean_tgt, mean_src, reduction="sum")
    zc_src, zc_tgt = z_src_d - mean_src, z_tgt - mean_tgt
    cov_src = (zc_src.T @ zc_src) / max(z_src_d.size(0) - 1, 1)
    cov_tgt = (zc_tgt.T @ zc_tgt) / max(z_tgt.size(0) - 1, 1)
    loss_cov = F.mse_loss(cov_tgt, cov_src, reduction="sum") / D
    loss_coral = loss_mean + loss_cov

    # ---- 2. Mahalanobis tới K tâm ----
    mu_c = model.mu_c.detach()
    var_c = model.var_c.detach()
    diff = z_tgt.unsqueeze(1) - mu_c.unsqueeze(0)
    maha_full = (diff.pow(2) / var_c.unsqueeze(0)).sum(dim=2)
    maha_mean = maha_full / D

    logits = -0.5 * maha_mean / temperature
    log_p = torch.log_softmax(logits, dim=1)
    p = log_p.exp()

    with torch.no_grad():
        p_ng = p.detach()
        f_k = p_ng.sum(dim=0).clamp(min=eps)
        q = p_ng.pow(2) / f_k.unsqueeze(0)
        q = q / q.sum(dim=1, keepdim=True).clamp(min=eps)

        min_d = maha_mean.min(dim=1).values
        med_d = min_d.median()
        mad_d = (min_d - med_d).abs().median() * 1.4826 + eps
        w = torch.sigmoid(-(min_d - (med_d + 0.5 * mad_d)) / (mad_d + eps))

        # Bán kính tin cậy lấy từ CHÍNH scRNA thật: quantile của khoảng cách tới tâm gần nhất
        z_real = z_src_d if n_src_real is None else z_src_d[:n_src_real]
        d_real = ((z_real.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2)
                  / var_c.unsqueeze(0)).sum(dim=2) / D
        r = torch.quantile(d_real.min(dim=1).values, pull_quantile).sqrt().clamp(min=1e-3)

    # Lực 1: CE (giảm hệ số)
    loss_ce = -(q * log_p).sum(dim=1)

    # Lực 2: pull dùng maha_mean, vùng chết trong bán kính r, ngoài r dùng pseudo-Huber
    d_k = torch.sqrt(maha_mean + eps)                 # (N_t, K), cùng thang với r
    excess = F.relu(d_k - r)
    loss_pull = (q * _pseudo_huber(excess, r)).sum(dim=1)

    loss_cluster = (w * (pull_weight * loss_pull + ce_weight * loss_ce)).sum() / (w.sum() + eps)

    # ---- 3. Khớp phương sai nội cụm ST với scRNA (chống nén quá mức) ----
    loss_var = z_tgt.new_zeros(())
    if src_class_var is not None:
        qw = q * w.unsqueeze(1)                                        # (N_t, K), không gradient
        n_k = qw.sum(dim=0)                                            # (K,)
        denom = n_k.clamp(min=eps).unsqueeze(1)
        mean_k = (qw.T @ z_tgt) / denom                                # (K, D)
        sq = (z_tgt.unsqueeze(1) - mean_k.unsqueeze(0)).pow(2)         # (N_t, K, D)
        var_k = ((qw.unsqueeze(2) * sq).sum(dim=0) / denom).mean(dim=1)  # (K,)
        ok = src_class_var_ok & (n_k >= var_min_n)
        if ok.any():
            loss_var = (torch.log(var_k[ok] + eps)
                        - torch.log(src_class_var[ok] + eps)).pow(2).mean()

    total = loss_cluster + radial_weight * loss_coral + var_weight * loss_var
    if return_parts:
        return total, {"var": loss_var.item(), "r": r.item(),
                       "pull": ((w * loss_pull).sum() / (w.sum() + eps)).item()}
    return total

# =====================================================================
# 3. GIAI ĐOẠN 1: PRETRAIN VAE & KHỞI TẠO THAM SỐ GMM
# =====================================================================

def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def pretrain_vae(model, dataloader_rna, optimizer, epochs, device, dataloader_spa=None):
    model.to(device)
    model.train()
    history = []
    spa_iter = _infinite_loader(dataloader_spa) if dataloader_spa is not None else None

    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x, _, _ in dataloader_rna:
            x = x.to(device)

            mu, log_var = model.encoder(x)
            z = model.reparameterize(mu, log_var)
            x_hat = model.decoder_src(z)
            loss = reconstruction_loss(x_hat, x)

            if spa_iter is not None:
                x_spa, _, _ = next(spa_iter)
                x_spa = x_spa.to(device)
                mu_s, log_var_s = model.encoder(x_spa)
                z_s = model.reparameterize(mu_s, log_var_s)
                x_hat_s = model.decoder_tgt(z_s)
                # Khớp tâm và phương sai ngay từ Pretrain để 2 decoder không đẩy 2 miền lệch nhau
                loss_anchor = (
                    F.mse_loss(mu_s.mean(0), mu.detach().mean(0), reduction="sum")
                    + F.mse_loss(mu_s.std(0), mu.detach().std(0), reduction="sum")
                )
                loss = loss + reconstruction_loss(x_hat_s, x_spa) + 0.5 * loss_anchor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(n_batches, 1)
        history.append(epoch_loss)
        print(f"[Pretrain] Epoch {epoch + 1}/{epochs} - recon_loss: {epoch_loss:.4f}")

    return history


@torch.no_grad()
def initialize_gmm_parameters(model, dataloader_rna, n_clusters, device, seed=0):
    """
    Tính chữ ký biểu hiện gene trung bình (Pseudobulk Signature) cho từng loại tế bào c
    từ dữ liệu scRNA-seq và nạp vào model.cluster_signatures để mạng h_beta sử dụng.
    """
    model.to(device)
    model.eval()

    x_sum = torch.zeros(n_clusters, model.input_dim, device=device)
    counts = torch.zeros(n_clusters, device=device)
    y_list = []

    for x, y, _ in dataloader_rna:
        x = x.to(device)
        y = y.to(device).long()
        y_list.append(y)

        valid = (y >= 0) & (y < n_clusters)
        if valid.any():
            one_hot = F.one_hot(y[valid], num_classes=n_clusters).to(x.dtype)  # (B, K)
            x_sum += one_hot.T @ x[valid]                                      # (K, input_dim)
            counts += one_hot.sum(dim=0)                                       # (K,)

    # 1. Lưu chữ ký biểu hiện gene trung bình \bar{x}_c^{sc} vào buffer của mô hình
    signatures = x_sum / counts.clamp(min=1.0).unsqueeze(1)
    model.cluster_signatures.copy_(signatures)

    # 2. Khởi tạo xác suất tiên nghiệm pi_c theo tỷ lệ mẫu của từng loại tế bào
    pi = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-6)
    pi = pi / pi.sum()
    model.pi_logits.data.copy_(torch.log(pi))

    # 3. Vì cụm c tương ứng trực tiếp với nhãn y = c nên label_to_cluster là ánh xạ đồng nhất [0, 1, ..., K-1]
    label_to_cluster = torch.arange(n_clusters, dtype=torch.long, device=device)
    mu_c, var_c = model.compute_gmm_params()

    return {
        "pi": pi,
        "mu_c": mu_c,
        "var_c": var_c,
        "labels": torch.cat(y_list, dim=0),
        "label_to_cluster": label_to_cluster,
    }


# =====================================================================
# 4. GIAI ĐOẠN 2: HUẤN LUYỆN ĐỒNG THỜI VỚI OPTIMAL TRANSPORT
# =====================================================================

def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def generate_latent_augmentation(model, labels, label_to_cluster=None, truncation=2.0):
    classes, counts = torch.unique(labels, return_counts=True)
    mean_count = counts.float().mean()
    minority_mask = counts.float() < mean_count
    if not minority_mask.any():
        return None

    target = math.ceil(mean_count.item())
    minority_classes = classes[minority_mask]
    n_needed = (target - counts[minority_mask]).clamp(min=0)

    if label_to_cluster is None:
        cluster_ids = minority_classes
    else:
        cluster_ids = label_to_cluster.to(labels.device)[minority_classes]

    valid_mask = cluster_ids >= 0
    cluster_ids = cluster_ids[valid_mask]
    n_needed = n_needed[valid_mask]

    cluster_idx = torch.repeat_interleave(cluster_ids, n_needed)
    if cluster_idx.numel() == 0:
        return None

    z_aug = model.sample_augmented_latent(
        cluster_idx, cluster_idx.numel(), truncation=truncation
    )
    return z_aug.detach()


def train_joint_gmm_vae(model, dataloader_rna, dataloader_spa, optimizer, epochs,
                        lambda_1, lambda_2, device,
                        label_to_cluster=None, truncation=2.0,
                        gmm_on_target=False,
                        align_temp=0.1, radial_weight=0.5,
                        warmup_epochs=10, pull_weight=1.0, ce_weight=0.5,
                        pull_quantile=0.9, var_weight=0.5, var_ema=0.9,
                        ot_mode=None, ot_reg=None, ot_tau_s=None, ot_tau_t=None,
                        ot_iters=None, **kwargs):
    """
    Huấn luyện chung GMM-VAE (Giai đoạn 2) với Energy-Gated Contrastive GMM Alignment
    + Radial Moment Matching thay cho Optimal Transport.

    Args:
        lambda_1: Trọng số GMM KL loss.
        lambda_2: Trọng số alignment loss (contrastive + radial).
        label_to_cluster: Ánh xạ nhãn scRNA-seq -> cụm GMM để neo tâm cụm.
        align_temp: Nhiệt độ Boltzmann của alignment loss.
        radial_weight: Trọng số thành phần Radial Moment Matching.
        ot_*, **kwargs: được chấp nhận nhưng bỏ qua (tương thích ngược).
    """
    model.to(device)
    model.train()
    spa_iter = _infinite_loader(dataloader_spa)
    history = []

    if label_to_cluster is not None:
        label_to_cluster = label_to_cluster.to(device)

    
    K = model.n_clusters
    src_var = torch.zeros(K, device=device)                       # EMA phương sai lớp của scRNA
    src_ok = torch.zeros(K, dtype=torch.bool, device=device)
    total_warm = max(warmup_epochs * len(dataloader_rna), 1)
    global_step = 0

    for epoch in range(epochs):
        sums = {"total": 0.0, "recon": 0.0, "gmm": 0.0, "ot": 0.0, "n_aug": 0,
                "var": 0.0, "pull": 0.0}
        n_steps = 0

        for x_rna, y_rna, _ in dataloader_rna:
            x_spa, _, _ = next(spa_iter)
            x_rna, y_rna, x_spa = x_rna.to(device), y_rna.to(device), x_spa.to(device)

            out_rna = model(x_rna, "src")
            out_spa = model(x_spa, "tgt")

            # 1. Reconstruction Loss trên cả hai miền (Tương ứng số hạng E_q[log p_theta(x|z)] trong L_sc^sup)
            loss_recon = (
                reconstruction_loss(out_rna["x_hat"], x_rna)
                + reconstruction_loss(out_spa["x_hat"], x_spa)
            )

            # 2. Nhánh Có giám sát cho scRNA-seq (L_align + L_sc^cls) tối ưu đồng thời Encoder và mạng h_beta
            if label_to_cluster is not None:
                cluster_rna = label_to_cluster[y_rna]
            else:
                cluster_rna = y_rna

            loss_gmm, gmm_parts = supervised_gmm_loss_and_cls(
                mu=out_rna["mu"],
                log_var=out_rna["log_var"],
                z_sample=out_rna["z"],
                y_rna=cluster_rna,
                model=model,
                lambda_cls=kwargs.get("lambda_cls", 1.0),
            )

            if gmm_on_target:
                loss_gmm = loss_gmm + gmm_kl_loss(
                    out_spa["mu"], out_spa["log_var"], out_spa["gamma"], model
                )
            # 3. Tăng cường dữ liệu cho các lớp thiểu số trên miền nguồn
            z_aug = generate_latent_augmentation(
                model, y_rna, label_to_cluster, truncation
            )
            n_aug = 0
            if z_aug is not None:
                z_src_all = torch.cat([out_rna["mu"], z_aug], dim=0)
                n_aug = z_aug.size(0)
            else:
                z_src_all = out_rna["mu"]

            # 4. Energy-Gated Contrastive GMM Alignment + Radial Moment Matching
            with torch.no_grad():
                v_b, ok_b = class_var_stats(out_rna["mu"].detach(), cluster_rna, K)
                first, upd = ok_b & ~src_ok, ok_b & src_ok
                src_var[first] = v_b[first]
                src_var[upd] = var_ema * src_var[upd] + (1 - var_ema) * v_b[upd]
                src_ok |= ok_b
            loss_align, a_parts = gmm_contrastive_alignment_loss(
                z_src=z_src_all,
                z_tgt=out_spa["mu"],
                model=model,
                temperature=align_temp,
                radial_weight=radial_weight,
                n_src_real=out_rna["mu"].size(0),
                pull_weight=pull_weight, ce_weight=ce_weight, pull_quantile=pull_quantile,
                src_class_var=src_var, src_class_var_ok=src_ok, var_weight=var_weight,
                return_parts=True,
            )

            ramp = min(1.0, (global_step + 1) / total_warm)   # warm-up tuyến tính theo step
            global_step += 1
            loss_total = loss_recon + lambda_1 * loss_gmm + (lambda_2 * ramp) * loss_align

            loss_total = loss_recon + lambda_1 * loss_gmm + lambda_2 * loss_align

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["gmm"] += loss_gmm.item()
            sums["ot"] += loss_align.item()
            sums["var"] += a_parts["var"]
            sums["pull"] += a_parts["pull"]
            sums["n_aug"] += n_aug
            n_steps += 1

        n_steps = max(n_steps, 1)
        stats = {k: v / n_steps for k, v in sums.items()}
        history.append(stats)
        print(
            f"[Joint-Align] Epoch {epoch + 1}/{epochs} - total: {stats['total']:.4f} | "
            f"recon: {stats['recon']:.4f} | gmm: {stats['gmm']:.4f} | "
            f"align: {stats['ot']:.4f} | z_aug/batch: {stats['n_aug']:.1f}"
        )

    return history

# =====================================================================
# 5. SUY LUẬN & PHÁT HIỆN OOD SAU HUẤN LUYỆN (POST-TRAINING INFERENCE)
# =====================================================================

@torch.no_grad()
def extract_mu(model, dataset, device, batch_size=512):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    return torch.cat([model.encoder(x.to(device))[0] for x, _, _ in loader], dim=0)


def build_spatial_neighbors(coords, k=15):
    from sklearn.neighbors import NearestNeighbors
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn_.kneighbors(coords)
    return torch.as_tensor(idx[:, 1:], dtype=torch.long)


@torch.no_grad()
def compute_prior(model, mu, mode="spatial", nbr=None, alpha=0.5, floor=1e-3):
    K = model.n_clusters
    if mode == "learned":
        return model.pi.detach()

    uniform = torch.full((K,), 1.0 / K, device=mu.device)
    if mode == "uniform":
        return uniform
    if mode != "spatial" or nbr is None:
        raise ValueError("mode='spatial' cần truyền nbr (láng giềng không gian)")

    gamma0 = model.get_gamma(mu, pi=uniform)
    neigh = gamma0[nbr.to(mu.device)].mean(dim=1)
    prior = (1.0 - alpha) * uniform.unsqueeze(0) + alpha * neigh
    prior = prior.clamp(min=floor)
    return prior / prior.sum(dim=1, keepdim=True)


@torch.no_grad()
def dual_rejection(model, mu, prior, tau_H=0.5, chi2_conf=0.95):
    gamma = model.get_gamma(mu, pi=prior)
    H = -(gamma * torch.log(gamma + 1e-10)).sum(dim=1) / math.log(model.n_clusters)
    cluster = gamma.argmax(dim=1)

    d2 = model.mahalanobis_sq(mu).gather(1, cluster.unsqueeze(1)).squeeze(1)
    d2_thr = float(chi2_dist.ppf(chi2_conf, df=model.latent_dim))

    accept = (d2 < d2_thr) & (H < tau_H)
    pseudo = torch.where(accept, cluster, torch.full_like(cluster, -1))
    return {
        "pseudo_labels": pseudo,
        "cluster": cluster,
        "gamma": gamma,
        "entropy": H,
        "d2": d2,
        "d2_thr": d2_thr,
        "accept": accept,
    }


@torch.no_grad()
def predict_spatial(model, dataset_spa, device, nbr=None, prior_mode="spatial",
                    alpha=0.5, tau_H=0.5, chi2_conf=0.95, batch_size=512):
    """
    Hàm suy luận chạy 1 lần duy nhất sau khi kết thúc Giai đoạn 2 (train_joint_gmm_vae).
    Trả về kết quả gán nhãn cụm (0 .. K-1) cho các tế bào Known và -1 cho các tế bào Unknown (OOD).
    """
    model.to(device)
    model.eval()
    mu_all = extract_mu(model, dataset_spa, device, batch_size=batch_size)
    prior = compute_prior(model, mu_all, mode=prior_mode, nbr=nbr, alpha=alpha)
    res = dual_rejection(model, mu_all, prior, tau_H=tau_H, chi2_conf=chi2_conf)

    acc_ratio = res["accept"].float().mean().item()
    print(
        f"[Inference] Known (accepted): {acc_ratio:.1%} | "
        f"Unknown (OOD): {1.0 - acc_ratio:.1%} | chi2 thr: {res['d2_thr']:.1f}"
    )
    return res
