import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2 as chi2_dist


# ==============================================================================
# 1. KIẾN TRÚC MẠNG CƠ SỞ (ENCODER, DECODER, GMM-VAE)
# ==============================================================================

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
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
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
        self.out_act = nn.Sigmoid()

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
        # Dữ liệu thưa: khởi tạo bias âm để sigmoid(bias) ≈ 0.12
        nn.init.constant_(self.out_layer.bias, -2.0)

    def forward(self, z):
        h = self.backbone(z)
        return self.out_act(self.out_layer(h))


class GMM_VAE(nn.Module):
    """
    GMM-VAE tích hợp Prototype-Guided Optimal Transport cho bài toán Open-Set Domain Adaptation.
    Sử dụng Encoder chung và hai Decoder riêng biệt cho scRNA-seq (src) và ST (tgt).
    Prior của z là hỗn hợp Gauss hiệp phương sai chéo với K thành phần đại diện cho K nhãn Known.

    Args:
        encoder, decoder_src, decoder_tgt: các module mạng nơ-ron.
        n_clusters: số cụm K của GMM (bằng số kiểu tế bào Known ở miền nguồn).
        latent_dim: số chiều không gian ẩn.
        mu_init_std: độ lệch chuẩn khi khởi tạo ngẫu nhiên tâm cụm.
    """

    LOG_VAR_C_MIN, LOG_VAR_C_MAX = -6.0, 4.0

    def __init__(self, encoder, decoder_src, decoder_tgt,
                 n_clusters, latent_dim, mu_init_std=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder_src = decoder_src
        self.decoder_tgt = decoder_tgt
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Tham số GMM đại diện cho K cụm tế bào Known
        self.pi_logits = nn.Parameter(torch.zeros(n_clusters))
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim) * mu_init_std)
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    # ------------------------------------------------------------------ GMM params
    @property
    def pi(self):
        return torch.softmax(self.pi_logits, dim=0)  # (K,)

    @property
    def var_c(self):
        return torch.exp(self.log_var_c.clamp(self.LOG_VAR_C_MIN, self.LOG_VAR_C_MAX))  # (K, D)

    @torch.no_grad()
    def init_gmm_params(self, centers, var=None):
        """Khởi tạo tâm cụm và phương sai từ bên ngoài. centers: (K, D)."""
        self.mu_c.copy_(centers)
        if var is not None:
            self.log_var_c.copy_(torch.log(var).clamp(self.LOG_VAR_C_MIN, self.LOG_VAR_C_MAX))

    # ------------------------------------------------------------------ core ops
    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def get_gamma(self, z, pi=None, temperature=1.0):
        """Tính xác suất hậu nghiệm p(c=k | z) trên K cụm Known: shape (B, K)."""
        z = z.unsqueeze(1)             # (B, 1, D)
        mu_c = self.mu_c.unsqueeze(0)  # (1, K, D)
        var_c = self.var_c.unsqueeze(0)

        log_gauss = -0.5 * (
            math.log(2 * math.pi) + torch.log(var_c) + (z - mu_c).pow(2) / var_c
        ).sum(dim=2)

        if pi is None:
            pi = self.pi
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)

        log_joint = torch.log(pi + 1e-10) + log_gauss
        return torch.softmax(log_joint / temperature, dim=1)

    def mahalanobis_sq(self, z):
        """Khoảng cách Mahalanobis bình phương từ điểm z (hoặc mu) tới K cụm: shape (B, K)."""
        diff = z.unsqueeze(1) - self.mu_c.unsqueeze(0)
        return (diff.pow(2) / self.var_c.unsqueeze(0)).sum(dim=2)

    def expected_mahalanobis_sq(self, mu, log_var, detach_gmm=True):
        """
        Kỳ vọng khoảng cách Mahalanobis bình phương E_{q(z|x)}[ ||z - mu_k||^2_{Sigma_k^{-1}} ].
        Dùng làm ma trận chi phí vận chuyển C_{jk} cho Optimal Transport: shape (B, K).
        """
        mu_c = self.mu_c.detach() if detach_gmm else self.mu_c
        var_c = self.var_c.detach() if detach_gmm else self.var_c

        diff_sq = (mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2)      # (B, K, D)
        var_z = log_var.exp().unsqueeze(1)                          # (B, 1, D)
        return ((diff_sq + var_z) / var_c.unsqueeze(0)).sum(dim=2)  # (B, K)

    @torch.no_grad()
    def sample_augmented_latent(self, cluster_idx, num_samples, truncation=2.0):
        """
        Sinh z_aug ~ N(mu_c, var_c) với nhiễu chuẩn bị cắt trong [-truncation, truncation].
        """
        if isinstance(cluster_idx, int):
            cluster_idx = torch.full((num_samples,), cluster_idx, dtype=torch.long,
                                     device=self.mu_c.device)
        else:
            cluster_idx = cluster_idx.to(self.mu_c.device)
            num_samples = cluster_idx.numel()

        mu = self.mu_c[cluster_idx]
        std = self.var_c[cluster_idx].sqrt()

        eps = torch.empty_like(mu)
        nn.init.trunc_normal_(eps, mean=0.0, std=1.0, a=-truncation, b=truncation)
        return mu + eps * std

    # ------------------------------------------------------------------ forward
    def forward(self, x, domain="src"):
        """
        x: (B, input_dim). domain: 'src' (scRNA-seq) hoặc 'tgt' (ST) để chọn decoder.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        decoder = self.decoder_src if domain == "src" else self.decoder_tgt
        x_hat = decoder(z)
        gamma = self.get_gamma(z)

        return {
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "gamma": gamma,
        }


# ==============================================================================
# 2. CÁC HÀM MẤT MÁT & BỘ GIẢI DUSTBIN UNBALANCED OPTIMAL TRANSPORT
# ==============================================================================

def reconstruction_loss(x_hat, x):
    return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)


def gmm_kl_loss(mu, log_var, gamma, model, eps=1e-10):
    """Hàm mất mát ELBO KL chuẩn của GMM-VAE."""
    mu = mu.unsqueeze(1)
    log_var = log_var.unsqueeze(1)
    mu_c = model.mu_c.unsqueeze(0)
    var_c = model.var_c.unsqueeze(0)

    term = 0.5 * (torch.log(var_c) + log_var.exp() / var_c + (mu - mu_c).pow(2) / var_c).sum(dim=2)
    expected = (gamma * term).sum(dim=1)
    kl_cat = (gamma * (torch.log(model.pi + eps).unsqueeze(0) - torch.log(gamma + eps))).sum(dim=1)
    entropy_q = 0.5 * (1.0 + log_var.squeeze(1)).sum(dim=1)

    return (expected - kl_cat - entropy_q).mean()


def supervised_source_cluster_loss(mu, log_var, gamma, target_clusters, model, detach_gmm=False):
    """
    Neo trực tiếp từng tế bào miền nguồn (scRNA-seq) vào đúng cụm tương ứng với nhãn thật.
    Ngăn hiện tượng trôi tâm cụm (cluster switching) và giữ biên phân lớp sắc nét.
    """
    valid = target_clusters >= 0
    if not valid.any():
        return mu.new_zeros(())

    mu_v = mu[valid]
    lv_v = log_var[valid]
    gamma_v = gamma[valid]
    c_idx = target_clusters[valid]

    mu_k = model.mu_c[c_idx].detach() if detach_gmm else model.mu_c[c_idx]
    var_k = model.var_c[c_idx].detach() if detach_gmm else model.var_c[c_idx]

    # 1. KL( q(z|x_src) || N(mu_c, var_c) )
    kl_gauss = 0.5 * (
        torch.log(var_k) - lv_v + lv_v.exp() / var_k + (mu_v - mu_k).pow(2) / var_k - 1.0
    ).sum(dim=1).mean()

    # 2. Cross-Entropy trên xác suất phân cụm hậu nghiệm gamma
    ce_cluster = F.nll_loss(torch.log(gamma_v + 1e-10), c_idx)

    return kl_gauss + ce_cluster


@torch.no_grad()
def solve_dustbin_unbalanced_ot(cost_matrix, ood_threshold, prior_pi=None,
                                reg_eps=0.1, tau_col=0.5, n_iters=25):
    """
    Giải bài toán Unbalanced Optimal Transport với cột hấp thụ OOD (Dustbin Sink) trên miền log.

    Args:
        cost_matrix:   (B, K) khoảng cách Mahalanobis kỳ vọng từ B tế bào ST tới K cụm Known.
        ood_threshold: ngưỡng chi phí xi (thường lấy từ phân vị Chi-square df=latent_dim).
        prior_pi:      (K,) phân phối tiên nghiệm của K cụm Known.
        reg_eps:       hệ số điều chuẩn entropy (Sinkhorn temperature).
        tau_col:       hệ số nới lỏng biên KL cho K cột cụm Known (Unbalanced OT).
        n_iters:       số vòng lặp Sinkhorn.

    Returns:
        T_known: (B, K) kế hoạch vận chuyển vào K cụm Known.
        u_ood:   (B,) xác suất tế bào thuộc vùng Unknown (cột thứ K+1).
    """
    B, K = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    # Chuẩn hóa thang đo chi phí theo số chiều để reg_eps ổn định số học
    scale = float(cost_matrix.mean().clamp(min=1.0).item()) if ood_threshold <= 0 else float(ood_threshold)
    C_known_scaled = cost_matrix / scale
    dustbin_col = torch.full((B, 1), float(ood_threshold) / scale, device=device, dtype=dtype)

    # Ma trận chi phí mở rộng C_tilde: shape (B, K + 1)
    C_ext = torch.cat([C_known_scaled, dustbin_col], dim=1)

    if prior_pi is None:
        log_b_known = torch.full((K,), -math.log(K), device=device, dtype=dtype)
    else:
        pi_norm = prior_pi.detach().clamp(min=1e-6)
        pi_norm = pi_norm / pi_norm.sum()
        log_b_known = torch.log(pi_norm)

    # Biến đối ngẫu miền log: f cho hàng (B,), g cho cột (K + 1,)
    f = torch.zeros(B, device=device, dtype=dtype)
    g = torch.zeros(K + 1, device=device, dtype=dtype)
    rho = tau_col / (tau_col + reg_eps) if tau_col > 0 else 0.0

    for _ in range(n_iters):
        # 1. Cập nhật hàng (ràng buộc tổng khối lượng mỗi tế bào ST bằng 1)
        # f_j = -eps * logsumexp_k( (g_k - C_ext_{jk}) / eps )
        f = -reg_eps * torch.logsumexp((g.unsqueeze(0) - C_ext) / reg_eps, dim=1)

        # 2. Cập nhật K cột Known (nới lỏng biên bằng KL-divergence)
        if rho > 0.0:
            log_col_sum = torch.logsumexp((f.unsqueeze(1) - C_ext[:, :K]) / reg_eps, dim=0) - math.log(B)
            g[:K] = rho * reg_eps * (log_b_known - log_col_sum)

        # 3. Cột thứ K+1 (Dustbin OOD): giữ nguyên g[K] = 0 để dung lượng hấp thụ OOD tự do
        g[K] = 0.0

    # Tính ma trận vận chuyển tối ưu T* (B, K + 1)
    log_T = (f.unsqueeze(1) + g.unsqueeze(0) - C_ext) / reg_eps
    T_ext = torch.softmax(log_T, dim=1)

    T_known = T_ext[:, :K]  # (B, K)
    u_ood = T_ext[:, K]     # (B,)
    return T_known, u_ood


def ot_open_set_alignment_loss(mu_spa, lv_spa, model, chi2_conf=0.95,
                               reg_eps=0.1, tau_col=0.5, ood_margin_weight=0.5,
                               detach_gmm_in_ot=True):
    """
    Tính toán đồng thời:
      1. Kế hoạch vận chuyển Dustbin Unbalanced OT.
      2. loss_ot_align: kéo các tế bào ST Known (w_j cao) về đúng cụm Gaussian nguồn.
      3. loss_ood_margin: đẩy các tế bào ST Unknown (u_j cao) ra khỏi bán kính Chi-square của K cụm Known.
    """
    d2_thr = float(chi2_dist.ppf(chi2_conf, df=model.latent_dim))

    # Bước 1: Tính ma trận chi phí và giải kế hoạch vận chuyển T* (không truyền gradient qua solver)
    with torch.no_grad():
        cost_mat = model.expected_mahalanobis_sq(mu_spa, lv_spa, detach_gmm=True)
        T_known, u_ood = solve_dustbin_unbalanced_ot(
            cost_mat,
            ood_threshold=d2_thr,
            prior_pi=model.pi,
            reg_eps=reg_eps,
            tau_col=tau_col,
        )

    # Bước 2: Loss căn chỉnh và phân cụm cho các tế bào ST Known (weighted KL divergence)
    mu_c = model.mu_c.detach() if detach_gmm_in_ot else model.mu_c
    var_c = model.var_c.detach() if detach_gmm_in_ot else model.var_c

    # Ma trận KL( q(z_j | x_spa) || N(mu_k, var_k) ): shape (B, K)
    mu_exp = mu_spa.unsqueeze(1)        # (B, 1, D)
    lv_exp = lv_spa.unsqueeze(1)        # (B, 1, D)
    mu_c_exp = mu_c.unsqueeze(0)        # (1, K, D)
    var_c_exp = var_c.unsqueeze(0)      # (1, K, D)

    pairwise_kl = 0.5 * (
        torch.log(var_c_exp) - lv_exp
        + lv_exp.exp() / var_c_exp
        + (mu_exp - mu_c_exp).pow(2) / var_c_exp
        - 1.0
    ).sum(dim=2)  # (B, K)

    # Với tế bào Unknown (T_known ≈ 0), lực kéo này tự động triệt tiêu
    loss_ot_align = (T_known * pairwise_kl).sum(dim=1).mean()

    # Bước 3: Loss đẩy biên cô lập cho các tế bào ST Unknown (Hinge Margin + ổn định phương sai)
    d2_all = ((mu_exp - mu_c_exp).pow(2) / var_c_exp).sum(dim=2)  # (B, K)
    min_d2, _ = d2_all.min(dim=1)                                 # Khoảng cách tới cụm Known gần nhất

    # Đẩy mu_spa của tế bào Unknown ra ngoài mặt cầu Mahalanobis bán kính d2_thr
    hinge_repulsion = F.relu(d2_thr - min_d2)
    # Giữ phương sai của tế bào Unknown gần 1.0 để tránh suy biến khi không bị kéo vào cụm
    var_reg_ood = 0.5 * (lv_spa.exp() - lv_spa - 1.0).sum(dim=1)

    loss_ood = (u_ood * (hinge_repulsion + 0.1 * var_reg_ood)).mean()

    total_ot_loss = loss_ot_align + ood_margin_weight * loss_ood
    inlier_ratio = (1.0 - u_ood).mean().item()

    return {
        "loss_ot_total": total_ot_loss,
        "loss_ot_align": loss_ot_align,
        "loss_ood": loss_ood,
        "inlier_ratio": inlier_ratio,
        "T_known": T_known,
        "u_ood": u_ood,
    }


# ==============================================================================
# 3. QUY TRÌNH HUẤN LUYỆN 2 GIAI ĐOẠN (PRETRAIN & JOINT OT ALIGNMENT)
# ==============================================================================

def pretrain_vae(model, dataloader_rna, optimizer, epochs, device, dataloader_spa=None):
    """
    Giai đoạn 1: Pretrain Encoder và Decoder trên miền nguồn (và tùy chọn miền đích).
    """
    model.to(device)
    model.train()
    history = []
    spa_iter = _infinite_loader(dataloader_spa) if dataloader_spa is not None else None

    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x_rna, _, _ in dataloader_rna:
            x_rna = x_rna.to(device)

            mu_r, lv_r = model.encoder(x_rna)
            z_r = model.reparameterize(mu_r, lv_r)
            x_r_hat = model.decoder_src(z_r)
            loss = reconstruction_loss(x_r_hat, x_rna)

            if spa_iter is not None:
                x_spa, _, _ = next(spa_iter)
                x_spa = x_spa.to(device)
                mu_s, lv_s = model.encoder(x_spa)
                z_s = model.reparameterize(mu_s, lv_s)
                x_s_hat = model.decoder_tgt(z_s)
                loss = loss + reconstruction_loss(x_s_hat, x_spa)

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
    Khởi tạo tham số GMM (pi, mu_c, var_c) từ biểu diễn tiềm ẩn của miền nguồn
    và thiết lập ánh xạ song ánh giữa nhãn tế bào gốc (y) và chỉ số cụm (cluster).
    """
    model.to(device)
    model.eval()

    z_list, y_list = [], []
    for x, y, _ in dataloader_rna:
        mu, _ = model.encoder(x.to(device))
        z_list.append(mu)
        y_list.append(y)
    z_all = torch.cat(z_list, dim=0)
    y_all = torch.cat(y_list, dim=0).to(device)

    labels_np = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        z_all.cpu().numpy()
    )
    labels = torch.from_numpy(labels_np).long().to(device)

    one_hot = F.one_hot(labels, num_classes=n_clusters).to(z_all.dtype)
    counts = one_hot.sum(dim=0).clamp(min=1.0)

    pi = counts / counts.sum()
    mu_c = (one_hot.T @ z_all) / counts.unsqueeze(1)
    diff = z_all - mu_c[labels]
    var_c = (one_hot.T @ diff.pow(2)) / counts.unsqueeze(1)

    var_c = var_c.clamp(min=float(np.exp(model.LOG_VAR_C_MIN)), max=float(np.exp(model.LOG_VAR_C_MAX)))
    pi = pi.clamp(min=1e-6)
    pi = pi / pi.sum()

    model.mu_c.data.copy_(mu_c)
    model.log_var_c.data.copy_(torch.log(var_c))
    model.pi_logits.data.copy_(torch.log(pi))

    n_labels = int(y_all.max().item()) + 1
    contingency = torch.zeros(n_labels, n_clusters, device=device)
    contingency.index_put_((y_all, labels), torch.ones_like(y_all, dtype=torch.float), accumulate=True)
    row, col = linear_sum_assignment(-contingency.cpu().numpy())

    label_to_cluster = torch.full((n_labels,), -1, dtype=torch.long)
    label_to_cluster[torch.as_tensor(row)] = torch.as_tensor(col)

    cluster_to_label = torch.full((n_clusters,), -1, dtype=torch.long)
    cluster_to_label[torch.as_tensor(col)] = torch.as_tensor(row)

    return {
        "pi": pi,
        "mu_c": mu_c,
        "var_c": var_c,
        "labels": labels,
        "label_to_cluster": label_to_cluster,
        "cluster_to_label": cluster_to_label,
    }


def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def generate_latent_augmentation(model, labels, label_to_cluster=None, truncation=2.0):
    """
    Sinh mẫu tiềm ẩn z_aug và nhãn cụm c_aug cho các lớp thiểu số trong batch miền nguồn.
    """
    classes, counts = torch.unique(labels, return_counts=True)
    mean_count = counts.float().mean()
    minority_mask = counts.float() < mean_count
    if not minority_mask.any():
        return None, None

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
        return None, None

    z_aug = model.sample_augmented_latent(cluster_idx, cluster_idx.numel(), truncation=truncation)
    return z_aug.detach(), cluster_idx.detach()


def train_joint_gmm_vae(model, dataloader_rna, dataloader_spa, optimizer, epochs,
                        lambda_1, lambda_2, device,
                        label_to_cluster=None, truncation=2.0,
                        chi2_conf=0.95, ot_reg_eps=0.1, ot_tau_col=0.5,
                        ood_margin_weight=0.5, sup_weight=1.0):
    """
    Giai đoạn 2 (Huấn luyện chính):
      - Tái tạo biểu hiện gene cho cả hai miền (loss_recon).
      - Định hình cụm GMM và neo nhãn có giám sát trên miền nguồn + mẫu tăng cường z_aug (loss_gmm).
      - Căn chỉnh miền chọn lọc bằng Dustbin Unbalanced OT: kéo tế bào ST Known vào đúng cụm
        và đẩy tế bào ST Unknown ra vùng ngoại vi (loss_ot).
    """
    model.to(device)
    model.train()
    spa_iter = _infinite_loader(dataloader_spa)
    if label_to_cluster is not None:
        label_to_cluster = label_to_cluster.to(device)

    history = []

    for epoch in range(epochs):
        sums = {
            "total": 0.0, "recon": 0.0, "gmm": 0.0,
            "ot_align": 0.0, "ood": 0.0, "inlier_ratio": 0.0, "n_aug": 0
        }
        n_steps = 0

        for x_rna, y_rna, _ in dataloader_rna:
            x_spa, _, _ = next(spa_iter)
            x_rna, y_rna, x_spa = x_rna.to(device), y_rna.to(device), x_spa.to(device)

            out_rna = model(x_rna, domain="src")
            out_spa = model(x_spa, domain="tgt")

            # 1. Reconstruction Loss trên cả 2 miền
            loss_recon = (
                reconstruction_loss(out_rna["x_hat"], x_rna)
                + reconstruction_loss(out_spa["x_hat"], x_spa)
            )

            # 2. GMM Prior Loss + Supervised Prototype Anchoring trên miền nguồn
            loss_gmm = gmm_kl_loss(out_rna["mu"], out_rna["log_var"], out_rna["gamma"], model)
            target_c = y_rna if label_to_cluster is None else label_to_cluster[y_rna]
            loss_sup = supervised_source_cluster_loss(
                out_rna["mu"], out_rna["log_var"], out_rna["gamma"], target_c, model
            )

            # Tăng cường dữ liệu vùng ẩn cho các cụm thiểu số để giữ biên phân lớp cân bằng
            z_aug, c_aug = generate_latent_augmentation(model, y_rna, label_to_cluster, truncation)
            n_aug = 0
            if z_aug is not None:
                gamma_aug = model.get_gamma(z_aug)
                loss_sup = loss_sup + 0.5 * F.nll_loss(torch.log(gamma_aug + 1e-10), c_aug)
                n_aug = z_aug.size(0)

            loss_gmm_total = loss_gmm + sup_weight * loss_sup

            # 3. Dustbin Unbalanced Optimal Transport (Căn chỉnh Known & Đẩy biên Unknown)
            ot_out = ot_open_set_alignment_loss(
                out_spa["mu"],
                out_spa["log_var"],
                model=model,
                chi2_conf=chi2_conf,
                reg_eps=ot_reg_eps,
                tau_col=ot_tau_col,
                ood_margin_weight=ood_margin_weight,
                detach_gmm_in_ot=True,
            )
            loss_ot_total = ot_out["loss_ot_total"]

            # 4. Tổng hợp hàm mục tiêu
            loss_total = loss_recon + lambda_1 * loss_gmm_total + lambda_2 * loss_ot_total

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["gmm"] += loss_gmm_total.item()
            sums["ot_align"] += ot_out["loss_ot_align"].item()
            sums["ood"] += ot_out["loss_ood"].item()
            sums["inlier_ratio"] += ot_out["inlier_ratio"]
            sums["n_aug"] += n_aug
            n_steps += 1

        n_steps = max(n_steps, 1)
        stats = {k: v / n_steps for k, v in sums.items()}
        history.append(stats)
        print(
            f"[Joint-OT] Epoch {epoch + 1}/{epochs} - total: {stats['total']:.4f} | "
            f"recon: {stats['recon']:.4f} | gmm: {stats['gmm']:.4f} | "
            f"ot_align: {stats['ot_align']:.4f} | ood: {stats['ood']:.4f} | "
            f"known_spa: {stats['inlier_ratio']:.1%} | z_aug: {stats['n_aug']:.1f}"
        )

    return history


# ==============================================================================
# 4. SUY LUẬN HẬU HUẤN LUYỆN & PHÂN LOẠI CÓ LỌC OOD (INFERENCE)
# ==============================================================================

@torch.no_grad()
def extract_latent_stats(model, dataset, device, batch_size=512):
    """Trích xuất toàn bộ (mu, log_var) của tập dữ liệu."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.to(device)
    model.eval()
    mu_list, lv_list = [], []
    for batch in loader:
        x = batch[0].to(device)
        mu, lv = model.encoder(x)
        mu_list.append(mu)
        lv_list.append(lv)
    return torch.cat(mu_list, dim=0), torch.cat(lv_list, dim=0)


@torch.no_grad()
def extract_mu(model, dataset, device, batch_size=512):
    mu, _ = extract_latent_stats(model, dataset, device, batch_size=batch_size)
    return mu


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

    nbr = nbr.to(mu.device)
    gamma0 = model.get_gamma(mu, pi=uniform)
    neigh = gamma0[nbr].mean(dim=1)
    prior = (1.0 - alpha) * uniform.unsqueeze(0) + alpha * neigh
    prior = prior.clamp(min=floor)
    return prior / prior.sum(dim=1, keepdim=True)


@torch.no_grad()
def dual_rejection(model, mu, prior, tau_H=0.5, chi2_conf=0.95, cluster_to_label=None):
    """
    Phân loại tế bào miền đích thành các nhãn cụ thể (Known) hoặc -1 (Unknown)
    kết hợp kiểm định kép (Entropy + Khoảng cách Mahalanobis Chi-square) và kế hoạch vận chuyển OT.
    """
    gamma = model.get_gamma(mu, pi=prior)
    H = -(gamma * torch.log(gamma + 1e-10)).sum(dim=1) / math.log(model.n_clusters)
    cluster = gamma.argmax(dim=1)

    d2_matrix = model.mahalanobis_sq(mu)
    d2 = d2_matrix.gather(1, cluster.unsqueeze(1)).squeeze(1)
    d2_thr = float(chi2_dist.ppf(chi2_conf, df=model.latent_dim))

    # Giải thêm kế hoạch vận chuyển Dustbin OT trên toàn bộ tập dữ liệu để lấy xác suất OOD
    T_known, u_ood = solve_dustbin_unbalanced_ot(
        d2_matrix, ood_threshold=d2_thr, prior_pi=model.pi
    )

    accept = (d2 < d2_thr) & (H < tau_H) & (u_ood < 0.5)
    pseudo_clusters = torch.where(accept, cluster, torch.full_like(cluster, -1))

    # Ánh xạ từ chỉ số cụm GMM sang nhãn tế bào thực tế (nếu truyền cluster_to_label)
    if cluster_to_label is not None:
        c2l = cluster_to_label.to(mu.device)
        pred_labels = torch.where(accept, c2l[cluster], torch.full_like(cluster, -1))
    else:
        pred_labels = pseudo_clusters

    return {
        "pred_labels": pred_labels,
        "pseudo_labels": pseudo_clusters,
        "cluster": cluster,
        "gamma": gamma,
        "T_known": T_known,
        "u_ood": u_ood,
        "entropy": H,
        "d2": d2,
        "d2_thr": d2_thr,
        "accept": accept,
    }


@torch.no_grad()
def predict_spatial_domain(model, dataset_spa, device, nbr=None,
                           prior_mode="spatial", alpha=0.5,
                           tau_H=0.5, chi2_conf=0.95, cluster_to_label=None,
                           batch_size=512):
    """
    Hàm tiện ích gọi trực tiếp sau khi kết thúc train_joint_gmm_vae để lấy kết quả dự đoán cuối cùng.
    """
    mu_all = extract_mu(model, dataset_spa, device, batch_size=batch_size)
    prior = compute_prior(model, mu_all, mode=prior_mode, nbr=nbr, alpha=alpha)
    return dual_rejection(
        model, mu_all, prior,
        tau_H=tau_H, chi2_conf=chi2_conf,
        cluster_to_label=cluster_to_label,
    )
