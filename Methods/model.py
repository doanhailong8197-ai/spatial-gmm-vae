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

class SharedDomainDecoder(nn.Module):
    """
    Thân Decoder và ma trận trọng số gene dùng chung cho cả 2 miền (khóa chặt hệ tọa độ ẩn z),
    kết hợp với hệ số khuếch đại (log_scale) và độ lệch nền (bias) riêng cho từng miền
    để hấp thụ sai số thiết bị giữa scRNA-seq và Spatial.
    """
    def __init__(self, latent_dim, hidden_dims=(256, 512), output_dim=2000, dropout=0.0):
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

        # 1. Thân mạng và trọng số ánh xạ gene DÙNG CHUNG cho cả src và tgt
        self.shared_backbone = nn.Sequential(*layers)
        self.shared_gene_weight = nn.Linear(current_dim, output_dim, bias=False)

        # 2. Tham số hiệu chỉnh thiết bị RIÊNG cho từng miền (chỉ 2 vector kích thước G)
        self.log_scale_src = nn.Parameter(torch.zeros(output_dim))
        self.bias_src = nn.Parameter(torch.full((output_dim,), -2.0))

        self.log_scale_tgt = nn.Parameter(torch.zeros(output_dim))
        self.bias_tgt = nn.Parameter(torch.full((output_dim,), -2.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_domain(self, z, domain="src"):
        h = self.shared_backbone(z)
        gene_logits = self.shared_gene_weight(h)  # (B, G) - Biểu diễn sinh học chung

        if domain == "src":
            scale = torch.clamp(self.log_scale_src, min=-2.0, max=2.0).exp()
            out = gene_logits * scale + self.bias_src
        else:
            scale = torch.clamp(self.log_scale_tgt, min=-2.0, max=2.0).exp()
            out = gene_logits * scale + self.bias_tgt

        return torch.sigmoid(out)

    def reg_loss(self):
        """Điều chuẩn nhẹ để scale của 2 miền không trôi quá xa nhau."""
        return F.mse_loss(self.log_scale_tgt, self.log_scale_src)


class _DomainDecoderView(nn.Module):
    """Wrapper mỏng giúp giữ nguyên cú pháp model.decoder_src(z) và model.decoder_tgt(z)."""
    def __init__(self, shared_decoder, domain):
        super().__init__()
        self.shared_decoder = shared_decoder
        self.domain = domain

    def forward(self, z):
        return self.shared_decoder.forward_domain(z, self.domain)
        
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

def gmm_contrastive_alignment_loss(
    z_src, z_tgt, model,
    x_tgt=None,
    temperature=0.1,
    radial_weight=1.0,
    eps=1e-8,
    n_src_real=None,
    pull_weight=1.0,
    ce_weight=0.5,
    pull_quantile=0.85,
    gate_quantile=0.95,
    gate_softness=0.25,
    src_class_var=None,
    src_class_var_ok=None,
    var_weight=0.5,
    var_min_n=5,
    gene_guide_weight=0.5,
    repel_weight=1.0,
    ood_prior=0.15,
    return_parts=False
):
    """
    Hàm căn chỉnh miền nâng cấp:
    1. Multi-modal Cost (Mahalanobis ẩn + Cosine biểu hiện gene) đưa vào Dustbin Sinkhorn OT.
    2. Lực Hút (Pull + CE) chỉ tác động lên các mẫu Inlier (trọng số w_in).
    3. Lực Đẩy (Repulsive Margin Loss) đẩy các mẫu OOD (trọng số w_ood) ra ngoài bán kính chi-square.
    4. Prototype-Matched Moment Alignment: Khớp tâm và phương sai của Inlier theo đúng tỷ lệ cụm
       thực tế của ST thay vì Deep CORAL toàn cục.
    """
    D = z_tgt.size(1)
    z_src_d = z_src.detach()
    mu_c = model.mu_c.detach()   # (K, D)
    var_c = model.var_c.detach() # (K, D)

    # ---- 1. Khoảng cách Mahalanobis trung bình chiều trên không gian ẩn ----
    diff = z_tgt.unsqueeze(1) - mu_c.unsqueeze(0)                     # (N_t, K, D)
    maha_full = (diff.pow(2) / (var_c.unsqueeze(0) + eps)).sum(dim=2) # (N_t, K)
    maha_mean = maha_full / D                                         # (N_t, K)

    logits = -0.5 * maha_mean / temperature
    log_p = torch.log_softmax(logits, dim=1)

    # ---- 2. Xác định bán kính cụm từ miền nguồn & Giải Dustbin OT ----
    with torch.no_grad():
        z_real = z_src_d if n_src_real is None else z_src_d[:n_src_real]
        d_real = ((z_real.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2)
                  / (var_c.unsqueeze(0) + eps)).sum(dim=2) / D
        d_real_min = d_real.min(dim=1).values

        # Bán kính hút (trong cụm) và ngưỡng ranh giới OOD
        r_in = torch.quantile(d_real_min, pull_quantile).sqrt().clamp(min=1e-3)
        d_thr = torch.quantile(d_real_min, gate_quantile).clamp(min=0.5)
        # Bán kính đẩy OOD: nằm ngoài ngưỡng chi-square chuẩn hóa theo chiều D
        chi2_norm_thr = float(chi2_dist.ppf(0.95, df=D)) / D
        r_out = max(d_thr.sqrt().item() * 1.25, math.sqrt(chi2_norm_thr))

        # Kết hợp chi phí biểu hiện gene (Cosine distance) để dẫn đường khi 2 miền chưa overlap
        cost_latent = maha_mean / (d_thr + eps)  # Chuẩn hóa về thang ~1.0 tại ranh giới cụm
        if x_tgt is not None and gene_guide_weight > 0.0:
            x_norm = F.normalize(x_tgt.detach(), p=2, dim=1)
            sig_norm = F.normalize(model.cluster_signatures.detach(), p=2, dim=1)
            cos_sim = x_norm @ sig_norm.T                       # (N_t, K) trong [-1, 1]
            cost_gene = 1.0 - cos_sim                           # (N_t, K) trong [0, 2]
            # Chuẩn hóa min-max theo hàng để tương thích thang đo với cost_latent
            cost_gene = cost_gene / (cost_gene.median().clamp(min=0.1))
            combined_cost = (1.0 - gene_guide_weight) * cost_latent + gene_guide_weight * cost_gene
        else:
            combined_cost = cost_latent

        # Ngưỡng chi phí vào cột Dustbin (Unknown): mẫu có cost > 1.0 (tức vượt d_thr) sẽ ưu tiên vào OOD
        q, w_in_ot, w_ood_ot = _sinkhorn_with_dustbin(
            combined_cost,
            dustbin_cost=1.05,
            reg=max(temperature, 0.08),
            ood_prior=ood_prior,
            cluster_prior=model.pi.detach(),
        )

        # Kết hợp cổng năng lượng mượt và xác suất từ Dustbin OT
        min_d = maha_mean.min(dim=1).values
        w_gate = torch.sigmoid((d_thr - min_d) / (gate_softness * d_thr + eps))
        w_in = (0.5 * w_in_ot + 0.5 * w_gate).clamp(min=0.0, max=1.0)
        w_ood = (1.0 - w_in).clamp(min=0.0, max=1.0)

    # ---- 3. Lực HÚT cho tế bào Known (Inliers) ----
    loss_ce = -(q * log_p).sum(dim=1)
    d_k = torch.sqrt(maha_mean + eps)                           # (N_t, K)
    excess_in = F.relu(d_k - r_in)
    loss_pull = (q * _pseudo_huber(excess_in, r_in)).sum(dim=1)

    loss_inlier = (w_in * (pull_weight * loss_pull + ce_weight * loss_ce)).sum() / (w_in.sum() + eps)

    # ---- 4. Lực ĐẨY (Repulsive Margin) tách tế bào Unknown (OOD) ra khỏi cụm Known ----
    # Đẩy khoảng cách tới cụm gần nhất vượt ra ngoài bán kính an toàn r_out
    d_min = d_k.min(dim=1).values                               # (N_t,)
    margin_violation = F.relu(r_out - d_min)
    loss_repel = (w_ood * margin_violation.pow(2)).sum() / (w_ood.sum() + eps)

    # ---- 5. Prototype-Conditional Moment Alignment (Thay cho Deep CORAL toàn cục) ----
    # Chỉ khớp các điểm Inlier với tâm kỳ vọng theo phân bố cụm thực tế của miền ST (tránh bẫy lệch tỷ lệ lớp & OOD)
    w_in_norm = (w_in / (w_in.sum() + eps)).unsqueeze(1)        # (N_t, 1)
    expected_tgt_anchors = q @ mu_c                             # (N_t, D) tâm mục tiêu của từng điểm ST
    mean_tgt_inlier = (w_in_norm * z_tgt).sum(dim=0)
    mean_anchor_inlier = (w_in_norm * expected_tgt_anchors).sum(dim=0)
    loss_coral_inlier = F.mse_loss(mean_tgt_inlier, mean_anchor_inlier, reduction="sum")

    # ---- 6. Khớp phương sai nội cụm ST với scRNA (chống nén cụm quá mức) ----
    loss_var = z_tgt.new_zeros(())
    if src_class_var is not None:
        qw = q * w_in.unsqueeze(1)                              # (N_t, K)
        n_k = qw.sum(dim=0)                                     # (K,)
        denom = n_k.clamp(min=eps).unsqueeze(1)
        mean_k = (qw.T @ z_tgt) / denom                         # (K, D)
        sq = (z_tgt.unsqueeze(1) - mean_k.unsqueeze(0)).pow(2)  # (N_t, K, D)
        var_k = ((qw.unsqueeze(2) * sq).sum(dim=0) / denom).mean(dim=1)
        ok = src_class_var_ok & (n_k >= var_min_n)
        if ok.any():
            loss_var = (torch.log(var_k[ok] + eps) - torch.log(src_class_var[ok] + eps)).pow(2).mean()

    total = (
        loss_inlier
        + repel_weight * loss_repel
        + radial_weight * loss_coral_inlier
        + var_weight * loss_var
    )
    if return_parts:
        return total, {
            "var": loss_var.item(),
            "r": r_in.item(),
            "pull": ((w_in * loss_pull).sum() / (w_in.sum() + eps)).item(),
            "repel": loss_repel.item(),
            "ood_frac": (w_ood > 0.5).float().mean().item(),
        }
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


def initialize_gmm_parameters(model, dataloader_rna, n_clusters, device,
                              warmup_steps=200, lr_hbeta=1e-3, seed=0):
    """
    1. Tính chữ ký biểu hiện gene trung bình (Pseudobulk Signature) cho K cụm.
    2. Tính tâm thực nghiệm (emp_mu, emp_var) của scRNA-seq trên Encoder đã pretrain.
    3. Khớp ấm (warm-start) mạng h_beta để mu_c, var_c trùng với đám mây ẩn thực tế
       ngay trước khi bước vào Giai đoạn 2.
    """
    model.to(device)
    model.eval()

    x_sum = torch.zeros(n_clusters, model.input_dim, device=device)
    z_sum = torch.zeros(n_clusters, model.latent_dim, device=device)
    z_sq_sum = torch.zeros(n_clusters, model.latent_dim, device=device)
    counts = torch.zeros(n_clusters, device=device)
    y_list = []

    with torch.no_grad():
        for x, y, _ in dataloader_rna:
            x = x.to(device)
            y = y.to(device).long()
            y_list.append(y)

            valid = (y >= 0) & (y < n_clusters)
            if valid.any():
                x_v, y_v = x[valid], y[valid]
                mu_v, _ = model.encoder(x_v)
                one_hot = F.one_hot(y_v, num_classes=n_clusters).to(x.dtype)  # (B, K)

                x_sum += one_hot.T @ x_v
                z_sum += one_hot.T @ mu_v
                z_sq_sum += one_hot.T @ (mu_v ** 2)
                counts += one_hot.sum(dim=0)

        denom = counts.clamp(min=1.0).unsqueeze(1)
        signatures = x_sum / denom
        model.cluster_signatures.copy_(signatures)

        emp_mu = z_sum / denom
        emp_var = (z_sq_sum / denom - emp_mu ** 2).clamp(
            min=model.SIGMA_MIN ** 2 + 1e-4, max=model.SIGMA_MAX ** 2 - 1e-4
        )

        pi = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-6)
        pi = pi / pi.sum()
        model.pi_logits.data.copy_(torch.log(pi))

    # --- Bước quan trọng: Huấn luyện ấm h_beta khớp với (emp_mu, emp_var) ---
    hbeta_params = (
        list(model.h_beta.parameters())
        + list(model.mu_head.parameters())
        + list(model.sigma_head.parameters())
    )
    opt_hb = torch.optim.Adam(hbeta_params, lr=lr_hbeta)
    has_samples = counts > 0

    for _ in range(warmup_steps):
        mu_c, var_c = model.compute_gmm_params()
        loss_hb = (
            F.mse_loss(mu_c[has_samples], emp_mu[has_samples])
            + F.mse_loss(var_c[has_samples], emp_var[has_samples])
        )
        opt_hb.zero_grad()
        loss_hb.backward()
        opt_hb.step()

    with torch.no_grad():
        mu_c, var_c = model.compute_gmm_params()

    label_to_cluster = torch.arange(n_clusters, dtype=torch.long, device=device)
    return {
        "pi": pi,
        "mu_c": mu_c.detach(),
        "var_c": var_c.detach(),
        "labels": torch.cat(y_list, dim=0),
        "label_to_cluster": label_to_cluster,
    }

def _sinkhorn_with_dustbin(cost_matrix, dustbin_cost, reg=0.1,
                           ood_prior=0.15, cluster_prior=None, n_iters=25):
    """
    Giải bài toán Optimal Transport có kênh rác (Dustbin / Unknown cột K+1):
    - cost_matrix: (N_t, K) chi phí giữa tế bào đích và K cụm Known.
    - dustbin_cost: (N_t, 1) hoặc scalar ngưỡng chi phí để đẩy mẫu vào kênh Unknown.
    Trả về:
        q_known: (N_t, K) phân phối gán cụm mềm đã chuẩn hóa trên các cụm Known.
        w_in:    (N_t,) xác suất mẫu đích thuộc tập Known (Inlier weight).
        w_ood:   (N_t,) xác suất mẫu đích là Unknown / OOD.
    """
    N_t, K = cost_matrix.shape
    device = cost_matrix.device

    if not torch.is_tensor(dustbin_cost):
        dustbin_col = torch.full((N_t, 1), float(dustbin_cost), device=device)
    else:
        dustbin_col = dustbin_cost.view(-1, 1).expand(N_t, 1)

    # Ma trận chi phí mở rộng (N_t, K + 1)
    C_ext = torch.cat([cost_matrix, dustbin_col], dim=1)

    # Log-kernel Boltzmann
    log_P = -C_ext / max(reg, 1e-4)

    # Phân phối biên của cột: (1 - ood_prior) phân bổ cho K cụm, ood_prior cho cột Unknown
    if cluster_prior is None:
        col_known = torch.full((K,), (1.0 - ood_prior) / K, device=device)
    else:
        col_known = (cluster_prior / cluster_prior.sum().clamp(min=1e-8)) * (1.0 - ood_prior)
    col_target = torch.cat([col_known, torch.tensor([ood_prior], device=device)], dim=0)
    log_col_target = torch.log(col_target.clamp(min=1e-8)).unsqueeze(0)  # (1, K+1)

    # Chuẩn hóa Sinkhorn xen kẽ hàng - cột trong miền log (ổn định số học)
    # Lưu ý: Nới lỏng chuẩn hóa cột (soft marginal relaxation) để chịu được lệch tỷ lệ lớp ST
    col_relax = 0.5
    for _ in range(n_iters):
        # Chuẩn hóa hàng (mỗi tế bào có tổng xác suất = 1)
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        # Chuẩn hóa cột có hệ số co giãn (Unbalanced / Relaxed column marginal)
        log_col_sum = torch.logsumexp(log_P, dim=0, keepdim=True) - math.log(max(N_t, 1))
        log_P = log_P + col_relax * (log_col_target - log_col_sum)

    log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
    P = log_P.exp()  # (N_t, K + 1)

    w_ood = P[:, K].clamp(min=0.0, max=1.0)
    w_in = (1.0 - w_ood).clamp(min=0.0, max=1.0)

    q_known = P[:, :K]
    q_known = q_known / q_known.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return q_known, w_in, w_ood


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
                        pull_quantile=0.85, var_weight=0.5, var_ema=0.9,
                        gate_quantile=0.95, gate_softness=0.25,
                        repel_weight=1.0, ood_prior=0.15,
                        **kwargs):
    model.to(device)
    model.train()
    spa_iter = _infinite_loader(dataloader_spa)
    history = []

    if label_to_cluster is not None:
        label_to_cluster = label_to_cluster.to(device)

    K = model.n_clusters
    src_var = torch.zeros(K, device=device)
    src_ok = torch.zeros(K, dtype=torch.bool, device=device)
    total_warm = max(warmup_epochs * len(dataloader_rna), 1)
    total_steps = max(epochs * len(dataloader_rna), 1)
    global_step = 0

    for epoch in range(epochs):
        sums = {"total": 0.0, "recon": 0.0, "gmm": 0.0, "ot": 0.0, "n_aug": 0,
                "var": 0.0, "pull": 0.0, "repel": 0.0, "ood_frac": 0.0}
        n_steps = 0

        for x_rna, y_rna, _ in dataloader_rna:
            x_spa, _, _ = next(spa_iter)
            x_rna, y_rna, x_spa = x_rna.to(device), y_rna.to(device), x_spa.to(device)

            out_rna = model(x_rna, "src")
            out_spa = model(x_spa, "tgt")

            # 1. Reconstruction Loss + Neo chữ ký tâm cụm qua decoder_src
            sig_hat = model.decoder_src(out_rna["mu_c"])
            loss_sig_anchor = reconstruction_loss(sig_hat, model.cluster_signatures)
            loss_recon = (
                reconstruction_loss(out_rna["x_hat"], x_rna)
                + reconstruction_loss(out_spa["x_hat"], x_spa)
                + 0.2 * loss_sig_anchor
            )

            # 2. Nhánh Có giám sát cho scRNA-seq (L_align + L_sc^cls)
            cluster_rna = label_to_cluster[y_rna] if label_to_cluster is not None else y_rna
            loss_gmm, _ = supervised_gmm_loss_and_cls(
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

            # 3. Tăng cường dữ liệu cho các lớp thiểu số trên miền nguồn (Đã sửa để đưa vào z_src_all)
            z_aug = generate_latent_augmentation(model, y_rna, label_to_cluster, truncation)
            n_aug = 0
            if z_aug is not None:
                z_src_all = torch.cat([out_rna["mu"], z_aug], dim=0)
                n_aug = z_aug.size(0)
            else:
                z_src_all = out_rna["mu"]

            # 4. Cập nhật EMA phương sai nội cụm miền nguồn
            with torch.no_grad():
                v_b, ok_b = class_var_stats(out_rna["mu"].detach(), cluster_rna, K)
                first, upd = ok_b & ~src_ok, ok_b & src_ok
                src_var[first] = v_b[first]
                src_var[upd] = var_ema * src_var[upd] + (1 - var_ema) * v_b[upd]
                src_ok |= ok_b

            # Hệ số dẫn đường bằng biểu hiện gene giảm dần khi 2 miền đã khớp (từ 0.6 xuống 0.15)
            progress = min(1.0, global_step / total_steps)
            gene_guide_w = 0.6 * (1.0 - progress) + 0.15 * progress

            # 5. Gọi hàm căn chỉnh 1 lần duy nhất với đầy đủ z_src_all và x_tgt
            loss_align, a_parts = gmm_contrastive_alignment_loss(
                z_src=z_src_all,
                z_tgt=out_spa["mu"],
                model=model,
                x_tgt=x_spa,
                temperature=align_temp,
                radial_weight=radial_weight,
                n_src_real=out_rna["mu"].size(0),
                pull_weight=pull_weight,
                ce_weight=ce_weight,
                pull_quantile=pull_quantile,
                gate_quantile=gate_quantile,
                gate_softness=gate_softness,
                src_class_var=src_var,
                src_class_var_ok=src_ok,
                var_weight=var_weight,
                gene_guide_weight=gene_guide_w,
                repel_weight=repel_weight,
                ood_prior=ood_prior,
                return_parts=True,
            )

            # Sửa lỗi ghi đè ramp: giữ nguyên hiệu lực của warm-up tuyến tính
            ramp = min(1.0, (global_step + 1) / total_warm)
            global_step += 1
            loss_total = loss_recon + lambda_1 * loss_gmm + (lambda_2 * ramp) * loss_align

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["gmm"] += loss_gmm.item()
            sums["ot"] += loss_align.item()
            sums["var"] += a_parts["var"]
            sums["pull"] += a_parts["pull"]
            sums["repel"] += a_parts["repel"]
            sums["ood_frac"] += a_parts["ood_frac"]
            sums["n_aug"] += n_aug
            n_steps += 1

        n_steps = max(n_steps, 1)
        stats = {k: v / n_steps for k, v in sums.items()}
        history.append(stats)
        print(
            f"[Joint-Align] Epoch {epoch + 1}/{epochs} - total: {stats['total']:.4f} | "
            f"recon: {stats['recon']:.4f} | gmm: {stats['gmm']:.4f} | "
            f"align: {stats['ot']:.4f} (pull: {stats['pull']:.3f}, repel: {stats['repel']:.3f}) | "
            f"est_OOD: {stats['ood_frac']:.1%}"
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
