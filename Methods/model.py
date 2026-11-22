import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2 as chi2_dist


# =====================================================================
# 1. KIẾN TRÚC MẠNG: ENCODER, DECODER, DISCRIMINATOR & GMM-VAE
# =====================================================================

class DSBN1d(nn.Module):
    """Domain-Specific BatchNorm1d (tùy chọn khi cần tách thống kê 2 miền)."""
    def __init__(self, num_features):
        super().__init__()
        self.bn_src = nn.BatchNorm1d(num_features)
        self.bn_tgt = nn.BatchNorm1d(num_features)

    def forward(self, x, domain="src"):
        return self.bn_src(x) if domain == "src" else self.bn_tgt(x)


class Encoder(nn.Module):
    """
    Encoder của GMM-VAE: (x, domain) -> (mu, log_var).

    Args:
        input_dim:     số đặc trưng đầu vào (số gene chung hoặc số pixel).
        hidden_dims:   danh sách kích thước các lớp ẩn, giảm dần.
        latent_dim:    số chiều không gian ẩn.
        dropout:       tỉ lệ dropout trong các khối ẩn.
        input_dropout: tỉ lệ denoising dropout đầu vào (tăng độ bền với dropout kỹ thuật).
        norm_type:     'layernorm' (mặc định, bất biến với label shift) hoặc 'dsbn'.
    """

    def __init__(self, input_dim, hidden_dims=(512, 256, 128), latent_dim=32,
                 dropout=0.1, input_dropout=0.15, norm_type="layernorm"):
        super().__init__()
        self.input_dim = input_dim
        self.norm_type = norm_type
        self.in_drop = nn.Dropout(input_dropout) if input_dropout > 0.0 else nn.Identity()

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()

        current_dim = input_dim
        for h_dim in hidden_dims:
            self.linears.append(nn.Linear(current_dim, h_dim))
            if norm_type == "dsbn":
                self.norms.append(DSBN1d(h_dim))
            else:
                self.norms.append(nn.LayerNorm(h_dim))
            self.acts.append(nn.GELU())
            self.drops.append(nn.Dropout(dropout))
            current_dim = h_dim

        self.backbone = self.linears
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_log_var = nn.Linear(current_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.fc_log_var.weight)
        nn.init.zeros_(self.fc_log_var.bias)

    def forward(self, x, domain="src"):
        h = self.in_drop(x) if self.training else x
        for lin, norm, act, drop in zip(self.linears, self.norms, self.acts, self.drops):
            h = lin(h)
            h = norm(h, domain) if self.norm_type == "dsbn" else norm(h)
            h = drop(act(h))
        mu = self.fc_mu(h)
        log_var = torch.clamp(self.fc_log_var(h), min=-4.0, max=4.0)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """
    Decoder của GMM-VAE: z -> x_hat.

    Args:
        out_act: 'linear' (cho dữ liệu Z-score / chuẩn hóa tổng quát)
                 hoặc 'sigmoid' (cho dữ liệu nằm trong [0, 1] như MNIST).
    """

    def __init__(self, latent_dim, hidden_dims=(128, 256, 512), output_dim=2000,
                 dropout=0.0, out_act="linear"):
        super().__init__()
        self.out_act_name = out_act
        self.use_sigmoid = (out_act == "sigmoid")

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
        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

        self._init_weights()

    @classmethod
    def from_encoder_dims(cls, encoder_hidden_dims, latent_dim, output_dim,
                          dropout=0.0, out_act="linear"):
        return cls(
            latent_dim,
            tuple(reversed(encoder_hidden_dims)),
            output_dim,
            dropout=dropout,
            out_act=out_act,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if self.use_sigmoid:
            nn.init.constant_(self.out_layer.bias, -2.0)

    def forward(self, z):
        h = self.backbone(z)
        return self.out_act(self.out_layer(h))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """Mạng phân biệt miền trong không gian ẩn dùng Spectral Normalization."""
    def __init__(self, latent_dim, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.utils.spectral_norm(nn.Linear(curr_dim, h_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            curr_dim = h_dim
        layers.append(nn.utils.spectral_norm(nn.Linear(curr_dim, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze(-1)


class GMM_VAE(nn.Module):
    """
    GMM-VAE tổng quát kết hợp:
      - Hình học Cosine + Latent Norm bất biến thang đo + Prior GMM.
      - Khóa kiểm chứng sinh ngược xuyên miền (decoder_src) chống nuốt OOD.
      - Bộ đệm hiệu chuẩn hoàn toàn từ tập nguồn (zero target leakage).
    """
    SIGMA_MIN, SIGMA_MAX = 0.35, 0.75

    def __init__(self, encoder, decoder_src, decoder_tgt,
                 n_clusters, latent_dim, input_dim=None, prior_hidden_dim=256,
                 disc_hidden_dims=(128, 64)):
        super().__init__()
        self.encoder = encoder
        self.decoder_src = decoder_src
        self.decoder_tgt = decoder_tgt
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        if input_dim is None:
            input_dim = getattr(encoder, "input_dim", encoder.backbone[0].in_features)
        self.input_dim = input_dim

        self.pi_logits = nn.Parameter(torch.zeros(n_clusters))
        self.register_buffer("cluster_signatures", torch.zeros(n_clusters, input_dim))
        self.register_buffer("emp_mu_c", torch.zeros(n_clusters, latent_dim))
        self.register_buffer("src_is_nonneg", torch.tensor(False))

        # Thống kê tập nguồn dùng làm ngưỡng dự phòng cho kịch bản Closed-Set (0% OOD)
        self.register_buffer("src_ood_q95", torch.tensor(0.50))
        self.register_buffer("src_ood_spread", torch.tensor(0.20))
        self.register_buffer("src_ver_q95", torch.tensor(0.60))
        self.register_buffer("src_ver_spread", torch.tensor(0.20))

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

        self.discriminator = DomainDiscriminator(
            latent_dim=latent_dim, hidden_dims=disc_hidden_dims, dropout=0.1
        )

        nn.init.orthogonal_(self.mu_head.weight, gain=2.0)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

    @property
    def pi(self):
        return torch.softmax(self.pi_logits, dim=0)

    def compute_gmm_params(self):
        h = self.h_beta(self.cluster_signatures)
        mu_c = self.mu_head(h)
        min_sq, max_sq = self.SIGMA_MIN ** 2, self.SIGMA_MAX ** 2
        var_c = min_sq + (max_sq - min_sq) * torch.sigmoid(self.sigma_head(h))
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

    def latent_cosine_sim(self, z, mu_c=None, eps=1e-8):
        if mu_c is None:
            mu_c = self.mu_c
        return F.normalize(z, p=2, dim=1, eps=eps) @ F.normalize(mu_c, p=2, dim=1, eps=eps).T

    def get_log_posterior(self, z, pi=None, temperature=1.0, mu_c=None, var_c=None,
                          use_uniform_prior=True):
        if mu_c is None or var_c is None:
            mu_c, var_c = self.compute_gmm_params()

        z_exp = z.unsqueeze(1)
        mu_exp = mu_c.unsqueeze(0)
        var_exp = var_c.unsqueeze(0)

        log_gauss = -0.5 * (
            math.log(2 * math.pi)
            + torch.log(var_exp + 1e-6)
            + (z_exp - mu_exp).pow(2) / (var_exp + 1e-6)
        ).sum(dim=2)

        if pi is None:
            if use_uniform_prior:
                pi = torch.full((self.n_clusters,), 1.0 / self.n_clusters, device=z.device)
            else:
                pi = self.pi
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)

        log_joint = torch.log(pi + 1e-10) + log_gauss
        return F.log_softmax(log_joint / temperature, dim=1)

    def get_gamma(self, z, pi=None, temperature=1.0, use_uniform_prior=True):
        log_post = self.get_log_posterior(
            z, pi=pi, temperature=temperature, use_uniform_prior=use_uniform_prior
        )
        return log_post.exp()

    def get_spherical_gamma(self, z, pi=None, temperature=0.12, use_uniform_prior=True, use_emp_mu=True):
        """Hậu nghiệm cụm dựa trên Latent Cosine (vMF) — miễn nhiễm với sụp đổ phương sai."""
        centers = self.emp_mu_c if use_emp_mu else self.mu_c
        sim = self.latent_cosine_sim(z, mu_c=centers)
        logits = sim / max(temperature, 1e-3)
        if pi is not None:
            if pi.dim() == 1:
                pi = pi.unsqueeze(0)
            logits = logits + torch.log(pi.clamp(min=1e-10))
        elif not use_uniform_prior:
            logits = logits + torch.log(self.pi.clamp(min=1e-10)).unsqueeze(0)
        return F.softmax(logits, dim=1)

    def mahalanobis_sq(self, z):
        mu_c, var_c = self.compute_gmm_params()
        diff = z.unsqueeze(1) - mu_c.unsqueeze(0)
        return (diff.pow(2) / (var_c.unsqueeze(0) + 1e-6)).sum(dim=2)

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

    def forward(self, x, domain="src"):
        mu, log_var = self.encoder(x, domain=domain)
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
# 2. TIỆN ÍCH CÂN BẰNG LỚP & CÁC HÀM MẤT MÁT TỔNG QUÁT
# =====================================================================

def make_balanced_loader(dataset, batch_size=128, power=0.85, drop_last=True):
    """Lấy mẫu cân bằng lớp cho tập nguồn (giảm tác hại Label Shift)."""
    labels = np.asarray(dataset.labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {c: 1.0 / (cnt ** power) for c, cnt in zip(classes, counts)}
    sample_weights = torch.as_tensor(
        [class_weights[y] for y in labels], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last and (len(dataset) > batch_size),
    )


def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _pearson_sim(a, b, eps=1e-8):
    """
    Tương quan Pearson theo từng mẫu (trừ trung bình theo chiều đặc trưng trước khi tính Cosine).
    Bất biến biên độ và loại bỏ độ lệch nền dương -> dùng thống nhất cho cả Z-score ST lẫn ảnh [0, 1] MNIST.
    """
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    return F.cosine_similarity(a_c, b_c, dim=1, eps=eps)


def _feature_cosine_sim(x, signatures, is_nonneg=False, eps=1e-8):
    """Cosine giữa mẫu x (B, G) và chữ ký cụm (K, G)."""
    if is_nonneg:
        x_use = x - x.mean(dim=0, keepdim=True)
        sig_use = signatures - signatures.mean(dim=0, keepdim=True)
    else:
        x_use = x
        sig_use = signatures
    return F.normalize(x_use, p=2, dim=1, eps=eps) @ F.normalize(sig_use, p=2, dim=1, eps=eps).T


def _attenuation_invariant_cost(sim_matrix, top_q=0.90, min_peak=0.10):
    """Chi phí tuyến tính bất biến suy hao c(i, k) = 1 - sim(i, k) / Q_top(max_k sim)."""
    max_sim = sim_matrix.max(dim=1).values
    peak = torch.quantile(max_sim, top_q).clamp(min=min_peak)
    return 1.0 - sim_matrix / peak, peak


def _log_attenuation_score(sim_1d, top_q=0.98, min_peak=0.10, peak_override=None):
    """
    Điểm OOD trên thang nén log bất biến suy hao:
        s(i) = -log( clip(sim(i) / peak, 0.05, 1.20) )
    - Nén đuôi trái của các lớp Known marker mạnh, giãn ranh giới giữa lớp Known khó và OOD.
    - Không tạo đỉnh nhọn giả tại 0.0 (cho phép đuôi âm mượt tới -0.18).
    """
    if peak_override is not None:
        peak = peak_override if torch.is_tensor(peak_override) else sim_1d.new_tensor(float(peak_override))
        peak = peak.clamp(min=min_peak)
    else:
        peak = torch.quantile(sim_1d, top_q).clamp(min=min_peak)
    rel = (sim_1d / peak).clamp(min=0.05, max=1.20)
    return -torch.log(rel), peak


def reconstruction_loss(x_hat, x, eps=1e-7):
    """Hàm tái tạo chuẩn hóa phương sai — bất biến với số chiều và thang đo đầu vào."""
    x_var = x.detach().var(dim=0, unbiased=False).mean().clamp(min=0.02)
    mse_norm = F.mse_loss(x_hat, x, reduction="mean") / x_var
    cos_dist = 1.0 - F.cosine_similarity(x_hat, x, dim=1, eps=eps).mean()
    return 10.0 * mse_norm + 10.0 * cos_dist


@torch.no_grad()
def mse_recon(x_hat, x):
    return F.mse_loss(x_hat, x, reduction="sum").item() / x.size(0)


def gmm_kl_loss(mu, log_var, gamma, model, eps=1e-10):
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


def supervised_gmm_loss_and_cls(mu, log_var, z_sample, y_rna, model,
                                lambda_cls=1.0, free_radius=0.50,
                                label_smoothing=0.05, cls_temp=2.0,
                                min_src_std=0.18, var_floor_weight=15.0, eps=1e-8):
    """
    Supervised GMM KL + Classification có sàn phương sai nội cụm chống sụp đổ Z_rna:
      - free_radius=0.50: không ép mu_src tụ thành điểm tuyệt đối quanh mu_c.
      - label_smoothing=0.05 & cls_temp=2.0: dịu hóa gradient phân loại.
      - min_src_std=0.18: giữ Ref_std nội cụm tự nhiên (~0.15-0.20) thay vì co về 0.02.
    """
    valid = (y_rna >= 0) & (y_rna < model.n_clusters)
    if not valid.any():
        zero = mu.new_zeros(())
        return zero, {"align_kl": 0.0, "cls_ce": 0.0}

    mu = mu[valid]
    log_var = log_var[valid]
    z_sample = z_sample[valid]
    y_true = y_rna[valid].long()

    mu_c, var_c = model.compute_gmm_params()
    target_mu = mu_c[y_true]
    target_var = var_c[y_true]
    tilde_var = log_var.exp()

    sq_norm = (mu - target_mu).pow(2) / (target_var + eps)
    mean_term = 0.02 * sq_norm + 0.98 * F.relu(sq_norm - free_radius ** 2)
    var_term = torch.log(target_var + eps) + tilde_var / (target_var + eps) - 1.0 - log_var

    loss_align = 0.5 * (var_term + mean_term).sum(dim=1).mean()
    log_p_c_given_z = model.get_log_posterior(z_sample, temperature=cls_temp, mu_c=mu_c, var_c=var_c)
    loss_cls = F.cross_entropy(log_p_c_given_z, y_true, label_smoothing=label_smoothing)

    # Sàn phương sai nội cụm chống sụp đổ mu_src thành hạt điểm
    loss_var_floor = mu.new_zeros(())
    if min_src_std > 0.0 and var_floor_weight > 0.0:
        penalties = []
        for k in range(model.n_clusters):
            mk = (y_true == k)
            if mk.sum() >= 4:
                std_k = mu[mk].std(dim=0, unbiased=True).mean()
                penalties.append(F.relu(min_src_std - std_k).pow(2))
        if penalties:
            loss_var_floor = torch.stack(penalties).mean()

    total_sup = loss_align + lambda_cls * loss_cls + var_floor_weight * loss_var_floor
    return total_sup, {"align_kl": loss_align.item(), "cls_ce": loss_cls.item()}


def _pseudo_huber(x, delta):
    return delta ** 2 * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


@torch.no_grad()
def class_var_stats(mu, y, n_clusters, min_n=3):
    var = mu.new_zeros(n_clusters)
    ok = torch.zeros(n_clusters, dtype=torch.bool, device=mu.device)
    for k in range(n_clusters):
        m = mu[y == k]
        if m.size(0) >= min_n:
            var[k] = m.var(dim=0, unbiased=True).mean()
            ok[k] = True
    return var, ok


@torch.no_grad()
def _sinkhorn_with_dustbin(cost_matrix, dustbin_cost=0.56, reg=0.08,
                           ood_prior=None, cluster_prior=None, n_iters=15,
                           adaptive_ood=True):
    """Entropic Unbalanced OT tự do biên cột (Zero Target Leakage khi adaptive_ood=True)."""
    N_t, K = cost_matrix.shape
    device = cost_matrix.device

    if not torch.is_tensor(dustbin_cost):
        dustbin_col = torch.full((N_t, 1), float(dustbin_cost), device=device)
    else:
        dustbin_col = dustbin_cost.view(-1, 1).expand(N_t, 1)

    C_ext = torch.cat([cost_matrix, dustbin_col], dim=1)
    log_P = -C_ext / max(reg, 1e-4)

    if adaptive_ood or ood_prior is None:
        P = F.softmax(log_P, dim=1)
        w_ood = P[:, K].clamp(min=0.0, max=1.0)
        w_in = (1.0 - w_ood).clamp(min=0.0, max=1.0)
        q_known = F.softmax(-cost_matrix / max(reg, 1e-4), dim=1)
        return q_known, w_in, w_ood

    col_known = torch.full((K,), (1.0 - ood_prior) / K, device=device)
    col_target = torch.cat([col_known, torch.tensor([ood_prior], device=device)], dim=0)
    log_col_target = torch.log(col_target.clamp(min=1e-8)).unsqueeze(0)

    col_relax_vec = torch.full((1, K + 1), 0.02, device=device)
    col_relax_vec[0, K] = 0.50

    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        log_col_sum = torch.logsumexp(log_P, dim=0, keepdim=True) - math.log(max(N_t, 1))
        log_P = log_P + col_relax_vec * (log_col_target - log_col_sum)

    log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
    P = log_P.exp()
    w_ood = P[:, K].clamp(min=0.0, max=1.0)
    w_in = (1.0 - w_ood).clamp(min=0.0, max=1.0)
    q_known = P[:, :K]
    q_known = q_known / q_known.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return q_known, w_in, w_ood


def _adaptive_ood_threshold(scores_np, src_q95=0.50, src_spread=0.20,
                            ashman_min=1.6, min_weight=0.03, return_stats=False):
    """
    Xác định ngưỡng OOD hoàn toàn bất biến miền & không rò rỉ target:
      1. Winsorize [p1, p99] ngăn 1-2% ngoại lai chiếm mất 1 cụm GMM.
      2. Kiểm định hai đỉnh bằng BIC(2-GMM) < BIC(1-GMM) - 10, Ashman's D >= 1.6,
         tỉ trọng tối thiểu >= 3% và khoảng cách hai đỉnh m1 - m0 >= max(0.20, 0.5 * src_spread).
      3. Nếu hai đỉnh (Open-Set): dùng giao điểm đồng tiên nghiệm (Maximum-Likelihood boundary)
         của 2-GMM(full) -> không bị kéo lệch khi tỉ lệ Known/Unknown bất cân xứng (5% hay 40% OOD).
      4. Nếu đơn đỉnh (Closed-Set 0% Unknown): dùng ngưỡng ngoại lai vững (med + 2.5 * MAD_std).
    """
    vals = np.asarray(scores_np, dtype=np.float64)
    med = float(np.median(vals))
    mad_std = float(1.4826 * np.median(np.abs(vals - med)))
    fallback_thr = max(med + 2.5 * mad_std, float(src_q95) * 1.25)

    if len(vals) < 30 or np.std(vals) < 1e-6:
        out = (fallback_thr, "unimodal-fallback", 0.0)
        return (*out, {"m0": med, "m1": fallback_thr, "is_bimodal": False}) if return_stats else out

    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    v_fit = np.clip(vals, lo, hi).reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, covariance_type="full", reg_covar=1e-4, random_state=42).fit(v_fit)
    gmm2 = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-4, random_state=42, n_init=5).fit(v_fit)

    bic_gain = float(gmm1.bic(v_fit) - gmm2.bic(v_fit))
    means = gmm2.means_.ravel()
    vars_ = np.maximum(gmm2.covariances_.ravel(), 1e-6)
    weights = gmm2.weights_.ravel()

    o = np.argsort(means)
    m0, m1 = float(means[o[0]]), float(means[o[1]])
    v0, v1 = float(vars_[o[0]]), float(vars_[o[1]])
    w0, w1 = float(weights[o[0]]), float(weights[o[1]])
    ashman_d = math.sqrt(2.0) * abs(m1 - m0) / math.sqrt(v0 + v1)

    min_gap = max(0.20, 0.5 * float(src_spread))
    if bic_gain > 10.0 and ashman_d >= ashman_min and min(w0, w1) >= min_weight and (m1 - m0) >= min_gap:
        grid = np.linspace(m0, m1, 1000)
        ll0 = -0.5 * (np.log(v0) + (grid - m0) ** 2 / v0)
        ll1 = -0.5 * (np.log(v1) + (grid - m1) ** 2 / v1)
        thr_ml = float(grid[int(np.argmin(np.abs(ll0 - ll1)))])
        out = (thr_ml, f"2-GMM-ML (w={w0:.2f}/{w1:.2f})", float(ashman_d))
        return (*out, {"m0": m0, "m1": m1, "w0": w0, "w1": w1, "is_bimodal": True}) if return_stats else out

    out = (fallback_thr, "unimodal-fallback", float(ashman_d))
    return (*out, {"m0": med, "m1": fallback_thr, "is_bimodal": False}) if return_stats else out


@torch.no_grad()
def estimate_target_verify_thresholds(model, dataset_spa, device, use_crec=False, batch_size=512):
    """
    Hiệu chuẩn không rò rỉ cặp ngưỡng Cổng 3 vùng (thr_in, thr_out) trên toàn bộ tập target
    đầu mỗi epoch Pha B/C:
      - s_ver < thr_in  -> Inlier tin cậy (kéo vào lõi cụm Known).
      - thr_in <= s_ver <= thr_out -> Vùng đệm trung lập (không đẩy nhầm lớp Known khó như L2/3).
      - s_ver > thr_out -> OOD tin cậy (chủ động đẩy góc trực giao + tách vỏ cầu).
      - Nếu phân phối đơn đỉnh (Closed-Set 0% OOD) -> thr_out rất cao -> gần như không đẩy mẫu nào.
    """
    was_training = model.training
    model.eval()

    loader = DataLoader(dataset_spa, batch_size=batch_size, shuffle=False, drop_last=False)
    xs, mus = [], []
    for x, _, _ in loader:
        x = x.to(device)
        xs.append(x)
        if use_crec:
            mus.append(model.encoder(x, domain="tgt")[0])
    X_all = torch.cat(xs, dim=0)

    sim_feat = _feature_cosine_sim(
        X_all, model.cluster_signatures, is_nonneg=bool(model.src_is_nonneg.item())
    ).max(dim=1).values
    s_feat, peak_feat = _log_attenuation_score(sim_feat, top_q=0.98, min_peak=0.10)

    if use_crec:
        mu_all = torch.cat(mus, dim=0)
        x_hat_src = model.decoder_src(mu_all)
        sim_crec = _pearson_sim(x_hat_src, X_all)
        s_crec, peak_crec = _log_attenuation_score(sim_crec, top_q=0.98, min_peak=0.10)
        s_ver = 0.5 * s_feat + 0.5 * s_crec
    else:
        peak_crec = X_all.new_tensor(0.10)
        s_ver = s_feat

    thr_ml, mode, ash_d, info = _adaptive_ood_threshold(
        s_ver.cpu().numpy(),
        src_q95=float(model.src_ver_q95.item()),
        src_spread=float(model.src_ver_spread.item()),
        return_stats=True,
    )

    if info["is_bimodal"]:
        m0, m1 = info["m0"], info["m1"]
        # Vùng đệm rộng bảo vệ lớp Known khó khỏi bị đẩy nhầm thành OOD
        thr_in = thr_ml
        thr_out = max(thr_ml + 0.14, 0.28 * m0 + 0.72 * m1)
    else:
        thr_in = thr_ml
        thr_out = thr_ml * 1.25

    if was_training:
        model.train()

    return {
        "thr_in": float(thr_in),
        "thr_out": float(thr_out),
        "peak_feat": float(peak_feat.item()),
        "peak_crec": float(peak_crec.item()),
        "mode": mode,
        "ashman_d": float(ash_d),
    }


def gmm_adversarial_alignment_loss(
    z_src, z_tgt, model,
    y_src=None,
    x_tgt=None,
    temperature=0.10,
    adv_weight=1.0,
    grl_alpha=0.15,
    src_grl_ratio=0.1,
    radial_weight=1.0,
    eps=1e-8,
    n_src_real=None,
    pull_weight=1.2,
    ce_weight=1.0,
    src_class_var=None,
    src_class_var_ok=None,
    var_weight=0.4,
    var_min_n=5,
    gene_guide_weight=0.5,
    use_crec_gate=False,
    repel_weight=2.0,
    dustbin_cost=0.56,
    ood_prior=None,
    adaptive_ood=True,
    verify_cfg=None,
    tgt_pi_ema=None,
    return_parts=False,
    **kwargs
):
    """
    Căn chỉnh Open-Set tổng quát với Cổng 3 vùng tách biệt (Pull / Neutral / Repel):
      - Dùng thang nén -log bất biến suy hao được hiệu chuẩn toàn cục đầu mỗi epoch (verify_cfg).
      - Phá vỡ bẫy một chiều của max(cost_lat, cost_ver):
          + w_in_pull chỉ mở khi s_ver < thr_in -> chặn vòng lặp nuốt OOD.
          + w_ood_h chỉ mở khi s_ver > thr_out -> không đẩy nhầm lớp Known khó (L2/3).
      - Sửa dứt điểm Cluster Kissing bằng r_in = 0.20 + kéo lõi trơn 0.20 * maha_mean.
    """
    mu_c = model.mu_c.detach()
    var_c = model.var_c.detach()

    # ---- 1. Độ tương đồng góc Cosine & khoảng cách Mahalanobis theo Prior var_c ----
    sim_lat = model.latent_cosine_sim(z_tgt, mu_c=mu_c, eps=eps)           # (N_t, K)
    log_p = F.log_softmax(sim_lat / max(temperature, 0.05), dim=1)

    diff = z_tgt.unsqueeze(1) - mu_c.unsqueeze(0)                          # (N_t, K, D)
    maha_mean = (diff.pow(2) / (var_c.unsqueeze(0) + eps)).mean(dim=2)     # (N_t, K)

    # ---- 2. Gán cụm q & Cổng 3 vùng không rò rỉ (Pull / Neutral / Repel) ----
    with torch.no_grad():
        cost_latent, peak_lat = _attenuation_invariant_cost(sim_lat.detach(), top_q=0.90, min_peak=0.15)
        cost_lat_c = cost_latent.clamp(min=-0.25, max=2.0)

        if x_tgt is not None:
            sim_feat_mat = _feature_cosine_sim(
                x_tgt.detach(),
                model.cluster_signatures.detach(),
                is_nonneg=bool(model.src_is_nonneg.item()),
                eps=eps,
            )
            cost_gene, _ = _attenuation_invariant_cost(sim_feat_mat, top_q=0.90, min_peak=0.10)
            cost_gene_c = cost_gene.clamp(min=-0.25, max=2.0)
            assign_cost = (1.0 - gene_guide_weight) * cost_lat_c + gene_guide_weight * cost_gene_c
        else:
            sim_feat_mat = None
            assign_cost = cost_lat_c

        q, _, _ = _sinkhorn_with_dustbin(
            assign_cost,
            dustbin_cost=dustbin_cost,
            reg=0.08,
            ood_prior=ood_prior,
            adaptive_ood=adaptive_ood,
        )

        # Tính điểm xác thực s_ver trên thang -log đồng nhất với verify_cfg
        if x_tgt is not None and verify_cfg is not None and adaptive_ood:
            sim_feat_max = sim_feat_mat.max(dim=1).values
            s_feat_b, _ = _log_attenuation_score(
                sim_feat_max, top_q=0.98, min_peak=0.10, peak_override=verify_cfg["peak_feat"]
            )
            if use_crec_gate:
                x_hat_src = model.decoder_src(z_tgt.detach())
                sim_crec_b = _pearson_sim(x_hat_src, x_tgt.detach(), eps=eps)
                s_crec_b, _ = _log_attenuation_score(
                    sim_crec_b, top_q=0.98, min_peak=0.10, peak_override=verify_cfg["peak_crec"]
                )
                s_ver_b = 0.5 * s_feat_b + 0.5 * s_crec_b
            else:
                s_ver_b = s_feat_b

            thr_in = verify_cfg["thr_in"]
            thr_out = verify_cfg["thr_out"]

            # Cổng 3 vùng tách biệt trên s_ver_b (không bị sập theo cost_lat và không kẹp bẫy Known)
            p_in = torch.sigmoid((thr_in - s_ver_b) / 0.10)
            p_out = torch.sigmoid((s_ver_b - thr_out) / 0.10)
            w_in_pull = ((p_in - 0.50) / 0.50).clamp(min=0.0, max=1.0)
            w_ood_h = ((p_out - 0.50) / 0.50).clamp(min=0.0, max=1.0)

            ot_in_log = p_in.mean().item()
            ood_frac_log = (s_ver_b > thr_out).float().mean().item()
        else:
            # Nhánh dự phòng khi không truyền verify_cfg hoặc adaptive_ood=False
            cost_ood = cost_lat_c.min(dim=1).values
            w_in = torch.sigmoid((dustbin_cost - cost_ood) / 0.08).clamp(min=0.0, max=1.0)
            w_in_pull = ((w_in - 0.55) / 0.45).clamp(min=0.0, max=1.0)
            w_ood_h = ((1.0 - w_in - 0.55) / 0.45).clamp(min=0.0, max=1.0)
            ot_in_log = w_in.mean().item()
            ood_frac_log = (w_in < 0.5).float().mean().item()

        lat_inlier_rate = (cost_lat_c.min(dim=1).values < 0.50).float().mean().item()

    # ---- 3. Học đối kháng có trọng số chống Label Shift (IWAN) chỉ trên Inlier tin cậy ----
    alpha_eff = float(grl_alpha * adv_weight)
    if alpha_eff > 0.0 and w_in_pull.sum() >= 2.0:
        z_src_rev = grad_reverse(z_src, alpha=src_grl_ratio * alpha_eff)
        z_tgt_rev = grad_reverse(z_tgt, alpha=alpha_eff)

        logits_d_src = model.discriminator(z_src_rev)
        logits_d_tgt = model.discriminator(z_tgt_rev)

        lbl_src = torch.full_like(logits_d_src, 0.05)
        lbl_tgt = torch.full_like(logits_d_tgt, 0.95)

        loss_d_src_vec = F.binary_cross_entropy_with_logits(logits_d_src, lbl_src, reduction="none")
        if y_src is not None and tgt_pi_ema is not None and y_src.numel() == logits_d_src.numel():
            with torch.no_grad():
                counts_s = torch.bincount(y_src.long(), minlength=model.n_clusters).float().clamp(min=1.0)
                pi_s = counts_s / counts_s.sum()
                w_cls = (tgt_pi_ema.clamp(min=1e-3) / pi_s).clamp(min=0.2, max=5.0)
                w_src = w_cls[y_src.long()]
                w_src = w_src / w_src.mean().clamp(min=eps)
            loss_d_src = (w_src * loss_d_src_vec).mean()
        else:
            loss_d_src = loss_d_src_vec.mean()

        loss_d_tgt_vec = F.binary_cross_entropy_with_logits(logits_d_tgt, lbl_tgt, reduction="none")
        loss_d_tgt = (w_in_pull * loss_d_tgt_vec).sum() / (w_in_pull.sum() + eps)
        loss_adv = 0.5 * (loss_d_src + loss_d_tgt)

        with torch.no_grad():
            acc_d_src = (logits_d_src < 0.0).float().mean()
            acc_d_tgt = ((logits_d_tgt > 0.0).float() * w_in_pull).sum() / (w_in_pull.sum() + eps)
            disc_acc = 0.5 * (acc_d_src + acc_d_tgt).item()
    else:
        loss_adv = z_tgt.new_zeros(())
        disc_acc = 0.5

    # ---- 4. Lực HÚT Inlier sâu vào lõi cụm (xóa Cluster Kissing: r_in = 0.20 + 0.20 * maha) ----
    loss_ce = -(q * log_p).sum(dim=1)
    r_in = 0.20
    d_k = torch.sqrt(maha_mean + eps)
    excess_in = F.relu(d_k - r_in)
    loss_pull = (q * (0.20 * maha_mean + _pseudo_huber(excess_in, r_in))).sum(dim=1)

    if w_in_pull.sum() >= 1.0:
        inlier_denom = w_in_pull.sum().clamp(min=4.0)
        loss_inlier = (w_in_pull * (pull_weight * loss_pull + ce_weight * loss_ce)).sum() / inlier_denom
        pull_log = ((w_in_pull * loss_pull).sum() / inlier_denom).item()
    else:
        loss_inlier = z_tgt.new_zeros(())
        pull_log = 0.0

    # ---- 5. Lực ĐẨY OOD chủ động chỉ trên OOD tin cậy (w_ood_h) ----
    if repel_weight > 0.0 and w_ood_h.sum() >= 1.0:
        cos_margin = (0.30 * peak_lat).clamp(min=0.15, max=0.35)
        ang_violation = F.relu(sim_lat - cos_margin).pow(2).sum(dim=1)

        known_norm = mu_c.norm(dim=1).mean().clamp(min=1.0)
        tgt_norm = z_tgt.norm(dim=1)
        norm_violation = F.relu(tgt_norm - 0.55 * known_norm).pow(2) / (known_norm.pow(2) + eps)

        ood_denom = w_ood_h.sum().clamp(min=8.0)
        loss_repel = (w_ood_h * (3.0 * ang_violation + 1.0 * norm_violation)).sum() / ood_denom
    else:
        loss_repel = z_tgt.new_zeros(())

    # ---- 6. Khớp Tâm (loss_proto) & Phương sai nội cụm (loss_var) chỉ trên Inlier tin cậy ----
    qw = q * w_in_pull.unsqueeze(1)
    n_k = qw.sum(dim=0)
    denom = n_k.clamp(min=eps).unsqueeze(1)
    mean_k = (qw.T @ z_tgt) / denom

    active_k = n_k >= 3.0
    if active_k.any():
        loss_proto = F.mse_loss(mean_k[active_k], mu_c[active_k], reduction="sum") / active_k.sum().clamp(min=1)
    else:
        loss_proto = z_tgt.new_zeros(())

    loss_var = z_tgt.new_zeros(())
    if src_class_var is not None and var_weight > 0.0:
        sq = (z_tgt.unsqueeze(1) - mean_k.unsqueeze(0)).pow(2)
        var_k = ((qw.unsqueeze(2) * sq).sum(dim=0) / denom).mean(dim=1)
        ok = src_class_var_ok & (n_k >= var_min_n)
        if ok.any():
            loss_var = (torch.log(var_k[ok] + eps) - torch.log(src_class_var[ok].clamp(min=1e-2) + eps)).pow(2).mean()

    aux_align = (
        loss_inlier
        + repel_weight * loss_repel
        + adv_weight * radial_weight * loss_proto
        + var_weight * loss_var
    )

    if return_parts:
        return loss_adv, aux_align, {
            "adv": loss_adv.item(),
            "disc_acc": disc_acc,
            "var": loss_var.item(),
            "r": r_in,
            "pull": pull_log,
            "repel": loss_repel.item(),
            "ood_frac": ood_frac_log,
            "gate": lat_inlier_rate,
            "ot_in": ot_in_log,
            "n_k": n_k.detach(),
            "in_mask": (w_in_pull > 0.2).detach(),
        }
    return loss_adv + aux_align


gmm_contrastive_alignment_loss = gmm_adversarial_alignment_loss


@torch.no_grad()
def calibrate_source_stats(model, dataloader_rna, device, label_to_cluster=None):
    """
    Quét toàn bộ tập nguồn để chốt:
      - cluster_signatures, emp_mu_c, src_is_nonneg
      - src_ood_q95, src_ood_spread (cùng công thức 0.45*s_lat + 0.35*s_norm + 0.20*s_crec của predict_spatial)
      - src_ver_q95, src_ver_spread (cùng công thức 0.5*s_feat + 0.5*s_crec dùng trong huấn luyện).
    """
    was_training = model.training
    model.eval()
    ds = dataloader_rna.dataset
    full_loader = DataLoader(ds, batch_size=512, shuffle=False, drop_last=False)

    xs, mus, ys = [], [], []
    for x, y, _ in full_loader:
        x = x.to(device)
        y = y.to(device).long()
        if label_to_cluster is not None:
            y = label_to_cluster.to(device)[y]
        mu, _ = model.encoder(x, domain="src")
        xs.append(x)
        mus.append(mu)
        ys.append(y)

    X_all = torch.cat(xs, dim=0)
    mu_all = torch.cat(mus, dim=0)
    y_all = torch.cat(ys, dim=0)

    K = model.n_clusters
    D = model.latent_dim
    is_nonneg = bool((X_all.min() >= -1e-4).item())
    model.src_is_nonneg.fill_(is_nonneg)

    emp_mu = torch.zeros(K, D, device=device)
    sigs = torch.zeros(K, model.input_dim, device=device)
    for k in range(K):
        m = (y_all == k)
        if m.any():
            sigs[k] = X_all[m].mean(dim=0)
            emp_mu[k] = mu_all[m].mean(dim=0)

    model.cluster_signatures.copy_(sigs)
    model.emp_mu_c.copy_(emp_mu)

    # 1. Thống kê tập nguồn cho điểm OOD suy luận (0.45 s_lat + 0.35 s_norm + 0.20 s_crec)
    x_hat_all = model.decoder_src(mu_all)
    sim_lat_s = model.latent_cosine_sim(mu_all, mu_c=emp_mu).max(dim=1).values
    norm_s = mu_all.norm(dim=1)
    sim_crec_s = _pearson_sim(x_hat_all, X_all)
    sim_feat_s = _feature_cosine_sim(X_all, sigs, is_nonneg=is_nonneg).max(dim=1).values

    s_lat_s, _ = _log_attenuation_score(sim_lat_s, top_q=0.98, min_peak=0.15)
    s_norm_s, _ = _log_attenuation_score(norm_s, top_q=0.98, min_peak=0.10)
    s_crec_s, _ = _log_attenuation_score(sim_crec_s, top_q=0.98, min_peak=0.10)
    s_feat_s, _ = _log_attenuation_score(sim_feat_s, top_q=0.98, min_peak=0.10)

    s_ood_src = 0.45 * s_lat_s + 0.35 * s_norm_s + 0.20 * s_crec_s
    model.src_ood_q95.copy_(torch.quantile(s_ood_src, 0.95).clamp(min=0.15))
    model.src_ood_spread.copy_((torch.quantile(s_ood_src, 0.75) - torch.quantile(s_ood_src, 0.25)).clamp(min=0.05))

    # 2. Thống kê tập nguồn cho nhánh xác thực trong huấn luyện (0.5 s_feat + 0.5 s_crec)
    s_ver_src = 0.5 * s_feat_s + 0.5 * s_crec_s
    model.src_ver_q95.copy_(torch.quantile(s_ver_src, 0.95).clamp(min=0.15))
    model.src_ver_spread.copy_((torch.quantile(s_ver_src, 0.75) - torch.quantile(s_ver_src, 0.25)).clamp(min=0.05))

    if was_training:
        model.train()


@torch.no_grad()
def phase_diagnostics(model, dl_rna, dl_spa, device, label_to_cluster=None, n_batches=10):
    model.eval()
    mus_s, mus_t, ys = [], [], []
    for i, ((xr, yr, _), (xs, _, _)) in enumerate(zip(dl_rna, dl_spa)):
        if i >= n_batches:
            break
        mus_s.append(model.encoder(xr.to(device), domain="src")[0])
        mus_t.append(model.encoder(xs.to(device), domain="tgt")[0])
        yr = yr.to(device).long()
        ys.append(label_to_cluster[yr] if label_to_cluster is not None else yr)
    mu_s, mu_t, y = torch.cat(mus_s), torch.cat(mus_t), torch.cat(ys)

    D = model.latent_dim
    d_s = model.mahalanobis_sq(mu_s) / D
    d_t = model.mahalanobis_sq(mu_t) / D
    d_s_min, d_t_min = d_s.min(1).values, d_t.min(1).values
    d_thr = torch.quantile(d_s_min, 0.95).clamp(min=0.05)
    acc = (model.get_gamma(mu_s).argmax(1) == y).float().mean().item()

    sim_t = model.latent_cosine_sim(mu_t).max(1).values
    cos_q90 = torch.quantile(sim_t, 0.90).item()
    cos_q25 = torch.quantile(sim_t, 0.25).item()

    mu_c = model.mu_c
    dist = torch.cdist(mu_c, mu_c)
    dist.fill_diagonal_(float("inf"))
    sep = (dist.min() / mu_s.std(0).mean().clamp(min=1e-4)).item()

    rep = {
        "acc_src": acc,
        "sep": sep,
        "cos_q90/q25": f"{cos_q90:.2f}/{cos_q25:.2f}",
        "cos_tgt_q90": cos_q90,
        "d_thr": d_thr.item(),
        "dmin_tgt_med": d_t_min.median().item(),
        "ratio_tgt_thr": (d_t_min.median() / d_thr).item(),
        "std_src": mu_s.std(0).mean().item(),
        "std_tgt": mu_t.std(0).mean().item(),
    }
    model.train()
    return rep


# =====================================================================
# 3. GIAI ĐOẠN 1: PRETRAIN VAE & KHỞI TẠO THAM SỐ GMM
# =====================================================================

def pretrain_vae(model, dataloader_rna, optimizer, epochs, device,
                 dataloader_spa=None, kl_weight=1e-3):
    """Pretrain thuần tái tạo (+ KL nhẹ), không khớp mean/std toàn cục."""
    model.to(device)
    model.train()
    history = []
    spa_iter = _infinite_loader(dataloader_spa) if dataloader_spa is not None else None

    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x, _, _ in dataloader_rna:
            x = x.to(device)

            mu, log_var = model.encoder(x, domain="src")
            z = model.reparameterize(mu, log_var)
            x_hat = model.decoder_src(z)
            kl_src = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()
            loss = reconstruction_loss(x_hat, x) + kl_weight * kl_src

            if spa_iter is not None:
                x_spa, _, _ = next(spa_iter)
                x_spa = x_spa.to(device)
                mu_s, log_var_s = model.encoder(x_spa, domain="tgt")
                z_s = model.reparameterize(mu_s, log_var_s)
                x_hat_s = model.decoder_tgt(z_s)
                kl_tgt = -0.5 * (1.0 + log_var_s - mu_s.pow(2) - log_var_s.exp()).mean()
                loss = loss + reconstruction_loss(x_hat_s, x_spa) + kl_weight * kl_tgt

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(n_batches, 1)
        history.append(epoch_loss)
        print(f"[Pretrain] Epoch {epoch + 1}/{epochs} - recon_loss: {epoch_loss:.4f}")

    return history


def initialize_gmm_parameters(model, dataloader_rna, n_clusters, device,
                              warmup_steps=200, lr_hbeta=1e-3, seed=0):
    model.to(device)
    model.eval()

    ds = dataloader_rna.dataset
    full_loader = DataLoader(ds, batch_size=512, shuffle=False, drop_last=False)

    z_sq_sum = torch.zeros(n_clusters, model.latent_dim, device=device)
    counts = torch.zeros(n_clusters, device=device)
    y_list = []

    with torch.no_grad():
        for x, y, _ in full_loader:
            x = x.to(device)
            y = y.to(device).long()
            y_list.append(y)
            valid = (y >= 0) & (y < n_clusters)
            if valid.any():
                mu_v, _ = model.encoder(x[valid], domain="src")
                one_hot = F.one_hot(y[valid], num_classes=n_clusters).to(x.dtype)
                z_sq_sum += one_hot.T @ (mu_v ** 2)
                counts += one_hot.sum(dim=0)

        pi = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-6)
        pi = pi / pi.sum()
        model.pi_logits.data.copy_(torch.log(pi))

    calibrate_source_stats(model, dataloader_rna, device, label_to_cluster=None)

    emp_mu = model.emp_mu_c.clone()
    denom = counts.clamp(min=1.0).unsqueeze(1)
    emp_var = (z_sq_sum / denom - emp_mu ** 2).clamp(
        min=model.SIGMA_MIN ** 2 + 1e-4, max=model.SIGMA_MAX ** 2 - 1e-4
    )

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


# =====================================================================
# 4. GIAI ĐOẠN 2: HUẤN LUYỆN 3 PHA (A -> B -> C)
# =====================================================================

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

    cluster_ids = minority_classes if label_to_cluster is None else label_to_cluster.to(labels.device)[minority_classes]
    valid_mask = cluster_ids >= 0
    cluster_idx = torch.repeat_interleave(cluster_ids[valid_mask], n_needed[valid_mask])
    if cluster_idx.numel() == 0:
        return None

    return model.sample_augmented_latent(cluster_idx, cluster_idx.numel(), truncation=truncation).detach()


def train_phased_gmm_vae(model, dataloader_rna, dataloader_spa, optimizer,
                         lambda_1=1.5, lambda_2=2.0, device="cuda",
                         label_to_cluster=None, truncation=2.0,
                         # --- cấu hình pha ---
                         epochs_A=(6, 10), acc_stop_A=0.95,
                         epochs_B=(5, 8), ratio_stop_B=1.8,
                         epochs_C=15,
                         ramp_epochs=4,
                         # --- tham số alignment & adversarial ---
                         align_temp=0.10, radial_weight_B=1.5, radial_weight_C=2.0,
                         pull_weight=1.5, ce_weight=1.0,
                         var_weight=0.4, var_ema=0.9,
                         repel_weight=2.0,
                         dustbin_cost=0.56,        # Chỉ dùng dự phòng khi adaptive_ood=False
                         ood_prior=None,           # Không dùng mặc định (zero target leakage)
                         adaptive_ood=True,
                         grl_weight=0.15,          # Cường độ GRL dịu (IWAN); đặt 0.0 để tắt hẳn
                         use_latent_aug=False,
                         gene_guide_B=0.75, gene_guide_C=0.45,
                         lambda_cls=1.0, spread_weight=0.2,
                         free_radius=0.50, label_smoothing=0.05,
                         min_src_std=0.18,
                         # --- tham số báo cáo ---
                         total_var_rna=17.204, total_var_spa=22.754,
                         anchor_weight=0.2):
    model.to(device)
    model.train()

    opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing_disc_params = [p for p in model.discriminator.parameters() if id(p) not in opt_param_ids]
    if missing_disc_params:
        optimizer.add_param_group({"params": missing_disc_params})

    spa_iter = _infinite_loader(dataloader_spa)
    history = []
    if label_to_cluster is not None:
        label_to_cluster = label_to_cluster.to(device)

    K = model.n_clusters
    src_var = torch.ones(K, device=device) * 0.25
    src_ok = torch.zeros(K, dtype=torch.bool, device=device)
    tgt_pi_ema = torch.full((K,), 1.0 / K, device=device)

    steps_per_epoch = max(len(dataloader_rna), 1)
    phase, ep_in_phase, step_in_phase = "A", 0, 0
    total_epochs = epochs_A[1] + epochs_B[1] + epochs_C

    sum_keys = ["total", "recon", "gmm", "align", "adv", "disc_acc",
                "pull", "repel", "var", "ood_frac", "gate", "ot_in", "acc",
                "l_rna", "l_spa", "anchor", "mse_rna", "mse_spa"]

    sig_target = (
        model.cluster_signatures.clamp(0.0, 1.0)
        if getattr(model.decoder_src, "use_sigmoid", False)
        else model.cluster_signatures
    )
    verify_cfg = None

    for epoch in range(total_epochs):
        # Hiệu chuẩn ngưỡng Cổng 3 vùng (thr_in, thr_out) tự thích ứng đầu mỗi epoch Pha B/C
        if phase != "A" and adaptive_ood:
            new_vcfg = estimate_target_verify_thresholds(
                model, dataloader_spa.dataset, device,
                use_crec=(phase == "C"),
            )
            if verify_cfg is None or ep_in_phase == 0:
                verify_cfg = new_vcfg
            else:
                # Làm mượt EMA để ngưỡng không rung giữa các epoch
                for k_ema in ("thr_in", "thr_out", "peak_feat", "peak_crec"):
                    verify_cfg[k_ema] = 0.7 * verify_cfg[k_ema] + 0.3 * new_vcfg[k_ema]
                verify_cfg["mode"] = new_vcfg["mode"]
                verify_cfg["ashman_d"] = new_vcfg["ashman_d"]

        sums = {k: 0.0 for k in sum_keys}
        n_steps = 0

        for x_rna, y_rna, _ in dataloader_rna:
            x_spa, _, _ = next(spa_iter)
            x_rna, y_rna, x_spa = x_rna.to(device), y_rna.to(device), x_spa.to(device)

            out_rna = model(x_rna, "src")
            out_spa = model(x_spa, "tgt")

            # ---- 1. Reconstruction + GMM giám sát trên tập nguồn ----
            sig_hat = model.decoder_src(out_rna["mu_c"])
            loss_sig_anchor = reconstruction_loss(sig_hat, sig_target)
            l_rna = reconstruction_loss(out_rna["x_hat"], x_rna)
            l_spa = reconstruction_loss(out_spa["x_hat"], x_spa)
            loss_recon = l_rna + l_spa + anchor_weight * loss_sig_anchor

            mse_rna = mse_recon(out_rna["x_hat"].detach(), x_rna)
            mse_spa = mse_recon(out_spa["x_hat"].detach(), x_spa)

            cluster_rna = label_to_cluster[y_rna] if label_to_cluster is not None else y_rna
            loss_gmm, _ = supervised_gmm_loss_and_cls(
                out_rna["mu"], out_rna["log_var"], out_rna["z"],
                cluster_rna, model,
                lambda_cls=lambda_cls, free_radius=free_radius,
                label_smoothing=label_smoothing, min_src_std=min_src_std,
            )

            with torch.no_grad():
                v_b, ok_b = class_var_stats(out_rna["mu"].detach(), cluster_rna, K)
                first, upd = ok_b & ~src_ok, ok_b & src_ok
                src_var[first] = v_b[first]
                src_var[upd] = var_ema * src_var[upd] + (1.0 - var_ema) * v_b[upd]
                src_ok |= ok_b
                acc_b = (out_rna["gamma"].argmax(1) == cluster_rna).float().mean().item()

            loss_total = loss_recon + lambda_1 * loss_gmm

            # ---- 2. Xử lý căn chỉnh theo từng Pha ----
            if phase == "A":
                if grl_weight > 0.0:
                    logits_d_s = model.discriminator(out_rna["mu"].detach())
                    logits_d_t = model.discriminator(out_spa["mu"].detach())
                    loss_adv = 0.5 * (
                        F.binary_cross_entropy_with_logits(logits_d_s, torch.full_like(logits_d_s, 0.05))
                        + F.binary_cross_entropy_with_logits(logits_d_t, torch.full_like(logits_d_t, 0.95))
                    )
                    with torch.no_grad():
                        d_acc = 0.5 * ((logits_d_s < 0.0).float().mean() + (logits_d_t > 0.0).float().mean()).item()
                    loss_total = loss_total + loss_adv
                    loss_align = loss_adv.detach()
                else:
                    loss_adv = x_rna.new_zeros(())
                    loss_align = loss_adv
                    d_acc = 0.5

                a_parts = {"adv": loss_adv.item(), "disc_acc": d_acc,
                           "var": 0., "pull": 0., "repel": 0., "ood_frac": 0., "gate": 0., "ot_in": 0.}
            else:
                ramp = min(1.0, (step_in_phase + 1) / (ramp_epochs * steps_per_epoch))

                if use_latent_aug:
                    z_aug = generate_latent_augmentation(model, y_rna, label_to_cluster, truncation)
                    z_src_all = torch.cat([out_rna["mu"], z_aug], 0) if z_aug is not None else out_rna["mu"]
                    y_src_for_d = None
                else:
                    z_src_all = out_rna["mu"]
                    y_src_for_d = cluster_rna

                if phase == "B":
                    align_scale = ramp
                    cfg = dict(
                        use_crec_gate=False,
                        repel_weight=0.0,
                        var_weight=0.0,
                        adv_weight=radial_weight_B,
                        grl_alpha=grl_weight * align_scale,
                    )
                    gene_w = gene_guide_B
                else:
                    align_scale = 1.0
                    cfg = dict(
                        use_crec_gate=True,
                        repel_weight=repel_weight * ramp,
                        var_weight=var_weight * ramp,
                        adv_weight=radial_weight_C,
                        grl_alpha=grl_weight * align_scale,
                    )
                    gene_w = gene_guide_C

                loss_adv, aux_align, a_parts = gmm_adversarial_alignment_loss(
                    z_src=z_src_all, z_tgt=out_spa["mu"], model=model,
                    y_src=y_src_for_d, x_tgt=x_spa,
                    temperature=align_temp, n_src_real=out_rna["mu"].size(0),
                    pull_weight=pull_weight, ce_weight=ce_weight,
                    src_class_var=src_var, src_class_var_ok=src_ok,
                    gene_guide_weight=gene_w, dustbin_cost=dustbin_cost,
                    ood_prior=ood_prior, adaptive_ood=adaptive_ood,
                    verify_cfg=verify_cfg, tgt_pi_ema=tgt_pi_ema,
                    return_parts=True, **cfg
                )

                with torch.no_grad():
                    n_k_b = a_parts.pop("n_k")
                    in_mask = a_parts.pop("in_mask")
                    if n_k_b.sum() > 1.0:
                        pi_t_b = (n_k_b / n_k_b.sum()).clamp(min=1e-3)
                        pi_t_b = pi_t_b / pi_t_b.sum()
                        tgt_pi_ema = 0.9 * tgt_pi_ema + 0.1 * pi_t_b

                if in_mask.sum() >= 16:
                    std_s = out_rna["mu"].std(0).detach()
                    std_t = out_spa["mu"][in_mask].std(0)
                    loss_spread = F.relu(0.75 * std_s - std_t).pow(2).sum()
                else:
                    loss_spread = x_spa.new_zeros(())

                loss_align = loss_adv + lambda_2 * align_scale * (aux_align + spread_weight * loss_spread)
                loss_total = loss_total + loss_align

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            step_in_phase += 1

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["gmm"] += loss_gmm.item()
            sums["align"] += float(loss_align.item())
            sums["acc"] += acc_b
            sums["l_rna"] += l_rna.item()
            sums["l_spa"] += l_spa.item()
            sums["anchor"] += loss_sig_anchor.item()
            sums["mse_rna"] += mse_rna
            sums["mse_spa"] += mse_spa
            for k in ["adv", "disc_acc", "pull", "repel", "var", "ood_frac", "gate", "ot_in"]:
                sums[k] += a_parts[k]
            n_steps += 1

        stats = {k: v / max(n_steps, 1) for k, v in sums.items()}
        stats["r2_rna"] = 1.0 - stats["mse_rna"] / max(total_var_rna, 1e-8)
        stats["r2_spa"] = 1.0 - stats["mse_spa"] / max(total_var_spa, 1e-8)
        stats["phase"] = phase
        history.append(stats)
        ep_in_phase += 1

        msg = (f"[{phase}] Ep {epoch+1} | total {stats['total']:.3f} | recon {stats['recon']:.3f} "
               f"(rna {stats['l_rna']:.2f} spa {stats['l_spa']:.2f} anc {stats['anchor']:.2f}) | "
               f"gmm {stats['gmm']:.3f} | acc_src {stats['acc']:.1%} | d_acc {stats['disc_acc']:.1%} | "
               f"R2 rna/spa {stats['r2_rna']:.3f}/{stats['r2_spa']:.3f}")
        if phase != "A":
            thr_str = f" thr[{verify_cfg['thr_in']:.2f}/{verify_cfg['thr_out']:.2f}]" if verify_cfg is not None else ""
            msg += (f" | adv {stats['adv']:.3f} pull {stats['pull']:.3f} repel {stats['repel']:.3f} | "
                    f"lat_in {stats['gate']:.2f} ot_in {stats['ot_in']:.2f} est_OOD {stats['ood_frac']:.1%}{thr_str}")
        print(msg)

        # ---- Chuyển pha ----
        if phase == "A":
            if (ep_in_phase >= epochs_A[0] and stats["acc"] >= acc_stop_A) or ep_in_phase >= epochs_A[1]:
                calibrate_source_stats(model, dataloader_rna, device, label_to_cluster)
                rep = phase_diagnostics(model, dataloader_rna, dataloader_spa, device, label_to_cluster)
                print("  >>", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
                if epochs_B[1] == 0 and epochs_C == 0:
                    break
                phase, ep_in_phase, step_in_phase = "B", 0, 0
                verify_cfg = None
                print("=== Chuyển sang pha B (căn chỉnh Inlier bằng cổng 3 vùng tự thích ứng) ===")
        elif phase == "B":
            rep = phase_diagnostics(model, dataloader_rna, dataloader_spa, device, label_to_cluster)
            print("  >>", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
            if (ep_in_phase >= epochs_B[0] and rep["cos_tgt_q90"] >= 0.85 and rep["ratio_tgt_thr"] <= ratio_stop_B) or ep_in_phase >= epochs_B[1]:
                if epochs_C == 0:
                    break
                phase, ep_in_phase, step_in_phase = "C", 0, 0
                verify_cfg = None
                print("=== Chuyển sang pha C (khóa sinh ngược decoder_src + đẩy OOD tự thích ứng) ===")
        elif phase == "C" and ep_in_phase >= epochs_C:
            break

    calibrate_source_stats(model, dataloader_rna, device, label_to_cluster)
    return history


# =====================================================================
# 5. SUY LUẬN & PHÁT HIỆN OOD TỔNG QUÁT, KHÔNG RÒ RỈ (POST-TRAINING)
# =====================================================================

@torch.no_grad()
def extract_mu(model, dataset, device, batch_size=512, domain="tgt"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    return torch.cat([model.encoder(x.to(device), domain=domain)[0] for x, _, _ in loader], dim=0)


def build_spatial_neighbors(coords, k=15):
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k, max(len(coords) - 1, 1))
    nn_ = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
    _, idx = nn_.kneighbors(coords)
    return torch.as_tensor(idx[:, 1:], dtype=torch.long)


@torch.no_grad()
def compute_prior(model, mu, mode="spatial", nbr=None, alpha=0.5, floor=1e-3):
    K = model.n_clusters
    if mode == "learned":
        return model.pi.detach()

    uniform = torch.full((K,), 1.0 / K, device=mu.device)
    if mode == "uniform" or nbr is None:
        return uniform

    gamma0 = model.get_spherical_gamma(mu, pi=uniform, use_emp_mu=True)
    neigh = gamma0[nbr.to(mu.device)].mean(dim=1)
    prior = (1.0 - alpha) * uniform.unsqueeze(0) + alpha * neigh
    prior = prior.clamp(min=floor)
    return prior / prior.sum(dim=1, keepdim=True)


@torch.no_grad()
def predict_spatial(model, dataset_spa, device, nbr=None, prior_mode="spatial",
                    alpha=0.5, norm_ood_weight=0.35, crec_ood_weight=0.20,
                    spatial_ood_alpha=0.50, use_adaptive_thr=True,
                    batch_size=512, **kwargs):
    """
    Hàm suy luận & phát hiện OOD tổng quát (dùng chung cho ST lẫn MNIST):
      - Gán nhãn cụm Known bằng Spherical GMM (+ làm mịn không gian nếu nbr != None).
      - Điểm OOD tổng hợp trên thang nén log (-log / Q98):
            s_ood = w_lat * s_lat + norm_ood_weight * s_norm + crec_ood_weight * s_crec
        với mặc định (0.45 s_lat + 0.35 s_norm + 0.20 s_crec).
      - Làm mịn điểm OOD theo tọa độ không gian (tự động bỏ qua khi nbr=None như MNIST).
      - Tách ngưỡng tự thích ứng không rò rỉ bằng BIC + Ashman's D + 2-GMM(full) ML boundary.
    """
    model.to(device)
    model.eval()

    loader = DataLoader(dataset_spa, batch_size=batch_size, shuffle=False, drop_last=False)
    xs, mus = [], []
    for x, _, _ in loader:
        x = x.to(device)
        mu, _ = model.encoder(x, domain="tgt")
        xs.append(x)
        mus.append(mu)
    X_all = torch.cat(xs, dim=0)
    mu_all = torch.cat(mus, dim=0)

    # 1. Gán nhãn cụm Known bằng Spherical GMM (+ spatial prior nếu có nbr)
    effective_mode = prior_mode if nbr is not None else "uniform"
    prior = compute_prior(model, mu_all, mode=effective_mode, nbr=nbr, alpha=alpha)
    gamma = model.get_spherical_gamma(mu_all, pi=prior, use_emp_mu=True)
    H = -(gamma * torch.log(gamma + 1e-10)).sum(dim=1) / math.log(model.n_clusters)
    cluster = gamma.argmax(dim=1)

    # 2. Ba thành phần điểm OOD trên cùng thang nén log bất biến suy hao
    sim_lat = model.latent_cosine_sim(mu_all, mu_c=model.emp_mu_c).max(dim=1).values
    s_lat, _ = _log_attenuation_score(sim_lat, top_q=0.98, min_peak=0.15)

    z_norm = mu_all.norm(dim=1)
    s_norm, _ = _log_attenuation_score(z_norm, top_q=0.98, min_peak=0.10)

    if crec_ood_weight > 0.0:
        x_hat_src = model.decoder_src(mu_all)
        sim_crec = _pearson_sim(x_hat_src, X_all)
        s_crec, _ = _log_attenuation_score(sim_crec, top_q=0.98, min_peak=0.10)
    else:
        s_crec = torch.zeros_like(s_lat)

    lat_w = max(0.0, 1.0 - float(norm_ood_weight) - float(crec_ood_weight))
    ood_score = lat_w * s_lat + float(norm_ood_weight) * s_norm + float(crec_ood_weight) * s_crec

    # 3. Làm mịn điểm OOD theo tọa độ không gian (tự động bỏ qua khi nbr=None)
    if nbr is not None and effective_mode == "spatial" and spatial_ood_alpha > 0.0:
        nbr_dev = nbr.to(device)
        ood_score = (1.0 - spatial_ood_alpha) * ood_score + spatial_ood_alpha * ood_score[nbr_dev].mean(dim=1)

    # 4. Xác định ngưỡng OOD tự thích ứng không rò rỉ
    ood_np = ood_score.cpu().numpy()
    if use_adaptive_thr:
        ood_thr, thr_mode, ashman_d = _adaptive_ood_threshold(
            ood_np,
            src_q95=float(model.src_ood_q95.item()),
            src_spread=float(model.src_ood_spread.item()),
        )
    else:
        ood_thr, thr_mode, ashman_d = float(model.src_ood_q95.item() * 1.25), "source-q95", 0.0

    accept = ood_score < ood_thr
    pseudo = torch.where(accept, cluster, torch.full_like(cluster, -1))

    acc_ratio = accept.float().mean().item()
    print(
        f"[Inference] Known (accepted): {acc_ratio:.1%} | "
        f"Unknown (OOD): {1.0 - acc_ratio:.1%} | "
        f"ood_thr: {ood_thr:.3f} (mode={thr_mode}, Ashman_D={ashman_d:.2f})"
    )
    return {
        "pseudo_labels": pseudo,
        "cluster": cluster,
        "gamma": gamma,
        "entropy": H,
        "ood_score": ood_score,
        "s_lat": s_lat,
        "s_norm": s_norm,
        "s_crec": s_crec,
        "ood_thr": ood_thr,
        "accept": accept,
    }
