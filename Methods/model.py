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
        input_dim:   số đặc trưng đầu vào (số gene chung hoặc số pixel).
        hidden_dims: danh sách kích thước các lớp ẩn, giảm dần.
        latent_dim:  số chiều không gian ẩn.
        dropout:     tỉ lệ dropout trong các khối ẩn.
        norm_type:   'layernorm' (mặc định, bất biến với label shift) hoặc 'dsbn'.
    """

    def __init__(self, input_dim, hidden_dims=(512, 256, 128), latent_dim=32,
                 dropout=0.1, norm_type="layernorm"):
        super().__init__()
        self.input_dim = input_dim
        self.norm_type = norm_type

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

        # Giữ thuộc tính backbone cho tương thích ngược nếu cần truy xuất in_features
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
        h = x
        for lin, norm, act, drop in zip(self.linears, self.norms, self.acts, self.drops):
            h = lin(h)
            h = norm(h, domain) if self.norm_type == "dsbn" else norm(h)
            h = drop(act(h))
        mu = self.fc_mu(h)
        log_var = torch.clamp(self.fc_log_var(h), min=-6.0, max=4.0)
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
        latent_dim:  số chiều không gian ẩn.
        hidden_dims: kích thước các lớp ẩn, TĂNG dần.
        output_dim:  số chiều đầu ra (bằng input_dim của Encoder).
        dropout:     tỉ lệ dropout (mặc định 0).
        out_act:     'linear' (cho dữ liệu Z-score / chuẩn hóa tổng quát)
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
        """Tạo Decoder đối xứng với Encoder bằng cách đảo ngược hidden_dims."""
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
    GMM-VAE tổng quát tích hợp:
      - Prior GMM sinh từ chữ ký cụm qua mạng h_beta.
      - Bộ đệm thống kê thực nghiệm (emp_mu_c, emp_var_c, src_lat_q95, src_feat_high/low)
        giúp hiệu chuẩn OOD không rò rỉ dữ liệu target và miễn nhiễm với sụp đổ phương sai.
      - DomainDiscriminator tùy chọn cho căn chỉnh đối kháng có trọng số (IWAN).
    """
    SIGMA_MIN, SIGMA_MAX = 0.25, 0.75

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

        # Thống kê thực nghiệm trên tập nguồn (cập nhật khi khởi tạo, EMA trong train, chốt sau train)
        self.register_buffer("emp_mu_c", torch.zeros(n_clusters, latent_dim))
        self.register_buffer("emp_var_c", torch.ones(n_clusters, latent_dim) * 0.25)
        self.register_buffer("src_lat_q95", torch.tensor(1.5))
        self.register_buffer("src_feat_high", torch.tensor(0.7))
        self.register_buffer("src_feat_low", torch.tensor(0.3))
        self.register_buffer("src_is_nonneg", torch.tensor(False))

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

    def get_log_posterior(self, z, pi=None, temperature=1.0, mu_c=None, var_c=None,
                          use_uniform_prior=True, use_emp_stats=False):
        if mu_c is None or var_c is None:
            if use_emp_stats:
                mu_c, var_c = self.emp_mu_c, self.emp_var_c
            else:
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

    def get_gamma(self, z, pi=None, temperature=1.0, use_uniform_prior=True, use_emp_stats=False):
        log_post = self.get_log_posterior(
            z, pi=pi, temperature=temperature,
            use_uniform_prior=use_uniform_prior, use_emp_stats=use_emp_stats
        )
        return log_post.exp()

    def mahalanobis_sq(self, z):
        mu_c, var_c = self.compute_gmm_params()
        diff = z.unsqueeze(1) - mu_c.unsqueeze(0)
        return (diff.pow(2) / (var_c.unsqueeze(0) + 1e-6)).sum(dim=2)

    def emp_mahalanobis_sq(self, z):
        """Khoảng cách Mahalanobis bình phương chuẩn hóa theo phương sai thực nghiệm tập nguồn."""
        diff = z.unsqueeze(1) - self.emp_mu_c.unsqueeze(0)
        var_safe = self.emp_var_c.unsqueeze(0).clamp(min=1e-2)
        return (diff.pow(2) / var_safe).sum(dim=2)

    @torch.no_grad()
    def sample_augmented_latent(self, cluster_idx, num_samples, truncation=2.0):
        mu_c = self.emp_mu_c
        var_c = self.emp_var_c.clamp(min=1e-2)
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
    """
    Tạo DataLoader lấy mẫu cân bằng lớp cho tập tham chiếu có nhãn (chống Label Shift).
    power=1.0: cân bằng tuyệt đối; power=0.85: cân bằng mềm tránh overfit lớp quá nhỏ.
    """
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


def _feature_cosine_sim(x, signatures, is_nonneg=False, eps=1e-8):
    """
    Tính độ tương đồng Cosine giữa mẫu x (B, G) và chữ ký cụm signatures (K, G).
    - Nếu dữ liệu đã chuẩn hóa 0-mean (Z-score): tính Cosine trực tiếp.
    - Nếu dữ liệu không âm ([0, 1] hoặc log1p): trừ trung bình trước khi chuẩn hóa L2.
    """
    if is_nonneg:
        x_use = x - x.mean(dim=0, keepdim=True)
        sig_use = signatures - signatures.mean(dim=0, keepdim=True)
    else:
        x_use = x
        sig_use = signatures
    x_norm = F.normalize(x_use, p=2, dim=1, eps=eps)
    sig_norm = F.normalize(sig_use, p=2, dim=1, eps=eps)
    return x_norm @ sig_norm.T


def reconstruction_loss(x_hat, x, eps=1e-7):
    """
    Hàm tái tạo bất biến theo số chiều đặc trưng và thang đo (Z-score hoặc [0, 1]):
    10 * (MSE / Var_x) + 10 * Cosine_Distance -> luôn ở thang ~2-20 trên mọi bộ dữ liệu.
    """
    x_var = x.detach().var(dim=0, unbiased=False).mean().clamp(min=0.02)
    mse_norm = F.mse_loss(x_hat, x, reduction="mean") / x_var
    cos_dist = 1.0 - F.cosine_similarity(x_hat, x, dim=1, eps=eps).mean()
    return 10.0 * mse_norm + 10.0 * cos_dist


@torch.no_grad()
def mse_recon(x_hat, x):
    """MSE tổng trên từng mẫu, dùng tính R² = 1 - mse_recon / total_var."""
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
                                lambda_cls=1.0, free_radius=0.35, eps=1e-8):
    """
    Supervised GMM KL có cơ chế free-bits (hinge lõi mềm) chống sụp đổ tâm mu_src:
    - Ngoài bán kính free_radius * sigma_c: kéo mạnh về tâm cụm.
    - Trong vùng lõi: lực kéo dịu (0.05x) để giữ phương sai sinh học nội cụm tự nhiên.
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
    mean_term = 0.05 * sq_norm + 0.95 * F.relu(sq_norm - free_radius ** 2)
    var_term = torch.log(target_var + eps) + tilde_var / (target_var + eps) - 1.0 - log_var

    loss_align = 0.5 * (var_term + mean_term).sum(dim=1).mean()
    log_p_c_given_z = model.get_log_posterior(z_sample, mu_c=mu_c, var_c=var_c)
    loss_cls = F.nll_loss(log_p_c_given_z, y_true)

    total_sup_gmm = loss_align + lambda_cls * loss_cls
    return total_sup_gmm, {"align_kl": loss_align.item(), "cls_ce": loss_cls.item()}


def _pseudo_huber(x, delta):
    return delta ** 2 * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


@torch.no_grad()
def class_var_stats(mu, y, n_clusters, min_n=3):
    """Phương sai nội cụm theo chiều (K, D) và trung bình (K,) -> (var_mean[K], var_dim[K, D], ok[K])."""
    D = mu.size(1)
    var_mean = mu.new_zeros(n_clusters)
    var_dim = mu.new_zeros(n_clusters, D)
    ok = torch.zeros(n_clusters, dtype=torch.bool, device=mu.device)
    for k in range(n_clusters):
        m = mu[y == k]
        if m.size(0) >= min_n:
            vd = m.var(dim=0, unbiased=True)
            var_dim[k] = vd
            var_mean[k] = vd.mean()
            ok[k] = True
    return var_mean, var_dim, ok


@torch.no_grad()
def _sinkhorn_with_dustbin(cost_matrix, dustbin_cost=1.0, reg=0.08,
                           ood_prior=None, cluster_prior=None, n_iters=15,
                           adaptive_ood=True):
    """
    Entropic Unbalanced OT với cột Dustbin hiệu chuẩn không rò rỉ (Zero Target Leakage).

    - Khi adaptive_ood=True (mặc định):
        Không ép cột Dustbin nhận một tỉ lệ ood_prior cố định và không ép các cụm Known
        chia đều mẫu (tránh kéo OOD vào cụm vắng mặt hoặc đẩy Known vào Dustbin dưới Label Shift).
        Nghiệm Gibbs khép kín: mẫu rơi vào Dustbin khi chi phí nhỏ nhất vượt ngưỡng hiệu chuẩn dustbin_cost.
    - Khi adaptive_ood=False và ood_prior được truyền: giữ tương thích ngược kiểu Sinkhorn mềm.
    """
    N_t, K = cost_matrix.shape
    device = cost_matrix.device

    if not torch.is_tensor(dustbin_cost):
        dustbin_col = torch.full((N_t, 1), float(dustbin_cost), device=device)
    else:
        dustbin_col = dustbin_cost.view(-1, 1).expand(N_t, 1)

    C_ext = torch.cat([cost_matrix, dustbin_col], dim=1)  # (N_t, K + 1)
    log_P = -C_ext / max(reg, 1e-4)

    if adaptive_ood or ood_prior is None:
        P = F.softmax(log_P, dim=1)
        w_ood = P[:, K].clamp(min=0.0, max=1.0)
        w_in = (1.0 - w_ood).clamp(min=0.0, max=1.0)
        q_known = F.softmax(-cost_matrix / max(reg, 1e-4), dim=1)
        return q_known, w_in, w_ood

    # Nhánh tương thích ngược nếu người dùng chủ động truyền adaptive_ood=False
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
    pull_weight=1.0,
    ce_weight=0.5,
    pull_quantile=0.50,
    gate_quantile=0.92,
    gate_softness=0.30,
    src_class_var=None,
    src_class_var_ok=None,
    var_weight=0.5,
    var_min_n=5,
    gene_guide_weight=0.5,
    repel_weight=1.0,
    ood_prior=None,
    adaptive_ood=True,
    use_latent_gate=True,
    tgt_pi_ema=None,
    return_parts=False,
    **kwargs
):
    """
    Hàm căn chỉnh Open-Set không rò rỉ target:
      1. Khoảng cách Mahalanobis chuẩn hóa bằng phương sai thực nghiệm (emp_var_c) -> bất biến với độ co cụm.
      2. Chi phí đặc trưng (cost_gene) tự động hiệu chuẩn độ suy hao xuyên miền (attenuation rho).
      3. Dustbin OT không ép tỷ lệ OOD cứng (adaptive_ood=True).
      4. Học đối kháng có trọng số theo tỷ lệ lớp ước lượng của tập target (IWAN) chống Label Shift.
    """
    D = z_tgt.size(1)
    z_src_d = z_src.detach()
    mu_c = model.mu_c.detach()
    emp_var_c = model.emp_var_c.detach().clamp(min=1e-2)

    # ---- 1. Khoảng cách Mahalanobis chuẩn hóa theo phương sai thực nghiệm ----
    diff = z_tgt.unsqueeze(1) - mu_c.unsqueeze(0)                        # (N_t, K, D)
    maha_mean = (diff.pow(2) / (emp_var_c.unsqueeze(0) + eps)).mean(dim=2)  # (N_t, K)

    scale_maha = maha_mean.detach().median(dim=1, keepdim=True).values.clamp(min=0.2)
    logits = -0.5 * (maha_mean / scale_maha) / temperature
    log_p = torch.log_softmax(logits, dim=1)

    # ---- 2. Hiệu chuẩn chi phí Latent + Feature và giải Dustbin OT ----
    with torch.no_grad():
        z_real = z_src_d if n_src_real is None else z_src_d[:n_src_real]
        d_real = ((z_real.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2)
                  / (emp_var_c.unsqueeze(0) + eps)).mean(dim=2)
        d_real_min = d_real.min(dim=1).values

        # Không còn clamp(min=0.5) cứng; ngưỡng bám sát phân bố thực tế của tập nguồn
        r_in = torch.quantile(d_real_min, pull_quantile).sqrt().clamp(min=0.25, max=1.2)
        d_thr = torch.quantile(d_real_min, gate_quantile).clamp(min=0.25)
        r_out = max(d_thr.sqrt().item() * 1.35, 1.35)

        cost_latent = maha_mean / (d_thr + eps)

        if x_tgt is not None and gene_guide_weight > 0.0:
            sim_tgt = _feature_cosine_sim(
                x_tgt.detach(),
                model.cluster_signatures.detach(),
                is_nonneg=bool(model.src_is_nonneg.item()),
                eps=eps,
            )
            max_sim_tgt = sim_tgt.max(dim=1).values
            # Hiệu chuẩn độ suy hao xuyên miền từ đỉnh phân vị Q90 (luôn nằm trong lõi Known)
            t_high = torch.quantile(max_sim_tgt, 0.90).clamp(min=0.10)
            s_high = model.src_feat_high.clamp(min=0.10)
            s_low = model.src_feat_low
            rel_ratio = (s_low / s_high).clamp(min=0.15, max=0.85)
            t_low = (t_high * rel_ratio).clamp(min=-0.20, max=0.90)

            gene_thr = (1.0 - t_low).clamp(min=0.10)
            cost_gene = (1.0 - sim_tgt) / gene_thr
            combined_cost = (1.0 - gene_guide_weight) * cost_latent + gene_guide_weight * cost_gene
        else:
            combined_cost = cost_latent

        q, w_in_ot, w_ood_ot = _sinkhorn_with_dustbin(
            combined_cost,
            dustbin_cost=1.0,
            reg=max(temperature, 0.08),
            ood_prior=ood_prior,
            adaptive_ood=adaptive_ood,
        )

        min_d = maha_mean.min(dim=1).values
        w_gate = torch.sigmoid((d_thr - min_d) / (gate_softness * d_thr + eps))

        if use_latent_gate:
            w_in = torch.minimum(w_in_ot, w_gate).clamp(min=0.0, max=1.0)
        else:
            # Ở Pha B: chỉ dùng w_in_ot (dẫn đường mạnh bởi đặc trưng gốc), chưa siết bằng khoảng cách latent
            w_in = w_in_ot.clamp(min=0.0, max=1.0)

        w_ood = (1.0 - w_in).clamp(min=0.0, max=1.0)

    # ---- 3. Học đối kháng có trọng số chống Label Shift (IWAN) ----
    alpha_eff = float(grl_alpha * adv_weight)
    if alpha_eff > 0.0:
        z_src_rev = grad_reverse(z_src, alpha=src_grl_ratio * alpha_eff)
        z_tgt_rev = grad_reverse(z_tgt, alpha=alpha_eff)

        logits_d_src = model.discriminator(z_src_rev)
        logits_d_tgt = model.discriminator(z_tgt_rev)

        lbl_src = torch.full_like(logits_d_src, 0.05)
        lbl_tgt = torch.full_like(logits_d_tgt, 0.95)

        loss_d_src_vec = F.binary_cross_entropy_with_logits(logits_d_src, lbl_src, reduction="none")
        # Đánh trọng số nguồn theo phân bố cụm ước lượng của tập target inlier
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
        loss_d_tgt = (w_in * loss_d_tgt_vec).sum() / (w_in.sum() + eps)
        loss_adv = 0.5 * (loss_d_src + loss_d_tgt)

        with torch.no_grad():
            acc_d_src = (logits_d_src < 0.0).float().mean()
            acc_d_tgt = (logits_d_tgt > 0.0).float().mean()
            disc_acc = 0.5 * (acc_d_src + acc_d_tgt).item()
    else:
        loss_adv = z_tgt.new_zeros(())
        disc_acc = 0.5

    # ---- 4. Lực HÚT về tâm cụm chỉ trên các mẫu Inlier ----
    loss_ce = -(q * log_p).sum(dim=1)
    d_k = torch.sqrt(maha_mean + eps)
    excess_in = F.relu(d_k - r_in)
    loss_pull = (q * _pseudo_huber(excess_in, r_in)).sum(dim=1)
    loss_inlier = (w_in * (pull_weight * loss_pull + ce_weight * loss_ce)).sum() / (w_in.sum() + eps)

    # ---- 5. Lực ĐẨY (Repulsive Margin) đẩy OOD ra khỏi biên cụm Known ----
    d_min = d_k.min(dim=1).values
    margin_violation = F.relu(r_out - d_min)
    w_ood_h = ((w_ood - 0.45) / 0.55).clamp(0.0, 1.0)
    loss_repel = (w_ood_h * margin_violation.pow(2)).sum() / float(w_ood_h.numel())

    # ---- 6. Khớp Tâm Từng Cụm (loss_proto) & Phương sai nội cụm (loss_var) ----
    qw = q * w_in.unsqueeze(1)
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
            loss_var = (torch.log(var_k[ok] + eps) - torch.log(src_class_var[ok] + eps)).pow(2).mean()

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
            "r": r_in.item(),
            "pull": ((w_in * loss_pull).sum() / (w_in.sum() + eps)).item(),
            "repel": loss_repel.item(),
            "ood_frac": (w_ood > 0.5).float().mean().item(),
            "gate": w_gate.mean().item(),
            "ot_in": w_in_ot.mean().item(),
            "d_thr": d_thr.item(),
            "dmin_med": min_d.median().item(),
            "n_k": n_k.detach(),
        }
    return loss_adv + aux_align


gmm_contrastive_alignment_loss = gmm_adversarial_alignment_loss


@torch.no_grad()
def calibrate_source_stats(model, dataloader_rna, device, label_to_cluster=None):
    """
    Quét toàn bộ tập nguồn một lần (không lặp, không drop_last) ở chế độ eval()
    để chốt chính xác:
      - emp_mu_c, emp_var_c (tâm và phương sai thực nghiệm không gian ẩn)
      - src_lat_q95 (phân vị 95 khoảng cách Mahalanobis thực nghiệm của tập nguồn)
      - src_feat_high, src_feat_low (ngưỡng tương đồng Cosine không gian đặc trưng gốc)
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
    emp_var = torch.ones(K, D, device=device) * 0.25
    sigs = torch.zeros(K, model.input_dim, device=device)

    for k in range(K):
        m = (y_all == k)
        if m.any():
            sigs[k] = X_all[m].mean(dim=0)
            emp_mu[k] = mu_all[m].mean(dim=0)
            if m.sum() >= 2:
                emp_var[k] = mu_all[m].var(dim=0, unbiased=True).clamp(min=1e-2)

    model.cluster_signatures.copy_(sigs)
    model.emp_mu_c.copy_(emp_mu)
    model.emp_var_c.copy_(emp_var)

    # Phân vị 95 khoảng cách Mahalanobis thực nghiệm (chuẩn hóa theo D) trên tập nguồn
    valid = (y_all >= 0) & (y_all < K)
    diff_self = mu_all[valid] - emp_mu[y_all[valid]]
    d_self = (diff_self.pow(2) / emp_var[y_all[valid]].clamp(min=1e-2)).mean(dim=1)
    model.src_lat_q95.copy_(torch.quantile(d_self, 0.95).clamp(min=0.5))

    # Thống kê tương đồng Cosine theo từng lớp trên không gian đặc trưng gốc
    sim_all = _feature_cosine_sim(X_all[valid], sigs, is_nonneg=is_nonneg)
    sim_own = sim_all.gather(1, y_all[valid].unsqueeze(1)).squeeze(1)

    highs, lows = [], []
    for k in range(K):
        mk = (y_all[valid] == k)
        if mk.sum() >= 5:
            highs.append(torch.quantile(sim_own[mk], 0.90))
            lows.append(torch.quantile(sim_own[mk], 0.10))
    if highs:
        model.src_feat_high.copy_(torch.stack(highs).median().clamp(min=0.10))
        model.src_feat_low.copy_(torch.stack(lows).median())

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
    d_s = model.emp_mahalanobis_sq(mu_s) / D
    d_t = model.emp_mahalanobis_sq(mu_t) / D
    d_s_min, d_t_min = d_s.min(1).values, d_t.min(1).values
    d_thr = torch.quantile(d_s_min, 0.95).clamp(min=1e-2)
    acc = (model.get_gamma(mu_s).argmax(1) == y).float().mean().item()

    mu_c = model.mu_c
    dist = torch.cdist(mu_c, mu_c)
    dist.fill_diagonal_(float("inf"))
    sep = (dist.min() / mu_s.std(0).mean().clamp(min=1e-4)).item()

    rep = {
        "acc_src": acc,
        "sep": sep,
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
    """
    Pretrain thuần tái tạo (+ KL rất nhẹ giữ quy mô không gian ẩn ổn định).
    Đã GỠ BỎ hoàn toàn loss_anchor (khớp mean/std toàn cục) để không đẩy
    dữ liệu target vào vùng lớp đa số của tập nguồn dưới điều kiện Label Shift.
    """
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
    """
    1. Quét toàn bộ tập nguồn để tính chữ ký đặc trưng, tâm & phương sai thực nghiệm,
       và ngưỡng hiệu chuẩn OOD (zero leakage).
    2. Khớp ấm mạng h_beta với (emp_mu_c, emp_var_c).
    """
    model.to(device)
    model.eval()

    ds = dataloader_rna.dataset
    full_loader = DataLoader(ds, batch_size=512, shuffle=False, drop_last=False)

    counts = torch.zeros(n_clusters, device=device)
    y_list = []
    with torch.no_grad():
        for _, y, _ in full_loader:
            y = y.to(device).long()
            y_list.append(y)
            valid = (y >= 0) & (y < n_clusters)
            if valid.any():
                counts += torch.bincount(y[valid], minlength=n_clusters).float()

        pi = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-6)
        pi = pi / pi.sum()
        model.pi_logits.data.copy_(torch.log(pi))

    calibrate_source_stats(model, dataloader_rna, device, label_to_cluster=None)

    emp_mu = model.emp_mu_c.clone()
    emp_var = model.emp_var_c.clamp(
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


def train_phased_gmm_vae(model, dataloader_rna, dataloader_spa, optimizer,
                         lambda_1=1.5, lambda_2=2.0, device="cuda",
                         label_to_cluster=None, truncation=2.0,
                         # --- cấu hình pha ---
                         epochs_A=(6, 12), acc_stop_A=0.95,
                         epochs_B=(5, 10), ratio_stop_B=1.8,
                         epochs_C=15,
                         ramp_epochs=4,
                         # --- tham số alignment & adversarial ---
                         align_temp=0.10, radial_weight_B=1.0, radial_weight_C=1.0,
                         pull_weight=1.0, ce_weight=0.5, pull_quantile=0.50,
                         var_weight=0.5, var_ema=0.9,
                         gate_quantile=0.92, gate_softness=0.30,
                         repel_weight=1.5,
                         ood_prior=None,           # Không bắt buộc; chỉ dùng khi adaptive_ood=False
                         adaptive_ood=True,        # Mặc định True: không rò rỉ tỉ lệ OOD tập target
                         grl_weight=0.15,          # Cường độ GRL dịu tách biệt với lambda_2 (đặt 0.0 để tắt GRL)
                         use_latent_aug=False,     # Đặt False khi đã dùng make_balanced_loader
                         gene_guide_B=0.70, gene_guide_C=0.45,
                         lambda_cls=1.0, spread_weight=0.3,
                         free_radius=0.35,
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
    src_var = model.emp_var_c.mean(dim=1).clone()
    src_var_dim = model.emp_var_c.clone()
    src_ok = torch.ones(K, dtype=torch.bool, device=device)
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

    for epoch in range(total_epochs):
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
                cluster_rna, model, lambda_cls=lambda_cls, free_radius=free_radius
            )

            # Cập nhật thống kê phương sai & tâm thực nghiệm của tập nguồn theo EMA
            with torch.no_grad():
                v_mean_b, v_dim_b, ok_b = class_var_stats(out_rna["mu"].detach(), cluster_rna, K)
                src_var[ok_b] = var_ema * src_var[ok_b] + (1.0 - var_ema) * v_mean_b[ok_b]
                src_var_dim[ok_b] = var_ema * src_var_dim[ok_b] + (1.0 - var_ema) * v_dim_b[ok_b]
                src_ok |= ok_b
                model.emp_var_c.copy_(src_var_dim.clamp(min=1e-2))
                model.emp_mu_c.copy_(out_rna["mu_c"].detach())
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
                        use_latent_gate=False,
                        repel_weight=0.0,
                        var_weight=0.0,
                        adv_weight=radial_weight_B,
                        grl_alpha=grl_weight * align_scale,
                    )
                    gene_w = gene_guide_B
                else:  # phase C
                    align_scale = 1.0
                    cfg = dict(
                        use_latent_gate=True,
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
                    pull_quantile=pull_quantile, gate_quantile=gate_quantile,
                    gate_softness=gate_softness,
                    src_class_var=src_var, src_class_var_ok=src_ok,
                    gene_guide_weight=gene_w, ood_prior=ood_prior,
                    adaptive_ood=adaptive_ood, tgt_pi_ema=tgt_pi_ema,
                    return_parts=True, **cfg
                )

                with torch.no_grad():
                    n_k_b = a_parts.pop("n_k")
                    if n_k_b.sum() > 1.0:
                        pi_t_b = (n_k_b / n_k_b.sum()).clamp(min=1e-3)
                        pi_t_b = pi_t_b / pi_t_b.sum()
                        tgt_pi_ema = 0.9 * tgt_pi_ema + 0.1 * pi_t_b

                std_s = out_rna["mu"].std(0).detach()
                std_t = out_spa["mu"].std(0)
                loss_spread = F.relu(0.75 * std_s - std_t).pow(2).sum()

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
            msg += (f" | adv {stats['adv']:.3f} pull {stats['pull']:.3f} repel {stats['repel']:.3f} | "
                    f"gate {stats['gate']:.2f} ot_in {stats['ot_in']:.2f} est_OOD {stats['ood_frac']:.1%}")
        print(msg)

        # ---- Chuyển pha ----
        if phase == "A":
            if (ep_in_phase >= epochs_A[0] and stats["acc"] >= acc_stop_A) or ep_in_phase >= epochs_A[1]:
                calibrate_source_stats(model, dataloader_rna, device, label_to_cluster)
                rep = phase_diagnostics(model, dataloader_rna, dataloader_spa, device, label_to_cluster)
                print("  >>", {k: round(v, 3) for k, v in rep.items()})
                if epochs_B[1] == 0 and epochs_C == 0:
                    break
                phase, ep_in_phase, step_in_phase = "B", 0, 0
                print("=== Chuyển sang pha B (căn chỉnh Inlier dẫn đường đặc trưng, chưa đẩy OOD) ===")
        elif phase == "B":
            rep = phase_diagnostics(model, dataloader_rna, dataloader_spa, device, label_to_cluster)
            print("  >>", {k: round(v, 3) for k, v in rep.items()})
            if (ep_in_phase >= epochs_B[0] and rep["ratio_tgt_thr"] <= ratio_stop_B) or ep_in_phase >= epochs_B[1]:
                if epochs_C == 0:
                    break
                phase, ep_in_phase, step_in_phase = "C", 0, 0
                print("=== Chuyển sang pha C (căn chỉnh có chọn lọc Inlier + đẩy OOD) ===")
        elif phase == "C" and ep_in_phase >= epochs_C:
            break

    # Chốt thống kê tập nguồn chính xác 1 lần cuối sau khi huấn luyện xong
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

    gamma0 = model.get_gamma(mu, pi=uniform, use_emp_stats=True)
    neigh = gamma0[nbr.to(mu.device)].mean(dim=1)
    prior = (1.0 - alpha) * uniform.unsqueeze(0) + alpha * neigh
    prior = prior.clamp(min=floor)
    return prior / prior.sum(dim=1, keepdim=True)


def _adaptive_ood_threshold(scores_np, calibrated_thr=1.0, ashman_min=2.0,
                            thr_bounds=(0.75, 1.35)):
    """
    Xác định ngưỡng OOD không giám sát (không cần biết trước tỉ lệ OOD):
      - Khớp GMM 2 thành phần trên phân phối điểm OOD đã hiệu chuẩn.
      - Nếu hai đỉnh tách biệt rõ (Ashman's D >= 2.0 và mỗi cụm >= 3% mẫu): lấy giao điểm Bayes.
      - Nếu đơn đỉnh (ví dụ bài toán Closed-Set 0% Unknown): trả về ngưỡng hiệu chuẩn từ tập nguồn.
    """
    vals = np.asarray(scores_np, dtype=np.float64).reshape(-1, 1)
    if len(vals) < 30 or np.std(vals) < 1e-6:
        return float(calibrated_thr), "calibrated", 0.0

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42, n_init=5)
    gmm.fit(vals)
    means = gmm.means_.ravel()
    vars_ = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()

    order = np.argsort(means)
    m0, m1 = means[order[0]], means[order[1]]
    v0, v1 = max(vars_[order[0]], 1e-8), max(vars_[order[1]], 1e-8)
    w0, w1 = weights[order[0]], weights[order[1]]

    ashman_d = math.sqrt(2.0) * abs(m1 - m0) / math.sqrt(v0 + v1)

    if ashman_d >= ashman_min and min(w0, w1) >= 0.03:
        grid = np.linspace(m0, m1, 1000).reshape(-1, 1)
        probs = gmm.predict_proba(grid)[:, order[1]]  # Xác suất thuộc cụm OOD (đỉnh cao hơn)
        cross_idx = int(np.argmin(np.abs(probs - 0.5)))
        thr_gmm = float(grid[cross_idx, 0])
        thr_final = float(np.clip(thr_gmm, thr_bounds[0], thr_bounds[1]))
        return thr_final, "2-GMM", float(ashman_d)

    return float(calibrated_thr), "calibrated", float(ashman_d)


@torch.no_grad()
def predict_spatial(model, dataset_spa, device, nbr=None, prior_mode="spatial",
                    alpha=0.5, feat_ood_weight=0.45, spatial_ood_alpha=0.5,
                    calibrated_thr=1.05, use_adaptive_thr=True,
                    batch_size=512):
    """
    Hàm suy luận & phát hiện OOD tổng quát (dùng được cho cả Spatial Transcriptomics lẫn MNIST):
      - Gán nhãn cụm dựa trên phân phối hậu nghiệm thực nghiệm (+ làm mịn không gian nếu có nbr).
      - Tính điểm OOD kép (Latent Mahalanobis thực nghiệm + Feature Cosine hiệu chuẩn độ suy hao).
      - Làm mịn điểm OOD theo láng giềng không gian (nếu nbr != None và prior_mode == 'spatial').
      - Tách ngưỡng không rò rỉ bằng Ashman's D + 2-GMM / ngưỡng hiệu chuẩn tập nguồn.
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

    # 1. Gán nhãn cụm Known bằng thống kê thực nghiệm (+ ưu tiên không gian nếu có)
    effective_mode = prior_mode if nbr is not None else "uniform"
    prior = compute_prior(model, mu_all, mode=effective_mode, nbr=nbr, alpha=alpha)
    gamma = model.get_gamma(mu_all, pi=prior, use_emp_stats=True)
    H = -(gamma * torch.log(gamma + 1e-10)).sum(dim=1) / math.log(model.n_clusters)
    cluster = gamma.argmax(dim=1)

    # 2. Điểm OOD không gian ẩn (chuẩn hóa về mốc 1.0 theo phân vị 95 tập nguồn)
    D = model.latent_dim
    d2_all = model.emp_mahalanobis_sq(mu_all)
    d2_min = d2_all.gather(1, cluster.unsqueeze(1)).squeeze(1)
    s_lat = (d2_min / D) / model.src_lat_q95.clamp(min=0.5)

    # 3. Điểm OOD không gian đặc trưng gốc (hiệu chuẩn độ suy hao xuyên miền)
    if feat_ood_weight > 0.0:
        sim_tgt = _feature_cosine_sim(
            X_all, model.cluster_signatures,
            is_nonneg=bool(model.src_is_nonneg.item()),
        )
        max_sim_tgt = sim_tgt.max(dim=1).values
        t_high = torch.quantile(max_sim_tgt, 0.90).clamp(min=0.10)
        s_high = model.src_feat_high.clamp(min=0.10)
        s_low = model.src_feat_low
        rel_ratio = (s_low / s_high).clamp(min=0.15, max=0.85)
        t_low = (t_high * rel_ratio).clamp(min=-0.20, max=0.90)
        gene_thr = (1.0 - t_low).clamp(min=0.10)
        s_feat = (1.0 - max_sim_tgt) / gene_thr

        ood_score = (1.0 - feat_ood_weight) * s_lat + feat_ood_weight * s_feat
    else:
        ood_score = s_lat

    # 4. Làm mịn điểm OOD theo láng giềng không gian (chỉ khi có tọa độ không gian)
    if nbr is not None and effective_mode == "spatial" and spatial_ood_alpha > 0.0:
        nbr_dev = nbr.to(device)
        ood_score = (1.0 - spatial_ood_alpha) * ood_score + spatial_ood_alpha * ood_score[nbr_dev].mean(dim=1)

    # 5. Xác định ngưỡng OOD không rò rỉ
    ood_np = ood_score.cpu().numpy()
    if use_adaptive_thr:
        ood_thr, thr_mode, ashman_d = _adaptive_ood_threshold(
            ood_np, calibrated_thr=calibrated_thr
        )
    else:
        ood_thr, thr_mode, ashman_d = float(calibrated_thr), "calibrated", 0.0

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
        "d2": d2_min,
        "d2_thr": float(model.src_lat_q95.item() * D),
        "ood_score": ood_score,
        "ood_thr": ood_thr,
        "accept": accept,
    }
