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
    GMM-VAE với Encoder dùng chung và hai Decoder riêng (src: scRNA-seq / tgt: ST).
    Prior của không gian ẩn z là hỗn hợp Gauss hiệp phương sai chéo với K thành phần.

    Args:
        encoder, decoder_src, decoder_tgt: các module đã xây dựng.
        n_clusters: số cụm K của GMM.
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

        # Tham số GMM học được qua backprop
        self.pi_logits = nn.Parameter(torch.zeros(n_clusters))
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim) * mu_init_std)
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    @property
    def pi(self):
        return torch.softmax(self.pi_logits, dim=0)  # (K,)

    @property
    def var_c(self):
        return torch.exp(self.log_var_c.clamp(self.LOG_VAR_C_MIN, self.LOG_VAR_C_MAX))  # (K, D)

    @torch.no_grad()
    def init_gmm_params(self, centers, var=None):
        """Khởi tạo tâm cụm và phương sai từ bên ngoài."""
        self.mu_c.copy_(centers)
        if var is not None:
            self.log_var_c.copy_(torch.log(var))

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def get_gamma(self, z, pi=None, temperature=1.0):
        """Tính xác suất hậu nghiệm p(c | z) của từng mẫu thuộc về K cụm."""
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
        """Khoảng cách Mahalanobis bình phương từ z (B, D) tới K cụm -> (B, K)."""
        diff = z.unsqueeze(1) - self.mu_c.unsqueeze(0)
        return (diff.pow(2) / self.var_c.unsqueeze(0)).sum(dim=2)

    @torch.no_grad()
    def sample_augmented_latent(self, cluster_idx, num_samples, truncation=2.0):
        """
        Sinh z_aug ~ N(mu_c, var_c) với nhiễu chuẩn cắt trong [-truncation, truncation].
        """
        if isinstance(cluster_idx, int):
            cluster_idx = torch.full(
                (num_samples,), cluster_idx, dtype=torch.long, device=self.mu_c.device
            )
        else:
            cluster_idx = cluster_idx.to(self.mu_c.device)
            num_samples = cluster_idx.numel()

        mu = self.mu_c[cluster_idx]
        std = self.var_c[cluster_idx].sqrt()

        eps = torch.empty_like(mu)
        nn.init.trunc_normal_(eps, mean=0.0, std=1.0, a=-truncation, b=truncation)
        return mu + eps * std

    def forward(self, x, domain):
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


def supervised_gmm_kl_loss(mu, log_var, cluster_idx, model, detach_gmm=False):
    """
    Neo trực tiếp biểu diễn (mu, log_var) của miền nguồn vào đúng cụm k = label_to_cluster[y_rna].
    Giúp giữ cố định ngữ nghĩa từng cụm GMM xuyên suốt quá trình huấn luyện chung mà không cần Giai đoạn 3.
    """
    valid = cluster_idx >= 0
    if not valid.any():
        return mu.new_zeros(())

    mu = mu[valid]
    log_var = log_var[valid]
    cluster_idx = cluster_idx[valid]

    mu_k = model.mu_c[cluster_idx]
    var_k = model.var_c[cluster_idx]
    if detach_gmm:
        mu_k, var_k = mu_k.detach(), var_k.detach()

    kl = 0.5 * (
        torch.log(var_k) - log_var + log_var.exp() / var_k
        + (mu - mu_k).pow(2) / var_k - 1.0
    ).sum(dim=1)
    return kl.mean()


@torch.no_grad()
def sinkhorn_log_domain(cost_matrix, reg=0.1, tau_s=None, tau_t=None, num_iters=50, eps=1e-8):
    n_s, n_t = cost_matrix.shape
    dev, dt = cost_matrix.device, cost_matrix.dtype

    cost_matrix = torch.nan_to_num(cost_matrix, nan=0.0, posinf=1e6)
    C = cost_matrix / torch.median(cost_matrix).clamp(min=eps)

    log_a = torch.full((n_s,), -math.log(n_s), device=dev, dtype=dt)
    log_b = torch.full((n_t,), -math.log(n_t), device=dev, dtype=dt)
    f = torch.zeros(n_s, device=dev, dtype=dt)
    g = torch.zeros(n_t, device=dev, dtype=dt)

    rho_s = 1.0 if tau_s is None else tau_s / (tau_s + reg)
    rho_t = 1.0 if tau_t is None else tau_t / (tau_t + reg)

    for _ in range(num_iters):
        f = -rho_s * reg * torch.logsumexp(log_b.unsqueeze(0) + (g.unsqueeze(0) - C) / reg, dim=1)
        g = -rho_t * reg * torch.logsumexp(log_a.unsqueeze(1) + (f.unsqueeze(1) - C) / reg, dim=0)

    log_P = log_a.unsqueeze(1) + log_b.unsqueeze(0) + (f.unsqueeze(1) + g.unsqueeze(0) - C) / reg
    P = torch.exp(log_P)
    return P / P.sum().clamp(min=eps)


def ot_alignment_loss(z_src, z_tgt, model=None, ot_mode="hybrid",
                      reg=0.1, tau_s=1.0, tau_t=1.0, num_iters=50):
    """
    Hàm mất mát căn chỉnh miền bằng Optimal Transport (không phân chia cứng Known/Unknown).

    Args:
        z_src:   Biểu diễn miền nguồn (đã gộp z_aug nếu có), shape (N_s, D).
        z_tgt:   Biểu diễn miền đích (ST), shape (N_t, D).
        model:   Thực thể GMM_VAE (cần khi ot_mode là 'prototype' hoặc 'hybrid').
        ot_mode:
            - 'sample':    OT giữa các mẫu nguồn z_src và mẫu đích z_tgt.
            - 'prototype': OT giữa K cụm GMM (mu_c, var_c) và mẫu đích z_tgt.
            - 'hybrid':    Kết hợp cả Sample-level OT và Prototype-level OT.
        reg:     Hệ số entropy Sinkhorn (mặc định 0.1).
        tau_s, tau_t: Hệ số nới lỏng biên của Unbalanced OT (truyền None nếu muốn Balanced OT).
    """
    loss = z_tgt.new_zeros(())

    if ot_mode in ("sample", "hybrid"):
        # Chi phí khoảng cách bình phương trung bình theo chiều ẩn giữa mẫu nguồn và mẫu đích
        cost_sample = torch.cdist(z_src, z_tgt, p=2).pow(2) / z_tgt.size(1)
        P_sample = sinkhorn_log_domain(
            cost_sample.detach(), reg=reg, tau_s=tau_s, tau_t=tau_t, num_iters=num_iters
        )
        loss = loss + (P_sample * cost_sample).sum()

    if ot_mode in ("prototype", "hybrid"):
        if model is None:
            raise ValueError("Cần truyền `model` khi sử dụng ot_mode='prototype' hoặc 'hybrid'.")
        # Chi phí Mahalanobis chuẩn hóa từ K tâm cụm nguồn (detach tâm cụm để ST di chuyển về nguồn)
        diff = z_tgt.unsqueeze(1) - model.mu_c.detach().unsqueeze(0)       # (N_t, K, D)
        var_c = model.var_c.detach().unsqueeze(0)                          # (1, K, D)
        cost_proto = (diff.pow(2) / var_c).mean(dim=2).T                   # (K, N_t)
        P_proto = sinkhorn_log_domain(
            cost_proto.detach(), reg=reg, tau_s=tau_s, tau_t=tau_t, num_iters=num_iters
        )
        loss = loss + (P_proto * cost_proto).sum()

    if ot_mode == "hybrid":
        loss = 0.5 * loss

    return loss


# =====================================================================
# 3. GIAI ĐOẠN 1: PRETRAIN VAE & KHỞI TẠO THAM SỐ GMM
# =====================================================================

def pretrain_vae(model, dataloader_rna, optimizer, epochs, device):
    model.to(device)
    model.train()
    history = []

    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x, _, _ in dataloader_rna:
            x = x.to(device)

            mu, log_var = model.encoder(x)
            z = model.reparameterize(mu, log_var)
            x_hat = model.decoder_src(z)

            loss = reconstruction_loss(x_hat, x)

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

    var_c = var_c.clamp(
        min=float(np.exp(model.LOG_VAR_C_MIN)),
        max=float(np.exp(model.LOG_VAR_C_MAX)),
    )
    pi = pi.clamp(min=1e-6)
    pi = pi / pi.sum()

    model.mu_c.data.copy_(mu_c)
    model.log_var_c.data.copy_(torch.log(var_c))
    model.pi_logits.data.copy_(torch.log(pi))

    n_labels = int(y_all.max().item()) + 1
    contingency = torch.zeros(n_labels, n_clusters, device=device)
    contingency.index_put_(
        (y_all, labels), torch.ones_like(y_all, dtype=torch.float), accumulate=True
    )
    row, col = linear_sum_assignment(-contingency.cpu().numpy())
    label_to_cluster = torch.full((n_labels,), -1, dtype=torch.long)
    label_to_cluster[torch.as_tensor(row)] = torch.as_tensor(col)

    return {
        "pi": pi,
        "mu_c": mu_c,
        "var_c": var_c,
        "labels": labels,
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
                        gmm_on_target=False, ot_mode="hybrid",
                        ot_reg=0.1, ot_tau_s=1.0, ot_tau_t=1.0, ot_iters=50):
    """
    Huấn luyện chung GMM-VAE kết hợp căn chỉnh miền bằng Optimal Transport (Giai đoạn 2).
    Không thực hiện phân chia cứng Known/Unknown trong quá trình tối ưu.

    Args:
        lambda_1: Trọng số của GMM KL loss.
        lambda_2: Trọng số của Optimal Transport loss.
        label_to_cluster: Ánh xạ từ nhãn scRNA-seq sang chỉ số cụm GMM để neo cố định tâm cụm.
        ot_mode:  'hybrid' (mặc định), 'sample', hoặc 'prototype'.
        ot_reg:   Hệ số điều chuẩn entropy Sinkhorn (epsilon).
        ot_tau_s, ot_tau_t: Hệ số nới lỏng biên Unbalanced OT (đặt None nếu dùng Balanced OT).
    """
    model.to(device)
    model.train()
    spa_iter = _infinite_loader(dataloader_spa)
    history = []

    if label_to_cluster is not None:
        label_to_cluster = label_to_cluster.to(device)

    for epoch in range(epochs):
        sums = {"total": 0.0, "recon": 0.0, "gmm": 0.0, "ot": 0.0, "n_aug": 0}
        n_steps = 0

        for x_rna, y_rna, _ in dataloader_rna:
            x_spa, _, _ = next(spa_iter)
            x_rna, y_rna, x_spa = x_rna.to(device), y_rna.to(device), x_spa.to(device)

            out_rna = model(x_rna, "src")
            out_spa = model(x_spa, "tgt")

            # 1. Reconstruction Loss trên cả hai miền (bảo toàn thông tin biểu hiện gene)
            loss_recon = (
                reconstruction_loss(out_rna["x_hat"], x_rna)
                + reconstruction_loss(out_spa["x_hat"], x_spa)
            )

            # 2. GMM KL Loss trên miền nguồn (kèm neo nhãn nguồn nếu có label_to_cluster)
            loss_gmm = gmm_kl_loss(
                out_rna["mu"], out_rna["log_var"], out_rna["gamma"], model
            )
            if label_to_cluster is not None:
                cluster_rna = label_to_cluster[y_rna]
                loss_gmm = 0.5 * (
                    loss_gmm
                    + supervised_gmm_kl_loss(
                        out_rna["mu"], out_rna["log_var"], cluster_rna, model
                    )
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

            # 4. Optimal Transport Alignment Loss (liên tục, không cắt ngưỡng Known/Unknown)
            loss_ot = ot_alignment_loss(
                z_src=z_src_all,
                z_tgt=out_spa["mu"],
                model=model,
                ot_mode=ot_mode,
                reg=ot_reg,
                tau_s=ot_tau_s,
                tau_t=ot_tau_t,
                num_iters=ot_iters,
            )

            loss_total = loss_recon + lambda_1 * loss_gmm + lambda_2 * loss_ot

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["gmm"] += loss_gmm.item()
            sums["ot"] += loss_ot.item()
            sums["n_aug"] += n_aug
            n_steps += 1

        n_steps = max(n_steps, 1)
        stats = {k: v / n_steps for k, v in sums.items()}
        history.append(stats)
        print(
            f"[Joint-OT] Epoch {epoch + 1}/{epochs} - total: {stats['total']:.4f} | "
            f"recon: {stats['recon']:.4f} | gmm: {stats['gmm']:.4f} | "
            f"ot: {stats['ot']:.4f} | z_aug/batch: {stats['n_aug']:.1f}"
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
