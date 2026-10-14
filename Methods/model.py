import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.lambda_)
    
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
    

class GradientReverseFunction(torch.autograd.Function):
    """Forward: identity. Backward: nhân gradient với -lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GRL(nn.Module):
    """Gradient Reversal Layer. lambda_ có thể thay đổi giữa các bước huấn luyện."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    """
    Phân loại miền từ z, gắn GRL ở ngõ vào.
    Nhãn miền: 0 = ST (đích), 1 = scRNA-seq (nguồn).

    Args:
        latent_dim:  số chiều của z (bằng latent_dim của Encoder).
        hidden_dims: kích thước các lớp ẩn.
        dropout:     tỉ lệ dropout trong các khối ẩn.
    Returns:
        logits có shape (batch, 2), KHÔNG qua softmax.
    """

    def __init__(self, latent_dim, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        self.grl = GRL(lambda_=1.0)

        layers = []
        current_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 2))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z, lambda_=None):
        if lambda_ is not None:
            self.grl.set_lambda(lambda_)
        return self.net(self.grl(z))
            

class GMM_VAE(nn.Module):
    """
    GMM-VAE với encoder dùng chung, hai decoder riêng (src / tgt) và Domain Classifier (GRL).
    Prior của z là hỗn hợp Gauss hiệp phương sai chéo với K thành phần.

    Args:
        encoder, decoder_src, decoder_tgt, domain_clf: các module đã xây dựng.
        n_clusters: số cụm K của GMM.
        latent_dim: số chiều không gian ẩn.
        mu_init_std: độ lệch chuẩn khi khởi tạo ngẫu nhiên tâm cụm.
    """

    LOG_VAR_C_MIN, LOG_VAR_C_MAX = -6.0, 4.0

    def __init__(self, encoder, decoder_src, decoder_tgt, domain_clf,
                 n_clusters, latent_dim, mu_init_std=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder_src = decoder_src
        self.decoder_tgt = decoder_tgt
        self.domain_clf = domain_clf
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Tham số GMM, học được bằng backprop.
        # pi và var_c được tham số hóa qua logit / log để luôn hợp lệ (tổng pi = 1, var > 0).
        self.pi_logits = nn.Parameter(torch.zeros(n_clusters))                          # softmax -> pi
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim) * mu_init_std)     # tâm cụm
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))              # exp -> var_c

    # ------------------------------------------------------------------ GMM params
    @property
    def pi(self):
        return torch.softmax(self.pi_logits, dim=0)                                     # (K,)

    @property
    def var_c(self):
        return torch.exp(self.log_var_c.clamp(self.LOG_VAR_C_MIN, self.LOG_VAR_C_MAX))  # (K, D)

    @torch.no_grad()
    def init_gmm_params(self, centers, var=None):
        """Khởi tạo tâm cụm từ bên ngoài (ví dụ KMeans trên mu của encoder đã pretrain,
        hoặc trung bình mu theo từng nhãn nguồn). centers: (K, D)."""
        self.mu_c.copy_(centers)
        if var is not None:
            self.log_var_c.copy_(torch.log(var))

    # ------------------------------------------------------------------ core ops
    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def get_gamma(self, z):
        """
        gamma[n, c] = P(c | z_n) = pi_c N(z_n | mu_c, var_c) / sum_k pi_k N(z_n | mu_k, var_k)
        Tính trong log-space để ổn định số học. Trả về (B, K).
        """
        z = z.unsqueeze(1)                                   # (B, 1, D)
        mu_c = self.mu_c.unsqueeze(0)                        # (1, K, D)
        var_c = self.var_c.unsqueeze(0)                      # (1, K, D)

        log_gauss = -0.5 * (
            math.log(2 * math.pi) + torch.log(var_c) + (z - mu_c).pow(2) / var_c
        ).sum(dim=2)                                         # (B, K)
        log_joint = torch.log(self.pi + 1e-10).unsqueeze(0) + log_gauss
        return torch.softmax(log_joint, dim=1)

    @torch.no_grad()
    def sample_augmented_latent(self, cluster_idx, num_samples, truncation=2.0):
        """
        Sinh z_aug ~ N(mu_c, var_c) với nhiễu chuẩn bị cắt trong [-truncation, truncation].

        cluster_idx: int, hoặc LongTensor (num_samples,) chỉ định cụm cho từng mẫu.
        Trả về tensor (num_samples, latent_dim), đã detach khỏi đồ thị.
        """
        if isinstance(cluster_idx, int):
            cluster_idx = torch.full((num_samples,), cluster_idx, dtype=torch.long,
                                     device=self.mu_c.device)
        else:
            cluster_idx = cluster_idx.to(self.mu_c.device)
            num_samples = cluster_idx.numel()

        mu = self.mu_c[cluster_idx]                          # (N, D)
        std = self.var_c[cluster_idx].sqrt()                 # (N, D)

        eps = torch.empty_like(mu)
        nn.init.trunc_normal_(eps, mean=0.0, std=1.0, a=-truncation, b=truncation)
        return mu + eps * std

    # ------------------------------------------------------------------ forward
    def forward(self, x, domain, lambda_val=1.0):
        """
        x: (B, input_dim). domain: 'src' (scRNA-seq) hoặc 'tgt' (ST) để chọn decoder.
        Gọi riêng cho từng miền, sau đó ghép z (và nhãn miền) lại bên ngoài nếu cần.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        decoder = self.decoder_src if domain == "src" else self.decoder_tgt
        x_hat = decoder(z)

        gamma = self.get_gamma(z)
        domain_logits = self.domain_clf(z, lambda_=lambda_val)

        return {
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "gamma": gamma,
            "domain_logits": domain_logits,
        }
        
def gmm_kl_loss(mu, log_var, gamma, model, eps=1e-10):
    mu = mu.unsqueeze(1)
    log_var = log_var.unsqueeze(1)
    mu_c = model.mu_c.unsqueeze(0)
    var_c = model.var_c.unsqueeze(0)

    term = 0.5 * (torch.log(var_c) + log_var.exp() / var_c + (mu - mu_c).pow(2) / var_c).sum(dim=2)
    expected = (gamma * term).sum(dim=1)
    kl_cat = (gamma * (torch.log(model.pi + eps).unsqueeze(0) - torch.log(gamma + eps))).sum(dim=1)
    entropy_q = 0.5 * (1.0 + log_var.squeeze(1)).sum(dim=1)

    return (expected - kl_cat - entropy_q).mean()


def domain_adversarial_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def reconstruction_loss(x_hat, x):
    return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)


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

    return {"pi": pi, "mu_c": mu_c, "var_c": var_c, "labels": labels,
            "label_to_cluster": label_to_cluster}

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

    cluster_idx = torch.repeat_interleave(cluster_ids, n_needed)
    if cluster_idx.numel() == 0:
        return None

    z_aug = model.sample_augmented_latent(cluster_idx, cluster_idx.numel(), truncation=truncation)
    return z_aug.detach()


def train_joint_gmm_vae(model, dataloader_rna, dataloader_spa, optimizer, epochs,
                        lambda_1, lambda_2, lambda_grl, device,
                        label_to_cluster=None, truncation=2.0, gmm_on_target=False):
    model.to(device)
    model.train()
    spa_iter = _infinite_loader(dataloader_spa)
    history = []

    for epoch in range(epochs):
        sums = {"total": 0.0, "recon": 0.0, "gmm": 0.0, "domain": 0.0, "n_aug": 0}
        n_steps = 0

        for x_rna, y_rna, _ in dataloader_rna:
            x_spa, _, _ = next(spa_iter)
            x_rna, y_rna, x_spa = x_rna.to(device), y_rna.to(device), x_spa.to(device)

            out_rna = model(x_rna, "src", lambda_val=lambda_grl)
            out_spa = model(x_spa, "tgt", lambda_val=lambda_grl)

            loss_recon = reconstruction_loss(out_rna["x_hat"], x_rna) + \
                         reconstruction_loss(out_spa["x_hat"], x_spa)

            loss_gmm = gmm_kl_loss(out_rna["mu"], out_rna["log_var"], out_rna["gamma"], model)
            if gmm_on_target:
                loss_gmm = loss_gmm + gmm_kl_loss(out_spa["mu"], out_spa["log_var"], out_spa["gamma"], model)

            z_aug = generate_latent_augmentation(model, y_rna, label_to_cluster, truncation)

            z_parts = [out_rna["z"], out_spa["z"]]
            d_parts = [
                torch.ones(out_rna["z"].size(0), dtype=torch.long, device=device),
                torch.zeros(out_spa["z"].size(0), dtype=torch.long, device=device),
            ]
            n_aug = 0
            if z_aug is not None:
                z_parts.append(z_aug)
                d_parts.append(torch.ones(z_aug.size(0), dtype=torch.long, device=device))
                n_aug = z_aug.size(0)

            z_all = torch.cat(z_parts, dim=0)
            domain_labels = torch.cat(d_parts, dim=0)

            domain_logits = model.domain_clf(z_all, lambda_=lambda_grl)
            loss_domain = domain_adversarial_loss(domain_logits, domain_labels)

            loss_total = loss_recon + lambda_1 * loss_gmm + lambda_2 * loss_domain

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["gmm"] += loss_gmm.item()
            sums["domain"] += loss_domain.item()
            sums["n_aug"] += n_aug
            n_steps += 1

        n_steps = max(n_steps, 1)
        stats = {k: v / n_steps for k, v in sums.items()}
        history.append(stats)
        print(f"[Joint] Epoch {epoch + 1}/{epochs} - total: {stats['total']:.4f} | "
              f"recon: {stats['recon']:.4f} | gmm: {stats['gmm']:.4f} | "
              f"domain: {stats['domain']:.4f} | z_aug/batch: {stats['n_aug']:.1f}")

    return history