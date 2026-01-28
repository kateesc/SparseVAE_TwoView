import torch
import torch.nn as nn
import torch.nn.functional as F

# Tighter than 20 to avoid exp(20) precision explosions.
# 8.0 => precision in [exp(-8), exp(8)] ~ [3.35e-4, 2981]
# If you want even tighter: set to 6.0.
LOGVAR_BOUND = 8.0

class OmicSparseVAE(nn.Module):
    """
    SparseVAE for a single omic view.

    - Accepts a per-feature missingness mask as input, optionally concatenated to the data.
    - Decodes a full-length reconstruction, but **loss and learning only occur at observed (mask==1) positions**.
    - No imputation or denoising logic: missing data is a signal, not a gap to be filled.
    """
    def __init__(self, config, key):
        super().__init__()
        self.config = config
        self.key = key

        # Flag: Should the mask be concatenated to the input? (recommended when learning missingness patterns)
        self.use_mask_input = bool(self.config.get("use_mask_input", False))

        # Per-feature reconstruction noise (learned), initialized from data std.
        sig_init = torch.tensor(self.config["sigmas_init"], dtype=torch.float)
        sig_init = torch.clamp(sig_init, min=1e-6)
        self.log_sigmas = nn.Parameter(torch.log(sig_init))

        # Lower bound to avoid sigma -> 0.
        self.register_buffer("min_log_sigma", self.log_sigmas.min().detach().clone())

        self.sigma_prior_df = self.config["sig_df"]
        self.sigma_prior_scale = self.config["sig_scale"]

        # Spike-slab hyperparams (used in external E-step style update).
        self.lambda0 = 5
        self.lambda1 = 0.5
        self.a = 1
        self.b = config["input_dim"] / 10
        self.row_normalize = config["row_normalize"]

        pstar_init = torch.full((config["input_dim"], config["latent_dim"]), 0.5, dtype=torch.float)
        self.pstar = nn.Parameter(pstar_init, requires_grad=False)

        thetas_init = torch.rand(config["latent_dim"], dtype=torch.float)
        self.thetas = nn.Parameter(thetas_init, requires_grad=False)

        # Sparse generator mask parameters
        self.W = nn.Parameter(torch.randn(config["input_dim"], config["latent_dim"]), requires_grad=True)

        dim_before_latent = 100

        # Encoder input dimension depends on whether we concat mask.
        enc_in_dim = config["input_dim"] * (2 if self.use_mask_input else 1)

        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, dim_before_latent),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim_before_latent, dim_before_latent),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.zmean = nn.Linear(dim_before_latent, config["latent_dim"])
        self.zlogvar = nn.Linear(dim_before_latent, config["latent_dim"])

        # Generator (decoder) network
        self.generator = nn.Sequential(
            nn.Linear(config["latent_dim"], dim_before_latent, bias=False),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Packed per-feature output head (scales to large feature counts)
        self.out_weight = nn.Parameter(torch.empty(config["input_dim"], dim_before_latent))
        nn.init.xavier_uniform_(self.out_weight)
        self.out_bias = nn.Parameter(torch.zeros(config["input_dim"]))

        # Assert: This model should **not** be used for imputation or denoising; mask learning only.
        assert not hasattr(self.config, "artificial_dropout"), (
            "Detected possible config for denoising/imputation. This model is intended only for learning missingness patterns!")

    def get_generator_mask(self):
        """
        Returns generator W matrix (sparse factors).
        If row_normalize is set, W is normalized for interpretability/sparsity.
        """
        if self.row_normalize:
            W = self.W.abs() + 1e-6
            W = F.normalize(W, p=1, dim=-1)
        else:
            W = self.W
        return W

    def encode(self, x, m=None):
        """
        Encode observed data plus mask (optional) into latent mean/logvar.
        Missingness pattern is always present as a signal if `use_mask_input` is True.
        """
        if self.use_mask_input:
            if m is None:
                m = torch.ones_like(x)
            else:
                m = m.to(dtype=x.dtype)
            x_enc = torch.cat([x, m], dim=1)
        else:
            x_enc = x

        q = self.encoder(x_enc)
        z_mean = self.zmean(q)
        z_logvar_raw = self.zlogvar(q)
        z_logvar = LOGVAR_BOUND * torch.tanh(z_logvar_raw / LOGVAR_BOUND)
        return z_mean, z_logvar

    def reparameterize(self, mean, log_var):
        """
        Standard VAE reparameterization trick.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, chunk_size: int | None = None):
        """
        Full-length vector reconstruction: always outputs all features.
        IMPORTANT: Only observed (mask==1) entries are used for loss/learning.
        Missing positions are ignored in loss, not "imputed".
        """
        if chunk_size is None:
            target = 131072
            chunk_size = max(256, target // max(1, z.shape[0]))

        mask = self.get_generator_mask()  # (F, K)
        input_dim = self.config["input_dim"]
        B, K = z.shape

        x_mean_out = torch.empty((B, input_dim), device=z.device, dtype=z.dtype)
        W_all = self.out_weight  # (F, H)
        b_all = self.out_bias    # (F,)

        for start in range(0, input_dim, chunk_size):
            end = min(start + chunk_size, input_dim)
            chunk = end - start

            m = mask[start:end, :]                          # (chunk, K)
            z_masked = z.unsqueeze(1) * m.unsqueeze(0)      # (B, chunk, K)
            dec_in = z_masked.reshape(-1, K)                # (B*chunk, K)
            h = self.generator(dec_in).view(B, chunk, -1)   # (B, chunk, H)
            w = W_all[start:end, :].unsqueeze(0)            # (1, chunk, H)
            x_chunk = (h * w).sum(dim=-1) + b_all[start:end].unsqueeze(0)  # (B, chunk)
            x_mean_out[:, start:end] = x_chunk

        return x_mean_out

class POEMS(nn.Module):
    """
    2-view POEMS (NO PoE):
    - Encodes each view separately, mask encoding supported.
    - Fuses latents with EXACT 50/50 averaging (mean and variance).
    - Reconstruction loss and gradients are ONLY computed at mask==1 (observed) entries.
    - NO imputation/denoising, NO artificial masking, mask pattern is used as biological signal.
    """

    def __init__(self, batch_size, omic1_info, omic2_info, **kwargs):
        super().__init__()
        self.batch_size = batch_size

        # Submodules are always missingness-aware; mask is channel if use_mask_input == True.
        self.specific_modules = nn.ModuleDict({
            "specific1": OmicSparseVAE(omic1_info, key="specific1"),
            "specific2": OmicSparseVAE(omic2_info, key="specific2"),
        })

        # Dead legacy attributes: kept ONLY for strict logging compatibility, not used anywhere else.
        self.register_buffer("log_s1", torch.zeros(()))
        self.register_buffer("log_s2", torch.zeros(()))
        self.gate_temperature = 1.0
        self.prec_exponent = 0.0

        # Ensure no sigma learning: freeze log_sigmas (consistent with "no sigma weighting").
        for k in ["specific1", "specific2"]:
            self.specific_modules[k].log_sigmas.requires_grad_(False)

        # Assert: No denoising, no imputation, no artificial dropout
        assert all(not hasattr(self.specific_modules[m], "artificial_dropout") for m in self.specific_modules), (
            "Artificial dropout/denoising logic detected in specific_modules!")

    def reparameterize(self, mean, log_var):
        """
        Standard VAE reparam trick.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x1, x2, m1=None, m2=None):
        """
        Encode each view with masks. No PoE, simple average, mask is observed and patterns learned.
        """
        z1_mean, z1_log_var = self.specific_modules["specific1"].encode(x1, m1)
        z2_mean, z2_log_var = self.specific_modules["specific2"].encode(x2, m2)

        z1_log_var = torch.clamp(z1_log_var, -LOGVAR_BOUND, LOGVAR_BOUND)
        z2_log_var = torch.clamp(z2_log_var, -LOGVAR_BOUND, LOGVAR_BOUND)

        # Exact mean fusion (NO PoE), learn missing patterns only
        a1 = 0.5
        a2 = 0.5
        alphas = torch.full((x1.shape[0], 2), 0.5, device=x1.device, dtype=x1.dtype)
        mean_comb = a1 * z1_mean + a2 * z2_mean
        var1 = torch.exp(z1_log_var)
        var2 = torch.exp(z2_log_var)
        var_comb = a1 * var1 + a2 * var2 + 1e-8
        log_var_comb = torch.log(var_comb)
        z = self.reparameterize(mean_comb, log_var_comb)
        eff_w1 = torch.full_like(z1_log_var, 0.5)
        eff_w2 = torch.full_like(z2_log_var, 0.5)

        return mean_comb, log_var_comb, z, alphas, (z1_mean, z1_log_var, z2_mean, z2_log_var, eff_w1, eff_w2)

    def forward(self, x1, x2, m1=None, m2=None):
        """
        Forward pass, all views. Decodes full vectors for both SNPs and traits, losses only on observed.
        """
        zmean, zlogvar, z, alphas, aux = self.encode(x1, x2, m1=m1, m2=m2)
        x1_rec = self.specific_modules["specific1"].decode(z)
        x2_rec = self.specific_modules["specific2"].decode(z)
        return x1_rec, x2_rec, zmean, zlogvar, z, alphas, aux

    def reconstruction_loss_raw(self, x_pred, x, mask=None, eps=1e-8):
        """
        Computes raw reconstruction MSE.
        Only observed (mask==1) positions count for training and reporting.
        """
        err2 = (x_pred - x) ** 2
        if mask is None:
            return 0.5 * err2.mean()
        w = mask.to(dtype=err2.dtype)
        return 0.5 * (err2 * w).sum() / (w.sum() + eps)

    def kl_loss(self, z_mu, z_log_var):
        """
        Standard VAE KL (mean/variance), with clamping.
        """
        z_log_var = torch.clamp(z_log_var, -LOGVAR_BOUND, LOGVAR_BOUND)
        return -0.5 * torch.mean(1 + z_log_var - z_mu.pow(2) - torch.exp(z_log_var))

    def sigma_loss(self, net_key: str, bs: int, sigma_weight: float = 1.0, **kwargs):
        # Sigma regularization OFF (log_sigmas frozen)
        return torch.zeros((), device=next(self.parameters()).device)

    def mask_loss(self, net_key: str, bs: int, mask_weight: float = 1.0, normalize_by_numel: bool = True):
        """
        Mask regularization for sparsity/interpretability.
        """
        W = self.specific_modules[net_key].get_generator_mask()
        pstar = self.specific_modules[net_key].pstar
        lambda0 = self.specific_modules[net_key].lambda0
        lambda1 = self.specific_modules[net_key].lambda1

        loss = (lambda1 * pstar + lambda0 * (1 - pstar)) * W.abs()  # (F,K)
        loss = loss.mean() if normalize_by_numel else loss.sum() / float(bs)
        return mask_weight * loss

    def all_loss(
        self,
        x1, x2,
        m1=None, m2=None,
        x1_in=None, x2_in=None,
        m1_in=None, m2_in=None,
    ):
        """
        Compute all relevant losses; all reconstruction losses only scored at observed positions.
        """
        if x1_in is None: x1_in = x1
        if x2_in is None: x2_in = x2
        if m1_in is None: m1_in = m1
        if m2_in is None: m2_in = m2

        loss_dict = {"rec_loss": [], "kl_loss": [], "mask_loss": [], "sigma_loss": []}

        # Forward pass from provided input (obeys missingness pattern)
        x1_rec, x2_rec, zmean, zlogvar, z, alphas, aux = self.forward(x1_in, x2_in, m1=m1_in, m2=m2_in)
        z1_mean, z1_log_var, z2_mean, z2_log_var, eff_w1, eff_w2 = aux

        # Only observed entries count for reconstruction
        raw1 = self.reconstruction_loss_raw(x1_rec, x1, mask=m1)
        raw2 = self.reconstruction_loss_raw(x2_rec, x2, mask=m2)
        loss_dict["rec_loss"] = [raw1, raw2]

        # KL
        kl_comb = self.kl_loss(zmean, zlogvar)
        kl_v1 = self.kl_loss(z1_mean, z1_log_var)
        kl_v2 = self.kl_loss(z2_mean, z2_log_var)
        loss_dict["kl_loss"] = [kl_comb, kl_v1, kl_v2]

        bs = x1.shape[0]

        # Regularization
        mask_weight = 1.0
        loss_dict["mask_loss"] = [
            self.mask_loss("specific1", bs, mask_weight=mask_weight, normalize_by_numel=True),
            self.mask_loss("specific2", bs, mask_weight=mask_weight, normalize_by_numel=True),
        ]
        # No sigma regularization
        loss_dict["sigma_loss"] = [
            torch.zeros((), device=x1.device),
            torch.zeros((), device=x1.device),
        ]

        # Diagnostics
        eps = 1e-8
        entropy = -(alphas * torch.log(alphas + eps)).sum(dim=1).mean()
        loss_dict["alpha_entropy"] = entropy
        loss_dict["alpha_mean_v1"] = alphas[:, 0].mean()
        loss_dict["alpha_mean_v2"] = alphas[:, 1].mean()
        eff_mean = torch.stack([eff_w1.mean(), eff_w2.mean()])
        loss_dict["eff_mean_v1"] = eff_mean[0]
        loss_dict["eff_mean_v2"] = eff_mean[1]

        loss_dict["rec_loss_raw1"] = raw1.detach()
        loss_dict["rec_loss_raw2"] = raw2.detach()

        return loss_dict

    @torch.no_grad()
    def get_final_embedding(self, X_all, M_all=None):
        """
        Returns mean_comb and alphas for all samples, with mask-aware encoding.
        """
        d1 = self.specific_modules["specific1"].config["input_dim"]
        d2 = self.specific_modules["specific2"].config["input_dim"]

        dev = next(self.parameters()).device
        x1 = torch.tensor(X_all[:, :d1], dtype=torch.float, device=dev)
        x2 = torch.tensor(X_all[:, d1:d1 + d2], dtype=torch.float, device=dev)

        if M_all is None:
            m1 = None
            m2 = None
        else:
            m_all = torch.tensor(M_all, dtype=torch.float, device=dev)
            m1 = m_all[:, :d1]
            m2 = m_all[:, d1:d1 + d2]

        mean_comb, log_var_comb, z, alphas, _ = self.encode(x1, x2, m1=m1, m2=m2)
        return mean_comb, alphas

# END OF FILE: All model logic is strictly "missingness as signal," with no artificial masking, imputation, or denoising logic remaining.
