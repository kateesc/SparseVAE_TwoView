from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import numpy as np
import umap
import os
import torch
from matplotlib.colors import ListedColormap
import sys
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
import json
from typing import Dict, Any

# -----------------------------
# Helpers
# -----------------------------
def _split_views_np(A, d1, d2):
    return A[:, :d1], A[:, d1:d1+d2]

def _ensure_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _safe_makedirs(d):
    if d is not None:
        os.makedirs(d, exist_ok=True)

def _nanwhere(div_num, div_den, eps=1e-12):
    # returns num/den where den>0 else np.nan
    out = np.full_like(div_num, np.nan, dtype=np.float64)
    m = div_den > eps
    out[m] = div_num[m] / div_den[m]
    return out

@torch.no_grad()
def export_latent_scores_csv(
    model,
    X_all,
    M_all,
    sample_ids,
    y,
    out_dir,
    tag="test",
    batch_size=1024,
    include_view_latents=False,
):
    os.makedirs(out_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    X_all = np.asarray(X_all, dtype=np.float32)
    M_all = None if M_all is None else np.asarray(M_all, dtype=np.float32)
    sample_ids = np.asarray(sample_ids, dtype=str)
    y = np.asarray(y)

    d1 = model.specific_modules["specific1"].config["input_dim"]
    d2 = model.specific_modules["specific2"].config["input_dim"]
    K  = model.specific_modules["specific1"].config["latent_dim"]

    z_list = []
    a_list = []
    if include_view_latents:
        z1_list = []
        z2_list = []

    n = X_all.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        xb = torch.tensor(X_all[start:end], dtype=torch.float32, device=device)
        x1 = xb[:, :d1]
        x2 = xb[:, d1:d1 + d2]

        if M_all is not None:
            mb = torch.tensor(M_all[start:end], dtype=torch.float32, device=device)
            m1 = (mb[:, :d1] > 0.5).float()
            m2 = (mb[:, d1:d1 + d2] > 0.5).float()
        else:
            m1 = None
            m2 = None

        mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var = _poems_encode_unpack(
            model, x1, x2, m1=m1, m2=m2
        )

        z_list.append(mean_comb.detach().cpu().numpy())
        a_list.append(alphas.detach().cpu().numpy())
        if include_view_latents:
            z1_list.append(z1_mean.detach().cpu().numpy())
            z2_list.append(z2_mean.detach().cpu().numpy())

    Z = np.concatenate(z_list, axis=0)          # (n, K)
    A = np.concatenate(a_list, axis=0)          # (n, 2)

    # build dataframe
    df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(K)])
    df.insert(0, "sample_id", sample_ids)
    df["alpha_view1"] = A[:, 0]
    df["alpha_view2"] = A[:, 1]
    df["label"] = y

    if include_view_latents:
        Z1 = np.concatenate(z1_list, axis=0)
        Z2 = np.concatenate(z2_list, axis=0)
        for i in range(K):
            df[f"z1_{i}"] = Z1[:, i]
            df[f"z2_{i}"] = Z2[:, i]

    out_path = os.path.join(out_dir, f"LatentScores_{tag}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def orient_latent_scores_csv(in_csv, out_csv, signs):
    """
    Multiply z-columns by signs and write new CSV.
    Also flips z1_*/z2_* columns if present.
    """
    df = pd.read_csv(in_csv)
    K = len(signs)

    for k in range(K):
        col = f"z{k}"
        if col in df.columns:
            df[col] = df[col] * signs[k]
        # optional view-specific columns
        c1 = f"z1_{k}"
        c2 = f"z2_{k}"
        if c1 in df.columns: df[c1] = df[c1] * signs[k]
        if c2 in df.columns: df[c2] = df[c2] * signs[k]

    df.to_csv(out_csv, index=False)
    return out_csv

def orientation_by_top_correlated_trait(Z, X_traits, trait_names, min_abs_corr=0.0):
    """
    Z: (n, K) latent scores (numpy or torch)
    X_traits: (n, T) trait matrix (your View2; imputed+scaled is fine)
    trait_names: length T

    Returns:
      signs: (K,) in {+1,-1}
      info_df: per-latent anchor trait + corr
      corr: (T, K) correlation matrix
    """
    # to numpy
    if hasattr(Z, "detach"):
        Z = Z.detach().cpu().numpy()
    Z = np.asarray(Z, dtype=float)

    X = np.asarray(X_traits, dtype=float)

    n, K = Z.shape
    T = X.shape[1]

    # standardize columns to compute Pearson corr via dot products
    Zs = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    corr = (Xs.T @ Zs) / (n - 1)   # (T, K)

    signs = np.ones(K, dtype=int)
    rows = []
    for k in range(K):
        t_star = int(np.argmax(np.abs(corr[:, k])))
        r = float(corr[t_star, k])

        # if weak, don't force an orientation
        if abs(r) < min_abs_corr:
            s = 1
            anchor = ""
        else:
            s = 1 if r >= 0 else -1
            anchor = trait_names[t_star]

        signs[k] = s
        rows.append({"latent": k, "anchor_trait": anchor, "corr": r, "sign": int(s)})

    info_df = pd.DataFrame(rows)
    return signs, info_df, corr

def permutation_latent_trait_corr_pvals(
    Z, X_traits, trait_names=None,
    n_perm=1000, seed=0,
    two_sided=True,
    return_null_max=False,
    apply_bh=True,
    bh_scope="global",   # "global" or "within_latent"
    fdr_alpha=0.05,
):
    """
    Empirical p-values for corr(Z[:,k], X[:,t]) by permuting rows of X.
    Z: (n,K)
    X_traits: (n,T)
    Returns:
      df: columns = latent, trait, r_obs, p_emp
      optionally also p_emp_max (max-|r| across all pairs each perm) if return_null_max=True
    """
    Z = _ensure_np(Z).astype(np.float64)
    X = _ensure_np(X_traits).astype(np.float64)
    n, K = Z.shape
    n2, T = X.shape
    assert n == n2, f"Row mismatch: Z has {n}, X has {n2}"

    if trait_names is None:
        trait_names = [f"trait{t}" for t in range(T)]

    # standardize
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)

    r_obs = (Xs.T @ Zs) / (n - 1)     # (T,K)

    # counts for empirical p-values
    obs_cmp = np.abs(r_obs) if two_sided else r_obs
    counts = np.zeros((T, K), dtype=np.int64)

    if return_null_max:
        # null distribution of max statistic across all pairs (FWER style)
        obs_max = obs_cmp.max()
        max_count = 0

    rng = np.random.default_rng(seed)
    idx = np.arange(n)

    for _ in range(n_perm):
        rng.shuffle(idx)
        Xp = Xs[idx, :]               # permute rows of X
        r_p = (Xp.T @ Zs) / (n - 1)

        r_cmp = np.abs(r_p) if two_sided else r_p
        counts += (r_cmp >= obs_cmp)

        if return_null_max:
            if r_cmp.max() >= obs_max:
                max_count += 1

    # +1 smoothing
    p_emp = (counts + 1) / (n_perm + 1)

    rows = []
    for t in range(T):
        for k in range(K):
            rows.append({
                "trait": trait_names[t],
                "latent": k,
                "r_obs": float(r_obs[t, k]),
                "p_emp": float(p_emp[t, k]),
            })

    df = pd.DataFrame(rows)
    if return_null_max:
        df["p_emp_max"] = float((max_count + 1) / (n_perm + 1))

    # ---------- BH/FDR ----------
    if apply_bh:
        if bh_scope == "global":
            df["q_bh"] = bh_fdr_qvals(df["p_emp"].values)
        elif bh_scope == "within_latent":
            df["q_bh"] = np.nan
            for k, sub in df.groupby("latent"):
                df.loc[sub.index, "q_bh"] = bh_fdr_qvals(sub["p_emp"].values)
        else:
            raise ValueError("bh_scope must be 'global' or 'within_latent'")

        df[f"sig_fdr_{fdr_alpha:g}"] = df["q_bh"] <= float(fdr_alpha)

    return df
    
def permutation_latent_group_pvals(
    Z, y, group_a, group_b,
    n_perm=1000, seed=0,
    two_sided=True,
    apply_bh=True,
    fdr_alpha=0.05,
):
    """
    Empirical p-values for Δ = mean(Z|a) - mean(Z|b) by permuting y.
    Returns df with latent, diff_obs, p_emp.
    """
    Z = _ensure_np(Z).astype(np.float64)
    y = np.asarray(y)
    n, K = Z.shape

    mask_a = (y == group_a)
    mask_b = (y == group_b)
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        raise ValueError("One of the groups has zero samples.")

    diff_obs = Z[mask_a].mean(0) - Z[mask_b].mean(0)
    obs_cmp = np.abs(diff_obs) if two_sided else diff_obs

    rng = np.random.default_rng(seed)
    counts = np.zeros(K, dtype=np.int64)

    for _ in range(n_perm):
        yp = rng.permutation(y)
        ma = (yp == group_a)
        mb = (yp == group_b)
        # keep group sizes same on average; if labels are imbalanced this is still valid
        diff_p = Z[ma].mean(0) - Z[mb].mean(0)
        cmp_p = np.abs(diff_p) if two_sided else diff_p
        counts += (cmp_p >= obs_cmp)

    p_emp = (counts + 1) / (n_perm + 1)

    df = pd.DataFrame({
        "latent": np.arange(K),
        "diff_obs_meanA_minus_meanB": diff_obs,
        "p_emp": p_emp
    })

    if apply_bh:
        df["q_bh"] = bh_fdr_qvals(df["p_emp"].values)
        df[f"sig_fdr_{fdr_alpha:g}"] = df["q_bh"] <= float(fdr_alpha)

    return df

def bh_fdr_qvals(pvals):
    """
    Benjamini-Hochberg FDR correction.
    Returns q-values (adjusted p-values) same shape as input.
    """
    p = np.asarray(pvals, dtype=np.float64)
    shape = p.shape
    p = p.ravel()

    n = p.size
    order = np.argsort(p)
    ranked = p[order]

    # BH step-up
    q = ranked * n / (np.arange(1, n + 1))

    # enforce monotonicity of q-values
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out = np.empty_like(q)
    out[order] = q
    return out.reshape(shape)

@torch.no_grad()
def within_factor_feature_ablation(
    model,
    X_all,
    M_all,
    d1: int,
    d2: int,
    *,
    latent_k: int,
    feat_idx,                 # indices into VIEW1 features (SNPs): list/np array
    batch_size: int = 64,
    max_samples: int = 512,
    seed: int = 0,
    use_mean_latent: bool = True,     # decode from mean_comb (recommended)
    restrict: str = "subset_obs",     # "subset_obs" | "subset_all" | "full_obs" | "full_all"
    zero_pstar: bool = True,          # set pstar[S,k]=0 too (stronger “edge off”)
):
    """
    C) Within-factor feature ablation:
      - temporarily set W[feat_idx, latent_k] = 0 (and optionally pstar as well)
      - measure reconstruction MSE change

    restrict:
      subset_obs:   MSE on view1 features in feat_idx, ONLY observed entries (mask==1)
      subset_all:   MSE on view1 features in feat_idx, ALL entries (ignores mask)
      full_obs:     MSE on ALL view1 features, ONLY observed entries
      full_all:     MSE on ALL view1 features, ALL entries
    """
    device = next(model.parameters()).device
    model.eval()

    X = _ensure_np(X_all).astype(np.float32)
    if M_all is None:
        M = None
    else:
        M = _ensure_np(M_all).astype(np.float32)

    n = X.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=max_samples, replace=False)
        X = X[sel]
        if M is not None:
            M = M[sel]

    feat_idx = np.asarray(feat_idx, dtype=int)
    if feat_idx.size == 0:
        raise ValueError("feat_idx is empty.")

    # module for view1
    mod1 = model.specific_modules["specific1"]
    if not hasattr(mod1, "W"):
        raise AttributeError("specific1 module has no attribute W; cannot ablate edges by W[S,k]=0.")

    # save original values
    W = mod1.W
    old_W = W[feat_idx, latent_k].detach().clone()

    has_pstar = hasattr(mod1, "pstar") and (mod1.pstar is not None)
    if zero_pstar and has_pstar:
        P = mod1.pstar
        old_P = P[feat_idx, latent_k].detach().clone()
    else:
        old_P = None

    def _accumulate_mse(x1_true, x1_rec, m1):
        # choose which features
        if restrict.startswith("subset"):
            x1_true = x1_true[:, feat_idx]
            x1_rec  = x1_rec[:,  feat_idx]
            if m1 is not None:
                m1 = m1[:, feat_idx]

        err = (x1_rec - x1_true) ** 2

        if restrict.endswith("_all"):
            # ignore mask
            num = err.sum().item()
            den = float(err.numel())
        else:
            # observed-only
            if m1 is None:
                num = err.sum().item()
                den = float(err.numel())
            else:
                num = (err * m1).sum().item()
                den = m1.sum().item()

        return num, den

    base_num = base_den = 0.0
    abl_num  = abl_den  = 0.0

    try:
        # ---- pass through batches ----
        N = X.shape[0]
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            if M is not None:
                mb = torch.tensor(M[start:end], dtype=torch.float32, device=device)
                m1 = (mb[:, :d1] > 0.5).float()
                m2 = (mb[:, d1:d1 + d2] > 0.5).float()
            else:
                m1 = None
                m2 = None

            x1 = xb[:, :d1]
            x2 = xb[:, d1:d1 + d2]

            # encode
            mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var = _poems_encode_unpack(
                model, x1, x2, m1=m1, m2=m2
            )
            z_use = mean_comb if use_mean_latent else z

            # baseline decode (original W)
            x1_base = mod1.decode(z_use)
            bn, bd = _accumulate_mse(x1, x1_base, m1)
            base_num += bn
            base_den += bd

            # ablate edges for this batch (global params, but we flip and restore immediately)
            W[feat_idx, latent_k].data.zero_()
            if zero_pstar and has_pstar:
                mod1.pstar[feat_idx, latent_k].data.zero_()

            x1_abl = mod1.decode(z_use)
            an, ad = _accumulate_mse(x1, x1_abl, m1)
            abl_num += an
            abl_den += ad

            # restore for next batch
            W[feat_idx, latent_k].data.copy_(old_W)
            if zero_pstar and has_pstar:
                mod1.pstar[feat_idx, latent_k].data.copy_(old_P)

    finally:
        # safety restore (in case anything throws)
        W[feat_idx, latent_k].data.copy_(old_W)
        if zero_pstar and has_pstar:
            mod1.pstar[feat_idx, latent_k].data.copy_(old_P)

    base_mse = float(base_num / (base_den + 1e-12)) if base_den > 0 else float("nan")
    abl_mse  = float(abl_num  / (abl_den  + 1e-12)) if abl_den  > 0 else float("nan")

    return pd.DataFrame([{
        "latent": int(latent_k),
        "n_features_ablated": int(feat_idx.size),
        "restrict": restrict,
        "base_mse": base_mse,
        "ablated_mse": abl_mse,
        "delta_mse": float(abl_mse - base_mse) if np.isfinite(base_mse) and np.isfinite(abl_mse) else float("nan"),
        "zero_pstar": bool(zero_pstar),
        "max_samples": int(X.shape[0]),
    }])


def top_feature_indices_for_latent_view1(
    model,
    *,
    latent_k: int,
    top_n: int = 200,
    abs_w_threshold: float = 1e-3,
    pstar_threshold: float = 0.7,
    use_pstar: bool = True,
):
    """
    Convenience helper: get a SNP index set S for latent k.
    Uses |W| threshold and optional pstar threshold.
    Returns: np.array of feature indices (into view1).
    """
    mod1 = model.specific_modules["specific1"]
    W1 = mod1.get_generator_mask().detach().cpu().numpy()  # often W * pstar (good for selection)
    absw = np.abs(W1[:, latent_k])

    mask = absw > abs_w_threshold
    if use_pstar and hasattr(mod1, "pstar") and mod1.pstar is not None:
        p = mod1.pstar.detach().cpu().numpy()[:, latent_k]
        mask = mask & (p > pstar_threshold)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return idx

    # rank by |W|
    idx = idx[np.argsort(absw[idx])[::-1]]
    return idx[:top_n]
    
# ========================
# ACTIVE, MISSINGNESS-AS-SIGNAL ALIGNED FUNCTIONS
# ========================

def _plot_annotation_text(ax, text):
    # Lower left or upper left corner: customize as needed
    ax.annotate(text, xy=(0.01, 0.98), xycoords="axes fraction", fontsize=9, va="top", ha="left", alpha=0.9, bbox=dict(fc='white', alpha=0.4))





def _poems_encode_unpack(model, x1, x2, m1=None, m2=None):
    mean_comb, log_var_comb, z, alphas, view_stats = model.encode(x1, x2, m1=m1, m2=m2)

    if len(view_stats) == 4:
        z1_mean, z1_log_var, z2_mean, z2_log_var = view_stats
    elif len(view_stats) == 6:
        z1_mean, z1_log_var, z2_mean, z2_log_var, eff_w1, eff_w2 = view_stats
        # (eff_w1/eff_w2 optional; ignore or use if you want)
    else:
        raise ValueError(f"Unexpected view_stats length: {len(view_stats)}")

    return mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var


def _summarize_W(W: np.ndarray, thresholds=(1e-3, 1e-2, 1e-1)):
    """
    Returns a dict of numeric diagnostics for W.
    W is a numpy array (n_features, latent_dim).
    """
    absW = np.abs(W)
    flat = absW.ravel()

    out = {
        "shape0_n_features": W.shape[0],
        "shape1_latent_dim": W.shape[1],
        "max_absW": float(flat.max()),
        "mean_absW": float(flat.mean()),
        "median_absW": float(np.median(flat)),
        "q90_absW": float(np.quantile(flat, 0.90)),
        "q99_absW": float(np.quantile(flat, 0.99)),
        "q999_absW": float(np.quantile(flat, 0.999)),
    }

    for thr in thresholds:
        cnt = int((absW > thr).sum())
        frac = float((absW > thr).mean())
        out[f"active_cnt_gt_{thr:g}"] = cnt
        out[f"active_frac_gt_{thr:g}"] = frac

    return out


# 2-view maps
OMIC_COLOR_MAP = {
    "specific1": "#1f77b4",  # View1
    "specific2": "#ff7f0e",  # View2
}
OMIC_NAME_MAP = {
    "specific1": "View1",
    "specific2": "View2",
}

sparsity_threshold = 1e-2


def _safe_featnames(disease: str, view_idx: int, n_features: int):
    """
    Try to load <view_idx>_featname.csv. If missing, fall back to generic names.
    """
    path = os.path.join(sys.path[1], "data", disease, f"{view_idx}_featname.csv")
    try:
        names = pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()
        if len(names) != n_features:
            return [f"f{view_idx}_{i}" for i in range(n_features)]
        return names
    except Exception:
        return [f"f{view_idx}_{i}" for i in range(n_features)]


# -----------------------------
# (A) FIXED: visualize_Ws
# -----------------------------
def _thr_to_str(thr: float) -> str:
    # 0.001 -> "1e-03" etc (safe for filenames)
    return f"{thr:.0e}"


def visualize_Ws(test_model, dir=None, thresholds=(1e-3, 1e-2, 1e-1),
                 hist_sharey=False, hist_density=True, hist_log_y=False,
                 bins=100):
    """
    For each threshold, saves:
      - Ws_Row1_SparsityMask_thr<THR>.pdf
      - Ws_Row2_Histogram_thr<THR>.pdf

    Improvements vs your current version:
      - histogram y-axis NOT shared by default (so View2 won't look flat/missing)
      - optional density normalization (compares shapes even if feature counts differ)
      - optional log y-axis for rare active weights
      - if no weights exceed thr in a view, fall back to plotting all weights + annotate
    """
    if dir:
        os.makedirs(dir, exist_ok=True)

    if isinstance(thresholds, (float, int)):
        thresholds = (float(thresholds),)

    cmap_bin = ListedColormap(["white", "red"])
    omic_items = list(OMIC_COLOR_MAP.items())
    V = len(omic_items)

    # Grab raw W once
    Ws_raw = {}
    for omic_key in OMIC_COLOR_MAP.keys():
        Ws_raw[omic_key] = (
            test_model.specific_modules[omic_key]
            .get_generator_mask()
            .detach()
            .cpu()
        )

    # Optional diagnostics (kept from your file)
    diag_rows = []
    for omic_key in OMIC_COLOR_MAP.keys():
        W_np = Ws_raw[omic_key].numpy()
        d = _summarize_W(W_np, thresholds=thresholds)
        d["view"] = OMIC_NAME_MAP.get(omic_key, omic_key)
        diag_rows.append(d)

        print("\n[W DIAGNOSTICS]", d["view"])
        print("  shape:", (d["shape0_n_features"], d["shape1_latent_dim"]))
        print("  max|W|:", d["max_absW"])
        for thr in thresholds:
            print(f"  |W| > {thr:g}: cnt={d[f'active_cnt_gt_{thr:g}']:,}  frac={d[f'active_frac_gt_{thr:g}']:.6g}")

    if dir:
        pd.DataFrame(diag_rows).to_csv(os.path.join(dir, "W_diagnostics.csv"), index=False)

    for thr in thresholds:
        thr_str = _thr_to_str(thr)

        # ---- Row 1: activation masks ----
        fig, axes = plt.subplots(1, V, figsize=(6 * V, 4))
        axes = axes if V > 1 else [axes]

        for i, (omic_key, _) in enumerate(omic_items):
            ax = axes[i]
            W = Ws_raw[omic_key]
            W_active = (W.abs() > thr).int()

            ax.imshow(W_active.T, cmap=cmap_bin, aspect="auto")
            ax.set_title(f"{OMIC_NAME_MAP[omic_key]} active mask (|W| > {thr_str})", fontsize=12)
            ax.set_xlabel("Input features", fontsize=10)
            if i == 0:
                ax.set_ylabel("Latent factors", fontsize=10)

            ax.legend(handles=[mpatches.Patch(color="red", label=f"|W| > {thr}")], fontsize=9)

        plt.tight_layout()
        if dir:
            plt.savefig(os.path.join(dir, f"Ws_Row1_SparsityMask_thr{thr_str}.pdf"),
                        dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # ---- Row 2: histograms ----
        fig, axes = plt.subplots(1, V, figsize=(6 * V, 4), sharey=hist_sharey)
        axes = axes if V > 1 else [axes]

        for i, (omic_key, color_value) in enumerate(omic_items):
            ax = axes[i]
            W = Ws_raw[omic_key].numpy()

            active_vals = W[np.abs(W) > thr]
            n_active = active_vals.size
            max_abs = float(np.max(np.abs(W))) if W.size else 0.0

            # If nothing passes the threshold, fall back to plotting all weights
            # so the plot isn't blank AND you get a clear signal that thr is too high.
            if n_active == 0:
                plot_vals = W.ravel()
                subtitle = f"(no |W|>{thr_str}; showing ALL W; max|W|={max_abs:.3g})"
            else:
                plot_vals = active_vals
                subtitle = f"(n={n_active:,} active; max|W|={max_abs:.3g})"

            ax.hist(plot_vals, bins=bins, color=color_value, alpha=0.7, density=hist_density)
            ax.set_title(f"{OMIC_NAME_MAP[omic_key]} histogram {subtitle}", fontsize=10)
            ax.set_xlabel("W value", fontsize=10)
            if i == 0:
                ax.set_ylabel("Density" if hist_density else "Frequency", fontsize=10)

            if hist_log_y:
                ax.set_yscale("log")

        plt.tight_layout()
        if dir:
            plt.savefig(os.path.join(dir, f"Ws_Row2_Histogram_thr{thr_str}.pdf"),
                        dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()



@torch.no_grad()
def _compute_alphas_and_effective_weights(model, X_all, M_all=None, batch_size=1024):
    device = next(model.parameters()).device
    model.eval()

    d1 = model.specific_modules["specific1"].config["input_dim"]
    d2 = model.specific_modules["specific2"].config["input_dim"]
    K  = model.specific_modules["specific1"].config["latent_dim"]

    X_all = np.asarray(X_all, dtype=np.float32)
    n = X_all.shape[0]

    if M_all is not None:
        M_all = np.asarray(M_all, dtype=np.float32)
        if M_all.shape != X_all.shape:
            raise ValueError(f"M_all shape {M_all.shape} != X_all shape {X_all.shape}")

    alpha_list = []
    eff_list = []
    eff_lat_list = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        xb = torch.tensor(X_all[start:end], dtype=torch.float, device=device)
        x1 = xb[:, :d1]
        x2 = xb[:, d1:d1 + d2]

        if M_all is not None:
            mb = torch.tensor(M_all[start:end], dtype=torch.float, device=device)
            m1 = (mb[:, :d1] > 0.5).float()
            m2 = (mb[:, d1:d1 + d2] > 0.5).float()
        else:
            m1 = None
            m2 = None

        # encode (your POEMS returns alphas and view stats)
        mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var = _poems_encode_unpack(
            model, x1, x2, m1=m1, m2=m2
        )

        # ------------------------------
        # NO PoE: "effective" == alpha
        # ------------------------------
        eff = alphas  # (b,2), should be exactly 0.5/0.5 in your design

        # per-latent effective weights: constant 0.5/0.5 (since fusion is fixed)
        bsz = alphas.shape[0]
        eff_lat = torch.full((bsz, 2, K), 0.5, device=alphas.device, dtype=alphas.dtype)

        alpha_list.append(alphas.detach().cpu().numpy())
        eff_list.append(eff.detach().cpu().numpy())
        eff_lat_list.append(eff_lat.detach().cpu().numpy())

    alpha_np = np.concatenate(alpha_list, axis=0)
    eff_np = np.concatenate(eff_list, axis=0)
    eff_lat_np = np.concatenate(eff_lat_list, axis=0)

    return alpha_np, eff_np, eff_lat_np



def plot_effective_fusion_weights(model, X_all, M_all=None, dir=None, batch_size=1024, tag="test"):
    if dir:
        os.makedirs(dir, exist_ok=True)

    alpha_np, eff_np, eff_lat = _compute_alphas_and_effective_weights(
        model, X_all, M_all=M_all, batch_size=batch_size
    )

    df = pd.DataFrame({
        "sample_index": np.arange(alpha_np.shape[0]),
        "alpha_view1": alpha_np[:, 0],
        "alpha_view2": alpha_np[:, 1],
        "eff_view1": eff_np[:, 0],
        "eff_view2": eff_np[:, 1],
    })
    if dir:
        df.to_csv(os.path.join(dir, f"EffectiveFusion_{tag}.csv"), index=False)

    n = alpha_np.shape[0]
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    axes[0].bar(idx, alpha_np[:, 0], label="View1", color=OMIC_COLOR_MAP["specific1"])
    axes[0].bar(idx, alpha_np[:, 1], bottom=alpha_np[:, 0], label="View2", color=OMIC_COLOR_MAP["specific2"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Fusion weights α (fixed 0.5/0.5; sanity check) ({tag})", fontsize=10)
    axes[0].set_ylabel("α", fontsize=9)
    axes[0].legend(fontsize=8)

    axes[1].bar(idx, eff_np[:, 0], label="View1", color=OMIC_COLOR_MAP["specific1"])
    axes[1].bar(idx, eff_np[:, 1], bottom=eff_np[:, 0], label="View2", color=OMIC_COLOR_MAP["specific2"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Effective fusion weights (same as α in No-PoE; sanity check) ({tag})", fontsize=10)
    axes[1].set_ylabel("weight", fontsize=9)
    axes[1].set_xlabel("Sample index", fontsize=9)

    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"EffectiveFusion_Alphas_vs_Effective_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    eff_lat_mean = eff_lat.mean(axis=0)  # (2,K)
    K = eff_lat_mean.shape[1]
    k_idx = np.arange(K)

    plt.figure(figsize=(10, 3))
    plt.bar(k_idx, eff_lat_mean[0], label="View1", color=OMIC_COLOR_MAP["specific1"])
    plt.bar(k_idx, eff_lat_mean[1], bottom=eff_lat_mean[0], label="View2", color=OMIC_COLOR_MAP["specific2"])
    plt.ylim(0, 1.05)
    plt.xlabel("Latent factor k")
    plt.ylabel("mean w_eff")
    plt.title(f"Mean fusion weight per latent factor (should be 0.5/0.5) ({tag})")
    plt.legend(fontsize=8)
    plt.tight_layout()

    if dir:
        plt.savefig(os.path.join(dir, f"EffectiveFusion_PerLatent_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



# -----------------------------
# (C) NEW: Trait × Latent matrix (+ CSV)
# -----------------------------
def plot_trait_latent_matrix(model, disease, dir=None, use_abs=True):
    """
    View2 is traits. Produces:
      - TraitLatent_Loadings.pdf
      - TraitLatent_Loadings.csv
    """
    if dir:
        os.makedirs(dir, exist_ok=True)

    W2 = model.specific_modules["specific2"].get_generator_mask().detach().cpu().numpy()  # (n_traits, K)
    trait_names = _safe_featnames(disease, view_idx=2, n_features=W2.shape[0])

    M = np.abs(W2) if use_abs else W2
    df = pd.DataFrame(M, index=trait_names, columns=[f"z{k}" for k in range(M.shape[1])])

    if dir:
        df.to_csv(os.path.join(dir, "TraitLatent_Loadings.csv"))

    plt.figure(figsize=(12, 4))
    sns.heatmap(df, cmap="coolwarm" if not use_abs else "viridis")
    plt.title("Trait × Latent loadings (View2 → latent)", fontsize=12)
    plt.xlabel("Latent factor")
    plt.ylabel("Trait")
    plt.tight_layout()

    if dir:
        plt.savefig(os.path.join(dir, "TraitLatent_Loadings.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -----------------------------
# (D) NEW: Trait → Top SNPs (trait-specific bridge)
# -----------------------------
def export_trait_top_snps(model, disease, dir=None, top_snps=50, top_latents=5, use_abs=True):
    """
    Builds a trait-specific SNP score using latent factors as bridges.

    For trait t and SNP s:
      score(s|t) = sum_k  |W2[t,k]| * |W1[s,k]|     (default, use_abs=True)

    Outputs:
      - TraitTopSNPs.csv   (trait-specific top SNP lists)
      - TraitTopLatents.csv
    """
    if dir:
        os.makedirs(dir, exist_ok=True)

    W1 = model.specific_modules["specific1"].get_generator_mask().detach().cpu().numpy()  # (n_snps, K)
    W2 = model.specific_modules["specific2"].get_generator_mask().detach().cpu().numpy()  # (n_traits, K)

    if use_abs:
        W1m = np.abs(W1)
        W2m = np.abs(W2)
    else:
        W1m = W1
        W2m = W2

    snp_names = _safe_featnames(disease, view_idx=1, n_features=W1.shape[0])
    trait_names = _safe_featnames(disease, view_idx=2, n_features=W2.shape[0])

    # Trait -> Top Latents
    lat_rows = []
    for t, trait in enumerate(trait_names):
        vals = W2m[t, :]
        idx = np.argsort(vals)[::-1][:top_latents]
        for k in idx:
            lat_rows.append({
                "trait": trait,
                "latent": int(k),
                "loading": float(W2m[t, k]),
                "signed_loading": float(W2[t, k]),
            })
    df_lat = pd.DataFrame(lat_rows)
    if dir:
        df_lat.to_csv(os.path.join(dir, "TraitTopLatents.csv"), index=False)

    # Trait -> Top SNPs
    rows = []
    for t, trait in enumerate(trait_names):
        w_t = W2m[t, :]               # (K,)
        scores = W1m @ w_t            # (n_snps,)

        top_idx = np.argsort(scores)[::-1][:top_snps]
        for rank, s in enumerate(top_idx, start=1):
            rows.append({
                "trait": trait,
                "rank": rank,
                "snp": snp_names[s],
                "score": float(scores[s]),
            })

    df = pd.DataFrame(rows)
    if dir:
        df.to_csv(os.path.join(dir, "TraitTopSNPs.csv"), index=False)

    return df, df_lat


# -----------------------------
# (E) NEW: Latent → Top SNPs & Top Traits
# -----------------------------
def export_latent_top_features(model, disease, dir=None, top_snps=10, top_traits=10):
    """
    For each latent factor k:
      - top SNPs by |W1[:,k]|
      - top traits by |W2[:,k]|
    Output:
      - LatentTopFeatures.csv
    """
    if dir:
        os.makedirs(dir, exist_ok=True)

    W1 = model.specific_modules["specific1"].get_generator_mask().detach().cpu().numpy()
    W2 = model.specific_modules["specific2"].get_generator_mask().detach().cpu().numpy()

    snp_names = _safe_featnames(disease, view_idx=1, n_features=W1.shape[0])
    trait_names = _safe_featnames(disease, view_idx=2, n_features=W2.shape[0])

    K = W1.shape[1]
    rows = []
    for k in range(K):
        snp_idx = np.argsort(np.abs(W1[:, k]))[::-1][:top_snps]
        trait_idx = np.argsort(np.abs(W2[:, k]))[::-1][:top_traits]

        rows.append({
            "latent": k,
            "top_snps": ";".join([snp_names[i] for i in snp_idx]),
            "top_snp_absW": ";".join([f"{abs(W1[i,k]):.4g}" for i in snp_idx]),
            "top_traits": ";".join([trait_names[i] for i in trait_idx]),
            "top_trait_absW": ";".join([f"{abs(W2[i,k]):.4g}" for i in trait_idx]),
        })

    df = pd.DataFrame(rows)
    if dir:
        df.to_csv(os.path.join(dir, "LatentTopFeatures.csv"), index=False)
    return df


# -----------------------------
# Your existing plotting functions below (unchanged)
# -----------------------------
def plot_tsne(latents, label, dir=None, tag=None, question_text=None):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(np.asarray(latents))
    plt.figure(figsize=(7,5))
    ax = plt.gca()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=label, alpha=0.7, cmap='viridis', s=30)
    plt.title("t-SNE on Latent Space", fontsize=16)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    if question_text is None:
        question_text = "Q: Do samples cluster by group in latent space?"
    _plot_annotation_text(ax, question_text)
    if dir:
        plt.savefig(os.path.join(dir, f'tsne{"" if not tag else "_"+tag}.png'), bbox_inches="tight", dpi=160)
        plt.close()
    else:
        plt.show()

def plot_umap(latents, label, dir=None, tag=None, question_text=None):
    import umap
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(np.asarray(latents))
    plt.figure(figsize=(7,5))
    ax = plt.gca()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, alpha=0.7, cmap='viridis', s=30)
    plt.title("UMAP on Latent Space", fontsize=16)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    if question_text is None:
        question_text = "Q: Are latent factors nonlinearly associated with label or trait?"
    _plot_annotation_text(ax, question_text)
    if dir:
        plt.savefig(os.path.join(dir, f'umap{"" if not tag else "_"+tag}.png'), bbox_inches="tight", dpi=160)
        plt.close()
    else:
        plt.show()

def visualize_final_embedding(embeddings, labels, dir=None):
    if isinstance(embeddings, torch.Tensor):
        em_np = embeddings.detach().cpu().numpy()
    else:
        em_np = np.asarray(embeddings)

    labels = np.asarray(labels)

    sort_idx = np.argsort(labels)
    sorted_embeddings = em_np[sort_idx]
    sorted_labels = labels[sort_idx]

    plt.clf()
    plt.figure(figsize=(6, 3))
    ax = sns.heatmap(sorted_embeddings, cmap="coolwarm", cbar=True, yticklabels=False)

    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=5)

    n_factors = sorted_embeddings.shape[1]
    ax.set_xticks(np.arange(n_factors) + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_factors)], fontsize=5, rotation=0)

    unique_clusters, counts = np.unique(sorted_labels, return_counts=True)
    boundaries = np.cumsum(counts)[:-1]
    for b in boundaries:
        ax.hlines(b, *ax.get_xlim(), colors="black", linestyles="dashed", linewidth=0.5)

    ax.set_xlabel("Latent factors", fontsize=6)
    ax.set_ylabel(f"Samples (sorted by label, n={len(labels)})", fontsize=6)
    plt.title("Latent embeddings heatmap", fontsize=7)

    if dir is not None:
        plt.savefig(os.path.join(dir, "final_em_test.pdf"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.clf()
    plt.figure(figsize=(12, 6))
    sns.heatmap(pd.DataFrame(em_np).corr(), cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Latent Dimension Correlation Matrix")
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Latent Dimensions")

    if dir is not None:
        plt.savefig(os.path.join(dir, "final_latent_corr_test.pdf"))
        plt.close()
    else:
        plt.show()


def plot_subtype_correlations(X1, X2, Z, y, subtype_names=None, dir=None):
    # Convert to numpy
    if isinstance(X1, torch.Tensor): X1 = X1.detach().cpu().numpy()
    if isinstance(X2, torch.Tensor): X2 = X2.detach().cpu().numpy()
    if isinstance(Z, torch.Tensor):  Z  = Z.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):  y  = y.detach().cpu().numpy()
    y = np.asarray(y)

    subtype_labels = np.unique(y)
    if subtype_names is None:
        subtype_names = [str(i) for i in subtype_labels]

    def mean_vectors(data):
        return {i: data[y == i].mean(axis=0) for i in subtype_labels}

    def corr_matrix(means_dict):
        mat = np.zeros((len(subtype_labels), len(subtype_labels)))
        for i in range(len(subtype_labels)):
            for j in range(len(subtype_labels)):
                r, _ = pearsonr(means_dict[subtype_labels[i]], means_dict[subtype_labels[j]])
                mat[i, j] = r
        return mat

    corr_1 = corr_matrix(mean_vectors(X1))
    corr_2 = corr_matrix(mean_vectors(X2))
    corr_z = corr_matrix(mean_vectors(Z))

    titles = ["View1 correlations", "View2 correlations", "Latent correlations"]
    mats = [corr_1, corr_2, corr_z]

    fig, axes = plt.subplots(1, 4, figsize=(3 * 4, 4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
                             squeeze=False)
    axes = axes[0]
    heat_axes = axes[:-1]
    cbar_ax = axes[-1]

    mappable = None
    for j, (ax, M, title) in enumerate(zip(heat_axes, mats, titles)):
        hm = sns.heatmap(
            M, annot=True, fmt=".2f",
            xticklabels=subtype_names, yticklabels=subtype_names,
            cmap="coolwarm", vmin=-1.0, vmax=1.0,
            cbar=False, ax=ax, linewidths=0.5, linecolor="white"
        )
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, pad=6)
        ax.tick_params(axis="x", labelrotation=30, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        if j == len(heat_axes) - 1:
            mappable = hm.collections[0]

        if j != 0:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

    if mappable is not None:
        cb = fig.colorbar(mappable, cax=cbar_ax)
        cb.ax.tick_params(labelsize=10)

    plt.tight_layout()

    if dir:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, "Subtype_Correlations.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_gating_alphas_stacked(alphas, dir=None):
    if isinstance(alphas, torch.Tensor):
        alphas_np = alphas.detach().cpu().numpy()
    else:
        alphas_np = np.asarray(alphas)

    n_samples = alphas_np.shape[0]
    indices = np.arange(n_samples)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(indices, alphas_np[:, 0], label="View1", color=OMIC_COLOR_MAP["specific1"])
    ax.bar(indices, alphas_np[:, 1], bottom=alphas_np[:, 0], label="View2", color=OMIC_COLOR_MAP["specific2"])

    ax.set_xlabel("Sample index", fontsize=6)
    ax.set_ylabel(r"$\alpha$", fontsize=6)
    ax.set_title("Fusion weights α (fixed 0.5/0.5; sanity check)", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6)

    plt.tight_layout()
    if dir:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, "GatingAlphas_StackedBar.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_importance(test_model, disease, omic_names=None, dir=None, top_k=15):
    omic_keys = ["specific1", "specific2"]
    if omic_names is None:
        omic_names = ["View1", "View2"]

    per_omic = []
    for idx, omic_key in enumerate(omic_keys, start=1):
        W = test_model.specific_modules[omic_key].get_generator_mask().detach().cpu().numpy()
        fi = np.sum(np.abs(W), axis=1)

        top_idx = np.argsort(fi)[::-1][:top_k]
        top_scores = fi[top_idx]

        featnames = _safe_featnames(disease, idx, W.shape[0])
        top_names = [featnames[i] for i in top_idx]

        W_clamped = W.copy()
        W_clamped[np.abs(W_clamped) <= sparsity_threshold] = 0.0
        W_top = W_clamped[top_idx, :]
        heatmap = W_top.T

        W_abs = np.abs(W)
        max_idx_per_lat = np.argmax(W_abs, axis=0)
        max_val_per_lat = W_abs[max_idx_per_lat, np.arange(W.shape[1])]
        max_name_per_lat = [featnames[i] for i in max_idx_per_lat]

        per_omic.append(dict(
            key=omic_key,
            display=omic_names[idx - 1],
            top_scores=top_scores,
            top_names=top_names,
            heatmap=heatmap,
            lat_vals=max_val_per_lat,
            lat_names=max_name_per_lat,
            latent_dim=W.shape[1],
        ))

    V = len(per_omic)

    # (1) Top-k bars
    fig, axes = plt.subplots(1, V, figsize=(6 * V, 5), squeeze=False)
    axes = axes[0]
    for j, info in enumerate(per_omic):
        ax = axes[j]
        ax.bar(range(top_k), info["top_scores"], color=OMIC_COLOR_MAP[info["key"]])
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(info["top_names"], rotation=30, ha="right", fontsize=10)
        ax.set_title(f"Top {top_k} features: {info['display']}", fontsize=14)
        if j == 0:
            ax.set_ylabel("Aggregated |W|", fontsize=12)

    plt.tight_layout()
    if dir:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, f"FeatureImportance_Top{top_k}_Bars.pdf"), dpi=300)
        plt.close(fig)
    else:
        plt.show()

    # (2) Heatmaps
    max_abs = max(np.max(np.abs(info["heatmap"])) for info in per_omic)
    vmin, vmax = -max_abs, max_abs

    fig, axes = plt.subplots(1, V + 1, figsize=(6 * V, 5),
                             gridspec_kw={"width_ratios": [1] * V + [0.05]},
                             squeeze=False)
    axes = axes[0]
    heat_axes = axes[:-1]
    cbar_ax = axes[-1]
    mappable = None

    for j, info in enumerate(per_omic):
        ax = heat_axes[j]
        hm = sns.heatmap(info["heatmap"], cmap="coolwarm", center=0, vmin=vmin, vmax=vmax,
                         linewidths=0.5, ax=ax, cbar=False)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(info["top_names"], rotation=30, ha="right", fontsize=10)
        ax.set_title(f"W heatmap: {info['display']}", fontsize=14)
        ax.set_ylabel("Latent factors" if j == 0 else "")

        if j == V - 1:
            mappable = hm.collections[0]

    if mappable is not None:
        fig.colorbar(mappable, cax=cbar_ax)

    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, "FeatureImportance_Heatmaps.pdf"), dpi=300)
        plt.close(fig)
    else:
        plt.show()

    # (3) Per-latent top feature
    fig, axes = plt.subplots(1, V, figsize=(6 * V, 7), squeeze=False)
    axes = axes[0]
    global_max = max(np.max(info["lat_vals"]) for info in per_omic)

    for j, info in enumerate(per_omic):
        ax = axes[j]
        y = np.arange(info["latent_dim"])
        ax.barh(y, info["lat_vals"], color=OMIC_COLOR_MAP[info["key"]])
        ax.set_yticks(y)
        ax.set_yticklabels(y, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, global_max * 1.25)

        for i, (val, name) in enumerate(zip(info["lat_vals"], info["lat_names"])):
            ax.text(val + 0.02, i, name, va="center", ha="left", fontsize=8)

        ax.set_title(f"Top feature per latent: {info['display']}", fontsize=14)
        ax.set_xlabel("Max |W|", fontsize=12)

    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, "FeatureImportance_LatentTop.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def plot_view_uncertainty(model, X_all, M_all=None, dir=None, batch_size=1024, tag="test"):
    if dir:
        os.makedirs(dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    d1 = model.specific_modules["specific1"].config["input_dim"]
    d2 = model.specific_modules["specific2"].config["input_dim"]

    X_all = np.asarray(X_all, dtype=np.float32)
    n = X_all.shape[0]

    if M_all is not None:
        M_all = np.asarray(M_all, dtype=np.float32)
        if M_all.shape != X_all.shape:
            raise ValueError(f"M_all shape {M_all.shape} != X_all shape {X_all.shape}")

    prec_exponent = float(getattr(model, "prec_exponent", 0.5))

    rows = []
    all_logvar1 = []
    all_logvar2 = []
    all_prec1 = []
    all_prec2 = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.tensor(X_all[start:end], dtype=torch.float, device=device)
        x1 = xb[:, :d1]
        x2 = xb[:, d1:d1 + d2]

        if M_all is not None:
            mb = torch.tensor(M_all[start:end], dtype=torch.float, device=device)
            m1 = mb[:, :d1]
            m2 = mb[:, d1:d1 + d2]
        else:
            m1 = None
            m2 = None
        
        if m1 is not None:
            m1 = (m1 > 0.5).float()
        if m2 is not None:
            m2 = (m2 > 0.5).float()
    

        mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var = _poems_encode_unpack(
            model, x1, x2, m1=m1, m2=m2
        )

        prec1 = torch.exp(-z1_log_var)
        prec2 = torch.exp(-z2_log_var)

        mean_logvar1 = z1_log_var.mean(dim=1)
        mean_logvar2 = z2_log_var.mean(dim=1)
        mean_prec1 = prec1.mean(dim=1)
        mean_prec2 = prec2.mean(dim=1)

        all_logvar1.append(z1_log_var.detach().cpu().numpy().ravel())
        all_logvar2.append(z2_log_var.detach().cpu().numpy().ravel())
        all_prec1.append(prec1.detach().cpu().numpy().ravel())
        all_prec2.append(prec2.detach().cpu().numpy().ravel())

        for i in range(end - start):
            rows.append({
                "sample_index": start + i,
                "alpha_view1": float(alphas[i, 0].item()),
                "alpha_view2": float(alphas[i, 1].item()),
                "mean_logvar_view1": float(mean_logvar1[i].item()),
                "mean_logvar_view2": float(mean_logvar2[i].item()),
                "mean_precision_view1": float(mean_prec1[i].item()),
                "mean_precision_view2": float(mean_prec2[i].item()),
            })

    df = pd.DataFrame(rows)
    if dir:
        df.to_csv(os.path.join(dir, f"ViewUncertainty_{tag}.csv"), index=False)

    logvar1 = np.concatenate(all_logvar1)
    logvar2 = np.concatenate(all_logvar2)
    prec1 = np.concatenate(all_prec1)
    prec2 = np.concatenate(all_prec2)

    plt.figure(figsize=(7, 3))
    plt.hist(logvar1, bins=100, alpha=0.6, label="View1 log_var", color=OMIC_COLOR_MAP["specific1"])
    plt.hist(logvar2, bins=100, alpha=0.6, label="View2 log_var", color=OMIC_COLOR_MAP["specific2"])
    plt.title(f"Latent log-variance distributions (diagnostic; not used for fusion) ({tag})")
    plt.xlabel("log_var")
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"ViewUncertainty_LogVarHist_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(7, 3))
    plt.hist(prec1, bins=100, alpha=0.6, label="View1 precision", color=OMIC_COLOR_MAP["specific1"])
    plt.hist(prec2, bins=100, alpha=0.6, label="View2 precision", color=OMIC_COLOR_MAP["specific2"])
    plt.title(f"Latent precision distributions exp(-log_var) (diagnostic; not used for fusion) ({tag})")
    plt.xlabel("precision")
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"ViewUncertainty_PrecisionHist_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(4, 3))
    plt.scatter(df["alpha_view2"], df["mean_precision_view2"], s=10, alpha=0.6)
    plt.title(f"alpha(View2) vs mean precision(View2) (diagnostic; alpha is fixed in No-PoE) ({tag})")
    plt.xlabel("alpha_view2")
    plt.ylabel("mean_precision_view2")
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"ViewUncertainty_Scatter_Alpha2_vs_Prec2_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_loss_history(dir=None, filename="loss_history.csv"):
    path = filename if dir is None else os.path.join(dir, filename)
    df = pd.read_csv(path)

    if "epoch" not in df.columns:
        df["epoch"] = np.arange(len(df))

    def _plot(cols, title, outname):
        cols = [c for c in cols if c in df.columns]
        if len(cols) == 0:
            return  # nothing to plot

        plt.figure(figsize=(7, 3))
        for c in cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend(fontsize=7)
        plt.tight_layout()

        if dir is not None:
            plt.savefig(os.path.join(dir, outname), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # Core losses
    _plot(["train_total_loss_all", "val_total_loss_all", "val_dn_total_loss_all"],
          "Total loss (train/val/val_dn)", "Loss_Total.pdf")
    _plot(["train_rec_loss_all", "val_rec_loss_all", "val_dn_rec_loss_all"],
          "Reconstruction loss", "Loss_Reconstruction.pdf")
    _plot(["train_kl_loss_all", "val_kl_loss_all", "val_dn_kl_loss_all"],
          "KL loss", "Loss_KL.pdf")
    _plot(["train_mask_loss_all", "val_mask_loss_all", "val_dn_mask_loss_all"],
          "Mask loss", "Loss_Mask.pdf")
    _plot(["train_sigma_loss_all", "val_sigma_loss_all", "val_dn_sigma_loss_all"],
          "Sigma loss", "Loss_Sigma.pdf")

    # Alpha diagnostics (50/50 check)
    _plot(["train_alpha_entropy", "val_alpha_entropy", "val_dn_alpha_entropy"],
          "Alpha entropy", "AlphaEntropy.pdf")

    _plot(["train_alpha_mean_v1", "train_alpha_mean_v2",
           "val_alpha_mean_v1", "val_alpha_mean_v2",
           "val_dn_alpha_mean_v1", "val_dn_alpha_mean_v2"],
          "Alpha means (expect ~0.5/0.5)", "AlphaMeans.pdf")

    # Effective diagnostics (in your no-PoE these should match alphas or be ~0.5/0.5)
    _plot(["train_eff_mean_v1", "train_eff_mean_v2",
           "val_eff_mean_v1", "val_eff_mean_v2",
           "val_dn_eff_mean_v1", "val_dn_eff_mean_v2"],
          "Effective means (no-PoE sanity)", "EffMeans.pdf")

    # Denoising gap plots
    _plot(["val_total_loss_all", "val_dn_total_loss_all"],
          "Val clean vs denoising total loss", "Loss_Val_CleanVsDN_Total.pdf")
    _plot(["val_rec_loss_all", "val_dn_rec_loss_all"],
          "Val clean vs denoising recon loss", "Loss_Val_CleanVsDN_Recon.pdf")

    # Print quick sanity summary (last epoch)
    last = df.iloc[-1]
    for prefix in ["train", "val", "val_dn"]:
        c1 = f"{prefix}_alpha_mean_v1"
        c2 = f"{prefix}_alpha_mean_v2"
        if c1 in df.columns and c2 in df.columns:
            print(f"[{prefix}] alpha_mean_v1={float(last[c1]):.4f} alpha_mean_v2={float(last[c2]):.4f} sum={float(last[c1]+last[c2]):.4f}")



def export_latent_trait_ranking(model, disease, dir=None, use_abs=True):
    """
    For each latent k, ranks ALL traits by |W2[trait,k]| (or signed if use_abs=False).
    Outputs:
      - LatentTraitRanking.csv  (long format)
      - LatentTraitRanking_Wide.csv (wide format: latent rows, trait columns)
    """
    if dir:
        os.makedirs(dir, exist_ok=True)

    W2 = model.specific_modules["specific2"].get_generator_mask().detach().cpu().numpy()  # (n_traits, K)
    trait_names = _safe_featnames(disease, view_idx=2, n_features=W2.shape[0])

    M = np.abs(W2) if use_abs else W2
    n_traits, K = M.shape

    rows = []
    for k in range(K):
        order = np.argsort(M[:, k])[::-1]  # descending
        for rank, t in enumerate(order, start=1):
            rows.append({
                "latent": k,
                "rank": rank,
                "trait": trait_names[t],
                "loading_abs" if use_abs else "loading": float(M[t, k]),
                "signed_loading": float(W2[t, k]),
            })

    df = pd.DataFrame(rows)

    if dir:
        df.to_csv(os.path.join(dir, "LatentTraitRanking.csv"), index=False)

        # wide view (nice for quick inspection)
        wide = pd.DataFrame(M.T, columns=trait_names)
        wide.insert(0, "latent", np.arange(K))
        wide.to_csv(os.path.join(dir, "LatentTraitRanking_Wide.csv"), index=False)

    return df


def export_latent_top_snps(
    model,
    disease,
    dir=None,
    top_snps=200,
    abs_w_threshold=1e-3,
    pstar_threshold=0.5,
    use_pstar=True,
):
    """
    For each latent k:
      - filters SNPs by (|W1[:,k]| > abs_w_threshold) AND optionally (pstar[:,k] > pstar_threshold)
      - ranks by |W1[:,k]|
    Outputs:
      - LatentTopSNPs.csv
      - LatentSNPCounts.csv
    """
    if dir:
        os.makedirs(dir, exist_ok=True)

    W1 = model.specific_modules["specific1"].get_generator_mask().detach().cpu().numpy()  # (n_snps, K)
    snp_names = _safe_featnames(disease, view_idx=1, n_features=W1.shape[0])

    # pstar is (n_snps, K)
    pstar = model.specific_modules["specific1"].pstar.detach().cpu().numpy()

    n_snps, K = W1.shape
    rows = []
    counts = []

    for k in range(K):
        absw = np.abs(W1[:, k])

        mask = absw > abs_w_threshold
        if use_pstar:
            mask = mask & (pstar[:, k] > pstar_threshold)

        idx = np.where(mask)[0]
        counts.append({
            "latent": k,
            "n_selected": int(idx.size),
            "abs_w_threshold": abs_w_threshold,
            "use_pstar": bool(use_pstar),
            "pstar_threshold": pstar_threshold if use_pstar else None,
        })

        if idx.size == 0:
            continue

        # rank selected by absW
        sel = idx[np.argsort(absw[idx])[::-1]]
        sel = sel[:top_snps]

        for rank, s in enumerate(sel, start=1):
            rows.append({
                "latent": k,
                "rank": rank,
                "snp": snp_names[s],
                "absW": float(absw[s]),
                "signedW": float(W1[s, k]),
                "pstar": float(pstar[s, k]),
            })

    df = pd.DataFrame(rows)
    df_counts = pd.DataFrame(counts)

    if dir:
        df.to_csv(os.path.join(dir, "LatentTopSNPs.csv"), index=False)
        df_counts.to_csv(os.path.join(dir, "LatentSNPCounts.csv"), index=False)

    return df, df_counts


def export_latent_fscore(Z, y, dir=None, tag="test"):
    """
    Ranks latent dimensions by how strongly they separate labels/classes y.
    Writes:
      - LatentImportance_Fscore_<tag>.csv
      - LatentImportance_Fscore_<tag>.pdf
    """
    if dir:
        os.makedirs(dir, exist_ok=True)

    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()
    y = np.asarray(y)

    # f_classif expects y to be discrete class labels
    F, p = f_classif(Z, y)
    order = np.argsort(F)[::-1]

    df = pd.DataFrame({
        "latent": np.arange(Z.shape[1]),
        "F_score": F,
        "p_value": p,
    }).sort_values("F_score", ascending=False)

    if dir:
        df.to_csv(os.path.join(dir, f"LatentImportance_Fscore_{tag}.csv"), index=False)

    # plot (bar of F-scores in ranked order)
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(order)), F[order])
    plt.xticks(np.arange(len(order)), [str(i) for i in order], fontsize=7)
    plt.xlabel("latent index (ranked)")
    plt.ylabel("F-score")
    plt.title(f"Latent importance by ANOVA F-score ({tag})")
    plt.tight_layout()

    if dir:
        plt.savefig(os.path.join(dir, f"LatentImportance_Fscore_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return df

    
@torch.no_grad()
def latent_ablation_importance(
    model,
    X_all,
    M_all=None,
    dir=None,
    tag="test",
    batch_size=128,
    max_samples=2000,
    seed=0,
):
    if dir:
        os.makedirs(dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    X_all = np.asarray(X_all, dtype=np.float32)
    if M_all is not None:
        M_all = np.asarray(M_all, dtype=np.float32)
        if M_all.shape != X_all.shape:
            raise ValueError(f"M_all shape {M_all.shape} != X_all shape {X_all.shape}")

    # subsample (apply SAME indices to mask)
    n = X_all.shape[0]
    if (max_samples is not None) and (n > max_samples):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_all = X_all[idx]
        if M_all is not None:
            M_all = M_all[idx]
        n = X_all.shape[0]

    d1 = model.specific_modules["specific1"].config["input_dim"]
    d2 = model.specific_modules["specific2"].config["input_dim"]
    K = model.specific_modules["specific1"].config["latent_dim"]

    base_sum1 = 0.0
    base_sum2 = 0.0
    base_den1 = 0.0
    base_den2 = 0.0

    ab_sum1 = np.zeros(K, dtype=np.float64)
    ab_sum2 = np.zeros(K, dtype=np.float64)
    ab_den1 = 0.0
    ab_den2 = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        xb = torch.tensor(X_all[start:end], dtype=torch.float, device=device)
        x1 = xb[:, :d1]
        x2 = xb[:, d1:d1 + d2]

        mb = None
        if M_all is not None:
            mb = torch.tensor(M_all[start:end], dtype=torch.float, device=device)
            m1 = mb[:, :d1]
            m2 = mb[:, d1:d1 + d2]
            den1 = float(m1.sum().item())
            den2 = float(m2.sum().item())
        else:
            m1 = None
            m2 = None
            den1 = float(x1.numel())
            den2 = float(x2.numel())

        if m1 is not None:
            m1 = (m1 > 0.5).float()
        if m2 is not None:
            m2 = (m2 > 0.5).float()
    

        # deterministic latent: mean_comb
        mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var = _poems_encode_unpack(
            model, x1, x2, m1=m1, m2=m2
        )


        # baseline recon
        x1b = model.specific_modules["specific1"].decode(mean_comb)
        x2b = model.specific_modules["specific2"].decode(mean_comb)

        err1 = (x1b - x1) ** 2
        err2 = (x2b - x2) ** 2

        if m1 is None:
            base_sum1 += float(err1.sum().item())
        else:
            base_sum1 += float((err1 * m1).sum().item())

        if m2 is None:
            base_sum2 += float(err2.sum().item())
        else:
            base_sum2 += float((err2 * m2).sum().item())

        base_den1 += den1
        base_den2 += den2

        # ablate each latent
        for k in range(K):
            z = mean_comb.clone()
            z[:, k] = 0.0

            x1k = model.specific_modules["specific1"].decode(z)
            x2k = model.specific_modules["specific2"].decode(z)

            err1k = (x1k - x1) ** 2
            err2k = (x2k - x2) ** 2

            if m1 is None:
                ab_sum1[k] += float(err1k.sum().item())
            else:
                ab_sum1[k] += float((err1k * m1).sum().item())

            if m2 is None:
                ab_sum2[k] += float(err2k.sum().item())
            else:
                ab_sum2[k] += float((err2k * m2).sum().item())

        ab_den1 += den1
        ab_den2 += den2

    base_mse1 = base_sum1 / max(1e-8, base_den1)
    base_mse2 = base_sum2 / max(1e-8, base_den2)
    base_mse_total = base_mse1 + base_mse2

    ab_mse1 = ab_sum1 / max(1e-8, ab_den1)
    ab_mse2 = ab_sum2 / max(1e-8, ab_den2)
    ab_mse_total = ab_mse1 + ab_mse2

    df = pd.DataFrame({
        "latent": np.arange(K),
        "base_mse_view1": base_mse1,
        "base_mse_view2": base_mse2,
        "base_mse_total": base_mse_total,
        "ablated_mse_view1": ab_mse1,
        "ablated_mse_view2": ab_mse2,
        "ablated_mse_total": ab_mse_total,
        "delta_mse_view1": ab_mse1 - base_mse1,
        "delta_mse_view2": ab_mse2 - base_mse2,
        "delta_mse_total": ab_mse_total - base_mse_total,
    }).sort_values("delta_mse_total", ascending=False)

    if dir:
        df.to_csv(os.path.join(dir, f"LatentImportance_Ablation_{tag}.csv"), index=False)

    plt.figure(figsize=(10, 3))
    order = df["latent"].values
    vals = df["delta_mse_total"].values
    plt.bar(np.arange(len(order)), vals)
    plt.xticks(np.arange(len(order)), [str(i) for i in order], fontsize=7)
    plt.xlabel("latent index (ranked)")
    plt.ylabel("Δ MSE (total)")
    plt.title(f"Latent importance by ablation (Δ recon error) ({tag})")
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"LatentImportance_Ablation_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return df



def plot_latent_pca_2d_3d(Z, y, dir=None, tag="test"):
    if dir:
        os.makedirs(dir, exist_ok=True)

    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()

    y = np.asarray(y).reshape(-1)

    if Z.shape[0] != y.shape[0]:
        raise ValueError(
            f"plot_latent_pca_2d_3d: Z has {Z.shape[0]} samples but y has {y.shape[0]} labels. "
            "Make sure you pass y_val with X_val_all, and y_test with X_test_all."
        )

    # make labels numeric for colormap (works for strings too)
    if np.issubdtype(y.dtype, np.number):
        y_codes = y.astype(int) if np.all(np.equal(np.mod(y, 1), 0)) else y
    else:
        y_codes, _ = pd.factorize(y)

    # 2D PCA
    p2 = PCA(n_components=2, random_state=0).fit_transform(Z)
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(p2[:, 0], p2[:, 1], c=y_codes, s=20, alpha=0.7, cmap="tab10")
    plt.title(f"Latent PCA 2D ({tag})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"LatentPCA2D_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # 3D PCA
    p3 = PCA(n_components=3, random_state=0).fit_transform(Z)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=y_codes, s=20, alpha=0.7, cmap="tab10")
    ax.set_title(f"Latent PCA 3D ({tag})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"LatentPCA3D_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================
# NEW: W and sigma diagnostics over epochs
# ============================================================

def summarize_W_np(W: np.ndarray, thresholds=(1e-4, 1e-3, 1e-2, 1e-1)) -> Dict[str, Any]:
    """
    Summarize a W matrix (n_features x latent_dim) as simple scalars.
    """
    absW = np.abs(W)
    flat = absW.ravel()

    summary = {
        "shape0_n_features": int(W.shape[0]),
        "shape1_latent_dim": int(W.shape[1]),
        "max_absW": float(flat.max()) if flat.size > 0 else 0.0,
        "mean_absW": float(flat.mean()) if flat.size > 0 else 0.0,
        "median_absW": float(np.median(flat)) if flat.size > 0 else 0.0,
        "q90_absW": float(np.quantile(flat, 0.90)) if flat.size > 0 else 0.0,
        "q99_absW": float(np.quantile(flat, 0.99)) if flat.size > 0 else 0.0,
        "q999_absW": float(np.quantile(flat, 0.999)) if flat.size > 0 else 0.0,
    }

    for thr in thresholds:
        mask = absW > thr
        cnt = int(mask.sum())
        frac = float(mask.mean()) if flat.size > 0 else 0.0
        summary[f"active_cnt_gt_{thr:g}"] = cnt
        summary[f"active_frac_gt_{thr:g}"] = frac

    return summary


def summarize_logsigmas_np(log_sigmas: np.ndarray, name: str) -> Dict[str, Any]:
    """
    Summarize a log_sigma vector/matrix.
    """
    flat = log_sigmas.ravel()
    if flat.size == 0:
        return {
            f"{name}_n": 0,
            f"{name}_min": 0.0,
            f"{name}_max": 0.0,
            f"{name}_mean": 0.0,
            f"{name}_median": 0.0,
            f"{name}_q10": 0.0,
            f"{name}_q90": 0.0,
        }

    return {
        f"{name}_n": int(flat.size),
        f"{name}_min": float(flat.min()),
        f"{name}_max": float(flat.max()),
        f"{name}_mean": float(flat.mean()),
        f"{name}_median": float(np.median(flat)),
        f"{name}_q10": float(np.quantile(flat, 0.10)),
        f"{name}_q90": float(np.quantile(flat, 0.90)),
    }


def export_W_and_sigma_stats(
    model,
    epoch: int,
    out_dir: str,
    view_keys=("specific1", "specific2"),
    thresholds=(1e-4, 1e-3, 1e-2, 1e-1),
):
    """
    Append diagnostics about W and log_sigmas to W_stats.csv.

    Call this from the training loop every N epochs.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for vk in view_keys:
        mod = model.specific_modules[vk]

        # W diagnostics
        W = mod.get_generator_mask().detach().cpu().numpy()
        W_summary = summarize_W_np(W, thresholds=thresholds)

        # log_sigma diagnostics
        log_sigmas = mod.log_sigmas.detach().cpu().numpy()
        sig_summary = summarize_logsigmas_np(log_sigmas, name=f"log_sigma_{vk}")

        row = {"epoch": epoch, "view": vk}
        row.update(W_summary)
        row.update(sig_summary)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "W_stats.csv")
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def plot_W_stats_over_epochs(out_dir: str):
    """
    Read W_stats.csv (generated by export_W_and_sigma_stats) and produce
    plots:

      - W_stats_max_mean_absW.pdf
      - W_stats_active_fracs.pdf
      - W_stats_logsigma_summary.pdf
    """
    csv_path = os.path.join(out_dir, "W_stats.csv")
    if not os.path.exists(csv_path):
        print(f"[plot_W_stats_over_epochs] No W_stats.csv found in {out_dir}")
        return

    df = pd.read_csv(csv_path)

    # 1) max_absW and mean_absW
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 4))
    for vk in df["view"].unique():
        sub = df[df["view"] == vk]
        ax.plot(sub["epoch"], sub["max_absW"], label=f"{vk} max|W|")
        ax.plot(sub["epoch"], sub["mean_absW"], linestyle="--", label=f"{vk} mean|W|")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|W|")
    ax.set_title("W magnitude over epochs")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "W_stats_max_mean_absW.pdf"))
    plt.close(fig)

    # 2) active fractions
    frac_cols = [c for c in df.columns if c.startswith("active_frac_gt_")]
    if frac_cols:
        plt.clf()
        fig, ax = plt.subplots(figsize=(7, 4))
        for vk in df["view"].unique():
            sub = df[df["view"] == vk]
            for col in frac_cols:
                ax.plot(sub["epoch"], sub[col], label=f"{vk} {col}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Active fraction")
        ax.set_title("Active fraction of W over epochs")
        ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "W_stats_active_fracs.pdf"))
        plt.close(fig)


    # 3) log_sigma summaries
    sig_cols = [c for c in df.columns if c.startswith("log_sigma_") and c.endswith(("_min", "_max", "_mean"))]
    if sig_cols:
        plt.clf()
        fig, ax = plt.subplots(figsize=(7, 4))

        for vk in df["view"].unique():
            sub = df[df["view"] == vk].sort_values("epoch")
            for col in sig_cols:
                y = pd.to_numeric(sub[col], errors="coerce")
                if y.notna().sum() >= 2:   # need at least 2 points to draw a line
                    ax.plot(sub["epoch"], y, label=f"{vk} {col}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("log_sigma stats")
        ax.set_title("log_sigma summary over epochs")
        ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "W_stats_logsigma_summary.pdf"))
        plt.close(fig)


# ============================================================
# NEW: loss history diagnostics
# ============================================================

def summarize_loss_history(out_dir: str, filename: str = "loss_history.csv"):
    """
    Quick summary of where each loss-like column reaches its minimum.
    Writes LossSummary.json in out_dir.
    """
    csv_path = os.path.join(out_dir, filename)
    if not os.path.exists(csv_path):
        print(f"[summarize_loss_history] No {filename} found in {out_dir}")
        return

    df = pd.read_csv(csv_path)
    summary = {"file": filename}

    cols_of_interest = [
        c for c in df.columns
        if "loss" in c or "kl_" in c or "sigma_" in c or "mask_" in c
    ]

    for col in cols_of_interest:
        series = df[col].values
        if len(series) == 0 or not np.isfinite(series).any():
            continue
        min_idx = int(np.nanargmin(series))
        min_val = float(series[min_idx])
        epoch = int(df["epoch"].iloc[min_idx])
        summary[col] = {"min_val": min_val, "min_epoch": epoch}

    json_path = os.path.join(out_dir, "LossSummary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summarize_loss_history] Wrote summary to {json_path}")


def plot_loss_components(out_dir: str, filename: str = "loss_history.csv"):
    """
    Multi-panel loss plots, complementing plot_loss_history.

    Produces:
      - LossComponents_total.pdf
      - LossComponents_rec_sigma_mask.pdf
    """
    csv_path = os.path.join(out_dir, filename)
    if not os.path.exists(csv_path):
        print(f"[plot_loss_components] No {filename} found in {out_dir}")
        return

    df = pd.read_csv(csv_path)
    epochs = df["epoch"].values

    # 1) total train/val
    train_total_cols = [c for c in df.columns if c.startswith("train_") and c.endswith("total_loss_all")]
    val_total_cols = [c for c in df.columns if c.startswith("val_") and c.endswith("total_loss_all")]

    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in train_total_cols:
        ax.plot(epochs, df[col].values, label=col)
    for col in val_total_cols:
        ax.plot(epochs, df[col].values, linestyle="--", label=col)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Train/Val total loss over epochs")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LossComponents_total.pdf"))
    plt.close(fig)

    # 2) rec vs sigma vs mask
    rec_cols = [c for c in df.columns if "rec_loss" in c]
    sigma_cols = [c for c in df.columns if "sigma_loss" in c]
    mask_cols = [c for c in df.columns if "mask_loss" in c]

    if rec_cols or sigma_cols or mask_cols:
        plt.clf()
        fig, ax = plt.subplots(figsize=(7, 4))
        for col in rec_cols:
            ax.plot(epochs, df[col].values, label=col)
        for col in sigma_cols:
            ax.plot(epochs, df[col].values, linestyle="--", label=col)
        for col in mask_cols:
            ax.plot(epochs, df[col].values, linestyle=":", label=col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss components")
        ax.set_title("Reconstruction vs sigma vs mask losses")
        ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "LossComponents_rec_sigma_mask.pdf"))
        plt.close(fig)

def plot_mask_loss_scaling(out_dir: str, filename="loss_history.csv"):
    import os, pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(out_dir, filename))
    epochs = df["epoch"].values

    plt.figure(figsize=(7,3))
    for col in ["train_mask_loss1","train_mask_loss2","val_mask_loss1","val_mask_loss2"]:
        if col in df.columns:
            plt.plot(epochs, df[col].values, label=col)
    plt.title("Mask loss (as used in training)")
    plt.xlabel("epoch"); plt.ylabel("mask loss")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "MaskLoss_TrainScale.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # If you log mask_loss_mean1/2:
    mean_cols = ["train_mask_loss_mean1","train_mask_loss_mean2","val_mask_loss_mean1","val_mask_loss_mean2"]
    if any(c in df.columns for c in mean_cols):
        plt.figure(figsize=(7,3))
        for col in mean_cols:
            if col in df.columns:
                plt.plot(epochs, df[col].values, label=col)
        plt.title("Mask loss normalized by |W| count (scale-invariant)")
        plt.xlabel("epoch"); plt.ylabel("mean mask penalty")
        plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "MaskLoss_Normalized.pdf"), dpi=300, bbox_inches="tight")
        plt.close()

def plot_kendall_calibration(out_dir: str, filename="loss_history.csv"):
    import os, pandas as pd, numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(out_dir, filename))
    epochs = df["epoch"].values

    # These exist already from your logging
    if not all(c in df.columns for c in ["train_log_s1","train_log_s2","train_rec_loss_raw1","train_rec_loss_raw2"]):
        print("[plot_kendall_calibration] Missing columns; check logging keys.")
        return

    pred_ratio = np.exp(2.0*(df["train_log_s1"].values - df["train_log_s2"].values))   # var1/var2
    obs_ratio  = (df["train_rec_loss_raw1"].values + 1e-12) / (df["train_rec_loss_raw2"].values + 1e-12)

    plt.figure(figsize=(7,3))
    plt.plot(epochs, pred_ratio, label="pred var ratio exp(2*(log_s1-log_s2))")
    plt.plot(epochs, obs_ratio,  label="obs loss ratio raw1/raw2", linestyle="--")
    plt.yscale("log")
    plt.title("Kendall vs observed recon ratio (train)")
    plt.xlabel("epoch"); plt.ylabel("ratio (log scale)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Kendall_Calibration_Ratios.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_kendall_contributions(out_dir: str, filename="loss_history.csv"):
    import os, pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(out_dir, filename))
    epochs = df["epoch"].values

    needed = ["train_kendall_w1","train_kendall_w2","train_rec_loss_raw1","train_rec_loss_raw2"]
    if not all(c in df.columns for c in needed):
        print("[plot_kendall_contributions] Missing columns; check logging.")
        return

    c1 = df["train_kendall_w1"] * df["train_rec_loss_raw1"]
    c2 = df["train_kendall_w2"] * df["train_rec_loss_raw2"]

    plt.figure(figsize=(7,3))
    plt.plot(epochs, c1, label="w1 * raw1")
    plt.plot(epochs, c2, label="w2 * raw2")
    plt.title("Kendall-weighted recon contributions (train)")
    plt.xlabel("epoch"); plt.ylabel("weighted recon term")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Kendall_WeightedContributions.pdf"), dpi=300, bbox_inches="tight")
    plt.close()



# ============================================================
# 1) Missingness distributions + mask heatmap subset
# ============================================================

def plot_missingness_histograms(M_all, omic1_dim, omic2_dim, out_dir, tag="train", question_text=None):
    mask1 = M_all[:, :omic1_dim]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mask1.mean(1), bins=30, color="#1f77b4", alpha=0.8)
    axes[0].set_title("Fraction Observed per Sample (SNPs)")
    axes[1].hist(mask1.mean(0), bins=30, color="#1f77b4", alpha=0.8)
    axes[1].set_title("Fraction Observed per SNP")
    fig.suptitle(f"Missingness Histograms ({tag}/view1)", fontsize=15)
    if question_text is None:
        question_text = "Q: How is missing data distributed across samples and features?"
    fig.text(0.01, 0.97, question_text, fontsize=9, va="top", ha="left")
    if out_dir:
        plt.savefig(os.path.join(out_dir, f'MissingnessHists_view1_{tag}.png'), bbox_inches="tight", dpi=160)
        plt.close(fig)
    else:
        plt.show()
        
def plot_mask_heatmap_subset(M_all, omic1_dim, omic2_dim, out_dir, tag="train", view="view1", n_features=None, question_text=None):
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    if view == "view1":
        mask = M_all[:, :omic1_dim]
    elif view == "view2":
        mask = M_all[:, omic1_dim:omic1_dim + (n_features or omic2_dim)]
    else:
        raise ValueError("view must be 'view1' or 'view2'")
    sns.heatmap(mask, cmap="Greys", cbar=False)
    plt.xlabel("Feature index")
    plt.ylabel("Sample index")
    plt.title(f"Missingness Heatmap ({tag}/{view})", fontsize=13)
    if question_text is None:
        question_text = "Q: What is the structure of missingness in samples and features?"
    _plot_annotation_text(ax, question_text)
    if out_dir:
        plt.savefig(os.path.join(out_dir, f'MissingnessHeatmap_{view}_{tag}.png'), bbox_inches="tight", dpi=160)
        plt.close()
    else:
        plt.show()

def plot_mask_latent_correlation(model, X_all, M_all, out_dir, tag="train", question_text=None):
    mean_comb, _ = model.get_final_embedding(X_all, M_all)
    mask_frac = M_all[:, :model.specific_modules["specific1"].config["input_dim"]].mean(1)
    for k in range(mean_comb.shape[1]):
        plt.figure()
        ax = plt.gca()
        plt.scatter(mask_frac, mean_comb[:, k], alpha=0.25)
        plt.xlabel("Fraction of SNPs observed (sample)")
        plt.ylabel(f"Value of latent dim {k}")
        plt.title(f"Latent {k} vs. SNP missingness (tag={tag})")
        text = (question_text or "Q: Does latent factor value reflect global missingness pattern?")
        _plot_annotation_text(ax, text)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mask_vs_latent{str(k)}_{tag}.png"), bbox_inches="tight", dpi=120)
        plt.close()
        
def plot_groupwise_mask_distribution(M_all, y, out_dir, tag="train", question_text=None):
    mask_frac = M_all.mean(1)
    group_labels = np.unique(y)
    data = [mask_frac[np.array(y) == g] for g in group_labels]
    plt.figure(figsize=(7,5))
    ax = plt.gca()
    plt.boxplot(data, labels=group_labels)
    plt.xlabel("Group")
    plt.ylabel("Fraction observed (per sample)")
    plt.title("Groupwise Masked Fraction")
    text = (question_text or "Q: Are any sample groups more missing overall? Technical covariate?")
    _plot_annotation_text(ax, text)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"groupwise_mask_fraction_{tag}.png"), bbox_inches="tight", dpi=140)
    plt.close()

def plot_mask_space_embedding(M_all, y, out_dir, tag="train", question_text=None):
    from sklearn.decomposition import PCA
    mask1 = M_all[:, :-1] if M_all.shape[1] > 1 else M_all
    pca = PCA(n_components=2)
    emb = pca.fit_transform(mask1)
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap='viridis', s=20, alpha=0.6)
    plt.title("PCA of Mask Patterns", fontsize=14)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    text = (question_text or
        "Q: Do samples cluster by group in MASK space? If so, missingness is informative.")
    _plot_annotation_text(ax, text)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"maskspace_pca_{tag}.png"), bbox_inches="tight", dpi=140)
    plt.close()


# ============================================================
# 2 & 3) Epoch curves from loss_history.csv (expects columns)
# ============================================================

def plot_epoch_recon_obs_vs_all(
    out_dir: str,
    filename: str = "loss_history.csv",
):
    """
    Plots observed-only vs all-entries recon MSE curves if present.

    Expected column names (recommendation):
      train_mse_obs1, train_mse_all1, val_mse_obs1, val_mse_all1
      train_mse_obs2, train_mse_all2, val_mse_obs2, val_mse_all2
    """
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        print(f"[plot_epoch_recon_obs_vs_all] Missing {path}")
        return

    df = pd.read_csv(path)
    epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))

    def _plot(view_idx):
        cols = [
            f"train_mse_obs{view_idx}", f"train_mse_all{view_idx}",
            f"val_mse_obs{view_idx}",   f"val_mse_all{view_idx}",
        ]
        have = [c for c in cols if c in df.columns]
        if not have:
            print(f"[plot_epoch_recon_obs_vs_all] No columns for view{view_idx}: {cols}")
            return

        plt.figure(figsize=(7, 3))
        for c in have:
            plt.plot(epochs, df[c].values, label=c)
        plt.title(f"Recon MSE: observed-only vs all entries (View{view_idx})")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ReconMSE_ObsVsAll_View{view_idx}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()

    _plot(1)
    _plot(2)


def _safe_div(num: float, den: float):
    return float(num / den) if den > 0 else float("nan")


@torch.no_grad()
def compute_recon_obs_all_imp_epoch_metrics(
    model,
    X_all,
    M_all,
    d1: int,
    d2: int,
    *,
    batch_size: int = 64,
    max_samples: int = 512,
    seed: int = 0,
    use_mean_latent: bool = True,
):
    """
    Computes MSE for:
      - observed entries only (mask==1)
      - imputed entries only (mask==0)
      - all entries

    Returns dict with:
      mse_obs1, mse_imp1, mse_all1, obs_frac1,
      mse_obs2, mse_imp2, mse_all2, obs_frac2
    """
    device = next(model.parameters()).device
    model.eval()

    X_all = np.asarray(X_all, dtype=np.float32)
    M_all = np.asarray(M_all, dtype=np.float32)

    n = X_all.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_all = X_all[idx]
        M_all = M_all[idx]
        n = X_all.shape[0]

    # accumulators
    obs_sum1 = imp_sum1 = all_sum1 = 0.0
    obs_cnt1 = imp_cnt1 = all_cnt1 = 0.0

    obs_sum2 = imp_sum2 = all_sum2 = 0.0
    obs_cnt2 = imp_cnt2 = all_cnt2 = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.tensor(X_all[start:end], dtype=torch.float32, device=device)
        mb = torch.tensor(M_all[start:end], dtype=torch.float32, device=device)

        x1 = xb[:, :d1]
        x2 = xb[:, d1:d1 + d2]
        m1 = mb[:, :d1]
        m2 = mb[:, d1:d1 + d2]
        # Ensure masks are exactly {0,1} (defensive; avoids float rounding issues)
        m1 = (m1 > 0.5).float()
        m2 = (m2 > 0.5).float()

        # deterministic latent is less noisy for monitoring
        mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var = _poems_encode_unpack(
            model, x1, x2, m1=m1, m2=m2
        )

        z_use = mean_comb if use_mean_latent else z

        x1_rec = model.specific_modules["specific1"].decode(z_use)
        x2_rec = model.specific_modules["specific2"].decode(z_use)

        err1 = (x1_rec - x1) ** 2
        err2 = (x2_rec - x2) ** 2

        # view1
        all_sum1 += err1.sum().item()
        all_cnt1 += float(err1.numel())

        obs_sum1 += (err1 * m1).sum().item()
        obs_cnt1 += m1.sum().item()

        im1 = (1.0 - m1)
        imp_sum1 += (err1 * im1).sum().item()
        imp_cnt1 += im1.sum().item()

        # view2
        all_sum2 += err2.sum().item()
        all_cnt2 += float(err2.numel())

        obs_sum2 += (err2 * m2).sum().item()
        obs_cnt2 += m2.sum().item()

        im2 = (1.0 - m2)
        imp_sum2 += (err2 * im2).sum().item()
        imp_cnt2 += im2.sum().item()

    out = {
        "mse_obs1": _safe_div(obs_sum1, obs_cnt1),
        "mse_imp1": _safe_div(imp_sum1, imp_cnt1),
        "mse_all1": _safe_div(all_sum1, all_cnt1),
        "obs_frac1": _safe_div(obs_cnt1, all_cnt1),

        "mse_obs2": _safe_div(obs_sum2, obs_cnt2),
        "mse_imp2": _safe_div(imp_sum2, imp_cnt2),
        "mse_all2": _safe_div(all_sum2, all_cnt2),
        "obs_frac2": _safe_div(obs_cnt2, all_cnt2),
    }
    return out



# ============================================================
# 4 & 5) Compute VAL/TEST metrics from model + masks
# ============================================================

@torch.no_grad()
def evaluate_recon_mask_metrics(
    model,
    X_all,
    M_all,
    d1,
    d2,
    batch_size=64,
    max_samples=None,
    seed=0,
    compute_effective=True,
):
    """
    Computes per-sample recon MSE components:
      - obs MSE (on mask==1)
      - imp MSE (on mask==0)
      - all MSE (on all entries)
    plus:
      - coverage counts
      - missing percentages
      - alphas (from model.encode)
      - effective fusion weights (optional)

    Returns dict of numpy arrays.
    """
    device = next(model.parameters()).device
    model.eval()

    X_all = _ensure_np(X_all).astype(np.float32)
    M_all = _ensure_np(M_all).astype(np.float32)

    n = X_all.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_all = X_all[idx]
        M_all = M_all[idx]
        n = X_all.shape[0]

    X1, X2 = _split_views_np(X_all, d1, d2)
    M1, M2 = _split_views_np(M_all, d1, d2)

    # storage
    obs1 = np.zeros(n, dtype=np.float64)
    imp1 = np.zeros(n, dtype=np.float64)
    all1 = np.zeros(n, dtype=np.float64)

    obs2 = np.zeros(n, dtype=np.float64)
    imp2 = np.zeros(n, dtype=np.float64)
    all2 = np.zeros(n, dtype=np.float64)

    cov1 = np.zeros(n, dtype=np.float64)
    cov2 = np.zeros(n, dtype=np.float64)

    alpha1 = np.zeros(n, dtype=np.float64)
    alpha2 = np.zeros(n, dtype=np.float64)

    eps = 1e-8

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        xb = torch.tensor(X_all[start:end], dtype=torch.float32, device=device)
        mb = torch.tensor(M_all[start:end], dtype=torch.float32, device=device)

        x1 = xb[:, :d1]
        x2 = xb[:, d1:d1+d2]
        m1 = mb[:, :d1]
        m2 = mb[:, d1:d1+d2]

        m1 = (m1 > 0.5).float()
        m2 = (m2 > 0.5).float()


        # deterministic: decode from mean_comb
        mean_comb, log_var_comb, z, alphas, z1_mean, z1_log_var, z2_mean, z2_log_var = _poems_encode_unpack(
            model, x1, x2, m1=m1, m2=m2
        )

        x1p = model.specific_modules["specific1"].decode(mean_comb)
        x2p = model.specific_modules["specific2"].decode(mean_comb)

        # errors
        e1 = (x1p - x1) ** 2
        e2 = (x2p - x2) ** 2

        # per-sample denominators
        den1_obs = m1.sum(dim=1)                         # observed count
        den1_imp = (1.0 - m1).sum(dim=1)                 # missing count
        den2_obs = m2.sum(dim=1)
        den2_imp = (1.0 - m2).sum(dim=1)

        # per-sample numerators
        num1_obs = (e1 * m1).sum(dim=1)
        num1_imp = (e1 * (1.0 - m1)).sum(dim=1)

        num2_obs = (e2 * m2).sum(dim=1)
        num2_imp = (e2 * (1.0 - m2)).sum(dim=1)

        # per-sample MSEs (nan if no entries)
        # per-sample MSEs with NaN when denom==0
        obs1_b = torch.where(
            den1_obs > 0,
            num1_obs / den1_obs,
            torch.full_like(num1_obs, float("nan"))
        )
        imp1_b = torch.where(
            den1_imp > 0,
            num1_imp / den1_imp,
            torch.full_like(num1_imp, float("nan"))
        )

        obs2_b = torch.where(
            den2_obs > 0,
            num2_obs / den2_obs,
            torch.full_like(num2_obs, float("nan"))
        )
        imp2_b = torch.where(
            den2_imp > 0,
            num2_imp / den2_imp,
            torch.full_like(num2_imp, float("nan"))
        )

        # write to numpy arrays
        obs1[start:end] = _ensure_np(obs1_b)
        imp1[start:end] = _ensure_np(imp1_b)
        all1[start:end] = _ensure_np(e1.mean(dim=1))

        obs2[start:end] = _ensure_np(obs2_b)
        imp2[start:end] = _ensure_np(imp2_b)
        all2[start:end] = _ensure_np(e2.mean(dim=1))


        cov1[start:end] = _ensure_np(den1_obs)
        cov2[start:end] = _ensure_np(den2_obs)

        alpha1[start:end] = _ensure_np(alphas[:, 0])
        alpha2[start:end] = _ensure_np(alphas[:, 1])

    miss_pct1 = 100.0 * (1.0 - cov1 / max(1.0, float(d1)))
    miss_pct2 = 100.0 * (1.0 - cov2 / max(1.0, float(d2)))

    out = dict(
        obs_mse1=obs1, imp_mse1=imp1, all_mse1=all1,
        obs_mse2=obs2, imp_mse2=imp2, all_mse2=all2,
        cov1=cov1, cov2=cov2,
        miss_pct1=miss_pct1, miss_pct2=miss_pct2,
        alpha1=alpha1, alpha2=alpha2,
    )

    # effective fusion weights (optional)
    if compute_effective:
        # reuse your existing util helper if present
        try:
            alpha_np, eff_np, _ = _compute_alphas_and_effective_weights(model, X_all, M_all=M_all, batch_size=1024)
            out["eff1"] = eff_np[:, 0]
            out["eff2"] = eff_np[:, 1]
        except Exception as e:
            print("[evaluate_recon_mask_metrics] Could not compute effective weights:", e)

    return out


def plot_coverage_vs_obs_error(metrics: dict, out_dir=None, tag="test"):
    """
    Produces:
      - CoverageVsObsError_View1_<tag>.pdf
      - CoverageVsObsError_View2_<tag>.pdf
    """
    _safe_makedirs(out_dir)

    # View1
    plt.figure(figsize=(5, 3))
    plt.scatter(metrics["cov1"], metrics["obs_mse1"], s=8, alpha=0.6)
    plt.xlabel("# observed entries (View1)")
    plt.ylabel("observed-only MSE (View1)")
    plt.title(f"Coverage vs observed recon error (View1) ({tag})")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"CoverageVsObsError_View1_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # View2
    plt.figure(figsize=(5, 3))
    plt.scatter(metrics["cov2"], metrics["obs_mse2"], s=8, alpha=0.6)
    plt.xlabel("# observed entries (View2)")
    plt.ylabel("observed-only MSE (View2)")
    plt.title(f"Coverage vs observed recon error (View2) ({tag})")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"CoverageVsObsError_View2_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_gating_vs_missingness(metrics: dict, out_dir=None, tag="test"):
    """
    Produces:
      - Alpha2_vs_Missing_View2_<tag>.pdf
      - Alpha2_vs_Missing_View1_<tag>.pdf
    """
    _safe_makedirs(out_dir)

    # alpha2 vs missing view2
    plt.figure(figsize=(5, 3))
    plt.scatter(metrics["miss_pct2"], metrics["alpha2"], s=8, alpha=0.6)
    plt.xlabel("% missing (View2)")
    plt.ylabel("alpha_view2")
    plt.title(f"alpha(View2) vs missingness(View2) ({tag})")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"Alpha2_vs_Missing_View2_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # alpha2 vs missing view1
    plt.figure(figsize=(5, 3))
    plt.scatter(metrics["miss_pct1"], metrics["alpha2"], s=8, alpha=0.6)
    plt.xlabel("% missing (View1)")
    plt.ylabel("alpha_view2")
    plt.title(f"alpha(View2) vs missingness(View1) ({tag})")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"Alpha2_vs_Missing_View1_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_effective_vs_missingness(metrics: dict, out_dir=None, tag="test"):
    """
    Produces (if eff2 exists):
      - Eff2_vs_Missing_View2_<tag>.pdf
      - Eff2_vs_Missing_View1_<tag>.pdf
    """
    _safe_makedirs(out_dir)
    if "eff2" not in metrics:
        print("[plot_effective_vs_missingness] metrics has no eff2; run evaluate_recon_mask_metrics(compute_effective=True).")
        return

    plt.figure(figsize=(5, 3))
    plt.scatter(metrics["miss_pct2"], metrics["eff2"], s=8, alpha=0.6)
    plt.xlabel("% missing (View2)")
    plt.ylabel("effective_weight_view2")
    plt.title(f"Effective(View2) vs missingness(View2) ({tag})")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"Eff2_vs_Missing_View2_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(5, 3))
    plt.scatter(metrics["miss_pct1"], metrics["eff2"], s=8, alpha=0.6)
    plt.xlabel("% missing (View1)")
    plt.ylabel("effective_weight_view2")
    plt.title(f"Effective(View2) vs missingness(View1) ({tag})")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"Eff2_vs_Missing_View1_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# ============================================================
# 6) Imputation baseline: training-mean baseline vs model
# ============================================================

def compute_masked_feature_means(X_train_all, M_train_all, chunk_size=16384):
    """
    Compute per-feature training mean using only observed entries.
    Returns means_all shape (D,).
    """
    X = _ensure_np(X_train_all).astype(np.float64)
    M = _ensure_np(M_train_all).astype(np.float64)

    D = X.shape[1]
    means = np.zeros(D, dtype=np.float64)

    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        Xc = X[:, start:end]
        Mc = M[:, start:end]
        num = (Xc * Mc).sum(axis=0)
        den = Mc.sum(axis=0)
        means[start:end] = np.where(den > 0, num / (den + 1e-12), 0.0)

    return means


def masked_baseline_obs_mse(X_all, M_all, means_all, chunk_size=16384):
    """
    Baseline observed-only MSE without materializing a full prediction matrix.
    Returns scalar observed-only MSE across all samples/features where mask==1.
    """
    X = _ensure_np(X_all).astype(np.float64)
    M = _ensure_np(M_all).astype(np.float64)
    means = _ensure_np(means_all).astype(np.float64)

    D = X.shape[1]
    num_total = 0.0
    den_total = 0.0

    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        Xc = X[:, start:end]
        Mc = M[:, start:end]
        mu = means[start:end][None, :]

        err2 = (Xc - mu) ** 2
        num_total += (err2 * Mc).sum()
        den_total += Mc.sum()

    return float(num_total / (den_total + 1e-12))


def plot_baseline_vs_model_bar(
    model_metrics: dict,
    baseline_mse_view1: float,
    baseline_mse_view2: float,
    out_dir=None,
    tag="test",
):
    """
    model_metrics: output of evaluate_recon_mask_metrics (contains obs_mse1/obs_mse2 per sample)
    Produces:
      - BaselineVsModel_ObsMSE_<tag>.pdf
    """
    _safe_makedirs(out_dir)
    model_mse1 = float(np.nanmean(model_metrics["obs_mse1"]))
    model_mse2 = float(np.nanmean(model_metrics["obs_mse2"]))

    labels = ["View1", "View2"]
    baseline = [baseline_mse_view1, baseline_mse_view2]
    model = [model_mse1, model_mse2]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(5, 3))
    plt.bar(x - w/2, baseline, width=w, label="baseline (train mean)")
    plt.bar(x + w/2, model, width=w, label="model")
    plt.xticks(x, labels)
    plt.ylabel("observed-only MSE")
    plt.title(f"Baseline vs Model (observed-only) ({tag})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"BaselineVsModel_ObsMSE_{tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================
# CSV plotters (summary figures for saved outputs)
# ============================================================

def plot_within_factor_feature_ablation(out_dir, csv_name="WithinFactorFeatureAblation_View1.csv", top_k=30):
    path = os.path.join(out_dir, csv_name)
    if not os.path.exists(path):
        print(f"[plot_within_factor_feature_ablation] Missing: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[plot_within_factor_feature_ablation] Empty: {path}")
        return

    required = ["latent", "delta_mse"]
    if not all(c in df.columns for c in required):
        print("[plot_within_factor_feature_ablation] Missing columns. Found:", list(df.columns))
        return

    df = df.sort_values("delta_mse", ascending=False)
    top = df.head(top_k)

    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(top)), top["delta_mse"].values)
    plt.xticks(np.arange(len(top)), top["latent"].astype(int).astype(str).values, fontsize=7)
    plt.xlabel("latent (ranked)")
    plt.ylabel("ΔMSE (ablated - base)")
    plt.title(f"Within-factor feature ablation (View1) top {top_k}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "WithinFactorFeatureAblation_View1_DeltaMSE_TopK.png"), dpi=200)
    plt.close()

    # optional: base vs ablated for same top_k
    if "base_mse" in df.columns and "ablated_mse" in df.columns:
        plt.figure(figsize=(10, 3))
        x = np.arange(len(top))
        plt.plot(x, top["base_mse"].values, marker="o", label="base_mse")
        plt.plot(x, top["ablated_mse"].values, marker="o", label="ablated_mse")
        plt.xticks(x, top["latent"].astype(int).astype(str).values, fontsize=7)
        plt.xlabel("latent (ranked)")
        plt.ylabel("MSE")
        plt.title(f"Within-factor ablation: base vs ablated (top {top_k})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "WithinFactorFeatureAblation_View1_BaseVsAblated_TopK.png"), dpi=200)
        plt.close()

def plot_latent_ablation_imputation_importance(out_dir, csv_name="LatentAblation_ImputationImportance_View1.csv", top_k=30):
    path = os.path.join(out_dir, csv_name)
    if not os.path.exists(path):
        print(f"[plot_latent_ablation_imputation_importance] Missing: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[plot_latent_ablation_imputation_importance] Empty: {path}")
        return

    # flexible column name
    delta_col_candidates = ["delta_drop_mse1", "delta_mse", "delta"]
    delta_col = next((c for c in delta_col_candidates if c in df.columns), None)

    if "latent" not in df.columns or delta_col is None:
        print("[plot_latent_ablation_imputation_importance] Missing columns. Found:", list(df.columns))
        return

    df = df.sort_values(delta_col, ascending=False)
    top = df.head(top_k)

    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(top)), top[delta_col].values)
    plt.xticks(np.arange(len(top)), top["latent"].astype(int).astype(str).values, fontsize=7)
    plt.xlabel("latent (ranked)")
    plt.ylabel(f"{delta_col} (ablated - base)")
    plt.title(f"Latent importance for imputation (View1 dropped entries) top {top_k}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LatentAblation_ImputationImportance_View1_TopK.png"), dpi=200)
    plt.close()

def plot_permtest_latent_group(out_dir, csv_name="PermTest_LatentGroup_test_groupOriented.csv"):
    path = os.path.join(out_dir, csv_name)
    if not os.path.exists(path):
        print(f"[plot_permtest_latent_group] Missing: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[plot_permtest_latent_group] Empty: {path}")
        return

    needed = ["latent", "diff_obs_meanA_minus_meanB", "p_emp"]
    if not all(c in df.columns for c in needed):
        print("[plot_permtest_latent_group] Missing columns. Found:", list(df.columns))
        return

    # effect size bar
    df2 = df.sort_values("diff_obs_meanA_minus_meanB", ascending=False)

    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(df2)), df2["diff_obs_meanA_minus_meanB"].values)
    plt.xticks(np.arange(len(df2)), df2["latent"].astype(int).astype(str).values, fontsize=7)
    plt.xlabel("latent")
    plt.ylabel("mean(groupA) - mean(groupB)")
    plt.title("Group contrast per latent (oriented so + means groupA)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "PermTest_LatentGroup_Effect.png"), dpi=200)
    plt.close()

    # -log10(p)
    plt.figure(figsize=(10, 3))
    p = np.clip(df2["p_emp"].values, 1e-300, 1.0)
    plt.bar(np.arange(len(df2)), -np.log10(p))
    plt.xticks(np.arange(len(df2)), df2["latent"].astype(int).astype(str).values, fontsize=7)
    plt.xlabel("latent")
    plt.ylabel("-log10(p_emp)")
    plt.title("Permutation p-values per latent")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "PermTest_LatentGroup_NegLog10p.png"), dpi=200)
    plt.close()

def plot_latent_group_ranking(out_dir, csv_name="LatentGroupRanking_test.csv", top_k=30):
    path = os.path.join(out_dir, csv_name)
    if not os.path.exists(path):
        print(f"[plot_latent_group_ranking] Missing: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[plot_latent_group_ranking] Empty: {path}")
        return

    # flexible
    dcol = "abs_cohen_d" if "abs_cohen_d" in df.columns else ("cohen_d" if "cohen_d" in df.columns else None)
    if "latent" not in df.columns or dcol is None:
        print("[plot_latent_group_ranking] Missing columns. Found:", list(df.columns))
        return

    top = df.head(top_k)

    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(top)), top[dcol].values)
    plt.xticks(np.arange(len(top)), top["latent"].astype(int).astype(str).values, fontsize=7)
    plt.xlabel("latent (ranked)")
    plt.ylabel(dcol)
    plt.title(f"Latent group separation ranking (top {top_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LatentGroupRanking_TopK.png"), dpi=200)
    plt.close()

def plot_latent_scores_by_group(out_dir, scores_csv="LatentScores_test.csv", max_latents=12):
    path = os.path.join(out_dir, scores_csv)
    if not os.path.exists(path):
        print(f"[plot_latent_scores_by_group] Missing: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[plot_latent_scores_by_group] Empty: {path}")
        return

    if "label" not in df.columns:
        print("[plot_latent_scores_by_group] Missing 'label' column. Found:", list(df.columns))
        return

    z_cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    if not z_cols:
        print("[plot_latent_scores_by_group] No z* columns found.")
        return

    z_cols = z_cols[:max_latents]

    # long format for seaborn
    dfl = df.melt(id_vars=["label"], value_vars=z_cols, var_name="latent", value_name="score")

    plt.figure(figsize=(12, 4))
    sns.boxplot(data=dfl, x="latent", y="score", hue="label")
    plt.title(f"Latent score distributions by group ({scores_csv})")
    plt.xlabel("latent")
    plt.ylabel("score")
    plt.legend(fontsize=8, title="label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"LatentScores_ByGroup_{os.path.splitext(scores_csv)[0]}.png"), dpi=200)
    plt.close()

def plot_latent_orientation_by_toptrait(out_dir, csv_name="LatentOrientation_byTopTrait.csv", top_k=32):
    path = os.path.join(out_dir, csv_name)
    if not os.path.exists(path):
        print(f"[plot_latent_orientation_by_toptrait] Missing: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[plot_latent_orientation_by_toptrait] Empty: {path}")
        return

    if not all(c in df.columns for c in ["latent", "corr"]):
        print("[plot_latent_orientation_by_toptrait] Missing columns. Found:", list(df.columns))
        return

    df = df.copy()
    df["abs_corr"] = np.abs(df["corr"])
    df = df.sort_values("abs_corr", ascending=False).head(top_k)

    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(df)), df["abs_corr"].values)
    plt.xticks(np.arange(len(df)), df["latent"].astype(int).astype(str).values, fontsize=7)
    plt.xlabel("latent (ranked)")
    plt.ylabel("|corr| with anchor trait")
    plt.title("Top-trait orientation strength per latent")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LatentOrientation_byTopTrait_absCorr.png"), dpi=200)
    plt.close()
    
def plot_all_csv_summaries(out_dir):
            plot_within_factor_feature_ablation(out_dir, "WithinFactorFeatureAblation_View1.csv")
            plot_latent_ablation_imputation_importance(out_dir, "LatentAblation_ImputationImportance_View1.csv")
            plot_permtest_latent_group(out_dir, "PermTest_LatentGroup_test_groupOriented.csv")
            plot_latent_group_ranking(out_dir, "LatentGroupRanking_test.csv")
            plot_latent_scores_by_group(out_dir, "LatentScores_test.csv")
            plot_latent_scores_by_group(out_dir, "LatentScores_test_oriented_byGroup.csv")
            plot_latent_orientation_by_toptrait(out_dir, "LatentOrientation_byTopTrait.csv")
    
