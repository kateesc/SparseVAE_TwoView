import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = os.path.abspath(os.path.join(Path(__file__).resolve(), ".."))
root_dir = os.path.abspath(os.path.join(cwd, "../.."))
sys.path.insert(1, root_dir)

print("Current working directory: ", cwd)
print("Updated path: ", root_dir)
print(sys.path)

import setup_seed
from load_data_mocs import load_data_mocs
from all_models.POEMS.models import POEMS
import torch.optim as optim
import wandb
from helper import EarlyStopper, perform_kmeans, perform_knn, get_sigma_params
import util
print("UTIL IMPORTED FROM:", util.__file__)

def _masked_mse_sse_cnt(x_pred, x_true, mask, eps=1e-8):
    err2 = (x_pred - x_true) ** 2
    w = mask.to(dtype=err2.dtype)
    sse = (err2 * w).sum().item()
    cnt = w.sum().item()
    return sse, cnt

def train_POEMS(
    lr_in, wd_in, batch_size_in, nepoch_in, is_wandb,
    experiment_note, disease, is_test, model_name="POEMS",
):
    setup_seed.setup_seed_21()

    patience = 30
    lr = lr_in
    wd = wd_in
    batch_size = batch_size_in
    nepoch = nepoch_in

    if is_test:
        trained_model_name = "trained-2view-model"
        project_name = trained_model_name
        out_dir = os.path.join(cwd, "results", trained_model_name) + "/"
        model_dir = os.path.join(cwd, "trained", trained_model_name) + "/"
    else:
        train_name = "train_POEMS_2view"
        project_name = f"{experiment_note}--{train_name}--{model_name}"
        out_dir = os.path.join(cwd, "results", project_name) + "/"
        model_dir = os.path.join(cwd, "trained", project_name) + "/"

    # ---- LOAD DATA (2 views) ----
    (omic1_dim, omic2_dim,
    X_train_all, X_val_all, X_test_all,
    y_train, y_val, y_test,
    n_clusters,
    M_train_all, M_val_all, M_test_all,
    sample_ids, idx_train, idx_val, idx_test) = load_data_mocs(disease=disease)

    base_gamma = 1.0
    cap = 200.0
    ratio = omic1_dim / max(1, omic2_dim)
    gamma_mask2 = base_gamma
    gamma_mask1 = base_gamma * min(ratio, cap)
    print(f"[mask-gamma] base_gamma={base_gamma} cap={cap} ratio={ratio:.2f} "
          f"gamma_mask1={gamma_mask1:.2f} gamma_mask2={gamma_mask2:.2f}")

    if is_wandb:
        wandb.init(project=project_name)
        wandb.config.update({
            "lr": lr, "wd": wd, "patience": patience,
            "n_epoch": nepoch, "batch_size": batch_size,
        }, allow_val_change=True)
        out_dir = out_dir + str(wandb.run.name) + "/"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train_all, dtype=torch.float),
        torch.tensor(M_train_all, dtype=torch.float),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val_all, dtype=torch.float),
        torch.tensor(M_val_all, dtype=torch.float),
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    util.plot_missingness_histograms(M_train_all, omic1_dim, omic2_dim, out_dir, tag="train")
    util.plot_missingness_histograms(M_val_all,   omic1_dim, omic2_dim, out_dir, tag="val")
    util.plot_missingness_histograms(M_test_all,  omic1_dim, omic2_dim, out_dir, tag="test")
    util.plot_mask_heatmap_subset(M_train_all, omic1_dim, omic2_dim, out_dir, tag="train", view="view1")
    util.plot_mask_heatmap_subset(M_train_all, omic1_dim, omic2_dim, out_dir, tag="train", view="view2", n_features=omic2_dim)

    means_all = util.compute_masked_feature_means(X_train_all, M_train_all)
    means1 = means_all[:omic1_dim]
    means2 = means_all[omic1_dim:omic1_dim + omic2_dim]
    b1 = util.masked_baseline_obs_mse(X_test_all[:, :omic1_dim], M_test_all[:, :omic1_dim], means1)
    b2 = util.masked_baseline_obs_mse(
        X_test_all[:, omic1_dim:omic1_dim + omic2_dim],
        M_test_all[:, omic1_dim:omic1_dim + omic2_dim],
        means2
    )

    if model_name != "POEMS":
        raise ValueError("Only model_name='POEMS' supported.")

    row_normalize = False

    sigmas_init_1 = np.std(X_train_all[:, :omic1_dim], axis=0) + 1e-6
    sig_df_1, sig_scale_1 = get_sigma_params(sigmas_init_1, disease)
    omic1_info = dict(
        input_dim=omic1_dim,
        latent_dim=32,
        sigmas_init=sigmas_init_1,
        sig_df=sig_df_1,
        sig_scale=sig_scale_1,
        row_normalize=row_normalize,
    )
    omic1_info["use_mask_input"] = True
    sigmas_init_2 = np.std(X_train_all[:, omic1_dim:omic1_dim + omic2_dim], axis=0) + 1e-6
    sig_df_2, sig_scale_2 = get_sigma_params(sigmas_init_2, disease)
    omic2_info = dict(
        input_dim=omic2_dim,
        latent_dim=32,
        sigmas_init=sigmas_init_2,
        sig_df=sig_df_2,
        sig_scale=sig_scale_2,
        row_normalize=row_normalize,
    )
    omic2_info["use_mask_input"] = True

    model = POEMS(batch_size, omic1_info, omic2_info).to(device)
    if is_wandb:
        wandb.config.update({
            "lambda0": model.specific_modules["specific1"].lambda0,
            "lambda1": model.specific_modules["specific1"].lambda1,
        }, allow_val_change=True)

    beta_kl = 1.0
    beta_kl_views = 0.1
    if is_wandb:
        wandb.config.update({
            "beta_kl": beta_kl,
            "beta_kl_views": beta_kl_views,
        }, allow_val_change=True)

    best_path = os.path.join(model_dir, "model_best.pth")
    last_path = os.path.join(model_dir, "model_last.pth")
    canonical_path = os.path.join(model_dir, "model.pth")
    def _safe_torch_save(state_dict, path):
        tmp = path + ".tmp"
        torch.save(state_dict, tmp)
        os.replace(tmp, path)

    import copy

    if not is_test:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        early_stopper = EarlyStopper(patience=patience, min_delta=0)
        best_score = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        loss_history = []

        for epoch in range(1, nepoch + 1):
            model.train()
            train_loss_dict = init_loss_dict("train", device=device)
            train_n = 0

            for data, mask in train_loader:
                data = data.to(device)
                mask = mask.to(device)

                bs = data.shape[0]
                train_n += bs
                optimizer.zero_grad(set_to_none=True)

                # ---- split views + masks ----
                x1 = data[:, :omic1_dim]
                x2 = data[:, omic1_dim:omic1_dim + omic2_dim]
                m1 = mask[:, :omic1_dim]
                m2 = mask[:, omic1_dim:omic1_dim + omic2_dim]

                # 🚫 NO artificial dropout! Only use original missing values.
                x1_in = x1
                m1_in = m1
                x2_in = x2
                m2_in = m2

                new_loss_values = model.all_loss(
                    x1, x2,
                    m1=m1, m2=m2,            # Loss/reconstruction mask (observed entries)
                    x1_in=x1_in, x2_in=x2_in,
                    m1_in=m1_in, m2_in=m2_in,  # Inference input (true missing only)
                )

                loss = update_loss_dict_return_total_loss(
                    loss_dict=train_loss_dict,
                    new_loss_values=new_loss_values,
                    beta_kl=beta_kl,
                    beta_kl_views=beta_kl_views,
                    mode="train",
                    bs=bs,
                    gamma_mask1=gamma_mask1,
                    gamma_mask2=gamma_mask2,
                    gamma_sigma=1.0,
                )
                loss.backward()
                optimizer.step()

                # E-step style parameter updates (retained)
                with torch.no_grad():
                    for net_key in ["specific1", "specific2"]:
                        mod = model.specific_modules[net_key]
                        mod.log_sigmas.data.clamp_(min=mod.min_log_sigma.item())
                        # spike-slab update, same as before
                        pstar = mod.pstar
                        thetas = mod.thetas
                        lambda0 = mod.lambda0
                        lambda1 = mod.lambda1
                        W = mod.get_generator_mask()
                        a = mod.a
                        b = mod.b
                        input_dim = mod.config["input_dim"]
                        for k in range(pstar.shape[1]):
                            num = thetas[k] * torch.exp(-lambda1 * W[:, k].abs())
                            den = num + (1 - thetas[k]) * torch.exp(-lambda0 * W[:, k].abs())
                            pstar[:, k].copy_(num / (den + 1e-12))
                            thetas[k].copy_((pstar[:, k].sum() + a - 1) / (a + b + input_dim - 2))

            train_loss_dict = {k: v / float(train_n) for k, v in train_loss_dict.items()}

            # ---- VALIDATION (ONLY ON TRUE NON-MISSING) ----
            model.eval()
            val_loss_dict = init_loss_dict("val", device=device)
            val_n = 0
            with torch.no_grad():
                for data, mask in val_loader:
                    data = data.to(device)
                    mask = mask.to(device)

                    bs = data.shape[0]
                    val_n += bs

                    x1 = data[:, :omic1_dim]
                    x2 = data[:, omic1_dim:omic1_dim + omic2_dim]
                    m1 = mask[:, :omic1_dim]
                    m2 = mask[:, omic1_dim:omic1_dim + omic2_dim]

                    # No denoising eval: only true missing handled
                    new_loss_values = model.all_loss(
                        x1, x2,
                        m1=m1, m2=m2,
                        x1_in=x1, x2_in=x2,
                        m1_in=m1, m2_in=m2,
                    )

                    _ = update_loss_dict_return_total_loss(
                        loss_dict=val_loss_dict,
                        new_loss_values=new_loss_values,
                        beta_kl=beta_kl,
                        beta_kl_views=beta_kl_views,
                        mode="val",
                        bs=bs,
                        gamma_mask1=gamma_mask1,
                        gamma_mask2=gamma_mask2,
                        gamma_sigma=1.0,
                    )

            val_loss_dict = {k: v / float(val_n) for k, v in val_loss_dict.items()}
            val_loss = val_loss_dict["val_total_loss_all"].item()
            # Early stopping on validation loss only
            early_stop_score = val_loss

            if early_stop_score < best_score:
                best_score = early_stop_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                _safe_torch_save({
                    "epoch": epoch,
                    "model": best_state,
                    "optim": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                }, best_path)

            is_early_stop = early_stopper.early_stop(early_stop_score)

            # ---- Diagnostics, Logging ----
            row = {
                "epoch": epoch,
                "early_stop_score": float(early_stop_score),
            }
            row.update({k: _to_float(v) for k, v in train_loss_dict.items()})
            row.update({k: _to_float(v) for k, v in val_loss_dict.items()})

            # add MSE on observed only for reporting
            row["train_mse_obs1"] = 2.0 * row["train_rec_loss1"]
            row["train_mse_obs2"] = 2.0 * row["train_rec_loss2"]
            row["val_mse_obs1"]   = 2.0 * row["val_rec_loss1"]
            row["val_mse_obs2"]   = 2.0 * row["val_rec_loss2"]
            loss_history.append(row)
            if epoch % 5 == 0 or epoch == 1:
                pd.DataFrame(loss_history).to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)

            if is_early_stop:
                print(f"Early stopping at epoch {epoch} (BEST score={best_score:.4f} @ epoch {best_epoch})")
                break

        # After training: load BEST model & save canonical weights
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(best_state)
        _safe_torch_save(model.state_dict(), canonical_path)
        print(f"Saved BEST checkpoint to {canonical_path} (best_epoch={best_epoch}, best_score={best_score:.4f})")
    else:
        best_epoch = 0
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
        elif os.path.exists(canonical_path):
            model.load_state_dict(torch.load(canonical_path, map_location=device, weights_only=True))
        elif os.path.exists(last_path):
            ckpt = torch.load(last_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
        else:
            raise FileNotFoundError("No checkpoint found (best/canonical/last).")

    # =========================
    # TEST EVAL (runs on BEST)
    # =========================
    model.eval()
    with torch.no_grad():
        final_embedding, alphas = model.get_final_embedding(X_test_all, M_test_all)
        util.plot_latent_pca_2d_3d(final_embedding, y_test, dir=out_dir, tag="test")
        util.export_latent_fscore(final_embedding, y_test, dir=out_dir, tag="test")
        util.latent_ablation_importance(
            model, X_test_all, M_all=M_test_all, dir=out_dir, tag="test",
            batch_size=64, max_samples=500, seed=0,
        )
        util.plot_view_uncertainty(model, X_test_all, M_all=M_test_all, dir=out_dir, batch_size=1024, tag="test")
        util.export_latent_scores_csv(
            model, X_train_all, M_train_all, sample_ids[idx_train], y_train,
            out_dir, tag="train", batch_size=1024, include_view_latents=False)
        util.export_latent_scores_csv(
            model, X_val_all, M_val_all, sample_ids[idx_val], y_val,
            out_dir, tag="val", batch_size=1024, include_view_latents=False)
        util.export_latent_scores_csv(
            model, X_test_all, M_test_all, sample_ids[idx_test], y_test,
            out_dir, tag="test", batch_size=1024, include_view_latents=False)
        # (other reporting/plotting unchanged)

def _to_float(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)

def update_loss_dict_return_total_loss(
    loss_dict, new_loss_values, beta_kl, beta_kl_views, mode, *,
    bs, gamma_mask1, gamma_mask2, gamma_sigma=1.0,
):
    w_bs = float(bs)
    rec_loss_all   = new_loss_values["rec_loss"][0] + new_loss_values["rec_loss"][1]
    mask_loss_all  = new_loss_values["mask_loss"][0] + new_loss_values["mask_loss"][1]
    sigma_loss_all = new_loss_values["sigma_loss"][0] + new_loss_values["sigma_loss"][1]
    kl_terms = new_loss_values["kl_loss"]
    assert len(kl_terms) == 3, f"Expected 3 KL terms, got {len(kl_terms)}"
    kl_comb, kl_v1, kl_v2 = kl_terms
    kl_loss_all = kl_comb + beta_kl_views * (kl_v1 + kl_v2)
    loss_dict[mode + "_rec_loss_all"]   += rec_loss_all.detach() * w_bs
    loss_dict[mode + "_mask_loss_all"]  += mask_loss_all.detach() * w_bs
    loss_dict[mode + "_sigma_loss_all"] += sigma_loss_all.detach() * w_bs
    loss_dict[mode + "_kl_loss_all"]    += kl_loss_all.detach() * w_bs
    loss_dict[mode + "_rec_loss1"]   += new_loss_values["rec_loss"][0].detach() * w_bs
    loss_dict[mode + "_rec_loss2"]   += new_loss_values["rec_loss"][1].detach() * w_bs
    loss_dict[mode + "_mask_loss1"]  += new_loss_values["mask_loss"][0].detach() * w_bs
    loss_dict[mode + "_mask_loss2"]  += new_loss_values["mask_loss"][1].detach() * w_bs
    loss_dict[mode + "_sigma_loss1"] += new_loss_values["sigma_loss"][0].detach() * w_bs
    loss_dict[mode + "_sigma_loss2"] += new_loss_values["sigma_loss"][1].detach() * w_bs
    loss_dict[mode + "_kl_comb"] += kl_comb.detach() * w_bs
    loss_dict[mode + "_kl_v1"]   += kl_v1.detach() * w_bs
    loss_dict[mode + "_kl_v2"]   += kl_v2.detach() * w_bs
    # alpha/effective diagnostics (for sanity checking only)
    for k in ["alpha_entropy", "alpha_balance_batch", "alpha_balance_sample", "alpha_mean_v1", "alpha_mean_v2",
              "eff_balance_batch", "eff_balance_sample", "eff_mean_v1", "eff_mean_v2",
              "rec_loss_raw1", "rec_loss_raw2"]:
        if k in new_loss_values:
            loss_dict[f"{mode}_{k}"] += new_loss_values[k].detach() * w_bs
    mask_loss_all_scaled = gamma_mask1 * new_loss_values["mask_loss"][0] + gamma_mask2 * new_loss_values["mask_loss"][1]
    sigma_loss_all_scaled = gamma_sigma * (new_loss_values["sigma_loss"][0] + new_loss_values["sigma_loss"][1])
    total_loss_all = rec_loss_all + beta_kl * kl_loss_all + mask_loss_all_scaled + sigma_loss_all_scaled
    loss_dict[mode + "_total_loss_all"] += total_loss_all.detach() * w_bs
    return total_loss_all

def init_loss_dict(mode, device=None):
    if device is None:
        device = torch.device("cpu")
    loss_dict = {}
    for k in [
        "rec_loss_all", "kl_loss_all", "mask_loss_all", "sigma_loss_all", "total_loss_all",
        "rec_loss1", "rec_loss2", "mask_loss1", "mask_loss2", "sigma_loss1", "sigma_loss2",
        "kl_comb", "kl_v1", "kl_v2",
        # Diagnostics
        "alpha_entropy", "alpha_balance_batch", "alpha_balance_sample", "alpha_mean_v1", "alpha_mean_v2",
        "eff_balance_batch", "eff_balance_sample", "eff_mean_v1", "eff_mean_v2",
        "rec_loss_raw1", "rec_loss_raw2",
    ]:
        loss_dict[f"{mode}_{k}"] = torch.zeros((), device=device)
    return loss_dict
