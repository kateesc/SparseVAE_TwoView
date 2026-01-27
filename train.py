#
#
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
    """
    Returns (sum_squared_error, count) for TRUE MSE aggregation.
    mask can be bool or float in {0,1}.
    """
    err2 = (x_pred - x_true) ** 2
    w = mask.to(dtype=err2.dtype)
    sse = (err2 * w).sum().item()
    cnt = w.sum().item()
    return sse, cnt


def train_POEMS(
    lr_in,
    wd_in,
    batch_size_in,
    nepoch_in,
    is_wandb,
    experiment_note,
    disease,
    is_test,
    model_name="POEMS",
):
    setup_seed.setup_seed_21()

    patience = 30
    lr = lr_in
    wd = wd_in
    batch_size = batch_size_in
    nepoch = nepoch_in

    # ---- naming/output dirs ----
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

    # ---- W&B ----
    if is_wandb:
        wandb.init(project=project_name)
        wandb.config.update({
            "lr": lr,
            "wd": wd,
            "patience": patience,
            "n_epoch": nepoch,
            "batch_size": batch_size,
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

    # ---- Model configs (2 views) ----
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

    # ---- loss weights ----
    beta_kl = 1.0
    beta_kl_views = 0.1

    if is_wandb:
        wandb.config.update({
            "beta_kl": beta_kl,
            "beta_kl_views": beta_kl_views,
        }, allow_val_change=True)

    # ---- checkpoint paths ----
    best_path = os.path.join(model_dir, "model_best.pth")
    last_path = os.path.join(model_dir, "model_last.pth")
    canonical_path = os.path.join(model_dir, "model.pth")  # we will write BEST here at the end            
    def _safe_torch_save(state_dict, path):
        tmp = path + ".tmp"
        torch.save(state_dict, tmp)
        os.replace(tmp, path)  # atomic on most OSes
        
    # =========================
    # TRAIN or LOAD
    # =========================
    import copy

    if not is_test:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        early_stopper = EarlyStopper(patience=patience, min_delta=0)

        best_score = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())

        loss_history = []

        warmup_epochs = 50
        p_mask_v1_max = 0.15
        mask_warmup = 20

        for epoch in range(1, nepoch + 1):

            # ---- TRAIN ----
            p_mask_v1 = p_mask_v1_max * min(1.0, epoch / mask_warmup)
            train_loss_dict = init_loss_dict("train", device=device)
            model.train()
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

                # ---- denoising corruption only on OBSERVED SNP entries (view1) ----
                drop1 = (torch.rand_like(m1) < p_mask_v1) & (m1 > 0.5)

                m1_in = m1.clone()
                m1_in[drop1] = 0.0        # tell encoder these are now "missing"

                if epoch == 1 and train_n <= bs:
                    print("p_mask_v1 target:", p_mask_v1,
                          "realized drop fraction (all entries):", drop1.float().mean().item())
                    obs = (m1 > 0.5)
                    realized_among_obs = (drop1.float().sum() / (obs.float().sum() + 1e-8)).item()
                    print("realized drop fraction (among observed):", realized_among_obs)

                x1_in = x1.clone()
                x1_in[drop1] = 0.0        # mean in z-scored space
                
                # view2 unchanged
                x2_in = x2
                m2_in = m2

                # ---- IMPORTANT: inference uses (x*_in, m*_in), reconstruction scored vs (x*, m*) ----
                new_loss_values = model.all_loss(
                    x1, x2,
                    m1=m1, m2=m2,            # reconstruction mask (true observed entries)
                    x1_in=x1_in, x2_in=x2_in,
                    m1_in=m1_in, m2_in=m2_in,  # inference mask
                )

                loss = update_loss_dict_return_total_loss(
                    loss_dict=train_loss_dict,
                    new_loss_values=new_loss_values,
                    beta_kl=beta_kl,
                    beta_kl_views=beta_kl_views,
                    mode="train",
                    bs=bs,
                )


                loss.backward()
                optimizer.step()

                # Manual spike-slab updates (E-step style)
                # (Still useful for W sparsity. OK to keep.)
                with torch.no_grad():
                    for net_key in ["specific1", "specific2"]:
                        mod = model.specific_modules[net_key]
                        mod.log_sigmas.data.clamp_(min=mod.min_log_sigma.item())

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

            # ---- VALIDATION ----
            model.eval()
            val_loss_dict = init_loss_dict("val", device=device)
            val_dn_loss_dict = init_loss_dict("val_dn", device=device)

            p_mask_v1_val = 0.15
            gen = torch.Generator(device=device)
            gen.manual_seed(12345 + epoch)  # deterministic per-epoch

            # Denoising eval metrics for view1
            dn_sse_obs_keep = 0.0
            dn_cnt_obs_keep = 0.0
            dn_sse_drop = 0.0
            dn_cnt_drop = 0.0

            with torch.no_grad():
                val_n = 0
                for data, mask in val_loader:
                    data = data.to(device)
                    mask = mask.to(device)

                    bs = data.shape[0]
                    val_n += bs

                    x1 = data[:, :omic1_dim]
                    x2 = data[:, omic1_dim:omic1_dim + omic2_dim]
                    m1 = mask[:, :omic1_dim]
                    m2 = mask[:, omic1_dim:omic1_dim + omic2_dim]

                    # (A) CLEAN validation
                    new_loss_values = model.all_loss(x1, x2, m1=m1, m2=m2)

                    _ = update_loss_dict_return_total_loss(
                        loss_dict=val_loss_dict,
                        new_loss_values=new_loss_values,
                        beta_kl=beta_kl,
                        beta_kl_views=beta_kl_views,
                        mode="val",
                        bs=bs,
                    )
                    

                    # (B) DENOISING validation (drop only observed SNPs in view1)
                    r = torch.rand(m1.shape, device=device, generator=gen)
                    drop1_val = (r < p_mask_v1_val) & (m1 > 0.5)

                    x1_in = x1.clone()
                    x1_in[drop1_val] = 0.0

                    m1_in = m1.clone()
                    m1_in[drop1_val] = 0.0

                    new_loss_values_dn = model.all_loss(
                        x1, x2,
                        m1=m1, m2=m2,
                        x1_in=x1_in, x2_in=x2,
                        m1_in=m1_in, m2_in=m2,
                    )

                    # ---- Extra logging: denoising/imputation MSE (TRUE MSE) on view1 ----
                    # Use reconstruction from the same corrupted inputs.
                    # We decode from the fused mean to reduce stochasticity noise.
                    mean_comb, log_var_comb, z_samp, alphas_tmp, aux_tmp = model.encode(
                        x1_in, x2, m1=m1_in, m2=m2
                    )
                    x1_rec = model.specific_modules["specific1"].decode(mean_comb)

                    # observed-but-kept entries (observed and NOT dropped)
                    mask_obs_keep = (m1 > 0.5) & (~drop1_val)

                    # artificially dropped entries (these were observed originally)
                    mask_drop = drop1_val

                    sse, cnt = _masked_mse_sse_cnt(x1_rec, x1, mask_obs_keep)
                    dn_sse_obs_keep += sse
                    dn_cnt_obs_keep += cnt

                    sse, cnt = _masked_mse_sse_cnt(x1_rec, x1, mask_drop)
                    dn_sse_drop += sse
                    dn_cnt_drop += cnt

                    _ = update_loss_dict_return_total_loss(
                        loss_dict=val_dn_loss_dict,
                        new_loss_values=new_loss_values_dn,
                        beta_kl=beta_kl,
                        beta_kl_views=beta_kl_views,
                        mode="val_dn",
                        bs=bs,
                    )

            val_loss_dict = {k: v / float(val_n) for k, v in val_loss_dict.items()}
            val_dn_loss_dict = {k: v / float(val_n) for k, v in val_dn_loss_dict.items()}

            # Final denoising metrics (TRUE MSE)
            val_dn_mse_obs1 = dn_sse_obs_keep / (dn_cnt_obs_keep + 1e-8)
            val_dn_mse_imp1 = dn_sse_drop / (dn_cnt_drop + 1e-8)  # "imputation" on dropped entries
            val_dn_imp_obs_gap1 = val_dn_mse_imp1 - val_dn_mse_obs1
            val_dn_imp_obs_ratio1 = val_dn_mse_imp1 / (val_dn_mse_obs1 + 1e-8)

            val_loss = val_loss_dict["val_total_loss_all"].item()
            val_dn_loss = val_dn_loss_dict["val_dn_total_loss_all"].item()

            early_stop_score = 0.5 * val_loss + 0.5 * val_dn_loss

            # track best on MIXED score
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

            # early stop on MIXED score
            is_early_stop = early_stopper.early_stop(early_stop_score)

            # ---- W/sigma diagnostics ----
            if (epoch % 10 == 0) or (epoch == 1):
                util.export_W_and_sigma_stats(model, epoch, out_dir)

            # ---- loss history ----
            row = {
                "epoch": epoch,
                "p_mask_v1": float(p_mask_v1),
                "p_mask_v1_val": float(p_mask_v1_val),
                "early_stop_score": float(early_stop_score),
            }
            row.update({k: _to_float(v) for k, v in train_loss_dict.items()})
            row.update({k: _to_float(v) for k, v in val_loss_dict.items()})
            row.update({k: _to_float(v) for k, v in val_dn_loss_dict.items()})
            row.update({
                "val_dn_mse_obs1": float(val_dn_mse_obs1),
                "val_dn_mse_imp1": float(val_dn_mse_imp1),
                "val_dn_imp_obs_gap1": float(val_dn_imp_obs_gap1),
                "val_dn_imp_obs_ratio1": float(val_dn_imp_obs_ratio1),
                "val_dn_drop_frac1": float(dn_cnt_drop / (dn_cnt_drop + dn_cnt_obs_keep + 1e-8)),
            })

            
            # rec_loss1/2 are 0.5*MSE on observed entries => multiply by 2 for true MSE
            row["train_mse_obs1"] = 2.0 * row["train_rec_loss1"]
            row["train_mse_obs2"] = 2.0 * row["train_rec_loss2"]
            row["val_mse_obs1"]   = 2.0 * row["val_rec_loss1"]
            row["val_mse_obs2"]   = 2.0 * row["val_rec_loss2"]

            # You don’t currently compute “all entries” MSE separately, so set equal to obs
            row["train_mse_all1"] = row["train_mse_obs1"]
            row["train_mse_all2"] = row["train_mse_obs2"]
            row["val_mse_all1"]   = row["val_mse_obs1"]
            row["val_mse_all2"]   = row["val_mse_obs2"]

            loss_history.append(row)

            if epoch % 5 == 0 or epoch == 1:
                pd.DataFrame(loss_history).to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)

            other_metrics_dict = {
                "min_log_sigma1": float(model.specific_modules["specific1"].log_sigmas.detach().min().cpu().item()),
                "min_log_sigma2": float(model.specific_modules["specific2"].log_sigmas.detach().min().cpu().item()),
                "w_min_abs_val1": float(model.specific_modules["specific1"].W.detach().abs().min().cpu().item()),
                "w_max_abs_val1": float(model.specific_modules["specific1"].W.detach().abs().max().cpu().item()),
                "w_min_abs_val2": float(model.specific_modules["specific2"].W.detach().abs().min().cpu().item()),
                "w_max_abs_val2": float(model.specific_modules["specific2"].W.detach().abs().max().cpu().item()),
                "beta_kl": float(beta_kl),
                "beta_kl_views": float(beta_kl_views),
                "best_score_so_far": float(best_score),
                "best_epoch_so_far": int(best_epoch),
            }

            # ---- LOG VAL metrics on BEST checkpoint ----
            if (epoch % 10 == 0) or is_early_stop:
                was_training = model.training
                current_state = copy.deepcopy(model.state_dict())

                if os.path.exists(best_path):
                    ckpt = torch.load(best_path, map_location=device, weights_only=True)
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(best_state)

                model.eval()
                with torch.no_grad():
                    final_embedding, alphas = model.get_final_embedding(X_val_all, M_val_all)

                final_embedding = final_embedding.detach()
                alphas = alphas.detach()

                print(
                    f"Epoch {epoch}: "
                    f"Train loss={train_loss_dict['train_total_loss_all'].item():.4f} "
                    f"Val={val_loss:.4f} ValDN={val_dn_loss:.4f} "
                    f"(BEST score={best_score:.4f} @ epoch {best_epoch})"
                )

                if not torch.isfinite(final_embedding).all():
                    n_nan = torch.isnan(final_embedding).sum().item()
                    n_inf = torch.isinf(final_embedding).sum().item()
                    print(f"[ERROR] final_embedding has NaN={n_nan}, Inf={n_inf} at epoch {epoch}")
                    model.load_state_dict(current_state)
                    if was_training:
                        model.train()
                    else:
                        model.eval()
                    break

                (kmeans_acc_mean, kmeans_acc_std,
                 kmeans_nmi_mean, kmeans_nmi_std,
                 silhouette_mean, silhouette_std) = perform_kmeans(final_embedding, y_val, n_clusters)
                knn_acc_mean, knn_acc_std = perform_knn(final_embedding, y_val, n_clusters)

                metrics_row = pd.DataFrame([{
                    "experiment": project_name if not is_test else trained_model_name,
                    "lr": lr,
                    "wd": wd,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "kmeans_acc_mean": kmeans_acc_mean,
                    "kmeans_acc_std": kmeans_acc_std,
                    "kmeans_nmi_mean": kmeans_nmi_mean,
                    "kmeans_nmi_std": kmeans_nmi_std,
                    "knn_acc_mean": knn_acc_mean,
                    "knn_acc_std": knn_acc_std,
                    "alpha1_mean": alphas[:, 0].mean().item(),
                    "alpha2_mean": alphas[:, 1].mean().item(),
                }])
                metrics_path = os.path.join(out_dir, "metrics_summary.csv")
                if not os.path.exists(metrics_path):
                    metrics_row.to_csv(metrics_path, index=False)
                else:
                    metrics_row.to_csv(metrics_path, mode="a", header=False, index=False)

                other_metrics_dict.update({
                    "val_kmeans_nmi_mean": kmeans_nmi_mean,
                    "val_knn_acc_mean": knn_acc_mean,
                    "val_alpha1_mean": alphas[:, 0].mean().item(),
                    "val_alpha2_mean": alphas[:, 1].mean().item(),
                    "val_silhouette_mean": silhouette_mean,
                    "val_silhouette_std": silhouette_std,
                })

                if is_wandb:
                    wandb.log(train_loss_dict | val_loss_dict | val_dn_loss_dict | other_metrics_dict | {"early_stop_score": early_stop_score}, step=epoch)


                # restore CURRENT model so training can continue
                model.load_state_dict(current_state)
                if was_training:
                    model.train()
                else:
                    model.eval()
                    
            _safe_torch_save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
            }, last_path)
            # ---- Early stop AFTER logging ----
            if is_early_stop:
                print(f"Early stopping at epoch {epoch} (BEST score={best_score:.4f} @ epoch {best_epoch})")
                break

        # ---- After training: load BEST and write canonical (weights only) ----
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
        else:
            # fallback if best_path was never written (rare)
            model.load_state_dict(best_state)

        _safe_torch_save(model.state_dict(), canonical_path)  # canonical = weights only
        print(f"Saved BEST checkpoint to {canonical_path} (best_epoch={best_epoch}, best_score={best_score:.4f})")

    else:
        best_epoch = 0  # default if unknown
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
            batch_size=64,
            max_samples=500,
            seed=0,
        )
        util.plot_view_uncertainty(model, X_test_all, M_all=M_test_all, dir=out_dir, batch_size=1024, tag="test")
        # --- LatentScores CSVs with IDs ---
        util.export_latent_scores_csv(
            model,
            X_train_all, M_train_all,
            sample_ids[idx_train],
            y_train,
            out_dir,
            tag="train",
            batch_size=1024,
            include_view_latents=False,   # set True if you want z1_*, z2_* columns
        )

        util.export_latent_scores_csv(
            model,
            X_val_all, M_val_all,
            sample_ids[idx_val],
            y_val,
            out_dir,
            tag="val",
            batch_size=1024,
            include_view_latents=False,
        )

        util.export_latent_scores_csv(
            model,
            X_test_all, M_test_all,
            sample_ids[idx_test],
            y_test,
            out_dir,
            tag="test",
            batch_size=1024,
            include_view_latents=False,
        )
        # --- OPTIONAL: LatentScores_all.csv in original samples.txt order ---
        n = len(sample_ids)
        d = X_train_all.shape[1]

        # rebuild full X/M/y in ORIGINAL order using idx_* mapping
        X_all = np.empty((n, d), dtype=X_train_all.dtype)
        M_all = np.empty((n, d), dtype=M_train_all.dtype)
        y_all = np.empty((n,), dtype=np.asarray(y_train).dtype)
        split = np.empty((n,), dtype=object)

        X_all[idx_train] = X_train_all
        M_all[idx_train] = M_train_all
        y_all[idx_train] = y_train
        split[idx_train] = "train"

        X_all[idx_val] = X_val_all
        M_all[idx_val] = M_val_all
        y_all[idx_val] = y_val
        split[idx_val] = "val"

        X_all[idx_test] = X_test_all
        M_all[idx_test] = M_test_all
        y_all[idx_test] = y_test
        split[idx_test] = "test"

        all_path = util.export_latent_scores_csv(
            model,
            X_all, M_all,
            sample_ids,
            y_all,
            out_dir,
            tag="all",
            batch_size=1024,
            include_view_latents=False,  # True if you want z1_*/z2_* columns
        )

        # add split column to that ALL file (optional but useful)
        df_all = pd.read_csv(all_path)
        df_all["split"] = split
        df_all.to_csv(all_path, index=False)
       

        final_embedding = final_embedding.detach()
        alphas = alphas.detach()

        if not torch.isfinite(final_embedding).all():
            n_nan = torch.isnan(final_embedding).sum().item()
            n_inf = torch.isinf(final_embedding).sum().item()
            raise RuntimeError(f"[ERROR] test final_embedding has NaN={n_nan}, Inf={n_inf}")

        np.savetxt(os.path.join(out_dir, "final_em_test.csv"),
                   final_embedding.cpu().numpy(), delimiter=",")

        (kmeans_acc_mean, kmeans_acc_std,
         kmeans_nmi_mean, kmeans_nmi_std,
         silhouette_mean, silhouette_std) = perform_kmeans(final_embedding, y_test, n_clusters)
        knn_acc_mean, knn_acc_std = perform_knn(final_embedding, y_test, n_clusters)

        metrics_row_test = pd.DataFrame([{
            "experiment": project_name if not is_test else trained_model_name,
            "lr": lr,
            "wd": wd,
            "batch_size": batch_size,
            "split": "test",
            "kmeans_acc_mean": kmeans_acc_mean,
            "kmeans_acc_std": kmeans_acc_std,
            "kmeans_nmi_mean": kmeans_nmi_mean,
            "kmeans_nmi_std": kmeans_nmi_std,
            "knn_acc_mean": knn_acc_mean,
            "knn_acc_std": knn_acc_std,
            "alpha1_mean": alphas[:, 0].mean().item(),
            "alpha2_mean": alphas[:, 1].mean().item(),
        }])
        metrics_row_test.to_csv(os.path.join(out_dir, "metrics_summary_test.csv"), index=False)

        util.plot_tsne(final_embedding, y_test, out_dir, tag="test")
        util.plot_umap(final_embedding, y_test, out_dir, tag="test")
        util.visualize_Ws(model, out_dir, thresholds=(1e-3, 1e-2, 1e-1))
        util.visualize_final_embedding(final_embedding, y_test, dir=out_dir)

        X1_test = X_test_all[:, :omic1_dim]
        X2_test = X_test_all[:, omic1_dim:omic1_dim + omic2_dim]

        # ==========================================================
        # Keep RAW embedding unchanged; create oriented COPIES only.
        # ==========================================================
        Z_raw = final_embedding.detach()  # torch tensor (n,K)
        Z_raw_np = Z_raw.cpu().numpy()
        K = Z_raw_np.shape[1]

        # ==========================================================
        # (1) TOP-TRAIT ORIENTATION (your existing logic)
        # Writes: LatentOrientation_byTopTrait.csv
        #         LatentScores_test_oriented_byTopTrait.csv
        # ==========================================================
        trait_names = util._safe_featnames(disease, view_idx=2, n_features=omic2_dim)

        signs_toptrait, orient_info, corr_mat = util.orientation_by_top_correlated_trait(
            Z_raw,        # (n,K)
            X2_test,      # (n,T)
            trait_names,
            min_abs_corr=0.05
        )
        orient_info.to_csv(os.path.join(out_dir, "LatentOrientation_byTopTrait.csv"), index=False)

        util.orient_latent_scores_csv(
            os.path.join(out_dir, "LatentScores_test.csv"),
            os.path.join(out_dir, "LatentScores_test_oriented_byTopTrait.csv"),
            signs_toptrait
        )

        # Optional: oriented copy for plots (do NOT overwrite final_embedding)
        Z_toptrait = Z_raw * torch.tensor(signs_toptrait, device=Z_raw.device).view(1, -1)
        # Example optional plots:
        # util.plot_latent_pca_2d_3d(Z_toptrait, y_test, dir=out_dir, tag="test_toptrait")
        util.plot_latent_pca_2d_3d(Z_toptrait, y_test, dir=out_dir, tag="test_toptrait")

        util.plot_tsne(Z_toptrait, y_test, out_dir, tag="test_toptrait")
        util.plot_umap(Z_toptrait, y_test, out_dir, tag="test_toptrait")

        df_p = util.permutation_latent_trait_corr_pvals(
            Z_toptrait, X2_test, trait_names=trait_names,
            n_perm=2000, seed=0, two_sided=True,
            return_null_max=True,
            apply_bh=True, bh_scope="global", fdr_alpha=0.05,
        )
        df_p.to_csv(os.path.join(out_dir, "PermTest_LatentTraitCorr_test_toptrait.csv"), index=False)

        # ==========================================================
        # (2) GROUP ORIENTATION (highland vs lowland) FOR ALL LATENTS
        # Writes: LatentGroupStats_test.csv
        #         LatentGroupContrast_test.csv
        #         LatentGroupRanking_test.csv
        #         LatentOrientation_byGroup_test.csv
        #         LatentScores_test_oriented_byGroup.csv
        # ==========================================================
        highland_label = 1   # <-- CHANGE to your encoding
        lowland_label  = 0   # <-- CHANGE to your encoding

        y = np.asarray(y_test)
        mask_hi = (y == highland_label)
        mask_lo = (y == lowland_label)
        if mask_hi.sum() == 0 or mask_lo.sum() == 0:
            raise RuntimeError(
                f"Group labels not found. highland_label={highland_label} (n={mask_hi.sum()}), "
                f"lowland_label={lowland_label} (n={mask_lo.sum()}). Check np.unique(y_test)."
            )

        Z_hi = Z_raw_np[mask_hi]
        Z_lo = Z_raw_np[mask_lo]
        n_hi, n_lo = Z_hi.shape[0], Z_lo.shape[0]

        # (A) Per-group mean/std/n for all latents
        rows = []
        for g in np.unique(y):
            m = (y == g)
            Zg = Z_raw_np[m]
            for k in range(K):
                rows.append({
                    "group": g,
                    "latent": k,
                    "mean": float(Zg[:, k].mean()),
                    "std": float(Zg[:, k].std(ddof=1)) if Zg.shape[0] > 1 else np.nan,
                    "n": int(Zg.shape[0]),
                })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "LatentGroupStats_test.csv"), index=False)

        
        # (B) Contrast + effect size + sign convention for ALL latents
        signs_group = np.ones(K, dtype=int)
        contrast_rows = []
        for k in range(K):
            mu_hi = Z_hi[:, k].mean()
            mu_lo = Z_lo[:, k].mean()
            diff = mu_hi - mu_lo

            # orient so "positive = highland" for EACH latent
            signs_group[k] = 1 if diff >= 0 else -1

            s_hi = Z_hi[:, k].std(ddof=1) if n_hi > 1 else np.nan
            s_lo = Z_lo[:, k].std(ddof=1) if n_lo > 1 else np.nan
            if n_hi > 1 and n_lo > 1 and np.isfinite(s_hi) and np.isfinite(s_lo):
                sp = np.sqrt(((n_hi - 1) * s_hi**2 + (n_lo - 1) * s_lo**2) / (n_hi + n_lo - 2))
                d = diff / (sp + 1e-12)
            else:
                d = np.nan

            contrast_rows.append({
                "latent": k,
                "mean_highland": float(mu_hi),
                "mean_lowland": float(mu_lo),
                "diff_highland_minus_lowland": float(diff),
                "abs_diff": float(abs(diff)),
                "cohen_d": float(d) if np.isfinite(d) else np.nan,
                "abs_cohen_d": float(abs(d)) if np.isfinite(d) else np.nan,
                "n_highland": int(n_hi),
                "n_lowland": int(n_lo),
                "sign_for_positive_highland": int(signs_group[k]),
            })

        df_contrast = pd.DataFrame(contrast_rows)
        df_contrast.to_csv(os.path.join(out_dir, "LatentGroupContrast_test.csv"), index=False)

        df_rank = df_contrast.sort_values("abs_cohen_d", ascending=False)
        df_rank.to_csv(os.path.join(out_dir, "LatentGroupRanking_test.csv"), index=False)

        pd.DataFrame({
            "latent": np.arange(K),
            "sign_for_positive_highland": signs_group
        }).to_csv(os.path.join(out_dir, "LatentOrientation_byGroup_test.csv"), index=False)

        util.orient_latent_scores_csv(
            os.path.join(out_dir, "LatentScores_test.csv"),
            os.path.join(out_dir, "LatentScores_test_oriented_byGroup.csv"),
            signs_group
        )
    
        # Optional: oriented copy for plots (do NOT overwrite final_embedding)
        Z_group = Z_raw * torch.tensor(signs_group, device=Z_raw.device).view(1, -1)
        df_g = util.permutation_latent_group_pvals(
        Z_group, y_test, group_a=highland_label, group_b=lowland_label,
        n_perm=2000, seed=0
        )
        df_g.to_csv(os.path.join(out_dir, "PermTest_LatentGroup_test_groupOriented.csv"), index=False)
        # Example optional plots:
        # util.plot_latent_pca_2d_3d(Z_group, y_test, dir=out_dir, tag="test_group")
        util.plot_latent_pca_2d_3d(Z_group, y_test, dir=out_dir, tag="test_group")
        util.plot_tsne(Z_group, y_test, out_dir, tag="test_group")
        util.plot_umap(Z_group, y_test, out_dir, tag="test_group")
        
        metrics_test = util.evaluate_recon_mask_metrics(model, X_test_all, M_test_all, omic1_dim, omic2_dim, batch_size=64)
        util.plot_coverage_vs_obs_error(metrics_test, out_dir, tag="test")
        util.plot_gating_vs_missingness(metrics_test, out_dir, tag="test")
        util.plot_effective_vs_missingness(metrics_test, out_dir, tag="test")
        util.plot_baseline_vs_model_bar(metrics_test, b1, b2, out_dir, tag="test")

        util.plot_subtype_correlations(
            X1_test, X2_test,
            final_embedding, y_test,
            subtype_names=None,
            dir=out_dir
        )
        util.plot_feature_importance(model, disease, omic_names=["View1", "View2"], dir=out_dir)
        util.plot_gating_alphas_stacked(alphas, dir=out_dir)
        util.plot_effective_fusion_weights(model, X_test_all, M_all=M_test_all, dir=out_dir, batch_size=1024, tag="test")
        
        # --- add these back (from version1) ---
        util.plot_trait_latent_matrix(model, disease, dir=out_dir, use_abs=True)
        util.export_trait_top_snps(model, disease, dir=out_dir, top_snps=50, top_latents=5, use_abs=True)
        util.export_latent_top_features(model, disease, dir=out_dir, top_snps=10, top_traits=10)

        # Optional: these still exist in util.py; in v2 they’ll be “neutral/flat” because Kendall is disabled
        loss_csv = os.path.join(out_dir, "loss_history.csv")
        if os.path.exists(loss_csv):
            util.plot_loss_history(out_dir)
            util.plot_mask_loss_scaling(out_dir)
            # util.plot_kendall_calibration(out_dir)
            # util.plot_kendall_contributions(out_dir)

        util.export_latent_trait_ranking(model, disease, dir=out_dir, use_abs=True)
        util.export_latent_top_snps(
            model, disease, dir=out_dir,
            top_snps=200,
            abs_w_threshold=1e-3,
            pstar_threshold=0.7,
            use_pstar=True,
        )
        # ==========================================================
        # C) Within-factor feature ablation for ALL latents (View1)
        # ==========================================================
        K = model.specific_modules["specific1"].config["latent_dim"]

        rows_featabl = []
        for k in range(K):
            S = util.top_feature_indices_for_latent_view1(
                model,
                latent_k=k,
                top_n=200,
                abs_w_threshold=1e-3,
                pstar_threshold=0.7,
                use_pstar=True,
            )

            # still write a row if empty (helps debug thresholds)
            if S.size == 0:
                rows_featabl.append(pd.DataFrame([{
                    "latent": int(k),
                    "n_features_ablated": 0,
                    "restrict": "subset_obs",
                    "base_mse": np.nan,
                    "ablated_mse": np.nan,
                    "delta_mse": np.nan,
                    "zero_pstar": True,
                    "max_samples": 500,
                    "note": "empty S_k (no features passed thresholds)"
                }]))
                continue

            df_k = util.within_factor_feature_ablation(
                model,
                X_test_all,
                M_test_all,
                omic1_dim,
                omic2_dim,
                latent_k=k,
                feat_idx=S,
                batch_size=64,
                max_samples=500,
                seed=0,
                restrict="subset_obs",
                zero_pstar=True,
            )
            rows_featabl.append(df_k)

        df_featabl = pd.concat(rows_featabl, ignore_index=True)
        df_featabl.to_csv(os.path.join(out_dir, "WithinFactorFeatureAblation_View1.csv"), index=False)

        df_ranked = df_featabl.sort_values("delta_mse", ascending=False)
        df_ranked.to_csv(os.path.join(out_dir, "WithinFactorFeatureAblation_View1_ranked.csv"), index=False)

        df_curve = util.denoising_difficulty_sweep_view1(
            model, X_val_all, M_val_all, omic1_dim, omic2_dim,
            p_list=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
            batch_size=64, seed=0, max_samples=500
        )
        df_curve.to_csv(os.path.join(out_dir, "DenoisingDifficultyCurve_View1.csv"), index=False)
        util.plot_denoising_difficulty_curve_view1(out_dir)    

        df_lat_imp = util.latent_ablation_imputation_importance_view1(
            model, X_val_all, M_val_all, omic1_dim, omic2_dim,
            p_mask=p_mask_v1_val,      # use same p as val_dn
            batch_size=64, seed=0, max_samples=500
        )
        df_lat_imp.to_csv(os.path.join(out_dir, "LatentAblation_ImputationImportance_View1.csv"), index=False)

    util.plot_W_stats_over_epochs(out_dir)
    util.summarize_loss_history(out_dir)
    util.plot_loss_components(out_dir)
    util.plot_epoch_recon_obs_vs_all(out_dir)
    util.plot_epoch_imputed_vs_observed_gap_ratio(out_dir)
    util.plot_denoising_gap(out_dir)
    util.plot_all_csv_summaries(out_dir)

    if is_wandb:
        wandb.log({
            "kmeans_acc_mean": kmeans_acc_mean,
            "kmeans_acc_std": kmeans_acc_std,
            "kmeans_nmi_mean": kmeans_nmi_mean,
            "kmeans_nmi_std": kmeans_nmi_std,
            "knn_acc_mean": knn_acc_mean,
            "knn_acc_std": knn_acc_std,
            "test_alpha1_mean": alphas[:, 0].mean().item(),
            "test_alpha2_mean": alphas[:, 1].mean().item(),
        }, step=best_epoch)
        wandb.finish()


def _to_float(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def update_loss_dict_return_total_loss(
    loss_dict,
    new_loss_values,
    beta_kl,
    beta_kl_views,
    mode,
    *,
    bs,
):
    w_bs = float(bs)

    # reconstruction / regularizers
    rec_loss_all   = new_loss_values["rec_loss"][0] + new_loss_values["rec_loss"][1]
    mask_loss_all  = new_loss_values["mask_loss"][0] + new_loss_values["mask_loss"][1]
    sigma_loss_all = new_loss_values["sigma_loss"][0] + new_loss_values["sigma_loss"][1]

    # KL: combined + per-view
    kl_terms = new_loss_values["kl_loss"]
    assert len(kl_terms) == 3, f"Expected 3 KL terms, got {len(kl_terms)}"
    kl_comb, kl_v1, kl_v2 = kl_terms
    kl_loss_all = kl_comb + beta_kl_views * (kl_v1 + kl_v2)

    # ----------------
    # logging
    # ----------------
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

    # alpha/effective diagnostics (for sanity checking; do NOT add to loss)
    for k in ["alpha_entropy","alpha_balance_batch","alpha_balance_sample","alpha_mean_v1","alpha_mean_v2",
              "eff_balance_batch","eff_balance_sample","eff_mean_v1","eff_mean_v2",
              "rec_loss_raw1","rec_loss_raw2"]:
        if k in new_loss_values:
            loss_dict[f"{mode}_{k}"] += new_loss_values[k].detach() * w_bs

    # total (NO alpha_reg)
    total_loss_all = rec_loss_all + beta_kl * kl_loss_all + mask_loss_all + sigma_loss_all
    loss_dict[mode + "_total_loss_all"] += total_loss_all.detach() * w_bs

    return total_loss_all


def init_loss_dict(mode, device=None):
    if device is None:
        device = torch.device("cpu")
    loss_dict = {}
    for k in [
        "rec_loss_all","kl_loss_all","mask_loss_all","sigma_loss_all","total_loss_all",
        "rec_loss1","rec_loss2","mask_loss1","mask_loss2","sigma_loss1","sigma_loss2",
        "kl_comb","kl_v1","kl_v2",

        # diagnostics only (no longer regularizers)
        "alpha_entropy","alpha_balance_batch","alpha_balance_sample","alpha_mean_v1","alpha_mean_v2",
        "eff_balance_batch","eff_balance_sample","eff_mean_v1","eff_mean_v2",
        "rec_loss_raw1","rec_loss_raw2",
    ]:
        loss_dict[f"{mode}_{k}"] = torch.zeros((), device=device)
    return loss_dict
