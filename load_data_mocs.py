import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _maybe_transpose_to_samples_rows(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    if df.shape[0] == n_samples:
        return df
    if df.shape[1] == n_samples:
        return df.T
    raise ValueError(
        f"Cannot match matrix to n_samples={n_samples}. Got shape={df.shape}. "
        f"Expected rows==n_samples or cols==n_samples."
    )

def _train_feature_means(X_train: np.ndarray, M_train: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # mean_j = sum_i X_ij * M_ij / sum_i M_ij  (only observed entries)
    num = (X_train * M_train).sum(axis=0)
    den = M_train.sum(axis=0)
    means = np.where(den > 0, num / (den + eps), 0.0)
    return means

def load_data_mocs(disease: str):
    print(os.path.join("data", disease))

    if disease != "plant":
        raise ValueError(f"Unknown disease/dataset key '{disease}'. Use disease='plant'.")

    base = os.path.join(sys.path[1], "data", disease)

    v1 = pd.read_csv(os.path.join(base, "1_all.csv"), header=None)
    v2 = pd.read_csv(os.path.join(base, "2_all.csv"), header=None)
    y_df = pd.read_csv(os.path.join(base, "labels_all.csv"), header=None)

    sample_ids = (
        pd.read_csv(os.path.join(base, "samples.txt"), header=None)
        .iloc[:, 0].astype(str).tolist()
    )

    y_true = y_df.iloc[:, 0].to_numpy()
    # if labels are strings, factorize; if already ints, this keeps them
    if not np.issubdtype(y_true.dtype, np.number):
        y_true, _ = pd.factorize(y_true)
    y_true = y_true.astype(int)

    if len(y_true) != len(sample_ids):
        raise ValueError(
            f"labels_all.csv has {len(y_true)} rows but samples.txt has {len(sample_ids)} rows."
        )

    # Fix orientation
    v1 = _maybe_transpose_to_samples_rows(v1, len(sample_ids))
    v2 = _maybe_transpose_to_samples_rows(v2, len(sample_ids))

    # enforce consistent row alignment (assumes sample_ids order matches files)
    v1.index = sample_ids
    v2.index = sample_ids

    # Convert to float; keep NaNs (don’t fill here!)
    X1_raw = v1.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X2_raw = v2.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    d1 = X1_raw.shape[1]
    d2 = X2_raw.shape[1]
    n = X1_raw.shape[0]

    # Masks: 1=observed, 0=missing
    M1 = np.isfinite(X1_raw).astype(np.float32)
    M2 = np.isfinite(X2_raw).astype(np.float32)

    n_clusters = int(len(np.unique(y_true)))

    # ----- Split by indices (keeps X and M consistent, no leakage) -----
    idx = np.arange(n)
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y_true, test_size=0.2, random_state=1, stratify=y_true
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train, y_train, test_size=0.2, random_state=1, stratify=y_train
    )

    # Split views + masks
    X1_train_raw, X2_train_raw = X1_raw[idx_train], X2_raw[idx_train]
    X1_val_raw,   X2_val_raw   = X1_raw[idx_val],   X2_raw[idx_val]
    X1_test_raw,  X2_test_raw  = X1_raw[idx_test],  X2_raw[idx_test]

    M1_train, M2_train = M1[idx_train], M2[idx_train]
    M1_val,   M2_val   = M1[idx_val],   M2[idx_val]
    M1_test,  M2_test  = M1[idx_test],  M2[idx_test]

    # ----- Scaling: fit scaler ONLY on observed entries of training set -----
    # To be sure, fit StandardScaler using mean-imputed, but DON'T use imputation for model input
    # Prepare imputed training data just for scaler fit
    def _impute_for_scaler(X, M):
        means = _train_feature_means(np.nan_to_num(X, nan=0.0), M)
        X_imp = X.copy()
        missing = (M < 0.5)
        X_imp[missing] = means[np.where(missing)[1]]
        return X_imp

    X1_train_for_scaler = _impute_for_scaler(X1_train_raw, M1_train)
    X2_train_for_scaler = _impute_for_scaler(X2_train_raw, M2_train)

    scaler1 = StandardScaler().fit(X1_train_for_scaler)
    scaler2 = StandardScaler().fit(X2_train_for_scaler)

    # Scale all splits (keep nans for missing data; models will use masks)
    X1_train_scaled = scaler1.transform(X1_train_raw)
    X2_train_scaled = scaler2.transform(X2_train_raw)
    X1_val_scaled   = scaler1.transform(X1_val_raw)
    X2_val_scaled   = scaler2.transform(X2_val_raw)
    X1_test_scaled  = scaler1.transform(X1_test_raw)
    X2_test_scaled  = scaler2.transform(X2_test_raw)

    # Fill missing with 0.0 after scaling (common for VAE masks; change if your model expects different)
    X1_train = np.nan_to_num(X1_train_scaled, nan=0.0)
    X2_train = np.nan_to_num(X2_train_scaled, nan=0.0)
    X1_val   = np.nan_to_num(X1_val_scaled, nan=0.0)
    X2_val   = np.nan_to_num(X2_val_scaled, nan=0.0)
    X1_test  = np.nan_to_num(X1_test_scaled, nan=0.0)
    X2_test  = np.nan_to_num(X2_test_scaled, nan=0.0)

    # Concatenate
    X_train_all = np.concatenate([X1_train, X2_train], axis=1)
    X_val_all   = np.concatenate([X1_val,   X2_val],   axis=1)
    X_test_all  = np.concatenate([X1_test,  X2_test],  axis=1)
    M_train_all = np.concatenate([M1_train, M2_train], axis=1)
    M_val_all   = np.concatenate([M1_val,   M2_val],   axis=1)
    M_test_all  = np.concatenate([M1_test,  M2_test],  axis=1)

    print("Train/Val/Test:", X_train_all.shape, X_val_all.shape, X_test_all.shape)
    print("Masks:", M_train_all.shape, M_val_all.shape, M_test_all.shape)
    print("n_clusters:", n_clusters)

    # --- also return sample ids + split indices ---
    sample_ids = np.asarray(sample_ids, dtype=str)
    idx_train = np.asarray(idx_train, dtype=int)
    idx_val   = np.asarray(idx_val, dtype=int)
    idx_test  = np.asarray(idx_test, dtype=int)

    return (
        d1, d2,
        X_train_all, X_val_all, X_test_all,
        y_train, y_val, y_test,
        n_clusters,
        M_train_all, M_val_all, M_test_all,
        sample_ids, idx_train, idx_val, idx_test
    )
