import os
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def stratified_resample(X, y, mask, n_samples):
    """
    Perform stratified resampling to ensure class distribution is kept for bootstrapping.

    Args:
        X: Input data matrix (features).
        y: Class labels.
        mask: Mask matrix for missing values.
        n_samples: Number of samples to resample.

    Returns:
        X_resampled, y_resampled, mask_resampled: Resampled data, labels, and mask.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_ratios = class_counts / class_counts.sum()

    X_resampled = []
    y_resampled = []
    mask_resampled = []

    for cls, ratio in zip(unique_classes, class_counts):
        cls_mask = (y == cls)
        n_cls_samples = int(ratio * n_samples)

        # Resample this class's samples
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]
        mask_cls = mask[cls_mask]

        X_cls_resampled, _, y_cls_resampled, _ = train_test_split(
            X_cls, y_cls, train_size=n_cls_samples, stratify=y_cls
        )

        X_resampled.append(X_cls_resampled)
        y_resampled.append(y_cls_resampled)
        mask_resampled.append(mask_cls[:len(y_cls_resampled)])

    return np.vstack(X_resampled), np.concatenate(y_resampled), np.vstack(mask_resampled)

def bootstrap_sparse_vae_with_splits(
    model_class,
    disease,
    train_data,
    val_data,
    test_data,
    num_samples=None,  # If None, use the full train set
    num_bootstrap=100,
    out_dir="./bootstrap_results",
    train_kwargs=None,
):
    """
    Perform bootstrap training for Sparse VAE using fixed train/val/test splits and save results for stability analysis.

    Args:
        model_class: The Sparse VAE class (e.g., POEMS or your model implementation).
        disease: Name of the dataset to bootstrap.
        train_data: Tuple (X_train, y_train, mask_train) for training.
        val_data: Tuple (X_val, y_val, mask_val) for validation.
        test_data: Tuple (X_test, y_test, mask_test) for testing.
        num_samples: Number of training samples (use full training set if None).
        num_bootstrap: Number of bootstrap replicates.
        out_dir: Directory to save results.
        train_kwargs: Dictionary of additional parameters for the Sparse VAE training function (e.g., hyperparameters).

    Saves:
        - W1, W2, pstar, and latent summaries for each bootstrap.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Extract training set
    X_train, y_train, mask_train = train_data
    n_samples = num_samples if num_samples else X_train.shape[0]  # Default to full training set

    for bootstrap_idx in range(1, num_bootstrap + 1):
        print(f"Bootstrap {bootstrap_idx}/{num_bootstrap}")

        # Create a subsampled training set using stratified resampling
        X_resampled, y_resampled, mask_resampled = stratified_resample(X_train, y_train, mask_train, n_samples)

        # Train Sparse VAE with the subsampled training set
        model = model_class(**train_kwargs)
        model.train(X_resampled, mask_resampled, val_data)  # Ensure validation data is passed here

        # Save model outputs for bootstrap run
        W1 = model.specific_modules["specific1"].W.detach().cpu().numpy()
        W2 = model.specific_modules["specific2"].W.detach().cpu().numpy()
        pstar1 = model.specific_modules["specific1"].pstar.detach().cpu().numpy()
        pstar2 = model.specific_modules["specific2"].pstar.detach().cpu().numpy()

        np.save(os.path.join(out_dir, f"bootstrap_{bootstrap_idx}_W1.npy"), W1)
        np.save(os.path.join(out_dir, f"bootstrap_{bootstrap_idx}_W2.npy"), W2)
        np.save(os.path.join(out_dir, f"bootstrap_{bootstrap_idx}_pstar1.npy"), pstar1)
        np.save(os.path.join(out_dir, f"bootstrap_{bootstrap_idx}_pstar2.npy"), pstar2)

        print(f"Bootstrap {bootstrap_idx} results saved.")        
