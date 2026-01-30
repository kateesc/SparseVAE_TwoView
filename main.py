import argparse
from train import train_POEMS
from bootstrap_sparse_vae import bootstrap_sparse_vae_with_splits
import util

def main():
    parser = argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nepoch", type=int, default=5000)

    # experiment / io naming
    parser.add_argument("--experiment_note", type=str, default="POEMS-2view")
    parser.add_argument("--disease", type=str, default="plant")

    # flags
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--is_test", action="store_true",
                        help="Use test naming/output dirs (trained-2view-model). "
                             "NOTE: this does NOT skip training; it only changes naming.")
    parser.add_argument("--test_only", action="store_true",
                        help="Load saved model and only run test eval (your train.py uses is_test for naming, "
                             "so test_only implies is_test).")
    
    # Bootstrapping options
    parser.add_argument("--use_bootstrap", action="store_true", help="Enable bootstrap training mode")
    parser.add_argument("--num_bootstrap", type=int, default=100, help="Number of bootstrap runs")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to subsample for each bootstrap")
    parser.add_argument("--out_dir", type=str, default="./bootstrap_results", help="Output directory for bootstrap results")

    args = parser.parse_args()

    if args.use_bootstrap:
        print("Running bootstrapping for Sparse VAE...")
    
        # Load the data and split into train/validation/test sets
        from util import load_data_mocs
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         mask_train, mask_val, mask_test) = load_data_mocs(disease=args.disease)

        # Print dataset shapes
        print(f"Dataset shapes: Train={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")
    
        # Handle sampling size
        num_samples = args.num_samples if args.num_samples else X_train.shape[0]

        # Perform Bootstrapping on the train set
        bootstrap_sparse_vae_with_splits(
            model_class=train_POEMS,
            disease=args.disease,
            train_data=(X_train, y_train, mask_train),
            val_data=(X_val, y_val, mask_val),
            test_data=(X_test, y_test, mask_test),
            num_samples=num_samples,
            num_bootstrap=args.num_bootstrap,
            out_dir=args.out_dir,
            train_kwargs={
                "lr": args.lr,
                "wd": args.wd,
                "batch_size": args.batch_size,
                "nepoch": min(args.nepoch, 50),   # Use fewer epochs for bootstrapping
            },
        )

        # Perform stability analysis on bootstrap results
        util.bootstrap_stability_analysis(out_dir=args.out_dir, num_bootstraps=args.num_bootstrap, top_n=10)
    else:
        print("Running standard Sparse VAE training...")
        is_test = bool(args.is_test)

        train_POEMS(
            lr_in=args.lr,
            wd_in=args.wd,
            batch_size_in=args.batch_size,
            nepoch_in=args.nepoch,
            is_wandb=bool(args.wandb),
            experiment_note=args.experiment_note,
            disease=args.disease,
            is_test=is_test,
            model_name="POEMS",
        )

if __name__ == "__main__":
    main()
