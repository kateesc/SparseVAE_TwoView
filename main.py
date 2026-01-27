import argparse
from train import train_POEMS

def main():
    parser = argparse.ArgumentParser()

    # training hyperparams
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

    args = parser.parse_args()

    is_test = bool(args.is_test)          # naming only
    # test_only = bool(args.test_only)      # skip training


    train_POEMS(
        lr_in=args.lr,
        wd_in=args.wd,
        batch_size_in=args.batch_size,
        nepoch_in=args.nepoch,
        is_wandb=bool(args.wandb),
        experiment_note=args.experiment_note,
        disease=args.disease,
        is_test=is_test,
        #test_only=test_only,
        model_name="POEMS",
    )

if __name__ == "__main__":
    main()
