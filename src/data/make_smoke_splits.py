import argparse
import json
from pathlib import Path

import pandas as pd


def main(
    root: str, train_size: int = 20, val_size: int = 10, smoke_dir: str = "splits_smoke"
):
    root = Path(root)
    splits_dir = root / "splits"
    smoke_dir = root / smoke_dir
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # Read existing splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    # Take first N items (deterministic)
    smoke_train = train_df.head(train_size)
    smoke_val = val_df.head(val_size)

    # Write smoke splits
    smoke_train.to_csv(smoke_dir / "train.csv", index=False)
    smoke_val.to_csv(smoke_dir / "val.csv", index=False)

    # Write metadata
    meta = {
        "seed": "deterministic_first_n",
        "counts": {"train": len(smoke_train), "val": len(smoke_val)},
        "source_splits": "splits",
    }
    (smoke_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote smoke splits to {smoke_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/")
    ap.add_argument("--train-size", type=int, default=20)
    ap.add_argument("--val-size", type=int, default=10)
    ap.add_argument("--smoke-dir", type=str, default="splits_smoke")
    args = ap.parse_args()
    main(args.root, args.train_size, args.val_size, args.smoke_dir)
