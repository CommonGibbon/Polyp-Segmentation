# src/data/make_splits.py
import argparse
import json
import random
from pathlib import Path

import pandas as pd

SEED = 42


def main(root: str, img_dir="images", mask_dir="masks", train=0.7, val=0.15):
    root = Path(root)
    imgs = sorted((root / img_dir).glob("*.jpg")) + sorted(
        (root / img_dir).glob("*.png")
    )
    pairs = []
    for img in imgs:
        mask = (root / mask_dir / img.name).with_suffix(".png")
        if mask.exists():
            pairs.append((str(img), str(mask)))
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train)
    n_val = int(n * val)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }
    out = root / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        pd.DataFrame(v, columns=["image", "mask"]).to_csv(out / f"{k}.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {"seed": SEED, "counts": {k: len(v) for k, v in splits.items()}}, indent=2
        )
    )
    print("Wrote splits to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/kvasir")
    args = ap.parse_args()
    main(args.root)
