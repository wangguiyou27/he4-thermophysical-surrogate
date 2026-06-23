from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--aug", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--aug-samples", type=int, default=220000)
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    aug = pd.read_csv(args.aug)
    cols = list(base.columns)
    aug = aug[cols]
    if args.aug_samples and args.aug_samples < len(aug):
        aug = (
            aug.groupby("phase_region", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max(1, round(args.aug_samples * len(g) / len(aug)))), random_state=args.seed))
            .reset_index(drop=True)
        )
        if len(aug) > args.aug_samples:
            aug = aug.sample(n=args.aug_samples, random_state=args.seed).reset_index(drop=True)
    out = pd.concat([base, aug], ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(
        {
            "base_rows": int(len(base)),
            "aug_rows": int(len(aug)),
            "out_rows": int(len(out)),
            "out": str(path),
            "phase_counts": out["phase_region"].value_counts().to_dict(),
        }
    )


if __name__ == "__main__":
    main()
