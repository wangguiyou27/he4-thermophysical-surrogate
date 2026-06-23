from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, sample_weights_from_temperature, split_indices
from helium_ml.metrics import regression_metrics, region_metrics, worst_point_table
from helium_ml.train_utils import build_model, predict_raw, save_artifacts, train_model


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg = dict(cfg)
        cfg["epochs"] = args.epochs

    data = prepare_data(cfg)
    if args.max_rows is not None and args.max_rows < len(data.raw):
        data.raw = data.raw.iloc[: args.max_rows].reset_index(drop=True)
        data.X_raw = data.X_raw[: args.max_rows]
        data.y_raw = data.y_raw[: args.max_rows]
        data.X_scaled = data.X_scaled[: args.max_rows]
        data.y_scaled = data.y_scaled[: args.max_rows]

    train_idx, val_idx, test_idx = split_indices(len(data.raw), cfg)
    sample_weights = sample_weights_from_temperature(data.X_raw[train_idx, 0], cfg)

    member_preds = []
    member_rows = []
    for member in range(args.members):
        seed = args.seed0 + member * 101
        set_seed(seed)
        model = build_model(cfg, data)
        model, history = train_model(
            model,
            data.X_scaled[train_idx],
            data.y_scaled[train_idx],
            data.X_scaled[val_idx],
            data.y_scaled[val_idx],
            cfg,
            sample_weights=sample_weights,
        )
        member_dir = out / f"member_{member}"
        save_artifacts(member_dir, model, data, cfg)
        pd.DataFrame(history).to_csv(member_dir / "history.csv", index=False)
        pred = predict_raw(model, data.X_scaled[test_idx], data)
        member_preds.append(pred)
        m = regression_metrics(data.y_raw[test_idx], pred, data.y_floor, data.target_cols)
        member_rows.append({"member": member, "seed": seed, **m["overall"]})
        print(json.dumps(member_rows[-1], ensure_ascii=False))

    y_pred = np.mean(np.stack(member_preds, axis=0), axis=0)
    metrics = regression_metrics(data.y_raw[test_idx], y_pred, data.y_floor, data.target_cols)
    regions = region_metrics(
        data.X_raw[test_idx, 0],
        data.y_raw[test_idx],
        y_pred,
        data.y_floor,
        cfg.get("temperature_regions", {}),
    )
    worst = worst_point_table(
        data.X_raw[test_idx],
        data.y_raw[test_idx],
        y_pred,
        data.y_floor,
        data.input_cols,
        data.target_cols,
        n=500,
    )

    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame(member_rows).to_csv(out / "member_metrics.csv", index=False)
    regions.to_csv(out / "region_metrics.csv", index=False)
    worst.to_csv(out / "worst_test_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["MAPE", "P95", "P99", "Max"], [
        metrics["overall"]["mape"],
        metrics["overall"]["p95"],
        metrics["overall"]["p99"],
        metrics["overall"]["max"],
    ])
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"Ensemble ({args.members} members)")
    fig.tight_layout()
    fig.savefig(out / "ensemble_summary.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

