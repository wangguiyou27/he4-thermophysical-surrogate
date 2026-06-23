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
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics
from helium_ml.train_utils import build_model, predict_raw, train_model


def select_random(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(pool, size=min(n, len(pool)), replace=False)


def select_uncertainty(model, X_pool: np.ndarray, pool: np.ndarray, n: int, n_forward: int = 12) -> np.ndarray:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.train()
    xb = torch.as_tensor(X_pool, dtype=torch.float32, device=dev)
    preds = []
    with torch.no_grad():
        for _ in range(n_forward):
            preds.append(model(xb).detach().cpu().numpy())
    std = np.std(np.stack(preds, axis=0), axis=0).mean(axis=1)
    top = np.argsort(std)[-min(n, len(pool)) :]
    return pool[top]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy", choices=["random", "uncertainty"], default="uncertainty")
    parser.add_argument("--initial", type=int, default=1000)
    parser.add_argument("--query", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
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

    rng = np.random.default_rng(cfg.get("seed", 42))
    initial = rng.choice(train_idx, size=min(args.initial, len(train_idx)), replace=False)
    labeled = set(initial.tolist())
    pool = np.array([i for i in train_idx if i not in labeled], dtype=int)

    rows = []
    model = None
    for round_id in range(args.rounds + 1):
        labeled_idx = np.array(sorted(labeled), dtype=int)
        model = build_model(cfg, data)
        model, _ = train_model(
            model,
            data.X_scaled[labeled_idx],
            data.y_scaled[labeled_idx],
            data.X_scaled[val_idx],
            data.y_scaled[val_idx],
            cfg,
        )
        y_pred = predict_raw(model, data.X_scaled[test_idx], data)
        metrics = regression_metrics(data.y_raw[test_idx], y_pred, data.y_floor, data.target_cols)
        row = {"round": round_id, "n_labeled": len(labeled_idx), **metrics["overall"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if round_id == args.rounds or len(pool) == 0:
            break

        if args.strategy == "random":
            chosen = select_random(pool, args.query, rng)
        else:
            probe_size = min(len(pool), max(args.query * 20, 5000))
            probe = rng.choice(pool, size=probe_size, replace=False)
            chosen = select_uncertainty(model, data.X_scaled[probe], probe, args.query)

        labeled.update(chosen.tolist())
        chosen_set = set(chosen.tolist())
        pool = np.array([i for i in pool if i not in chosen_set], dtype=int)

    curve = pd.DataFrame(rows)
    curve.to_csv(out / f"active_learning_{args.strategy}.csv", index=False)
    with (out / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["n_labeled"], curve["mape"], marker="o", label=args.strategy)
    ax.axhline(1.0, color="red", linestyle="--", label="1%")
    ax.axhline(5.0, color="orange", linestyle="--", label="5%")
    ax.set_xlabel("Labeled training samples")
    ax.set_ylabel("Overall MAPE (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"active_learning_{args.strategy}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
