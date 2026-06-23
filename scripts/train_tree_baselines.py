from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics, worst_point_table


def inverse_targets(y_scaled: np.ndarray, data) -> np.ndarray:
    y_proc = data.scaler_y.inverse_transform(y_scaled)
    y_raw = y_proc.copy()
    for idx in data.log_target_indices:
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx])
    for idx, offset in data.shifted_log_target_offsets.items():
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx]) - offset
    return y_raw


def fit_eval(name, model, x_train, y_train, x_test, y_true, data, out_dir, save_model):
    print(f"Training {name}...")
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_scaled = model.predict(x_test)
    pred_s = time.perf_counter() - t0
    y_pred = inverse_targets(pred_scaled, data)
    metrics = regression_metrics(y_true, y_pred, data.y_floor, data.target_cols)
    worst = worst_point_table(
        data.X_raw[-len(x_test) :],
        y_true,
        y_pred,
        data.y_floor,
        data.input_cols,
        data.target_cols,
        n=300,
    )
    row = {
        "model": name,
        "train_time_s": train_s,
        "predict_time_s": pred_s,
        "points_per_s": len(x_test) / pred_s,
        **metrics["overall"],
    }
    safe = name.lower().replace(" ", "_")
    with (out_dir / f"{safe}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    worst.to_csv(out_dir / f"{safe}_worst_points.csv", index=False)
    if save_model:
        joblib.dump(model, out_dir / f"{safe}.joblib", compress=3)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-sample", type=int, default=0)
    parser.add_argument("--test-sample", type=int, default=0)
    parser.add_argument("--trees", type=int, default=180)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    train_idx, _, test_idx = split_indices(len(data.X_scaled), cfg)
    rng = np.random.default_rng(args.seed)
    if args.train_sample > 0 and len(train_idx) > args.train_sample:
        train_idx = rng.choice(train_idx, size=args.train_sample, replace=False)
    if args.test_sample > 0 and len(test_idx) > args.test_sample:
        test_idx = rng.choice(test_idx, size=args.test_sample, replace=False)

    x_train = data.X_scaled[train_idx]
    y_train = data.y_scaled[train_idx]
    x_test = data.X_scaled[test_idx]
    y_true = data.y_raw[test_idx]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(
        fit_eval(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=args.trees,
                min_samples_leaf=args.min_samples_leaf,
                max_depth=args.max_depth,
                random_state=args.seed,
                n_jobs=args.n_jobs,
            ),
            x_train,
            y_train,
            x_test,
            y_true,
            data,
            out_dir,
            args.save_models,
        )
    )
    rows.append(
        fit_eval(
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=args.trees,
                min_samples_leaf=args.min_samples_leaf,
                max_depth=args.max_depth,
                random_state=args.seed,
                n_jobs=args.n_jobs,
            ),
            x_train,
            y_train,
            x_test,
            y_true,
            data,
            out_dir,
            args.save_models,
        )
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "tree_baseline_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
