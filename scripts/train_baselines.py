from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics
from helium_ml.model import MultiPropertyMLP


def inverse_targets(y_scaled: np.ndarray, data) -> np.ndarray:
    y_proc = data.scaler_y.inverse_transform(y_scaled)
    y_raw = y_proc.copy()
    for idx in data.log_target_indices:
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx])
    for idx, offset in data.shifted_log_target_offsets.items():
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx]) - offset
    return y_raw


def load_ann(run_dir: Path, data):
    metadata = joblib.load(run_dir / "metadata.joblib")
    cfg = metadata["config"]
    model = MultiPropertyMLP(
        input_dim=len(data.input_cols),
        output_dim=len(data.target_cols),
        hidden_layers=list(cfg["hidden_layers"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


def ann_predict_scaled(model, x_scaled: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            xb = torch.as_tensor(x_scaled[start : start + batch_size], dtype=torch.float32)
            chunks.append(model(xb).cpu().numpy())
    return np.vstack(chunks)


def time_predict(model, x: np.ndarray, repeats: int = 3):
    times = []
    pred = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = model.predict(x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), pred


def time_ann_predict(model, x: np.ndarray, batch_size: int, repeats: int = 3):
    times = []
    pred = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = ann_predict_scaled(model, x, batch_size)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), pred


def build_xgboost(seed: int, n_jobs: int):
    from xgboost import XGBRegressor

    base = XGBRegressor(
        n_estimators=350,
        max_depth=8,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.95,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=max(1, n_jobs),
        reg_lambda=1.0,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def summarize_model(name, train_time, pred_time, y_pred_raw, y_true_raw, data, n_train, n_test):
    metrics = regression_metrics(y_true_raw, y_pred_raw, data.y_floor, data.target_cols)
    row = {
        "model": name,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_time_s": float(train_time),
        "predict_time_s": float(pred_time),
        "points_per_s": float(n_test / pred_time),
        **metrics["overall"],
    }
    per_target_rows = []
    for target, vals in metrics["per_target"].items():
        per_target_rows.append({"model": name, "property": target, **vals})
    return row, per_target_rows, metrics


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    order = summary.sort_values("mape")["model"].tolist()
    x = np.arange(len(order))
    ordered = summary.set_index("model").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(x, ordered["mape"], color="#4C78A8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=20, ha="right")
    axes[0].set_ylabel("Overall MAPE (%)")
    axes[0].set_title("Accuracy comparison")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, ordered["points_per_s"], color="#F58518")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=20, ha="right")
    axes[1].set_ylabel("Predictions per second")
    axes[1].set_title("Inference throughput")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_accuracy_speed.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    width = 0.25
    for offset, col, label in [(-width, "p95", "P95"), (0, "p99", "P99"), (width, "max", "Max")]:
        ax.bar(x + offset, ordered[col], width=width, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Absolute percentage error (%)")
    ax.set_title("Tail-error comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_tail_errors.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/he4_single_phase.json")
    parser.add_argument("--ann-run", default="runs/he4_supervised_shifted_entropy")
    parser.add_argument("--out", default="runs/baseline_comparison")
    parser.add_argument("--train-sample", type=int, default=60000)
    parser.add_argument("--test-sample", type=int, default=40000)
    parser.add_argument("--gaussian-train-sample", type=int, default=25000)
    parser.add_argument("--gaussian-components", type=int, default=1500)
    parser.add_argument("--rf-trees", type=int, default=220)
    parser.add_argument("--extra-trees", type=int, default=220)
    parser.add_argument("--n-jobs", type=int, default=-1)
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

    models = []
    models.append(
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=args.rf_trees,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=args.seed,
                n_jobs=args.n_jobs,
            ),
            x_train,
            y_train,
        )
    )
    models.append(
        (
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=args.extra_trees,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=args.seed,
                n_jobs=args.n_jobs,
            ),
            x_train,
            y_train,
        )
    )
    models.append(("XGBoost", build_xgboost(args.seed, args.n_jobs), x_train, y_train))

    g_train_idx = train_idx
    if args.gaussian_train_sample > 0 and len(g_train_idx) > args.gaussian_train_sample:
        g_train_idx = rng.choice(g_train_idx, size=args.gaussian_train_sample, replace=False)
    gaussian = make_pipeline(
        Nystroem(
            kernel="rbf",
            gamma=0.6,
            n_components=args.gaussian_components,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        ),
        Ridge(alpha=1e-4),
    )
    models.append(
        (
            "Gaussian RBF Ridge",
            gaussian,
            data.X_scaled[g_train_idx],
            data.y_scaled[g_train_idx],
        )
    )

    summary_rows = []
    per_target_rows = []
    full_metrics = {}

    ann = load_ann(Path(args.ann_run), data)
    ann_time, ann_scaled = time_ann_predict(ann, x_test, int(cfg.get("batch_size", 8192)), repeats=3)
    ann_pred = inverse_targets(ann_scaled, data)
    row, rows, metrics = summarize_model(
        "Current MLP", 0.0, ann_time, ann_pred, y_true, data, len(train_idx), len(test_idx)
    )
    summary_rows.append(row)
    per_target_rows.extend(rows)
    full_metrics["Current MLP"] = metrics

    for name, model, xt, yt in models:
        print(f"Training {name} on {len(xt)} samples...")
        t0 = time.perf_counter()
        model.fit(xt, yt)
        train_time = time.perf_counter() - t0
        pred_time, pred_scaled = time_predict(model, x_test, repeats=3)
        pred_raw = inverse_targets(pred_scaled, data)
        row, rows, metrics = summarize_model(
            name, train_time, pred_time, pred_raw, y_true, data, len(xt), len(test_idx)
        )
        summary_rows.append(row)
        per_target_rows.extend(rows)
        full_metrics[name] = metrics
        if args.save_models:
            joblib.dump(model, out_dir / f"{name.replace(' ', '_').lower()}.joblib", compress=3)
        print(f"{name}: MAPE={row['mape']:.4f}%, P99={row['p99']:.4f}%, speed={row['points_per_s']:.0f} points/s")

    summary = pd.DataFrame(summary_rows)
    per_target = pd.DataFrame(per_target_rows)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False, encoding="utf-8-sig")
    per_target.to_csv(out_dir / "baseline_per_property_metrics.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, ensure_ascii=False, indent=2)
    plot_summary(summary, out_dir)

    print()
    print(summary.sort_values("mape").to_string(index=False))
    print(f"\nSaved baseline comparison to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
