from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics
from helium_ml.model import MultiPropertyMLP
from helium_ml.train_utils import predict_raw


def load_ann(run: Path, data):
    metadata = joblib.load(run / "metadata.joblib")
    cfg = metadata["config"]
    model = MultiPropertyMLP(
        input_dim=len(metadata["input_cols"]),
        output_dim=len(metadata["target_cols"]),
        hidden_layers=list(cfg["hidden_layers"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.eval()
    data.scaler_X = joblib.load(run / "scaler_X.joblib")
    data.scaler_y = joblib.load(run / "scaler_y.joblib")
    data.log_target_indices = metadata["log_target_indices"]
    data.shifted_log_target_offsets = metadata["shifted_log_target_offsets"]
    data.input_cols = metadata["input_cols"]
    data.target_cols = metadata["target_cols"]
    return model, metadata


def scaled_inputs(df: pd.DataFrame, metadata: dict, scaler_x):
    x = df[metadata["input_cols"]].to_numpy(dtype=np.float64)
    xp = x.copy()
    for col in metadata["config"].get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        xp[:, idx] = np.log10(np.clip(xp[:, idx], 1e-300, None))
    return scaler_x.transform(xp)


def summarize(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_floor, targets: list[str], seconds: float) -> dict:
    metrics = regression_metrics(y_true, y_pred, y_floor, targets)
    row = {"method": name, "predict_time_s": seconds, "points_per_s": len(y_true) / seconds}
    row.update(metrics["overall"])
    return row


def build_features(df: pd.DataFrame, mode: str, phase_weight: float) -> np.ndarray:
    t = df["Temperature (K)"].to_numpy(dtype=np.float64)
    p = df["Pressure (MPa)"].to_numpy(dtype=np.float64)
    cols = [
        np.log10(np.clip(t, 1e-12, None)),
        np.log10(np.clip(p, 1e-12, None)),
    ]
    if mode == "tp_phase":
        phase = df["phase_code"].to_numpy(dtype=np.float64) * phase_weight
        cols.append(phase)
    x = np.column_stack(cols)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    return (x - mean) / std, mean, std


def transform_features(df: pd.DataFrame, mode: str, phase_weight: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    t = df["Temperature (K)"].to_numpy(dtype=np.float64)
    p = df["Pressure (MPa)"].to_numpy(dtype=np.float64)
    cols = [
        np.log10(np.clip(t, 1e-12, None)),
        np.log10(np.clip(p, 1e-12, None)),
    ]
    if mode == "tp_phase":
        cols.append(df["phase_code"].to_numpy(dtype=np.float64) * phase_weight)
    return (np.column_stack(cols) - mean) / std


def idw_predict(tree: cKDTree, x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray, k: int, eps: float) -> np.ndarray:
    dist, idx = tree.query(x_query, k=k, workers=-1)
    if k == 1:
        return y_train[idx]
    dist = np.maximum(dist, eps)
    w = 1.0 / (dist**2)
    w_sum = w.sum(axis=1, keepdims=True)
    return (w[..., None] * y_train[idx]).sum(axis=1) / w_sum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/he4_single_phase_extended_2p2_phaseaware_critical_features.json")
    parser.add_argument("--ann-run", default="runs/he4_extended_2p2_phaseaware_critical_features_ann")
    parser.add_argument("--out", default="runs/precomputed_interpolation_benchmark")
    parser.add_argument("--max-train", type=int, default=220000)
    parser.add_argument("--max-test", type=int, default=50000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    train_idx, _, test_idx = split_indices(len(data.raw), cfg)

    rng = np.random.default_rng(123)
    if len(train_idx) > args.max_train:
        train_idx = rng.choice(train_idx, size=args.max_train, replace=False)
    if len(test_idx) > args.max_test:
        test_idx = rng.choice(test_idx, size=args.max_test, replace=False)

    model, metadata = load_ann(Path(args.ann_run), data)
    train_df = data.raw.iloc[train_idx].reset_index(drop=True)
    test_df = data.raw.iloc[test_idx].reset_index(drop=True)
    y_train = train_df[metadata["target_cols"]].to_numpy(dtype=np.float64)
    y_true = test_df[metadata["target_cols"]].to_numpy(dtype=np.float64)

    rows = []

    x_ann = scaled_inputs(test_df, metadata, data.scaler_X)
    t0 = time.perf_counter()
    y_ann = predict_raw(model, x_ann, data)
    rows.append(summarize("Critical-feature ANN", y_true, y_ann, data.y_floor, metadata["target_cols"], time.perf_counter() - t0))

    for mode in ["tp", "tp_phase"]:
        x_train, mean, std = build_features(train_df, mode, phase_weight=3.0)
        x_test = transform_features(test_df, mode, phase_weight=3.0, mean=mean, std=std)

        t0 = time.perf_counter()
        tree = cKDTree(x_train)
        build_seconds = time.perf_counter() - t0

        for k in [1, 4, 8, 16]:
            t0 = time.perf_counter()
            y_pred = idw_predict(tree, x_train, y_train, x_test, k=k, eps=1e-12)
            predict_seconds = time.perf_counter() - t0
            row = summarize(
                f"Precomputed KDTree {'T-P' if mode == 'tp' else 'T-P-phase'} k={k}",
                y_true,
                y_pred,
                data.y_floor,
                metadata["target_cols"],
                predict_seconds,
            )
            row["build_time_s"] = build_seconds
            row["n_database_points"] = len(train_df)
            row["n_test_points"] = len(test_df)
            rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "interpolation_vs_ann_summary.csv", index=False, encoding="utf-8-sig")

    best = summary.sort_values(["max", "mape"]).head(10)
    with (out / "interpolation_vs_ann_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_database_points": int(len(train_df)),
                "n_test_points": int(len(test_df)),
                "best_by_max": best.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(summary[["method", "points_per_s", "mape", "p99", "max", "within_5pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
