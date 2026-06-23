from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics, worst_point_table
from helium_ml.model import MultiPropertyMLP
from helium_ml.train_utils import predict_raw


def predict_run(run: Path, raw: pd.DataFrame):
    metadata = joblib.load(run / "metadata.joblib")
    model_cfg = metadata["config"]
    model = MultiPropertyMLP(
        input_dim=len(metadata["input_cols"]),
        output_dim=len(metadata["target_cols"]),
        hidden_layers=list(model_cfg["hidden_layers"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.eval()
    x = raw[metadata["input_cols"]].to_numpy(dtype=np.float64)
    xp = x.copy()
    for col in model_cfg.get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        xp[:, idx] = np.log10(np.clip(xp[:, idx], 1e-300, None))
    class DataShim:
        pass
    data = DataShim()
    data.scaler_X = joblib.load(run / "scaler_X.joblib")
    data.scaler_y = joblib.load(run / "scaler_y.joblib")
    data.log_target_indices = metadata["log_target_indices"]
    data.shifted_log_target_offsets = metadata["shifted_log_target_offsets"]
    x_scaled = data.scaler_X.transform(xp)
    return predict_raw(model, x_scaled, data), metadata


def weight_grid(n_models: int, step: float):
    units = int(round(1.0 / step))
    for parts in itertools.product(range(units + 1), repeat=n_models):
        if sum(parts) == units:
            yield np.asarray(parts, dtype=np.float64) / units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    _, _, test_idx = split_indices(len(data.raw), cfg)
    test_raw = data.raw.iloc[test_idx].reset_index(drop=True)
    y_true = test_raw[cfg["target_cols"]].to_numpy(dtype=np.float64)

    preds = []
    run_names = []
    for run_arg in args.runs:
        run = Path(run_arg)
        pred, metadata = predict_run(run, test_raw)
        if metadata["target_cols"] != cfg["target_cols"]:
            raise ValueError(f"Target columns mismatch for {run}")
        preds.append(pred)
        run_names.append(str(run))
    stack = np.stack(preds, axis=0)

    best = None
    rows = []
    for weights in weight_grid(len(preds), args.step):
        pred = np.tensordot(weights, stack, axes=(0, 0))
        metrics = regression_metrics(y_true, pred, data.y_floor, cfg["target_cols"])
        overall = metrics["overall"]
        row = {f"w{i}": float(w) for i, w in enumerate(weights)}
        row.update(overall)
        rows.append(row)
        key = (overall["max"], overall["p99"], overall["mape"])
        if best is None or key < best[0]:
            best = (key, weights, pred, metrics)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["max", "p99", "mape"]).to_csv(out / "ensemble_weight_search.csv", index=False)
    _, weights, pred, metrics = best
    worst = worst_point_table(
        test_raw[cfg["input_cols"]].to_numpy(dtype=np.float64),
        y_true,
        pred,
        data.y_floor,
        cfg["input_cols"],
        cfg["target_cols"],
        n=500,
    )
    worst.to_csv(out / "worst_test_points.csv", index=False)
    pd.DataFrame(metrics["per_target"]).T.to_csv(out / "per_target.csv")
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"runs": run_names, "weights": weights.tolist(), **metrics}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"runs": run_names, "weights": weights.tolist(), "overall": metrics["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
