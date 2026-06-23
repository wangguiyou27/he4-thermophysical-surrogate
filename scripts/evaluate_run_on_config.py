from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics, region_metrics, worst_point_table
from helium_ml.model import MultiPropertyMLP
from helium_ml.train_utils import predict_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    _, _, test_idx = split_indices(len(data.raw), cfg)

    run = Path(args.run)
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

    # Use the run's scalers/transforms by building a lightweight data object replacement.
    data.scaler_X = joblib.load(run / "scaler_X.joblib")
    data.scaler_y = joblib.load(run / "scaler_y.joblib")
    data.log_target_indices = metadata["log_target_indices"]
    data.shifted_log_target_offsets = metadata["shifted_log_target_offsets"]
    data.input_cols = metadata["input_cols"]
    data.target_cols = metadata["target_cols"]

    x_raw = data.raw[metadata["input_cols"]].to_numpy(dtype="float64")
    x_proc = x_raw.copy()
    for col in model_cfg.get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        import numpy as np

        x_proc[:, idx] = np.log10(np.clip(x_proc[:, idx], 1e-300, None))
    x_scaled = data.scaler_X.transform(x_proc)

    t0 = time.perf_counter()
    y_pred = predict_raw(model, x_scaled[test_idx], data)
    elapsed = time.perf_counter() - t0

    y_true = data.raw[metadata["target_cols"]].to_numpy(dtype="float64")[test_idx]
    x_test = x_raw[test_idx]
    metrics = regression_metrics(y_true, y_pred, data.y_floor, metadata["target_cols"])
    regions = region_metrics(x_test[:, 0], y_true, y_pred, data.y_floor, cfg.get("temperature_regions", {}))
    worst = worst_point_table(x_test, y_true, y_pred, data.y_floor, metadata["input_cols"], metadata["target_cols"], n=500)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"overall": metrics["overall"], "elapsed_s": elapsed}, f, indent=2, ensure_ascii=False)
    pd.DataFrame(metrics["per_target"]).T.to_csv(out / "per_target.csv")
    regions.to_csv(out / "region_metrics.csv", index=False)
    worst.to_csv(out / "worst_test_points.csv", index=False)
    print(json.dumps(metrics["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
