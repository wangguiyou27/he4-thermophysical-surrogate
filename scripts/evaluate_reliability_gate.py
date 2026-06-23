from __future__ import annotations

import argparse
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
from helium_ml.metrics import regression_metrics, relative_error
from helium_ml.model import MultiPropertyMLP
from helium_ml.train_utils import predict_raw


def load_model(run: Path):
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
    return model, metadata, joblib.load(run / "scaler_X.joblib"), joblib.load(run / "scaler_y.joblib")


def scaled_inputs(raw: pd.DataFrame, metadata, scaler_x):
    cfg = metadata["config"]
    x = raw[metadata["input_cols"]].to_numpy(dtype=np.float64)
    xp = x.copy()
    for col in cfg.get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        xp[:, idx] = np.log10(np.clip(xp[:, idx], 1e-300, None))
    return x, scaler_x.transform(xp)


def gate_mask(x_raw, rules):
    temp = x_raw[:, 0]
    pressure = x_raw[:, 1]
    mask = np.zeros(len(x_raw), dtype=bool)
    for rule in rules:
        m = np.ones(len(x_raw), dtype=bool)
        if "t_min" in rule:
            m &= temp >= rule["t_min"]
        if "t_max" in rule:
            m &= temp <= rule["t_max"]
        if "p_min" in rule:
            m &= pressure >= rule["p_min"]
        if "p_max" in rule:
            m &= pressure <= rule["p_max"]
        mask |= m
    return mask


def default_rules():
    return [
        {"name": "critical_cp_density_band", "t_min": 5.3, "t_max": 6.5, "p_min": 0.18, "p_max": 0.60},
        {"name": "near_critical_margin", "t_min": 5.3, "t_max": 5.8, "p_min": 0.12, "p_max": 0.70},
        {"name": "high_temperature_very_low_pressure", "t_min": 240.0, "t_max": 300.0, "p_min": 0.001, "p_max": 0.005},
        {"name": "upper_mid_temperature_low_pressure", "t_min": 200.0, "t_max": 260.0, "p_min": 0.001, "p_max": 0.005},
        {"name": "mid_temperature_low_pressure", "t_min": 140.0, "t_max": 245.0, "p_min": 0.001, "p_max": 0.0055},
        {"name": "high_temperature_high_pressure_density", "t_min": 240.0, "t_max": 300.0, "p_min": 2.5, "p_max": 3.0},
        {"name": "near_critical_high_pressure_entropy", "t_min": 5.3, "t_max": 6.0, "p_min": 2.0, "p_max": 3.0},
        {"name": "mid_temperature_high_pressure_density", "t_min": 50.0, "t_max": 130.0, "p_min": 2.7, "p_max": 3.0},
        {"name": "very_low_pressure_strip", "t_min": 5.3, "t_max": 300.0, "p_min": 0.001, "p_max": 0.0015},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    _, _, test_idx = split_indices(len(data.raw), cfg)
    model, metadata, scaler_x, scaler_y = load_model(Path(args.run))
    x_raw_all, x_scaled_all = scaled_inputs(data.raw, metadata, scaler_x)
    data.scaler_X = scaler_x
    data.scaler_y = scaler_y
    data.log_target_indices = metadata["log_target_indices"]
    data.shifted_log_target_offsets = metadata["shifted_log_target_offsets"]
    data.input_cols = metadata["input_cols"]
    data.target_cols = metadata["target_cols"]

    y_true = data.raw[metadata["target_cols"]].to_numpy(dtype=np.float64)[test_idx]
    y_pred = predict_raw(model, x_scaled_all[test_idx], data)
    x_test = x_raw_all[test_idx]
    err = relative_error(y_true, y_pred, data.y_floor) * 100.0

    rules = default_rules()
    fallback = gate_mask(x_test, rules)
    retained = ~fallback
    retained_metrics = regression_metrics(y_true[retained], y_pred[retained], data.y_floor, metadata["target_cols"])
    combined_err = err.copy()
    combined_err[fallback, :] = 0.0
    summary = {
        "n_test": int(len(test_idx)),
        "fallback_count": int(fallback.sum()),
        "fallback_fraction_pct": float(fallback.mean() * 100.0),
        "retained_count": int(retained.sum()),
        "retained_overall": retained_metrics["overall"],
        "combined_max_pct_assuming_refprop_fallback": float(combined_err.max()),
        "combined_p99_pct_assuming_refprop_fallback": float(np.percentile(combined_err.reshape(-1), 99)),
        "rules": rules,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"fallback": fallback, "Temperature (K)": x_test[:, 0], "Pressure (MPa)": x_test[:, 1]}).to_csv(
        out / "gate_points.csv", index=False
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
