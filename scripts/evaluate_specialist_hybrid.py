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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics, relative_error, worst_point_table
from helium_ml.model import MultiPropertyMLP
from helium_ml.train_utils import predict_raw


class LocalLinearInterpolator:
    def __init__(self):
        self.linear = None
        self.nearest = None

    def fit(self, x, y):
        xy = x[:, :2]
        self.linear = LinearNDInterpolator(xy, y)
        self.nearest = NearestNDInterpolator(xy, y)
        return self

    def predict(self, x):
        xy = x[:, :2]
        y = np.asarray(self.linear(xy), dtype=np.float64)
        missing = ~np.isfinite(y)
        if np.any(missing):
            y[missing] = self.nearest(xy[missing])
        return y


def load_mlp(run: Path, data):
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


def scale_x(df, metadata, scaler_x):
    x = df[metadata["input_cols"]].to_numpy(dtype=np.float64)
    xp = x.copy()
    for col in metadata["config"].get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        xp[:, idx] = np.log10(np.clip(xp[:, idx], 1e-300, None))
    return x, scaler_x.transform(xp)


def mask_region(df, bounds):
    mask = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in bounds.items():
        vals = df[col].to_numpy(dtype=np.float64)
        mask &= (vals >= lo) & (vals <= hi)
    return mask


def train_expert(name, train_df, aug_df, target, bounds, transform, seed, trees, model_type="extratrees", neighbors=12):
    if model_type == "knn":
        model_type = "linear"
    cols = ["Temperature (K)", "Pressure (MPa)", "phase_code"]
    df = pd.concat([train_df, aug_df], ignore_index=True)
    mask = mask_region(df, bounds)
    local = df.loc[mask].copy()
    x = local[cols].to_numpy(dtype=np.float64)
    x[:, 0] = np.log10(x[:, 0])
    x[:, 1] = np.log10(x[:, 1])
    y = local[target].to_numpy(dtype=np.float64)
    if transform == "log":
        y_fit = np.log10(np.clip(y, 1e-300, None))
    elif transform.startswith("shift"):
        offset = float(transform.split(":")[1])
        y_fit = np.log10(np.clip(y + offset, 1e-300, None))
    else:
        y_fit = y
    if model_type == "linear":
        model = LocalLinearInterpolator()
    elif model_type == "knn":
        model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=neighbors, weights="distance", p=2, n_jobs=4),
        )
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=trees,
            random_state=seed,
            n_jobs=4,
            min_samples_leaf=1,
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=trees,
            random_state=seed,
            n_jobs=4,
            min_samples_leaf=1,
        )
    t0 = time.perf_counter()
    model.fit(x, y_fit)
    train_s = time.perf_counter() - t0
    return {
        "name": name,
        "target": target,
        "bounds": bounds,
        "transform": transform,
        "model_type": model_type,
        "neighbors": int(neighbors),
        "model": model,
        "n_train": int(len(local)),
        "train_s": train_s,
    }


def apply_expert(pred, test_df, expert, target_cols):
    cols = ["Temperature (K)", "Pressure (MPa)", "phase_code"]
    mask = mask_region(test_df, expert["bounds"])
    if not np.any(mask):
        return 0
    x = test_df.loc[mask, cols].to_numpy(dtype=np.float64)
    x[:, 0] = np.log10(x[:, 0])
    x[:, 1] = np.log10(x[:, 1])
    y_fit = expert["model"].predict(x)
    transform = expert["transform"]
    if transform == "log":
        y = 10**y_fit
    elif transform.startswith("shift"):
        offset = float(transform.split(":")[1])
        y = 10**y_fit - offset
    else:
        y = y_fit
    pred[np.where(mask)[0], target_cols.index(expert["target"])] = y
    return int(mask.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mlp-run", required=True)
    parser.add_argument("--aug", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trees", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    train_idx, _, test_idx = split_indices(len(data.raw), cfg)
    model, metadata = load_mlp(Path(args.mlp_run), data)
    x_raw, x_scaled = scale_x(data.raw, metadata, data.scaler_X)
    y_base = predict_raw(model, x_scaled[test_idx], data)
    y_true = data.raw[metadata["target_cols"]].to_numpy(dtype=np.float64)[test_idx]
    test_df = data.raw.iloc[test_idx].reset_index(drop=True)
    train_df = data.raw.iloc[train_idx].reset_index(drop=True)
    aug_df = pd.read_csv(args.aug)
    pred = y_base.copy()

    cp_col = next(c for c in metadata["target_cols"] if c.startswith("Cp "))
    entropy_col = next(c for c in metadata["target_cols"] if c.startswith("Entropy "))
    viscosity_col = next(c for c in metadata["target_cols"] if c.startswith("Viscosity "))
    conductivity_col = next(c for c in metadata["target_cols"] if c.startswith("Thermal Conductivity "))

    experts_spec = [
        {
            "name": "critical_cp",
            "target": cp_col,
            "bounds": {"Temperature (K)": (5.20, 5.36), "Pressure (MPa)": (0.225, 0.265), "phase_code": (3, 3)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "liquid_near_critical_cp",
            "target": cp_col,
            "bounds": {"Temperature (K)": (5.05, 5.20), "Pressure (MPa)": (0.220, 0.250), "phase_code": (4, 4)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "vapor_near_critical_cp",
            "target": cp_col,
            "bounds": {"Temperature (K)": (5.00, 5.23), "Pressure (MPa)": (0.170, 0.230), "phase_code": (1, 1)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "gas_near_critical_cp",
            "target": cp_col,
            "bounds": {"Temperature (K)": (5.18, 5.36), "Pressure (MPa)": (0.220, 0.260), "phase_code": (2, 2)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "near_lambda_enthalpy",
            "target": "Enthalpy (kJ/kg)",
            "bounds": {"Temperature (K)": (2.20, 2.50), "Pressure (MPa)": (0.85, 1.35), "phase_code": (4, 4)},
            "transform": "raw",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "cold_liquid_enthalpy",
            "target": "Enthalpy (kJ/kg)",
            "bounds": {"Temperature (K)": (2.20, 4.60), "Pressure (MPa)": (0.08, 1.40), "phase_code": (4, 4)},
            "transform": "raw",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "cold_liquid_entropy",
            "target": entropy_col,
            "bounds": {"Temperature (K)": (4.00, 5.00), "Pressure (MPa)": (0.08, 0.60), "phase_code": (4, 4)},
            "transform": "raw",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "critical_enthalpy",
            "target": "Enthalpy (kJ/kg)",
            "bounds": {"Temperature (K)": (5.18, 5.45), "Pressure (MPa)": (0.220, 0.270), "phase_code": (3, 3)},
            "transform": "raw",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "high_pressure_critical_enthalpy",
            "target": "Enthalpy (kJ/kg)",
            "bounds": {"Temperature (K)": (5.15, 5.55), "Pressure (MPa)": (2.80, 3.00), "phase_code": (3, 3)},
            "transform": "raw",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "entropy_zero",
            "target": entropy_col,
            "bounds": {"Temperature (K)": (5.20, 6.25), "Pressure (MPa)": (1.30, 3.00), "phase_code": (3, 3)},
            "transform": "raw",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "density_lowT_highP",
            "target": "Density (kg/m^3)",
            "bounds": {"Temperature (K)": (5.30, 6.20), "Pressure (MPa)": (1.8, 3.0), "phase_code": (3, 3)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "density_highT_lowP",
            "target": "Density (kg/m^3)",
            "bounds": {"Temperature (K)": (180.0, 300.0), "Pressure (MPa)": (0.001, 0.006), "phase_code": (2, 2)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "near_critical_gas_density",
            "target": "Density (kg/m^3)",
            "bounds": {"Temperature (K)": (5.18, 5.36), "Pressure (MPa)": (0.220, 0.260), "phase_code": (2, 2)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "transport_highT_highP_viscosity",
            "target": viscosity_col,
            "bounds": {"Temperature (K)": (240.0, 300.0), "Pressure (MPa)": (2.0, 3.0), "phase_code": (2, 2)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "transport_highT_highP_conductivity",
            "target": conductivity_col,
            "bounds": {"Temperature (K)": (240.0, 300.0), "Pressure (MPa)": (2.0, 3.0), "phase_code": (2, 2)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "critical_conductivity",
            "target": conductivity_col,
            "bounds": {"Temperature (K)": (5.18, 5.36), "Pressure (MPa)": (0.220, 0.270), "phase_code": (3, 3)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
        {
            "name": "near_critical_gas_viscosity",
            "target": viscosity_col,
            "bounds": {"Temperature (K)": (5.18, 5.36), "Pressure (MPa)": (0.220, 0.260), "phase_code": (2, 2)},
            "transform": "log",
            "model_type": "knn",
            "neighbors": 10,
        },
    ]
    experts = []
    routed = {}
    for spec in experts_spec:
        expert = train_expert(**spec, train_df=train_df, aug_df=aug_df, seed=args.seed, trees=args.trees)
        n = apply_expert(pred, test_df, expert, metadata["target_cols"])
        expert_meta = {k: v for k, v in expert.items() if k != "model"}
        expert_meta["n_routed_test"] = n
        experts.append(expert_meta)
        routed[expert["name"]] = n
        print(expert_meta)

    metrics = regression_metrics(y_true, pred, data.y_floor, metadata["target_cols"])
    base_metrics = regression_metrics(y_true, y_base, data.y_floor, metadata["target_cols"])
    worst = worst_point_table(
        x_raw[test_idx],
        y_true,
        pred,
        data.y_floor,
        metadata["input_cols"],
        metadata["target_cols"],
        n=500,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "hybrid_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"base": base_metrics, "hybrid": metrics, "experts": experts}, f, ensure_ascii=False, indent=2)
    worst.to_csv(out / "hybrid_worst_points.csv", index=False)
    pd.DataFrame(metrics["per_target"]).T.to_csv(out / "hybrid_per_target.csv")
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print("base", json.dumps(base_metrics["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
