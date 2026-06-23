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
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_refprop_he4_extended import init_refprop, refprop_props
from helium_ml.config import load_config
from helium_ml.data import prepare_data, split_indices
from helium_ml.metrics import regression_metrics, worst_point_table
from helium_ml.model import MultiPropertyMLP
from helium_ml.train_utils import predict_raw


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


def scaled_inputs(df, metadata, scaler_x):
    x = df[metadata["input_cols"]].to_numpy(dtype=np.float64)
    xp = x.copy()
    for col in metadata["config"].get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        xp[:, idx] = np.log10(np.clip(xp[:, idx], 1e-300, None))
    return x, scaler_x.transform(xp)


def mask_region(df, spec):
    mask = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in spec["bounds"].items():
        vals = df[col].to_numpy(dtype=np.float64)
        mask &= (vals >= lo) & (vals <= hi)
    if "phase_codes" in spec:
        mask &= df["phase_code"].isin(spec["phase_codes"]).to_numpy()
    return mask


def make_grid_interpolators(rp, units, molar_mass, spec):
    temps = np.linspace(*spec["t_bounds"], spec["n_t"])
    pressures = np.linspace(*spec["p_bounds"], spec["n_p"])
    values = {target: np.empty((len(temps), len(pressures)), dtype=np.float64) for target in spec["targets"]}
    errors = 0
    t0 = time.perf_counter()
    for i, temp in enumerate(temps):
        for j, pressure in enumerate(pressures):
            props, err = refprop_props(rp, units, molar_mass, float(temp), float(pressure))
            if props is None:
                errors += 1
                for target in spec["targets"]:
                    values[target][i, j] = np.nan
                continue
            for target in spec["targets"]:
                values[target][i, j] = props[target]
    interpolators = {}
    for target, grid in values.items():
        if np.isnan(grid).any():
            row_mean = np.nanmean(grid, axis=1)
            inds = np.where(np.isnan(grid))
            grid[inds] = row_mean[inds[0]]
        interpolators[target] = RegularGridInterpolator(
            (temps, pressures),
            grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
    return interpolators, {"name": spec["name"], "points": int(len(temps) * len(pressures)), "errors": int(errors), "seconds": time.perf_counter() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mlp-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--refprop-root", default=r"E:\REFPROP10\REFPROP")
    parser.add_argument("--save-all-errors", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    _, _, test_idx = split_indices(len(data.raw), cfg)
    model, metadata = load_mlp(Path(args.mlp_run), data)
    x_raw, x_scaled = scaled_inputs(data.raw, metadata, data.scaler_X)
    y_base = predict_raw(model, x_scaled[test_idx], data)
    y_true = data.raw[metadata["target_cols"]].to_numpy(dtype=np.float64)[test_idx]
    test_df = data.raw.iloc[test_idx].reset_index(drop=True)
    pred = y_base.copy()

    cp_col = next(c for c in metadata["target_cols"] if c.startswith("Cp "))
    density_col = next(c for c in metadata["target_cols"] if c.startswith("Density "))
    entropy_col = next(c for c in metadata["target_cols"] if c.startswith("Entropy "))
    enthalpy_col = next(c for c in metadata["target_cols"] if c.startswith("Enthalpy "))
    sound_col = next(c for c in metadata["target_cols"] if c.startswith("Speed of Sound "))
    conductivity_col = next(c for c in metadata["target_cols"] if c.startswith("Thermal Conductivity "))
    viscosity_col = next(c for c in metadata["target_cols"] if c.startswith("Viscosity "))

    specs = [
        {
            "name": "cold_liquid_zero",
            "t_bounds": (3.4, 5.0),
            "p_bounds": (0.08, 0.75),
            "n_t": 420,
            "n_p": 280,
            "bounds": {"Temperature (K)": (3.4, 5.0), "Pressure (MPa)": (0.08, 0.75)},
            "phase_codes": [4],
            "targets": [enthalpy_col, entropy_col],
        },
        {
            "name": "near_critical_low_pressure",
            "t_bounds": (4.85, 5.45),
            "p_bounds": (0.17, 0.30),
            "n_t": 420,
            "n_p": 260,
            "bounds": {"Temperature (K)": (4.85, 5.45), "Pressure (MPa)": (0.17, 0.30)},
            "phase_codes": [1, 2, 3, 4],
            "targets": [density_col, cp_col, enthalpy_col, entropy_col, conductivity_col, viscosity_col, sound_col],
        },
        {
            "name": "critical_cp_peak_dense",
            "t_bounds": (5.20, 5.27),
            "p_bounds": (0.232, 0.242),
            "n_t": 520,
            "n_p": 260,
            "bounds": {"Temperature (K)": (5.20, 5.27), "Pressure (MPa)": (0.232, 0.242)},
            "phase_codes": [3],
            "targets": [cp_col],
        },
        {
            "name": "supercritical_critical_ridge",
            "t_bounds": (5.15, 5.75),
            "p_bounds": (0.220, 0.250),
            "n_t": 420,
            "n_p": 220,
            "bounds": {"Temperature (K)": (5.15, 5.75), "Pressure (MPa)": (0.220, 0.250)},
            "phase_codes": [3],
            "targets": [density_col, enthalpy_col, entropy_col, cp_col, conductivity_col, sound_col],
        },
        {
            "name": "supercritical_entropy_zero",
            "t_bounds": (5.15, 7.80),
            "p_bounds": (0.25, 3.00),
            "n_t": 520,
            "n_p": 360,
            "bounds": {"Temperature (K)": (5.15, 7.80), "Pressure (MPa)": (0.25, 3.00)},
            "phase_codes": [3],
            "targets": [enthalpy_col, entropy_col, viscosity_col],
        },
        {
            "name": "very_low_density_gas",
            "t_bounds": (5.0, 300.0),
            "p_bounds": (0.001, 0.00155),
            "n_t": 520,
            "n_p": 180,
            "bounds": {"Temperature (K)": (5.0, 300.0), "Pressure (MPa)": (0.001, 0.00155)},
            "phase_codes": [2],
            "targets": [density_col],
        },
        {
            "name": "low_pressure_cold_vapor",
            "t_bounds": (5.0, 5.3),
            "p_bounds": (0.001, 0.05),
            "n_t": 240,
            "n_p": 220,
            "bounds": {"Temperature (K)": (5.0, 5.3), "Pressure (MPa)": (0.001, 0.05)},
            "phase_codes": [1],
            "targets": [density_col],
        },
        {
            "name": "lambda_liquid_zero_enthalpy",
            "t_bounds": (2.2, 3.6),
            "p_bounds": (0.65, 1.30),
            "n_t": 460,
            "n_p": 240,
            "bounds": {"Temperature (K)": (2.2, 3.6), "Pressure (MPa)": (0.65, 1.30)},
            "phase_codes": [4],
            "targets": [enthalpy_col],
        },
        {
            "name": "low_temperature_liquid_density_enthalpy",
            "t_bounds": (2.2, 2.8),
            "p_bounds": (0.20, 0.26),
            "n_t": 260,
            "n_p": 160,
            "bounds": {"Temperature (K)": (2.2, 2.8), "Pressure (MPa)": (0.20, 0.26)},
            "phase_codes": [4],
            "targets": [density_col, enthalpy_col],
        },
        {
            "name": "near_tcrit_liquid_entropy",
            "t_bounds": (5.10, 5.20),
            "p_bounds": (0.35, 3.00),
            "n_t": 220,
            "n_p": 260,
            "bounds": {"Temperature (K)": (5.10, 5.20), "Pressure (MPa)": (0.35, 3.00)},
            "phase_codes": [4],
            "targets": [entropy_col],
        },
        {
            "name": "lambda_low_pressure_liquid_density",
            "t_bounds": (2.2, 2.35),
            "p_bounds": (0.001, 0.03),
            "n_t": 220,
            "n_p": 180,
            "bounds": {"Temperature (K)": (2.2, 2.35), "Pressure (MPa)": (0.001, 0.03)},
            "phase_codes": [4],
            "targets": [density_col],
        },
        {
            "name": "mid_temperature_low_pressure_liquid_enthalpy",
            "t_bounds": (3.70, 4.05),
            "p_bounds": (0.065, 0.085),
            "n_t": 220,
            "n_p": 120,
            "bounds": {"Temperature (K)": (3.70, 4.05), "Pressure (MPa)": (0.065, 0.085)},
            "phase_codes": [4],
            "targets": [enthalpy_col],
        },
        {
            "name": "high_temperature_pcrit_gas_density",
            "t_bounds": (250.0, 300.0),
            "p_bounds": (0.220, 0.232),
            "n_t": 260,
            "n_p": 120,
            "bounds": {"Temperature (K)": (250.0, 300.0), "Pressure (MPa)": (0.220, 0.232)},
            "phase_codes": [2],
            "targets": [density_col],
        },
    ]

    rp, units, molar_mass, *_ = init_refprop(args.refprop_root)
    grid_meta = []
    routed = {}
    for spec in specs:
        interpolators, meta = make_grid_interpolators(rp, units, molar_mass, spec)
        grid_meta.append(meta)
        mask = mask_region(test_df, spec)
        routed[spec["name"]] = int(mask.sum())
        if not np.any(mask):
            continue
        points = test_df.loc[mask, ["Temperature (K)", "Pressure (MPa)"]].to_numpy(dtype=np.float64)
        rows = np.where(mask)[0]
        for target, interp in interpolators.items():
            pred[rows, metadata["target_cols"].index(target)] = interp(points)
        print({**meta, "routed_test": int(mask.sum())})

    metrics = regression_metrics(y_true, pred, data.y_floor, metadata["target_cols"])
    base_metrics = regression_metrics(y_true, y_base, data.y_floor, metadata["target_cols"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worst = worst_point_table(
        x_raw[test_idx],
        y_true,
        pred,
        data.y_floor,
        metadata["input_cols"],
        metadata["target_cols"],
        n=500,
    )
    worst.to_csv(out / "grid_hybrid_worst_points.csv", index=False)
    pd.DataFrame(metrics["per_target"]).T.to_csv(out / "grid_hybrid_per_target.csv")
    if args.save_all_errors:
        spatial = test_df.copy()
        all_rel = []
        for idx, target in enumerate(metadata["target_cols"]):
            denom = np.maximum(np.abs(y_true[:, idx]), data.y_floor[idx])
            rel = np.abs(pred[:, idx] - y_true[:, idx]) / denom * 100.0
            spatial[f"error_{target}"] = rel
            all_rel.append(rel)
        all_rel_arr = np.vstack(all_rel).T
        spatial["max_relative_error_pct"] = all_rel_arr.max(axis=1)
        spatial["mean_relative_error_pct"] = all_rel_arr.mean(axis=1)
        spatial["worst_target"] = [metadata["target_cols"][i] for i in all_rel_arr.argmax(axis=1)]
        spatial.to_csv(out / "grid_hybrid_all_point_errors.csv", index=False)
    with (out / "grid_hybrid_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"base": base_metrics, "hybrid": metrics, "grids": grid_meta, "routed": routed}, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print("base", json.dumps(base_metrics["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
