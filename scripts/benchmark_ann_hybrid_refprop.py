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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_refprop_ann import ann_predict, init_refprop, load_ann, refprop_predict, time_call
from evaluate_grid_hybrid import make_grid_interpolators, mask_region
from helium_ml.config import load_config


def grid_specs(target_cols):
    density_col = next(c for c in target_cols if c.startswith("Density "))
    entropy_col = next(c for c in target_cols if c.startswith("Entropy "))
    enthalpy_col = next(c for c in target_cols if c.startswith("Enthalpy "))
    sound_col = next(c for c in target_cols if c.startswith("Speed of Sound "))
    cp_col = next(c for c in target_cols if c.startswith("Cp "))
    viscosity_col = next(c for c in target_cols if c.startswith("Viscosity "))
    conductivity_col = next(c for c in target_cols if c.startswith("Thermal Conductivity "))
    return [
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


def build_grids(refprop_root, target_cols):
    rp, units, molar_mass = init_refprop(refprop_root)
    grids = []
    meta = []
    for spec in grid_specs(target_cols):
        interpolators, item = make_grid_interpolators(rp, units, molar_mass, spec)
        grids.append((spec, interpolators))
        meta.append(item)
    return rp, units, molar_mass, grids, meta


def apply_hybrid(ann_pred, df, grids, target_cols):
    pred = ann_pred.copy()
    routed = {}
    for spec, interpolators in grids:
        mask = mask_region(df, spec)
        routed[spec["name"]] = int(mask.sum())
        if not np.any(mask):
            continue
        points = df.loc[mask, ["Temperature (K)", "Pressure (MPa)"]].to_numpy(dtype=np.float64)
        rows = np.where(mask)[0]
        for target, interp in interpolators.items():
            pred[rows, target_cols.index(target)] = interp(points)
    return pred, routed


def median_time(fn, repeats=5):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--refprop-root", default=r"E:\REFPROP10\REFPROP")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000, 100000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, scaler_x, scaler_y, metadata = load_ann(Path(args.run))
    rp, units, molar_mass, grids, grid_meta = build_grids(args.refprop_root, metadata["target_cols"])

    df = pd.read_csv(cfg["data_path"], usecols=metadata["input_cols"] + metadata["target_cols"]).dropna()
    max_n = min(max(args.sizes), len(df))
    df = df.sample(n=max_n, random_state=args.seed).reset_index(drop=True)
    rows = []
    routed_rows = []
    for n in args.sizes:
        n = min(n, len(df))
        sub = df.iloc[:n].reset_index(drop=True)
        x = sub[metadata["input_cols"]].to_numpy(dtype=np.float64)
        ann_t, ann_pred = median_time(
            lambda: ann_predict(model, scaler_x, scaler_y, metadata, x),
            repeats=args.repeats,
        )
        hybrid_t, hybrid_result = median_time(
            lambda: apply_hybrid(ann_pred, sub, grids, metadata["target_cols"]),
            repeats=args.repeats,
        )
        ref_t, ref_result = time_call(refprop_predict, rp, units, molar_mass, x, repeats=1)
        _, ref_errors = ref_result
        hybrid_pred, routed = hybrid_result
        rows.append(
            {
                "n_points": n,
                "ann_time_s": ann_t,
                "hybrid_extra_time_s": hybrid_t,
                "hybrid_total_time_s": ann_t + hybrid_t,
                "refprop_time_s": ref_t,
                "ann_points_per_s": n / ann_t,
                "hybrid_points_per_s": n / (ann_t + hybrid_t),
                "refprop_points_per_s": n / ref_t,
                "ann_speedup_vs_refprop": ref_t / ann_t,
                "hybrid_speedup_vs_refprop": ref_t / (ann_t + hybrid_t),
                "hybrid_slowdown_vs_ann": (ann_t + hybrid_t) / ann_t,
                "routed_point_events": int(sum(routed.values())),
                "refprop_error_count": len(ref_errors),
            }
        )
        routed_rows.append({"n_points": n, **routed})
    speed = pd.DataFrame(rows)
    speed.to_csv(out / "ann_hybrid_refprop_speed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(routed_rows).to_csv(out / "hybrid_routed_counts.csv", index=False, encoding="utf-8-sig")
    with (out / "grid_build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(grid_meta, f, ensure_ascii=False, indent=2)
    print(speed.to_string(index=False))
    print(f"Saved benchmark outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
