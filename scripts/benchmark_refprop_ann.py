from __future__ import annotations

import argparse
import json
import os
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.model import MultiPropertyMLP


def load_ann(run_dir: Path):
    metadata = joblib.load(run_dir / "metadata.joblib")
    scaler_x = joblib.load(run_dir / "scaler_X.joblib")
    scaler_y = joblib.load(run_dir / "scaler_y.joblib")
    cfg = metadata["config"]
    model = MultiPropertyMLP(
        input_dim=len(metadata["input_cols"]),
        output_dim=len(metadata["target_cols"]),
        hidden_layers=list(cfg["hidden_layers"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, scaler_x, scaler_y, metadata


def ann_predict(model, scaler_x, scaler_y, metadata, x_raw: np.ndarray, batch_size: int = 65536):
    cfg = metadata["config"]
    x_proc = x_raw.astype(np.float64).copy()
    for col in cfg.get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        x_proc[:, idx] = np.log10(np.clip(x_proc[:, idx], 1e-300, None))
    x_scaled = scaler_x.transform(x_proc)

    preds = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            xb = torch.as_tensor(x_scaled[start : start + batch_size], dtype=torch.float32)
            preds.append(model(xb).cpu().numpy())
    y_scaled = np.vstack(preds)
    y_proc = scaler_y.inverse_transform(y_scaled)
    y_raw = y_proc.copy()
    for idx in metadata["log_target_indices"]:
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx])
    for idx, offset in metadata["shifted_log_target_offsets"].items():
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx]) - offset
    return y_raw


def init_refprop(refprop_root: str):
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

    if not Path(refprop_root).exists():
        candidates = [
            os.environ.get("RPPREFIX"),
            os.environ.get("REFPROP_PATH"),
            r"E:\Personal document\Refprop",
            r"E:\Personal document\REFPROP10\REFPROP",
            r"C:\Program Files (x86)\REFPROP",
            r"C:\Program Files\REFPROP",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                refprop_root = candidate
                break
    os.environ["RPPREFIX"] = refprop_root
    if not any(Path(refprop_root).glob("*.dll")):
        raise FileNotFoundError(
            f"REFPROP shared library was not found in {refprop_root}. "
            "Please install REFPROP or pass --refprop-root to the folder containing REFPRP64.dll."
        )
    rp = REFPROPFunctionLibrary(refprop_root)
    rp.SETPATHdll(refprop_root)
    units = rp.GETENUMdll(0, "MOLAR BASE SI").iEnum
    molar_mass = rp.REFPROPdll("HELIUM", "", "M", units, 0, 0, 0, 0, [1.0]).Output[0]
    return rp, units, molar_mass


def refprop_predict(rp, units, molar_mass, x_raw: np.ndarray):
    prop_string = "D;S;H;W;CV;CP;ETA;TCX"
    comp = [1.0]
    out = np.full((len(x_raw), 8), np.nan, dtype=np.float64)
    errors = []
    for i, row in enumerate(x_raw):
        temp_k = row[0]
        pressure_mpa = row[1]
        pressure_pa = float(pressure_mpa) * 1e6
        res = rp.REFPROPdll("HELIUM", "TP", prop_string, units, 0, 0, float(temp_k), pressure_pa, comp)
        if res.ierr != 0:
            errors.append({"i": i, "T": float(temp_k), "P_MPa": float(pressure_mpa), "error": res.herr})
            continue
        vals = res.Output
        out[i, 0] = vals[0] * molar_mass
        out[i, 1] = vals[1] / molar_mass / 1e3
        out[i, 2] = vals[2] / molar_mass / 1e3
        out[i, 3] = vals[3]
        out[i, 4] = vals[4] / molar_mass / 1e3
        out[i, 5] = vals[5] / molar_mass / 1e3
        out[i, 6] = vals[6] * 1e6
        out[i, 7] = vals[7] * 1e3
    return out, errors


def time_call(fn, *args, repeats: int = 3):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def metric_summary(pred: np.ndarray, ref: np.ndarray, cols: list[str]):
    rows = []
    valid_rows = np.isfinite(ref).all(axis=1) & np.isfinite(pred).all(axis=1)
    pred = pred[valid_rows]
    ref = ref[valid_rows]
    for j, col in enumerate(cols):
        err = np.abs(pred[:, j] - ref[:, j])
        denom = np.maximum(np.abs(ref[:, j]), np.quantile(np.abs(ref[:, j]), 0.01))
        ape = err / np.maximum(denom, 1e-12) * 100.0
        ss_res = float(np.sum((pred[:, j] - ref[:, j]) ** 2))
        ss_tot = float(np.sum((ref[:, j] - np.mean(ref[:, j])) ** 2))
        rows.append(
            {
                "property": col,
                "mape_pct": float(np.mean(ape)),
                "p95_abs_pct": float(np.percentile(ape, 95)),
                "p99_abs_pct": float(np.percentile(ape, 99)),
                "max_abs_pct": float(np.max(ape)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            }
        )
    return rows


def plot_speed(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["n_points"], df["refprop_points_per_s"], marker="o", label="REFPROP")
    ax.plot(df["n_points"], df["ann_points_per_s"], marker="o", label="ANN")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of state points")
    ax.set_ylabel("Predictions per second")
    ax.set_title("REFPROP vs ANN inference throughput")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "refprop_ann_speed.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["n_points"], df["speedup_ann_vs_refprop"], marker="o", color="#4C78A8")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Number of state points")
    ax.set_ylabel("Speedup")
    ax.set_title("ANN speedup over REFPROP")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ann_speedup_over_refprop.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/he4_single_phase.json")
    parser.add_argument("--run", default="runs/he4_supervised_shifted_entropy")
    parser.add_argument("--out", default="runs/refprop_ann_benchmark")
    parser.add_argument("--refprop-root", default=r"E:\Personal document\REFPROP10\REFPROP")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 5000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["data_path"], usecols=cfg["input_cols"] + cfg["target_cols"]).dropna()
    df = df.sample(n=min(max(args.sizes), len(df)), random_state=args.seed).reset_index(drop=True)
    x_all = df[cfg["input_cols"]].to_numpy(dtype=np.float64)

    model, scaler_x, scaler_y, metadata = load_ann(run_dir)
    try:
        rp, units, molar_mass = init_refprop(args.refprop_root)
    except Exception as exc:
        message = {
            "status": "refprop_unavailable",
            "reason": str(exc),
            "next_step": "Install REFPROP or pass --refprop-root to the directory containing REFPRP64.dll, then rerun this script.",
        }
        with open(out_dir / "refprop_unavailable.json", "w", encoding="utf-8") as f:
            json.dump(message, f, ensure_ascii=False, indent=2)
        print(json.dumps(message, ensure_ascii=False, indent=2))
        return

    speed_rows = []
    largest_pred = None
    largest_ref = None
    largest_errors = None
    for n in args.sizes:
        x = x_all[:n]
        ann_time, ann_pred = time_call(
            ann_predict, model, scaler_x, scaler_y, metadata, x, repeats=args.repeats
        )
        ref_time, ref_result = time_call(refprop_predict, rp, units, molar_mass, x, repeats=1)
        ref_pred, errors = ref_result
        speed_rows.append(
            {
                "n_points": n,
                "ann_time_s": ann_time,
                "refprop_time_s": ref_time,
                "ann_points_per_s": n / ann_time,
                "refprop_points_per_s": n / ref_time,
                "speedup_ann_vs_refprop": ref_time / ann_time,
                "refprop_error_count": len(errors),
            }
        )
        largest_pred = ann_pred
        largest_ref = ref_pred
        largest_errors = errors

    speed_df = pd.DataFrame(speed_rows)
    speed_df.to_csv(out_dir / "refprop_ann_speed.csv", index=False, encoding="utf-8-sig")
    plot_speed(speed_df, out_dir)

    prop_rows = metric_summary(largest_pred, largest_ref, metadata["target_cols"])
    prop_df = pd.DataFrame(prop_rows)
    prop_df.to_csv(out_dir / "ann_vs_refprop_property_metrics.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "refprop_errors.json", "w", encoding="utf-8") as f:
        json.dump(largest_errors, f, ensure_ascii=False, indent=2)

    print(speed_df.to_string(index=False))
    print()
    print(prop_df.to_string(index=False))
    print(f"\nSaved benchmark outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
