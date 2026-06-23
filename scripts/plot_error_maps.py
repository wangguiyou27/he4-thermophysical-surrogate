from __future__ import annotations

import argparse
import sys
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
from helium_ml.data import load_clean_frame
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


def predict(model, scaler_x, scaler_y, metadata, x_raw, batch_size):
    cfg = metadata["config"]
    x_proc = x_raw.astype(np.float64).copy()
    for col in cfg.get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        x_proc[:, idx] = np.log10(np.clip(x_proc[:, idx], 1e-300, None))
    x_scaled = scaler_x.transform(x_proc)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            xb = torch.as_tensor(x_scaled[start : start + batch_size], dtype=torch.float32)
            chunks.append(model(xb).cpu().numpy())
    y_proc = scaler_y.inverse_transform(np.vstack(chunks))
    y_raw = y_proc.copy()
    for idx in metadata["log_target_indices"]:
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx])
    for idx, offset in metadata["shifted_log_target_offsets"].items():
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx]) - offset
    return y_raw


def relative_error_percent(y_pred, y_true):
    floors = np.quantile(np.abs(y_true), 0.01, axis=0)
    floors = np.maximum(floors, 1e-12)
    denom = np.maximum(np.abs(y_true), floors.reshape(1, -1))
    return np.abs(y_pred - y_true) / denom * 100.0


def binned_stat(x, y, value, x_edges, y_edges):
    stat = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
    ix = np.digitize(x, x_edges) - 1
    iy = np.digitize(y, y_edges) - 1
    for row in range(stat.shape[0]):
        for col in range(stat.shape[1]):
            mask = (ix == col) & (iy == row)
            if np.any(mask):
                stat[row, col] = np.mean(value[mask])
    return stat


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("^", "")
        .replace("·", "")
    )


def plot_overall_heatmap(temp, pressure, err, out_dir, x_edges, y_edges):
    stat = binned_stat(temp, pressure, np.mean(err, axis=1), x_edges, y_edges)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    mesh = ax.pcolormesh(x_edges, y_edges, stat, shading="auto", cmap="viridis")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title("Mean absolute percentage error over T-P plane")
    fig.colorbar(mesh, ax=ax, label="MAPE (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_error_heatmap.png", dpi=240)
    plt.close(fig)


def plot_property_heatmaps(temp, pressure, err, cols, out_dir, x_edges, y_edges):
    for j, col in enumerate(cols):
        stat = binned_stat(temp, pressure, err[:, j], x_edges, y_edges)
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        mesh = ax.pcolormesh(x_edges, y_edges, stat, shading="auto", cmap="magma")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (MPa)")
        ax.set_title(f"{col}: binned absolute percentage error")
        fig.colorbar(mesh, ax=ax, label="APE (%)")
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{safe_name(col)}.png", dpi=220)
        plt.close(fig)


def plot_worst_points(temp, pressure, err, cols, out_dir, top_n):
    flat = err.reshape(-1)
    order = np.argsort(flat)[::-1][:top_n]
    row_idx = order // err.shape[1]
    prop_idx = order % err.shape[1]
    worst = pd.DataFrame(
        {
            "Temperature (K)": temp[row_idx],
            "Pressure (MPa)": pressure[row_idx],
            "property": [cols[i] for i in prop_idx],
            "absolute_percentage_error": flat[order],
        }
    )
    worst.to_csv(out_dir / "worst_points_spatial.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(
        worst["Temperature (K)"],
        worst["Pressure (MPa)"],
        c=worst["absolute_percentage_error"],
        s=28,
        cmap="plasma",
        edgecolors="black",
        linewidths=0.25,
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title(f"Top {top_n} worst prediction points")
    fig.colorbar(sc, ax=ax, label="APE (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "worst_points_tp_scatter.png", dpi=240)
    plt.close(fig)


def plot_region_box(temp, err, regions, out_dir):
    rows = []
    mean_err = np.mean(err, axis=1)
    for name, bounds in regions.items():
        lo, hi = bounds
        mask = (temp >= lo) & (temp < hi)
        for val in mean_err[mask]:
            rows.append({"region": name, "mean_ape_pct": val})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = list(regions.keys())
    data = [df[df["region"] == label]["mean_ape_pct"].to_numpy() for label in labels]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Mean APE across properties (%)")
    ax.set_title("Error distribution by temperature region")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "region_error_boxplot.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/he4_single_phase.json")
    parser.add_argument("--run", default="runs/he4_supervised_shifted_entropy")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample", type=int, default=120000)
    parser.add_argument("--bins-t", type=int, default=45)
    parser.add_argument("--bins-p", type=int, default=36)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir / "error_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_frame(cfg)
    if args.sample > 0 and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    x_raw = df[cfg["input_cols"]].to_numpy(dtype=np.float64)
    y_true = df[cfg["target_cols"]].to_numpy(dtype=np.float64)
    model, scaler_x, scaler_y, metadata = load_ann(run_dir)
    y_pred = predict(model, scaler_x, scaler_y, metadata, x_raw, args.batch_size)
    err = relative_error_percent(y_pred, y_true)

    temp = df["Temperature (K)"].to_numpy(dtype=np.float64)
    pressure = df["Pressure (MPa)"].to_numpy(dtype=np.float64)
    t_edges = np.linspace(temp.min(), temp.max(), args.bins_t + 1)
    p_edges = np.linspace(pressure.min(), pressure.max(), args.bins_p + 1)

    plot_overall_heatmap(temp, pressure, err, out_dir, t_edges, p_edges)
    plot_property_heatmaps(temp, pressure, err, cfg["target_cols"], out_dir, t_edges, p_edges)
    plot_worst_points(temp, pressure, err, cfg["target_cols"], out_dir, args.top_n)
    plot_region_box(temp, err, cfg.get("temperature_regions", {}), out_dir)

    summary = []
    for j, col in enumerate(cfg["target_cols"]):
        summary.append(
            {
                "property": col,
                "mape_pct": float(np.mean(err[:, j])),
                "p95_abs_pct": float(np.percentile(err[:, j], 95)),
                "p99_abs_pct": float(np.percentile(err[:, j], 99)),
                "max_abs_pct": float(np.max(err[:, j])),
            }
        )
    pd.DataFrame(summary).to_csv(out_dir / "error_map_metric_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved error maps to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
