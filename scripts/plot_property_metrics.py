from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def short_name(name: str) -> str:
    mapping = {
        "Density (kg/m^3)": "Density",
        "Entropy (kJ/(kg路K))": "Entropy",
        "Entropy (kJ/(kg·K))": "Entropy",
        "Enthalpy (kJ/kg)": "Enthalpy",
        "Speed of Sound (m/s)": "Sound speed",
        "Cv (kJ/(kg路K))": "Cv",
        "Cv (kJ/(kg·K))": "Cv",
        "Cp (kJ/(kg路K))": "Cp",
        "Cp (kJ/(kg·K))": "Cp",
        "Viscosity (渭Pa路s)": "Viscosity",
        "Viscosity (μPa·s)": "Viscosity",
        "Thermal Conductivity (mW/(m路K))": "Therm. cond.",
        "Thermal Conductivity (mW/(m·K))": "Therm. cond.",
    }
    return mapping.get(name, name)


def load_metrics(run_dir: Path) -> pd.DataFrame:
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = []
    for target, values in metrics["per_target"].items():
        row = {"property": short_name(target), "full_property": target}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def bar_labels(ax, bars, fmt="{:.2f}") -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            " " + fmt.format(width),
            va="center",
            ha="left",
            fontsize=8,
        )


def make_property_error_figure(df: pd.DataFrame, out_path: Path) -> None:
    df_plot = df.copy()
    y = np.arange(len(df_plot))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    specs = [
        ("mape", "MAPE (%)", "#2f6fbb"),
        ("p99", "P99 error (%)", "#d9822b"),
        ("max", "Maximum error (%)", "#b33a3a"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        bars = ax.barh(y, df_plot[col], color=color, alpha=0.88)
        ax.set_title(title)
        ax.set_xlabel("Relative error (%)")
        ax.grid(axis="x", alpha=0.25)
        bar_labels(ax, bars)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_plot["property"])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_accuracy_figure(df: pd.DataFrame, out_path: Path) -> None:
    df_plot = df.copy()
    x = np.arange(len(df_plot))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.bar(x - width / 2, df_plot["within_1pct"], width, label="< 1%", color="#4c78a8")
    ax1.bar(x + width / 2, df_plot["within_5pct"], width, label="< 5%", color="#54a24b")
    ax1.set_ylabel("Prediction fraction (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot["property"], rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.plot(x, df_plot["r2"], color="#8f3f97", marker="o", linewidth=2, label="R2")
    ax2.set_ylabel("R2")
    r2_min = max(0.995, float(df_plot["r2"].min()) - 0.0005)
    ax2.set_ylim(r2_min, 1.00005)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory containing metrics.json")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to the run directory.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(run_dir)
    df.to_csv(out_dir / "property_metrics_table.csv", index=False)
    make_property_error_figure(df, out_dir / "property_error_bars.png")
    make_accuracy_figure(df, out_dir / "property_accuracy_r2.png")
    print(df[["property", "mape", "p95", "p99", "max", "r2", "within_5pct"]].to_string(index=False))


if __name__ == "__main__":
    main()

