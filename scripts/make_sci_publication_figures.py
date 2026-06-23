from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "paper_outputs" / "he4_2p2_hybrid_paper"
TABLES = PKG / "tables"
OUT = PKG / "figures_sci"


COLORS = {
    "blue": "#2f5f8f",
    "orange": "#c47c2c",
    "green": "#4f8a5b",
    "red": "#b44a4a",
    "purple": "#6f5a9a",
    "gray": "#4b5563",
    "light_gray": "#e5e7eb",
}


PHASE_COLORS = {
    "Single-phase gas": "#4b78a8",
    "Supercritical": "#7b6aa8",
    "Subcooled liquid": "#4f8a5b",
    "Superheated vapor": "#d08b3e",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.55)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def make_dataset_phase_map() -> None:
    data_path = ROOT / "data" / "he4data_single_phase_extended_2p2_critical_features.csv"
    cols = ["Temperature (K)", "Pressure (MPa)", "phase_region"]
    data = pd.read_csv(data_path, usecols=cols)
    if len(data) > 90000:
        data = data.sample(90000, random_state=12)
    labels = {
        "single_phase_gas": "Single-phase gas",
        "supercritical": "Supercritical",
        "subcooled_liquid": "Subcooled liquid",
        "superheated_vapor": "Superheated vapor",
    }
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    for key, label in labels.items():
        part = data[data["phase_region"] == key]
        ax.scatter(
            part["Temperature (K)"],
            part["Pressure (MPa)"],
            s=2.0,
            alpha=0.28,
            linewidths=0,
            color=PHASE_COLORS[label],
            label=label,
            rasterized=True,
        )
    ax.set_xlim(2.2, 300)
    ax.set_ylim(0, 3.02)
    ax.set_xlabel("Temperature, T (K)")
    ax.set_ylabel("Pressure, P (MPa)")
    ax.legend(frameon=False, loc="upper right", markerscale=3.2, handletextpad=0.2)
    clean_axes(ax)
    save(fig, "fig1_dataset_phase_map_sci")


def make_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 1.85))
    ax.axis("off")
    labels = [
        "REFPROP\nsampling",
        "Single-phase\nfiltering",
        "Feature\nengineering",
        "Critical-feature\nANN",
        "Local hybrid\ncorrection",
        "Eight-property\nprediction",
    ]
    xs = np.linspace(0.06, 0.86, len(labels))
    w, h, y = 0.125, 0.36, 0.43
    fills = ["#eef5ff", "#eef8f0", "#fff6e8", "#f5f1ff", "#fff0f0", "#eef8f8"]
    edges = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"], "#407878"]
    for i, (x, label) in enumerate(zip(xs, labels)):
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, fc=fills[i], ec=edges[i], lw=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes, ha="center", va="center", fontsize=8)
        if i < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.01, y + h / 2),
                xytext=(x + w + 0.01, y + h / 2),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#374151"),
            )
    ax.text(
        0.5,
        0.16,
        "Inputs: T, P, phase code and critical features; outputs: density, entropy, enthalpy, sound speed, Cv, Cp, viscosity and thermal conductivity.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        color="#374151",
    )
    save(fig, "fig2_model_workflow_sci")


def make_property_metrics() -> None:
    df = pd.read_csv(TABLES / "table_3_hybrid_per_property_metrics.csv")
    order = ["Density", "Entropy", "Enthalpy", "Sound speed", "Cv", "Cp", "Viscosity", "Therm. cond."]
    df["property"] = pd.Categorical(df["property"], order, ordered=True)
    df = df.sort_values("property")
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    specs = [
        ("mape", "MAPE (%)", COLORS["blue"], 0.36),
        ("p99", "P99 error (%)", COLORS["orange"], 1.65),
        ("max", "Maximum error (%)", COLORS["red"], 5.0),
    ]
    for ax, (col, title, color, xmax) in zip(axes, specs):
        ax.barh(y, df[col], height=0.62, color=color, alpha=0.9)
        ax.set_xlabel(title)
        ax.set_xlim(0, xmax)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["property"])
    axes[0].invert_yaxis()
    save(fig, "fig3_per_property_errors_sci")

    fig, ax1 = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(df))
    ax1.bar(x - 0.18, df["within_1pct"], width=0.36, color=COLORS["blue"], label="Within 1%")
    ax1.bar(x + 0.18, df["within_5pct"], width=0.36, color=COLORS["green"], label="Within 5%")
    ax1.set_ylim(95, 100.4)
    ax1.set_ylabel("Fraction of predictions (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["property"], rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, df["r2"], color=COLORS["red"], marker="o", ms=3, lw=1.1, label="$R^2$")
    ax2.set_ylim(max(0.99995, df["r2"].min() - 0.00001), 1.000003)
    ax2.set_ylabel("$R^2$")
    clean_axes(ax1)
    ax2.spines["top"].set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="lower left", ncol=1)
    save(fig, "fig3_r2_thresholds_sci")


def heatmap_grid(df: pd.DataFrame, value_col: str, t_bins: np.ndarray, p_bins: np.ndarray) -> np.ndarray:
    tmp = df[["Temperature (K)", "Pressure (MPa)", value_col]].dropna().copy()
    tmp["t_bin"] = pd.cut(tmp["Temperature (K)"], t_bins, include_lowest=True)
    tmp["p_bin"] = pd.cut(tmp["Pressure (MPa)"], p_bins, include_lowest=True)
    grid = tmp.pivot_table(index="p_bin", columns="t_bin", values=value_col, aggfunc="mean", observed=False)
    return grid.to_numpy(dtype=float)


def make_error_heatmaps() -> None:
    df = pd.read_csv(TABLES / "table_3_hybrid_all_point_errors.csv")
    t_bins = np.linspace(2.2, 300, 86)
    p_bins = np.linspace(0, 3.0, 70)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    specs = [
        ("max_relative_error_pct", "Maximum error over outputs", 2.0),
        ("mean_relative_error_pct", "Mean error over outputs", 0.45),
    ]
    for ax, (col, title, vmax) in zip(axes, specs):
        arr = heatmap_grid(df, col, t_bins, p_bins)
        im = ax.pcolormesh(t_bins, p_bins, arr, shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel("Temperature, T (K)")
        ax.set_ylabel("Pressure, P (MPa)")
        ax.set_title(title)
        clean_axes(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.02)
        cb.set_label("Mean bin error (%)")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save(fig, "fig4_hybrid_error_heatmaps_sci")


def make_hybrid_comparison() -> None:
    overall = pd.read_csv(TABLES / "table_2_overall_ann_vs_hybrid.csv")
    per = pd.read_csv(TABLES / "table_3_ann_vs_hybrid_per_property.csv")
    order = ["Density", "Entropy", "Enthalpy", "Sound speed", "Cv", "Cp", "Viscosity", "Therm. cond."]
    pivot = per.pivot(index="property", columns="model", values="max").loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(order))
    width = 0.36
    axes[0].bar(x - width / 2, pivot["Critical-feature ANN"], width, color="#8aa6c8", label="ANN")
    axes[0].bar(x + width / 2, pivot["Local hybrid"], width, color=COLORS["red"], label="Hybrid")
    axes[0].axhline(5, color="#111827", lw=0.8, ls="--", label="5% target")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Maximum error (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=38, ha="right")
    axes[0].legend(frameon=False, loc="upper right")
    clean_axes(axes[0])

    metrics = ["mape", "p99", "max"]
    labels = ["MAPE", "P99", "Max"]
    x2 = np.arange(len(metrics))
    ann = overall[overall["model"] == "Critical-feature ANN"].iloc[0]
    hyb = overall[overall["model"] == "Local hybrid"].iloc[0]
    axes[1].bar(x2 - 0.18, [ann[m] for m in metrics], 0.36, color="#8aa6c8", label="ANN")
    axes[1].bar(x2 + 0.18, [hyb[m] for m in metrics], 0.36, color=COLORS["red"], label="Hybrid")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Relative error (%)")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, loc="upper left")
    clean_axes(axes[1])
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save(fig, "fig5_ann_hybrid_comparison_sci")


def make_baseline_comparison() -> None:
    df = pd.read_csv(TABLES / "table_4_baseline_summary.csv")
    final = pd.DataFrame(
        [
            {
                "model": "Local hybrid",
                "mape": 0.13065211874252647,
                "p99": 0.9414324568953982,
                "max": 4.651398824088986,
                "points_per_s": 237308.61653348134,
            }
        ]
    )
    df = pd.concat([df[["model", "mape", "p99", "max", "points_per_s"]], final], ignore_index=True)
    order = ["Local hybrid", "Current MLP", "Random Forest", "Extra Trees", "Gaussian RBF Ridge", "XGBoost"]
    df["model"] = pd.Categorical(df["model"], order, ordered=True)
    df = df.sort_values("model")
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.05), sharey=True)
    specs = [("mape", "MAPE (%)", COLORS["blue"]), ("max", "Maximum error (%)", COLORS["red"]), ("points_per_s", "Throughput (points s$^{-1}$)", COLORS["green"])]
    for ax, (col, title, color) in zip(axes, specs):
        ax.barh(y, df[col], color=color, alpha=0.88)
        ax.set_xlabel(title)
        if col == "points_per_s":
            ax.set_xscale("log")
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["model"])
    axes[0].invert_yaxis()
    save(fig, "fig6_baseline_comparison_sci")


def make_speed_benchmark() -> None:
    df = pd.read_csv(TABLES / "table_5_speed_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    axes[0].plot(df["n_points"], df["ann_points_per_s"], marker="o", ms=3, lw=1.2, color=COLORS["blue"], label="ANN")
    axes[0].plot(df["n_points"], df["hybrid_points_per_s"], marker="s", ms=3, lw=1.2, color=COLORS["red"], label="Hybrid")
    axes[0].plot(df["n_points"], df["refprop_points_per_s"], marker="^", ms=3, lw=1.2, color=COLORS["gray"], label="REFPROP")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of state points")
    axes[0].set_ylabel("Throughput (points s$^{-1}$)")
    axes[0].legend(frameon=False)
    clean_axes(axes[0])

    axes[1].plot(df["n_points"], df["ann_speedup_vs_refprop"], marker="o", ms=3, lw=1.2, color=COLORS["blue"], label="ANN")
    axes[1].plot(df["n_points"], df["hybrid_speedup_vs_refprop"], marker="s", ms=3, lw=1.2, color=COLORS["red"], label="Hybrid")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of state points")
    axes[1].set_ylabel("Speedup over REFPROP")
    axes[1].legend(frameon=False)
    clean_axes(axes[1])
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save(fig, "fig7_speed_benchmark_sci")


def write_index() -> None:
    files = sorted(OUT.glob("*.svg"))
    lines = [
        "# SCI-style figures",
        "",
        "All figures are exported as both high-resolution PNG and editable SVG.",
        "",
    ]
    for path in files:
        lines.append(f"- `{path.name}`")
    (OUT / "README_figures_sci.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    make_dataset_phase_map()
    make_workflow()
    make_property_metrics()
    make_error_heatmaps()
    make_hybrid_comparison()
    make_baseline_comparison()
    make_speed_benchmark()
    write_index()
    print(f"Saved SCI-style figures to {OUT}")


if __name__ == "__main__":
    main()
