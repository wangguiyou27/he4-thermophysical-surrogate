from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PKG = ROOT / "paper_outputs" / "he4_2p2_hybrid_paper"
TABLES = PKG / "tables"
OUT = PKG / "poster_figures"

BLUE = "#1f4e79"
BLUE2 = "#5b8db8"
ORANGE = "#d9892b"
RED = "#b84a4a"
GREEN = "#4c8b5a"
GRAY = "#4b5563"
LIGHT_GRAY = "#d6dce5"
PURPLE = "#7463a8"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
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


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig1_dataset_phase_map() -> None:
    data_path = ROOT / "data" / "he4data_single_phase_extended_2p2_critical_features.csv"
    data = pd.read_csv(data_path, usecols=["Temperature (K)", "Pressure (MPa)", "phase_region"])

    labels = {
        "single_phase_gas": ("Single-phase gas", "#2f6da3"),
        "superheated_vapor": ("Superheated vapor", "#d97706"),
        "subcooled_liquid": ("Subcooled liquid", "#238b45"),
        "supercritical": ("Supercritical", "#6d4aa2"),
    }

    # Balanced sampling keeps the plot readable on a poster and prevents the
    # dominant gas region from visually burying the other phase regions.
    shown = []
    for phase in labels:
        part = data[data["phase_region"] == phase]
        shown.append(part.sample(min(len(part), 7000), random_state=42))
    plot_data = pd.concat(shown, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.7))
    for phase, (label, color) in labels.items():
        part = plot_data[plot_data["phase_region"] == phase]
        ax.scatter(
            part["Temperature (K)"],
            part["Pressure (MPa)"],
            s=18,
            alpha=0.78,
            color=color,
            edgecolors="white",
            linewidths=0.18,
            label=label,
            rasterized=True,
        )

    ax.set_title("Dataset coverage and phase distribution", pad=14, fontweight="bold", fontsize=27)
    ax.set_xlabel("Temperature / K", fontsize=23)
    ax.set_ylabel("Pressure / MPa", fontsize=23)
    ax.set_xlim(2.2, 300)
    ax.set_ylim(0, 3.02)
    clean_axes(ax, "both")
    ax.tick_params(axis="both", labelsize=18)
    legend = ax.legend(
        frameon=True,
        loc="upper right",
        markerscale=1.9,
        fontsize=16.5,
        borderpad=0.5,
        labelspacing=0.45,
        handletextpad=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cbd5e1")
    legend.get_frame().set_alpha(0.92)

    # Direct labels help viewers identify the four regions without relying only
    # on the legend after the figure is embedded in a large poster.
    region_labels = [
        (215, 0.36, "Single-phase\ngas", labels["single_phase_gas"][1]),
        (36, 0.055, "Superheated\nvapor", labels["superheated_vapor"][1]),
        (32, 1.75, "Subcooled\nliquid", labels["subcooled_liquid"][1]),
        (116, 2.48, "Supercritical", labels["supercritical"][1]),
    ]
    for x, y, text, color in region_labels:
        ax.text(
            x,
            y,
            text,
            fontsize=16.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.0, alpha=0.88),
        )

    ax.text(
        0.02,
        0.03,
        "Balanced display sample from 358,588 REFPROP-based single-phase helium-4 states",
        transform=ax.transAxes,
        fontsize=15.5,
        color=GRAY,
    )
    save_all(fig, "poster_fig1_dataset_phase_map")


def box(ax, x, y, w, h, text, fc="#eef5ff", ec=BLUE, fs=15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=fs)


def make_fig2_workflow() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 8.0))
    ax.axis("off")
    ax.set_title("Reliability-aware surrogate modeling workflow", pad=16, fontweight="bold")

    steps = [
        ("REFPROP\ndata generation", "#eef5ff", BLUE),
        ("Single-phase\nfiltering", "#eef8f1", GREEN),
        ("Phase-aware and\ncritical-feature\nconstruction", "#fff7e6", ORANGE),
        ("Critical-feature\nANN prediction", "#f4f0ff", PURPLE),
        ("Local hybrid\ncorrection", "#fff0f0", RED),
        ("Eight-property\noutputs", "#eef8f8", "#397878"),
    ]
    x, w, h = 0.18, 0.64, 0.105
    ys = np.linspace(0.80, 0.16, len(steps))
    for i, (label, fc, ec) in enumerate(steps):
        box(ax, x, ys[i], w, h, label, fc=fc, ec=ec, fs=16)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.50, ys[i + 1] + h + 0.01),
                xytext=(0.50, ys[i] - 0.012),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"),
            )

    ax.text(
        0.50,
        0.055,
        "Phase information and critical features are introduced before ANN prediction;\nlocal hybrid correction suppresses high-error tail points.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color=GRAY,
    )
    save_all(fig, "poster_fig2_model_workflow")


def make_fig3_tail_error_reduction() -> None:
    labels = ["Critical-feature\nANN", "Local\nhybrid"]
    values = [46.10, 4.6514]
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    bars = ax.bar(labels, values, color=[BLUE2, ORANGE], width=0.55)
    ax.set_title("Tail-error reduction by local hybrid correction", pad=12, fontweight="bold")
    ax.set_ylabel("Maximum relative error / %")
    ax.set_ylim(0, 52)
    clean_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.2f}%", ha="center", va="bottom", fontsize=20, fontweight="bold")
    ax.annotate(
        "46.10% → 4.65%",
        xy=(1, values[1] + 1.0),
        xytext=(0.50, 35),
        textcoords="data",
        ha="center",
        fontsize=19,
        fontweight="bold",
        color=RED,
        arrowprops=dict(arrowstyle="->", lw=2.0, color=RED),
    )
    ax.text(0.5, 0.05, "Worst-case error reduced below the 5% target.", transform=ax.transAxes, ha="center", fontsize=15, color=GRAY)
    save_all(fig, "poster_fig3_tail_error_reduction")


def make_fig4_metric_cards() -> None:
    metrics = [
        ("0.1307%", "MAPE", "#eef5ff", BLUE),
        ("0.9414%", "P99 error", "#fff7e6", ORANGE),
        ("4.6514%", "Max error", "#fff0f0", RED),
        ("100%", "Within 5%", "#eef8f1", GREEN),
        ("17.45×", "Speedup vs REFPROP", "#f4f0ff", PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.axis("off")
    ax.set_title("Hybrid model performance", pad=14, fontweight="bold")
    positions = [(0.06, 0.58), (0.53, 0.58), (0.06, 0.31), (0.53, 0.31), (0.295, 0.04)]
    for (value, label, fc, ec), (x, y) in zip(metrics, positions):
        w = 0.41 if label != "Speedup vs REFPROP" else 0.41
        h = 0.20
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            transform=ax.transAxes,
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.125, value, transform=ax.transAxes, ha="center", va="center", fontsize=27, fontweight="bold", color="#0f172a")
        ax.text(x + w / 2, y + 0.052, label, transform=ax.transAxes, ha="center", va="center", fontsize=16, color=GRAY)
    save_all(fig, "poster_fig4_metric_cards")


def make_fig5_per_property_max_error() -> None:
    df = pd.read_csv(TABLES / "table_3_hybrid_per_property_metrics.csv")
    df = df[["property", "max"]].copy()
    order = ["Density", "Entropy", "Enthalpy", "Sound speed", "Cv", "Cp", "Viscosity", "Therm. cond."]
    df["property"] = pd.Categorical(df["property"], order, ordered=True)
    df = df.sort_values("max", ascending=True)

    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    y = np.arange(len(df))
    ax.barh(y, df["max"], color=BLUE2, height=0.58)
    ax.axvline(5, color=RED, linestyle="--", linewidth=1.7)
    ax.text(5.04, len(df) - 0.25, "5% threshold", color=RED, fontsize=16, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(df["property"])
    ax.set_xlim(0, 5.6)
    ax.set_xlabel("Maximum relative error / %")
    ax.set_title("All eight properties below 5% maximum error", pad=12, fontweight="bold")
    clean_axes(ax, "x")
    for i, value in enumerate(df["max"]):
        ax.text(value + 0.06, i, f"{value:.2f}%", va="center", fontsize=14, color=GRAY)
    save_all(fig, "poster_fig5_per_property_max_error")


def make_fig6_speedup() -> None:
    labels = ["REFPROP", "ANN", "Local hybrid"]
    values = [1.0, 18.23, 17.45]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    bars = ax.bar(labels, values, color=[GRAY, BLUE2, ORANGE], width=0.58)
    ax.set_title("Computational speedup", pad=12, fontweight="bold")
    ax.set_ylabel("Speedup relative to REFPROP / ×")
    ax.set_ylim(0, 21)
    clean_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.2f}×" if value > 1 else "1×", ha="center", fontsize=19, fontweight="bold")
    ax.text(
        0.5,
        0.06,
        "Hybrid retains 17.45× speedup with only 4.5% slowdown vs ANN.",
        transform=ax.transAxes,
        ha="center",
        fontsize=14.5,
        color=GRAY,
    )
    save_all(fig, "poster_fig6_speedup")


def make_fig7_hybrid_error_tail() -> None:
    hybrid_df = pd.read_csv(TABLES / "table_3_hybrid_all_point_errors.csv")
    ann_df = load_or_create_ann_all_point_errors()
    hybrid_errors = np.sort(hybrid_df["max_relative_error_pct"].to_numpy(dtype=float))
    ann_errors = np.sort(ann_df["max_relative_error_pct"].to_numpy(dtype=float))
    percentiles = np.linspace(90, 100, 700)
    hybrid_vals = np.percentile(hybrid_errors, percentiles)
    ann_vals = np.percentile(ann_errors, percentiles)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(percentiles, ann_vals, color=BLUE, linewidth=2.4, label="Critical-feature ANN")
    ax.plot(percentiles, hybrid_vals, color=ORANGE, linewidth=2.8, label="Local hybrid")
    ax.axhline(5, color=RED, linestyle="--", linewidth=1.6, label="5% threshold")
    ax.scatter(
        [99, 100],
        [np.percentile(hybrid_errors, 99), hybrid_errors[-1]],
        color=ORANGE,
        s=50,
        zorder=3,
        clip_on=False,
    )
    ax.scatter(
        [99, 100],
        [np.percentile(ann_errors, 99), ann_errors[-1]],
        color=BLUE,
        s=44,
        zorder=3,
        clip_on=False,
    )
    ax.annotate(
        "ANN max = 46.10%",
        xy=(100, ann_errors[-1]),
        xytext=(96.6, 40.5),
        fontsize=14,
        color=BLUE,
        arrowprops=dict(arrowstyle="->", lw=1.3, color=BLUE),
    )
    ax.annotate(
        "Hybrid max = 4.65%",
        xy=(100, hybrid_errors[-1]),
        xytext=(95.8, 10.0),
        fontsize=14,
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", lw=1.3, color=ORANGE),
    )
    ax.set_title("Hybrid error-tail suppression", pad=12, fontweight="bold")
    ax.set_xlabel("Error percentile / %")
    ax.set_ylabel("Relative error / %")
    ax.set_xlim(90, 100.35)
    ax.set_ylim(0, 50)
    clean_axes(ax, "both")
    ax.legend(frameon=False, loc="upper left")
    save_all(fig, "poster_fig7_error_tail_suppression")


def load_or_create_ann_all_point_errors() -> pd.DataFrame:
    out_path = TABLES / "table_3_ann_all_point_errors.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    import joblib
    import torch

    from helium_ml.config import load_config
    from helium_ml.data import prepare_data, split_indices
    from helium_ml.model import MultiPropertyMLP
    from helium_ml.train_utils import predict_raw

    config_path = ROOT / "configs" / "he4_single_phase_extended_2p2_phaseaware_critical_features.json"
    ann_run = ROOT / "runs" / "he4_extended_2p2_phaseaware_critical_features_ann"
    cfg = load_config(config_path)
    data = prepare_data(cfg)
    _, _, test_idx = split_indices(len(data.raw), cfg)

    metadata = joblib.load(ann_run / "metadata.joblib")
    model_cfg = metadata["config"]
    model = MultiPropertyMLP(
        input_dim=len(metadata["input_cols"]),
        output_dim=len(metadata["target_cols"]),
        hidden_layers=list(model_cfg["hidden_layers"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(torch.load(ann_run / "model.pt", map_location="cpu"))
    model.eval()
    data.scaler_X = joblib.load(ann_run / "scaler_X.joblib")
    data.scaler_y = joblib.load(ann_run / "scaler_y.joblib")
    data.log_target_indices = metadata["log_target_indices"]
    data.shifted_log_target_offsets = metadata["shifted_log_target_offsets"]
    data.input_cols = metadata["input_cols"]
    data.target_cols = metadata["target_cols"]

    df = data.raw.iloc[test_idx].reset_index(drop=True)
    x = df[metadata["input_cols"]].to_numpy(dtype=np.float64)
    xp = x.copy()
    for col in metadata["config"].get("log_input_cols", []):
        idx = metadata["input_cols"].index(col)
        xp[:, idx] = np.log10(np.clip(xp[:, idx], 1e-300, None))
    x_scaled = data.scaler_X.transform(xp)
    y_true = df[metadata["target_cols"]].to_numpy(dtype=np.float64)
    y_pred = predict_raw(model, x_scaled, data)
    rel_errors = []
    for idx, _target in enumerate(metadata["target_cols"]):
        denom = np.maximum(np.abs(y_true[:, idx]), data.y_floor[idx])
        rel_errors.append(np.abs(y_pred[:, idx] - y_true[:, idx]) / denom * 100.0)
    rel = np.vstack(rel_errors).T
    out = df[["Temperature (K)", "Pressure (MPa)"]].copy()
    out["max_relative_error_pct"] = rel.max(axis=1)
    out["mean_relative_error_pct"] = rel.mean(axis=1)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def make_fig8_worst_point_distribution() -> None:
    df = pd.read_csv(TABLES / "table_3_hybrid_all_point_errors.csv", usecols=["Temperature (K)", "Pressure (MPa)", "max_relative_error_pct"])
    bg = df.sample(min(len(df), 60000), random_state=3)
    top = df.nlargest(350, "max_relative_error_pct")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(bg["Temperature (K)"], bg["Pressure (MPa)"], s=4, color="#cbd5e1", alpha=0.35, linewidths=0, label="Test samples", rasterized=True)
    ax.scatter(top["Temperature (K)"], top["Pressure (MPa)"], s=18, color=RED, alpha=0.82, linewidths=0, label="Top-error samples")
    ax.set_title("Location of high-error points", pad=12, fontweight="bold")
    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Pressure / MPa")
    ax.set_xlim(2.2, 300)
    ax.set_ylim(0, 3.02)
    clean_axes(ax, "both")
    ax.legend(frameon=False, loc="upper right")
    save_all(fig, "poster_fig8_worst_point_distribution")


def write_readme() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Poster Figures",
        "",
        "Generated according to the handoff requirements for a 0.9 m x 1.5 m vertical international conference poster.",
        "Each figure is exported as PDF, SVG, and 300 dpi PNG.",
        "",
    ]
    for path in sorted(OUT.glob("poster_fig*.pdf")):
        lines.append(f"- `{path.name}`")
    (OUT / "README_poster_figures.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    make_fig1_dataset_phase_map()
    make_fig2_workflow()
    make_fig3_tail_error_reduction()
    make_fig4_metric_cards()
    make_fig5_per_property_max_error()
    make_fig6_speedup()
    make_fig7_hybrid_error_tail()
    make_fig8_worst_point_distribution()
    write_readme()
    print(f"Saved poster figures to {OUT}")


if __name__ == "__main__":
    main()
