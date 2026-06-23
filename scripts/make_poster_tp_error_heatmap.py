from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "results" / "he4_single_phase_surrogate"
TABLES = PKG / "tables"
OUT = PKG / "figures" / "error_maps"


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


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_heatmap(
    df: pd.DataFrame,
    stem: str,
    log_temperature: bool,
    log_pressure: bool,
    log_error: bool,
    show_near_critical: bool = False,
) -> None:
    if log_temperature:
        t_bins = np.geomspace(2.2, 300.0, 115)
    else:
        t_bins = np.linspace(2.2, 300.0, 115)
    if log_pressure:
        p_bins = np.geomspace(0.001, 3.0, 95)
    else:
        p_bins = np.linspace(0.0, 3.0, 95)
    tmp = df.copy()
    tmp["t_bin"] = pd.cut(tmp["Temperature (K)"], t_bins, include_lowest=True)
    tmp["p_bin"] = pd.cut(tmp["Pressure (MPa)"], p_bins, include_lowest=True)
    grid = tmp.pivot_table(
        index="p_bin",
        columns="t_bin",
        values="max_relative_error_pct",
        aggfunc="mean",
        observed=False,
    )
    z = grid.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.35))
    if log_error:
        z_plot = np.clip(z, 0.01, 5.0)
        mesh = ax.pcolormesh(
            t_bins,
            p_bins,
            z_plot,
            cmap="YlOrRd",
            shading="auto",
            norm=LogNorm(vmin=0.01, vmax=5.0),
        )
    else:
        mesh = ax.pcolormesh(
            t_bins,
            p_bins,
            z,
            cmap="YlOrRd",
            shading="auto",
            vmin=0.0,
            vmax=5.0,
        )

    # Helium-4 critical point used in the feature construction.
    tc = 5.1953
    pc = 0.22746
    ax.scatter([tc], [pc], s=95, marker="*", color="#111827", edgecolors="white", linewidths=0.8, zorder=5)
    ax.annotate(
        "Critical point",
        xy=(tc, pc),
        xytext=(30, 0.62),
        textcoords="data",
        fontsize=15.5,
        color="#111827",
        arrowprops=dict(arrowstyle="->", lw=1.3, color="#111827"),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#9ca3af", alpha=0.92),
    )

    if show_near_critical:
        # Qualitative marker only; disabled for the clean poster version.
        near = Circle((tc, pc), radius=0.18, fill=False, ec="#111827", lw=1.5, ls="--", alpha=0.9, zorder=4)
        ax.add_patch(near)
        ax.text(
            10.5,
            0.16,
            "Near-critical\nregion",
            fontsize=14.5,
            color="#111827",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#9ca3af", alpha=0.88),
        )

    ax.set_title("T-P error map of the local hybrid model", pad=12, fontweight="bold")
    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Pressure / MPa")
    if log_temperature:
        ax.set_xscale("log")
        ax.set_xlim(2.2, 300.0)
        ax.set_xticks([2.2, 3, 5, 10, 30, 100, 300])
        ax.set_xticklabels(["2.2", "3", "5", "10", "30", "100", "300"])
    else:
        ax.set_xlim(2.2, 300.0)
    if log_pressure:
        ax.set_yscale("log")
        ax.set_ylim(0.001, 3.0)
        ax.set_yticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
        ax.set_yticklabels(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3"])
    else:
        ax.set_ylim(0.0, 3.0)
    ax.grid(color="#d1d5db", linewidth=0.75, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Hybrid relative error / %", fontsize=18)
    if log_error:
        cbar.set_ticks([0.01, 0.03, 0.1, 0.3, 1, 3, 5])
        cbar.set_ticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3", "5"])
    else:
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.ax.tick_params(labelsize=15)

    axes_note = []
    if log_temperature:
        axes_note.append("log-temperature axis")
    if log_pressure:
        axes_note.append("log-pressure axis")
    scale_note = ", ".join(axes_note) + "; " if axes_note else ""
    scale_note += "log color scale clipped to 0.01-5%" if log_error else "linear color scale clipped to 0-5%"
    ax.text(
        0.02,
        0.035,
        f"{scale_note}; bin value = mean maximum relative error over eight outputs.",
        transform=ax.transAxes,
        fontsize=13.2,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", alpha=0.88),
    )

    save_all(fig, stem)


def main() -> None:
    setup_style()
    df = pd.read_csv(
        TABLES / "table_3_hybrid_all_point_errors.csv",
        usecols=["Temperature (K)", "Pressure (MPa)", "max_relative_error_pct"],
    )
    make_heatmap(
        df,
        "poster_fig9_tp_hybrid_error_heatmap_linear_clean",
        log_temperature=False,
        log_pressure=False,
        log_error=False,
        show_near_critical=False,
    )
    print(f"Saved T-P error heatmap to {OUT}")


if __name__ == "__main__":
    main()
