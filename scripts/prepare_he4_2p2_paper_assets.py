from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROPERTY_SHORT = {
    "Density (kg/m^3)": "Density",
    "Entropy (kJ/(kg·K))": "Entropy",
    "Entropy (kJ/(kg路K))": "Entropy",
    "Enthalpy (kJ/kg)": "Enthalpy",
    "Speed of Sound (m/s)": "Sound speed",
    "Cv (kJ/(kg·K))": "Cv",
    "Cv (kJ/(kg路K))": "Cv",
    "Cp (kJ/(kg·K))": "Cp",
    "Cp (kJ/(kg路K))": "Cp",
    "Viscosity (μPa·s)": "Viscosity",
    "Viscosity (渭Pa路s)": "Viscosity",
    "Thermal Conductivity (mW/(m·K))": "Therm. cond.",
    "Thermal Conductivity (mW/(m路K))": "Therm. cond.",
}

PHASE_LABELS = {
    "superheated_vapor": "Superheated vapor",
    "single_phase_gas": "Single-phase gas",
    "supercritical": "Supercritical",
    "subcooled_liquid": "Subcooled liquid",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def short_property(name: str) -> str:
    return PROPERTY_SHORT.get(name, name)


def ensure_dirs(out: Path) -> dict[str, Path]:
    dirs = {
        "root": out,
        "figures": out / "figures",
        "tables": out / "tables",
        "model": out / "model",
        "copied": out / "copied_existing_figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def copy_if_exists(src: Path, dst: Path, manifest: list[dict], description: str) -> None:
    record = {
        "description": description,
        "source": str(src),
        "destination": str(dst),
        "status": "missing",
    }
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        record["status"] = "copied"
    manifest.append(record)


def metrics_to_rows(section: dict[str, float], model: str) -> dict[str, float | str]:
    keys = [
        "mape",
        "median",
        "p95",
        "p99",
        "max",
        "mean_r2",
        "mean_nrmse_range",
        "within_1pct",
        "within_5pct",
    ]
    row: dict[str, float | str] = {"model": model}
    for key in keys:
        row[key] = section.get(key, np.nan)
    return row


def per_target_frame(metrics: dict, model: str) -> pd.DataFrame:
    rows = []
    for target, values in metrics.items():
        row = {
            "model": model,
            "property": short_property(target),
            "full_property": target,
        }
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def make_dataset_tables(data: pd.DataFrame, tables: Path, figures: Path) -> None:
    input_cols = ["Temperature (K)", "Pressure (MPa)"]
    target_cols = [
        "Density (kg/m^3)",
        "Entropy (kJ/(kg·K))",
        "Enthalpy (kJ/kg)",
        "Speed of Sound (m/s)",
        "Cv (kJ/(kg·K))",
        "Cp (kJ/(kg·K))",
        "Viscosity (μPa·s)",
        "Thermal Conductivity (mW/(m·K))",
    ]

    summary = pd.DataFrame(
        [
            {"item": "number_of_samples", "value": len(data)},
            {"item": "temperature_min_K", "value": data["Temperature (K)"].min()},
            {"item": "temperature_max_K", "value": data["Temperature (K)"].max()},
            {"item": "pressure_min_MPa", "value": data["Pressure (MPa)"].min()},
            {"item": "pressure_max_MPa", "value": data["Pressure (MPa)"].max()},
            {"item": "number_of_inputs_for_final_model", "value": 9},
            {"item": "number_of_targets", "value": len(target_cols)},
        ]
    )
    save_table(summary, tables / "table_1_dataset_summary.csv")

    phase_counts = (
        data["phase_region"]
        .value_counts()
        .rename_axis("phase_region")
        .reset_index(name="n_samples")
    )
    phase_counts["phase_label"] = phase_counts["phase_region"].map(PHASE_LABELS).fillna(
        phase_counts["phase_region"]
    )
    phase_counts["fraction_pct"] = phase_counts["n_samples"] / len(data) * 100
    save_table(phase_counts, tables / "table_1_phase_counts.csv")

    range_rows = []
    for col in input_cols + target_cols:
        if col in data.columns:
            range_rows.append(
                {
                    "variable": col,
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                }
            )
    save_table(pd.DataFrame(range_rows), tables / "table_1_variable_ranges.csv")

    sample = data.sample(min(len(data), 70000), random_state=7)
    phase_order = [
        "subcooled_liquid",
        "superheated_vapor",
        "single_phase_gas",
        "supercritical",
    ]
    colors = {
        "subcooled_liquid": "#3b74af",
        "superheated_vapor": "#d97904",
        "single_phase_gas": "#48995a",
        "supercritical": "#7b5aa6",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for phase in phase_order:
        part = sample[sample["phase_region"] == phase]
        if part.empty:
            continue
        ax.scatter(
            part["Temperature (K)"],
            part["Pressure (MPa)"],
            s=3,
            alpha=0.35,
            color=colors[phase],
            label=PHASE_LABELS.get(phase, phase),
            rasterized=True,
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title("Single-phase helium-4 dataset domain")
    ax.set_xlim(2.2, 300)
    ax.set_ylim(0, 3.05)
    ax.grid(alpha=0.22)
    ax.legend(markerscale=4, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "fig_1_dataset_phase_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_property_figures(per_hybrid: pd.DataFrame, figures: Path) -> None:
    df = per_hybrid.copy()
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey=True)
    specs = [
        ("mape", "MAPE (%)", "#3f6fb5"),
        ("p99", "P99 error (%)", "#d8862f"),
        ("max", "Maximum error (%)", "#b94343"),
    ]
    for ax, (col, label, color) in zip(axes, specs):
        ax.barh(y, df[col], color=color, alpha=0.9)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.22)
        for i, val in enumerate(df[col]):
            ax.text(val, i, f" {val:.3g}", va="center", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["property"])
    axes[0].invert_yaxis()
    fig.suptitle("Per-property error metrics of the final hybrid model", y=1.02)
    fig.tight_layout()
    fig.savefig(figures / "fig_3_per_property_error_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10.4, 4.6))
    x = np.arange(len(df))
    width = 0.36
    ax1.bar(x - width / 2, df["within_1pct"], width, label="Within 1%", color="#4c78a8")
    ax1.bar(x + width / 2, df["within_5pct"], width, label="Within 5%", color="#59a14f")
    ax1.set_ylabel("Prediction fraction (%)")
    ax1.set_ylim(95, 100.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["property"], rotation=28, ha="right")
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(frameon=False, loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(x, df["r2"], color="#8f4b9b", marker="o", linewidth=2, label="R2")
    ax2.set_ylabel("R2")
    ax2.set_ylim(max(0.99995, df["r2"].min() - 0.00001), 1.000002)
    ax2.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figures / "fig_3_r2_and_within_thresholds.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_ann_hybrid_comparison_figures(
    per_base: pd.DataFrame, per_hybrid: pd.DataFrame, overall: pd.DataFrame, figures: Path
) -> None:
    combo = pd.concat([per_base, per_hybrid], ignore_index=True)
    pivot = combo.pivot(index="property", columns="model", values="max")
    order = per_hybrid["property"].tolist()
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    x = np.arange(len(order))
    width = 0.38
    ax.bar(x - width / 2, pivot["Critical-feature ANN"], width, label="ANN", color="#6b8ec1")
    ax.bar(x + width / 2, pivot["Local hybrid"], width, label="Hybrid", color="#c95252")
    ax.axhline(5, color="#202020", linestyle="--", linewidth=1, label="5% target")
    ax.set_ylabel("Maximum relative error (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=28, ha="right")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "fig_5_ann_vs_hybrid_max_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2))
    metrics = [("mape", "MAPE (%)"), ("p99", "P99 error (%)"), ("max", "Maximum error (%)")]
    colors = ["#4c78a8", "#f28e2b"]
    for ax, (col, title) in zip(axes, metrics):
        vals = overall[col].to_numpy(dtype=float)
        bars = ax.bar(overall["model"], vals, color=colors, alpha=0.92)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig_5_ann_vs_hybrid_overall_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_baseline_figure(baselines: pd.DataFrame, figures: Path) -> None:
    df = baselines.copy()
    order = ["Current MLP", "Random Forest", "Extra Trees", "XGBoost", "Gaussian RBF Ridge"]
    df["model"] = pd.Categorical(df["model"], order, ordered=True)
    df = df.sort_values("model")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    specs = [
        ("mape", "MAPE (%)", "#4c78a8"),
        ("max", "Maximum error (%)", "#c44e52"),
        ("points_per_s", "Throughput (points/s)", "#59a14f"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        ax.barh(df["model"].astype(str), df[col], color=color, alpha=0.9)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        if col == "points_per_s":
            ax.set_xscale("log")
    axes[0].invert_yaxis()
    fig.suptitle("Baseline model comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(figures / "fig_6_baseline_accuracy_speed_tail.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_speed_figure(speed: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    ax.plot(speed["n_points"], speed["ann_points_per_s"], marker="o", label="ANN")
    ax.plot(speed["n_points"], speed["hybrid_points_per_s"], marker="o", label="Hybrid")
    ax.plot(speed["n_points"], speed["refprop_points_per_s"], marker="o", label="REFPROP")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of state points")
    ax.set_ylabel("Throughput (points/s)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(speed["n_points"], speed["ann_speedup_vs_refprop"], marker="o", label="ANN")
    ax.plot(speed["n_points"], speed["hybrid_speedup_vs_refprop"], marker="o", label="Hybrid")
    ax.set_xscale("log")
    ax.set_xlabel("Number of state points")
    ax.set_ylabel("Speedup over REFPROP")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "fig_7_speed_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_route_and_worst_figures(
    metrics: dict, worst: pd.DataFrame, figures: Path, tables: Path
) -> None:
    routed = pd.DataFrame(
        [{"region": key, "routed_events": val} for key, val in metrics.get("routed", {}).items()]
    ).sort_values("routed_events", ascending=True)
    save_table(routed, tables / "table_6_hybrid_routed_counts.csv")

    if not routed.empty:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax.barh(routed["region"], routed["routed_events"], color="#7373ad")
        ax.set_xlabel("Routed point events")
        ax.set_title("Local hybrid correction routing")
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(figures / "fig_2_hybrid_routed_counts.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    grids = pd.DataFrame(metrics.get("grids", []))
    if not grids.empty:
        save_table(grids, tables / "table_6_hybrid_grid_regions.csv")

    if not worst.empty:
        worst_plot = worst.head(250).copy()
        worst_plot["property"] = worst_plot["target"].map(short_property)
        fig, ax = plt.subplots(figsize=(7.3, 5.1))
        scatter = ax.scatter(
            worst_plot["Temperature (K)"],
            worst_plot["Pressure (MPa)"],
            c=worst_plot["relative_error_pct"],
            s=20,
            cmap="magma_r",
            edgecolors="none",
            alpha=0.85,
        )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (MPa)")
        ax.set_title("Worst hybrid-error points in the T-P domain")
        ax.grid(alpha=0.22)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Relative error (%)")
        fig.tight_layout()
        fig.savefig(figures / "fig_5_hybrid_worst_points_tp.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_error_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    vmax: float | None = None,
) -> None:
    t_bins = np.linspace(2.2, 300.0, 91)
    p_bins = np.linspace(0.0, 3.0, 76)
    tmp = df[["Temperature (K)", "Pressure (MPa)", value_col]].dropna().copy()
    tmp["t_bin"] = pd.cut(tmp["Temperature (K)"], t_bins, include_lowest=True)
    tmp["p_bin"] = pd.cut(tmp["Pressure (MPa)"], p_bins, include_lowest=True)
    grid = tmp.pivot_table(index="p_bin", columns="t_bin", values=value_col, aggfunc="mean", observed=False)
    arr = grid.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    mesh = ax.pcolormesh(t_bins, p_bins, arr, cmap="viridis", shading="auto", vmax=vmax)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Mean relative error in bin (%)")
    ax.set_xlim(2.2, 300)
    ax.set_ylim(0, 3.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_hybrid_error_maps(all_errors: pd.DataFrame | None, figures: Path, tables: Path) -> None:
    if all_errors is None or all_errors.empty:
        return
    save_table(all_errors, tables / "table_3_hybrid_all_point_errors.csv")
    heatmap_dir = figures / "fig_4_hybrid_error_maps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    make_error_heatmap(
        all_errors,
        "max_relative_error_pct",
        "Hybrid maximum error over eight properties",
        heatmap_dir / "fig_4a_hybrid_max_error_heatmap.png",
        vmax=2.0,
    )
    make_error_heatmap(
        all_errors,
        "mean_relative_error_pct",
        "Hybrid mean error over eight properties",
        heatmap_dir / "fig_4b_hybrid_mean_error_heatmap.png",
        vmax=0.5,
    )

    selected = [
        ("Density", "Density"),
        ("Entropy", "Entropy"),
        ("Enthalpy", "Enthalpy"),
        ("Cp", "Cp"),
    ]
    error_cols = [c for c in all_errors.columns if c.startswith("error_")]
    for key, label in selected:
        candidates = [c for c in error_cols if key in c]
        if not candidates:
            continue
        make_error_heatmap(
            all_errors,
            candidates[0],
            f"Hybrid error heatmap: {label}",
            heatmap_dir / f"fig_4_{label.lower()}_error_heatmap.png",
            vmax=1.5,
        )


def write_index(out: Path, metrics: dict, speed: pd.DataFrame) -> None:
    hybrid = metrics["hybrid"]["overall"]
    last = speed.iloc[-1]
    lines = [
        "# He-4 2.2 K Hybrid Paper Assets",
        "",
        "This folder contains manuscript-ready tables and figures for the 2.2-300 K single-phase helium-4 ANN/hybrid paper line.",
        "",
        "## Core Result",
        "",
        f"- Hybrid overall MAPE: {hybrid['mape']:.4f}%.",
        f"- Hybrid P99 error: {hybrid['p99']:.4f}%.",
        f"- Hybrid maximum error: {hybrid['max']:.4f}%.",
        f"- Hybrid mean R2: {hybrid['mean_r2']:.8f}.",
        f"- Within 5%: {hybrid['within_5pct']:.1f}%.",
        f"- 100000-point hybrid speedup over REFPROP: {last['hybrid_speedup_vs_refprop']:.2f}x.",
        f"- Hybrid slowdown relative to ANN: {(last['hybrid_slowdown_vs_ann'] - 1) * 100:.2f}%.",
        "",
        "## Figure Mapping",
        "",
        "- Fig. 1: `figures/fig_1_dataset_phase_map.png`",
        "- Fig. 2: `figures/fig_2_hybrid_routed_counts.png` or use a schematic in the manuscript.",
        "- Fig. 3: `figures/fig_3_per_property_error_metrics.png` and `figures/fig_3_r2_and_within_thresholds.png`",
        "- Fig. 4: `figures/fig_4_hybrid_error_maps/`",
        "- Fig. 5: `figures/fig_5_ann_vs_hybrid_max_error.png`, `figures/fig_5_ann_vs_hybrid_overall_errors.png`, `figures/fig_5_hybrid_worst_points_tp.png`",
        "- Fig. 6: `figures/fig_6_baseline_accuracy_speed_tail.png`",
        "- Fig. 7: `figures/fig_7_speed_benchmark.png`",
        "",
        "## Table Mapping",
        "",
        "- Table 1: `tables/table_1_dataset_summary.csv`, `tables/table_1_phase_counts.csv`, `tables/table_1_variable_ranges.csv`",
        "- Table 2: `tables/table_2_overall_ann_vs_hybrid.csv`",
        "- Table 3: `tables/table_3_hybrid_per_property_metrics.csv`",
        "- Table 4: `tables/table_4_baseline_summary.csv`",
        "- Table 5: `tables/table_5_speed_summary.csv`",
        "- Additional hybrid details: `tables/table_6_hybrid_grid_regions.csv`, `tables/table_6_hybrid_routed_counts.csv`",
        "",
        "## Note",
        "",
        "The Fig. 4 heatmaps are based on the exported all-point hybrid errors from the final test split.",
    ]
    (out / "README_paper_assets.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/he4data_single_phase_extended_2p2_critical_features.csv")
    parser.add_argument("--ann-run", default="runs/he4_extended_2p2_phaseaware_critical_features_ann")
    parser.add_argument("--hybrid-run", default="runs/he4_extended_2p2_critical_features_grid_hybrid_v8_full_export")
    parser.add_argument("--speed-run", default="runs/critical_feature_hybrid_refprop_benchmark_v8")
    parser.add_argument("--baseline-run", default="runs/baseline_comparison_full")
    parser.add_argument("--config", default="configs/he4_single_phase_extended_2p2_phaseaware_critical_features.json")
    parser.add_argument("--out", default="paper_outputs/he4_2p2_hybrid_paper")
    args = parser.parse_args()

    out = Path(args.out)
    dirs = ensure_dirs(out)
    figures = dirs["figures"]
    tables = dirs["tables"]

    data = pd.read_csv(args.data)
    metrics = read_json(Path(args.hybrid_run) / "grid_hybrid_metrics.json")
    speed = pd.read_csv(Path(args.speed_run) / "ann_hybrid_refprop_speed.csv")
    baselines = pd.read_csv(Path(args.baseline_run) / "baseline_summary.csv")
    worst = pd.read_csv(Path(args.hybrid_run) / "grid_hybrid_worst_points.csv")
    all_errors_path = Path(args.hybrid_run) / "grid_hybrid_all_point_errors.csv"
    all_errors = pd.read_csv(all_errors_path) if all_errors_path.exists() else None

    make_dataset_tables(data, tables, figures)

    overall = pd.DataFrame(
        [
            metrics_to_rows(metrics["base"]["overall"], "Critical-feature ANN"),
            metrics_to_rows(metrics["hybrid"]["overall"], "Local hybrid"),
        ]
    )
    save_table(overall, tables / "table_2_overall_ann_vs_hybrid.csv")

    per_base = per_target_frame(metrics["base"]["per_target"], "Critical-feature ANN")
    per_hybrid = per_target_frame(metrics["hybrid"]["per_target"], "Local hybrid")
    save_table(per_hybrid, tables / "table_3_hybrid_per_property_metrics.csv")
    save_table(pd.concat([per_base, per_hybrid], ignore_index=True), tables / "table_3_ann_vs_hybrid_per_property.csv")

    save_table(baselines, tables / "table_4_baseline_summary.csv")
    save_table(speed, tables / "table_5_speed_summary.csv")
    save_table(worst, tables / "table_3_hybrid_worst_points.csv")

    make_property_figures(per_hybrid, figures)
    make_ann_hybrid_comparison_figures(per_base, per_hybrid, overall, figures)
    make_baseline_figure(baselines, figures)
    make_speed_figure(speed, figures)
    make_route_and_worst_figures(metrics, worst, figures, tables)
    make_hybrid_error_maps(all_errors, figures, tables)

    manifest: list[dict] = []
    copy_if_exists(Path(args.config), dirs["model"] / Path(args.config).name, manifest, "final model config")
    for name in ["model.pt", "scaler_X.joblib", "scaler_y.joblib", "metadata.joblib", "history.csv", "training_curve.png"]:
        subdir = "model" if name.endswith((".pt", ".joblib")) else "copied"
        copy_if_exists(Path(args.ann_run) / name, dirs[subdir] / name, manifest, f"ANN run {name}")
    for name in ["baseline_accuracy_speed.png", "baseline_tail_errors.png"]:
        copy_if_exists(Path(args.baseline_run) / name, dirs["copied"] / "baselines" / name, manifest, name)
    ann_error_dir = Path(args.ann_run) / "error_maps"
    for src in sorted(ann_error_dir.glob("*.png")):
        copy_if_exists(src, dirs["copied"] / "ann_error_maps" / src.name, manifest, f"ANN error map {src.name}")

    save_table(pd.DataFrame(manifest), out / "manifest.csv")
    write_index(out, metrics, speed)

    print(f"Prepared paper assets in {out.resolve()}")
    print(f"Figures: {len(list(figures.glob('*.png')))} generated")
    print(f"Tables: {len(list(tables.glob('*.csv')))} generated")


if __name__ == "__main__":
    main()
