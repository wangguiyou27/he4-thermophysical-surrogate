from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def copy_file(src: Path, dst: Path, records: list[dict], category: str, description: str):
    if not src.exists():
        records.append(
            {
                "category": category,
                "description": description,
                "source": str(src),
                "destination": "",
                "status": "missing",
            }
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    records.append(
        {
            "category": category,
            "description": description,
            "source": str(src),
            "destination": str(dst),
            "status": "copied",
        }
    )


def copy_many(pattern_dir: Path, pattern: str, dst_dir: Path, records: list[dict], category: str):
    for src in sorted(pattern_dir.glob(pattern)):
        copy_file(src, dst_dir / src.name, records, category, src.stem)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(out_dir: Path, best_run: Path, refprop_run: Path, records: list[dict]):
    metrics = read_json(best_run / "metrics.json")
    speed_csv = refprop_run / "refprop_ann_speed.csv"
    prop_csv = refprop_run / "ann_vs_refprop_property_metrics.csv"
    baseline_csv = out_dir / "tables" / "baselines" / "baseline_summary.csv"

    lines = [
        "# Paper Output Package",
        "",
        "This folder collects the final model artifacts, tables, and figures for the single-phase helium-4 ANN surrogate paper.",
        "",
        "## Final Model Accuracy",
        "",
    ]
    if metrics:
        overall = metrics.get("overall", metrics)
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
        for key in keys:
            if key in overall:
                lines.append(f"- {key}: {overall[key]}")
    else:
        lines.append("- metrics.json was not found.")

    lines += ["", "## Large REFPROP vs ANN Benchmark", ""]
    if speed_csv.exists():
        speed = pd.read_csv(speed_csv)
        last = speed.iloc[-1]
        lines.append(
            f"- Largest benchmark: {int(last['n_points'])} state points, "
            f"ANN speedup = {last['speedup_ann_vs_refprop']:.2f}x."
        )
        lines.append(
            f"- ANN throughput = {last['ann_points_per_s']:.0f} points/s; "
            f"REFPROP throughput = {last['refprop_points_per_s']:.0f} points/s."
        )
    else:
        lines.append("- refprop_ann_speed.csv was not found.")

    lines += ["", "## ANN vs REFPROP Property Metrics", ""]
    if prop_csv.exists():
        prop = pd.read_csv(prop_csv)
        for _, row in prop.iterrows():
            lines.append(
                f"- {row['property']}: MAPE {row['mape_pct']:.4f}%, "
                f"P99 {row['p99_abs_pct']:.4f}%, R2 {row['r2']:.6f}"
            )
    else:
        lines.append("- ann_vs_refprop_property_metrics.csv was not found.")

    lines += ["", "## Baseline Model Comparison", ""]
    if baseline_csv.exists():
        base = pd.read_csv(baseline_csv).sort_values("mape")
        for _, row in base.iterrows():
            lines.append(
                f"- {row['model']}: MAPE {row['mape']:.4f}%, "
                f"P99 {row['p99']:.4f}%, Max {row['max']:.4f}%, "
                f"throughput {row['points_per_s']:.0f} points/s."
            )
    else:
        lines.append("- baseline_summary.csv was not found.")

    lines += [
        "",
        "## Folder Layout",
        "",
        "- `model/`: trained model, scalers, metadata, and config.",
        "- `figures/accuracy/`: training curve, property metrics, and region/error maps.",
        "- `figures/speed/`: REFPROP vs ANN throughput and speedup figures.",
        "- `figures/baselines/`: baseline accuracy-speed and tail-error comparison figures.",
        "- `tables/`: CSV metrics and benchmark tables.",
        "- `manifest.csv`: copied artifact list.",
        "",
    ]
    copied = sum(1 for item in records if item["status"] == "copied")
    missing = sum(1 for item in records if item["status"] == "missing")
    lines.append(f"Copied artifacts: {copied}; missing artifacts: {missing}.")

    (out_dir / "README_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-run", default="runs/he4_supervised_shifted_entropy")
    parser.add_argument("--refprop-run", default="runs/refprop_ann_benchmark_large")
    parser.add_argument("--baseline-run", default="runs/baseline_comparison_full")
    parser.add_argument("--config", default="configs/he4_single_phase.json")
    parser.add_argument("--out", default="results/final_he4_single_phase")
    args = parser.parse_args()

    best_run = Path(args.best_run)
    refprop_run = Path(args.refprop_run)
    baseline_run = Path(args.baseline_run)
    config = Path(args.config)
    out_dir = Path(args.out)
    records: list[dict] = []

    copy_file(config, out_dir / "model" / config.name, records, "config", "training config")
    for name in ["model.pt", "scaler_X.joblib", "scaler_y.joblib", "metadata.joblib"]:
        copy_file(best_run / name, out_dir / "model" / name, records, "model", name)

    for name in [
        "metrics.json",
        "property_metrics_table.csv",
        "region_metrics.csv",
        "worst_test_points.csv",
        "history.csv",
    ]:
        copy_file(best_run / name, out_dir / "tables" / name, records, "accuracy_table", name)

    for name in [
        "training_curve.png",
        "property_error_bars.png",
        "property_accuracy_r2.png",
    ]:
        copy_file(best_run / name, out_dir / "figures" / "accuracy" / name, records, "accuracy_figure", name)

    error_map_dir = best_run / "error_maps"
    copy_many(error_map_dir, "*.png", out_dir / "figures" / "accuracy" / "error_maps", records, "error_map")
    copy_many(error_map_dir, "*.csv", out_dir / "tables" / "error_maps", records, "error_map_table")

    for name in [
        "refprop_ann_speed.csv",
        "ann_vs_refprop_property_metrics.csv",
        "refprop_errors.json",
    ]:
        copy_file(refprop_run / name, out_dir / "tables" / name, records, "refprop_table", name)
    for name in ["refprop_ann_speed.png", "ann_speedup_over_refprop.png"]:
        copy_file(refprop_run / name, out_dir / "figures" / "speed" / name, records, "speed_figure", name)

    for name in [
        "baseline_summary.csv",
        "baseline_per_property_metrics.csv",
        "baseline_metrics.json",
    ]:
        copy_file(baseline_run / name, out_dir / "tables" / "baselines" / name, records, "baseline_table", name)
    for name in ["baseline_accuracy_speed.png", "baseline_tail_errors.png"]:
        copy_file(baseline_run / name, out_dir / "figures" / "baselines" / name, records, "baseline_figure", name)

    manifest = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    write_summary(out_dir, best_run, refprop_run, records)

    print(f"Organized paper outputs at: {out_dir.resolve()}")
    print(manifest["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
