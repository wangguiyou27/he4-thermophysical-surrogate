from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, text, width=2.15, height=0.82, fc="#f7f9fc", ec="#365f91"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.045,rounding_size=0.035",
        linewidth=1.35,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#1f2933",
        linespacing=1.25,
    )
    return box


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.25,
        color="#333333",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="results/he4_single_phase_surrogate/figures/poster/fig_2_model_workflow.png",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.2, 4.7))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    y = 2.45
    w = 1.85
    h = 0.78
    x_positions = [0.35, 2.45, 4.55, 6.65, 8.75, 10.85]

    labels = [
        "REFPROP\ndata generation",
        "Single-phase\nfiltering",
        "Feature\nconstruction",
        "Multi-output\nANN training",
        "Local hybrid\ncorrection",
        "Eight-property\nprediction",
    ]
    colors = ["#e9f2ff", "#eef8ee", "#fff5e5", "#f4efff", "#ffecec", "#eaf7f7"]
    edges = ["#3f6fb5", "#4d8b57", "#c28222", "#7b5aa6", "#b94343", "#397c7c"]

    boxes = []
    for x, label, fc, ec in zip(x_positions, labels, colors, edges):
        boxes.append(add_box(ax, (x, y), label, width=w, height=h, fc=fc, ec=ec))

    for i in range(len(x_positions) - 1):
        add_arrow(ax, (x_positions[i] + w, y + h / 2), (x_positions[i + 1], y + h / 2))

    # Detail notes.
    detail_y = 0.85
    detail_h = 0.85
    details = [
        ("T: 2.2-300 K\nP: 0.001-3 MPa", x_positions[0], "#e9f2ff", "#3f6fb5"),
        ("Remove two-phase\nand near-saturation states", x_positions[1], "#eef8ee", "#4d8b57"),
        ("phase_code +\ncritical features", x_positions[2], "#fff5e5", "#c28222"),
        ("density, entropy,\nenthalpy, sound speed,\nCv, Cp, viscosity, k", x_positions[5] - 0.18, "#eaf7f7", "#397c7c"),
    ]
    for text, x, fc, ec in details:
        add_box(ax, (x, detail_y), text, width=w + 0.28, height=detail_h, fc=fc, ec=ec)
        add_arrow(ax, (x + (w + 0.28) / 2, detail_y + detail_h), (x + w / 2, y))

    # Hybrid note spanning ANN and correction.
    note = FancyBboxPatch(
        (6.55, 0.45),
        4.15,
        0.55,
        boxstyle="round,pad=0.04,rounding_size=0.03",
        linewidth=1.1,
        edgecolor="#6b7280",
        facecolor="#f8fafc",
    )
    ax.add_patch(note)
    ax.text(
        8.625,
        0.725,
        "ANN as global surrogate; local grids suppress tail errors in high-risk regions",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#374151",
    )
    add_arrow(ax, (8.62, 1.0), (8.65, y))

    ax.text(
        6.6,
        4.25,
        "Workflow of the critical-feature ANN with local hybrid correction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )

    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
