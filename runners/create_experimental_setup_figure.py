#!/usr/bin/env python3
"""Generate a publication-grade figure for validation/evaluation setup."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("results/figures")
PNG_PATH = OUT_DIR / "experimental_setup_validation_evaluation.png"
PDF_PATH = OUT_DIR / "experimental_setup_validation_evaluation.pdf"

INK = "#1F2937"
ACCENT = "#1E6091"
SOFT = "#F8FAFC"


def add_box(ax, xy, wh, title, body, face=SOFT, edge=INK, lw=1.1, tsize=8.8, bsize=7.7):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=tsize, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center", fontsize=bsize, color=INK)


def add_arrow(ax, start, end, color=INK, lw=1.1, style="-|>", scale=11):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=scale,
            linewidth=lw,
            color=color,
        )
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})

    fig, ax = plt.subplots(figsize=(13.8, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.03, 0.95, "Validation and Evaluation Setup", fontsize=13.0, weight="bold", color=INK)
    ax.text(0.03, 0.915, "Datasets, hardware, baselines, and protocol used for fair comparison", fontsize=8.6, color="#475569")

    # Row 1: Data and infrastructure
    add_box(
        ax,
        (0.04, 0.72),
        (0.28, 0.15),
        "Datasets",
        "Small/medium: WikiCS, Coauthor-Physics,\nCoauthor-CS, DeezerEurope, Foursquare\nLarge: reddit, ogbn-products",
        edge=ACCENT,
        face="#EEF6FD",
        lw=1.2,
    )

    add_box(
        ax,
        (0.36, 0.72),
        (0.28, 0.15),
        "Cluster and Runtime",
        "AWS EMR (5 workers), 20 executors, 60 vCPUs\nSpark + YARN + Delta Lake\nExecutor memory 28g, overhead 12g",
        edge=ACCENT,
        face="#EEF6FD",
        lw=1.2,
    )

    add_box(
        ax,
        (0.68, 0.72),
        (0.28, 0.15),
        "Model Families",
        "GraphSAGE, GATv2, ARMA, ASAP\nCommunity models and CAAN variants\nPhase 3.7/3.8 global path",
        edge=ACCENT,
        face="#EEF6FD",
        lw=1.2,
    )

    # Row 2: Protocol and fairness controls
    add_box(
        ax,
        (0.04, 0.46),
        (0.28, 0.17),
        "Validation Protocol",
        "Fixed train/val/test masks\nMultiple partition strategies (LPA, Louvain)\nConsistent preprocessing contracts",
    )

    add_box(
        ax,
        (0.36, 0.46),
        (0.28, 0.17),
        "Fairness Controls",
        "Matched partition bounds and resources\nSame data source and feature schema\nCompared under identical cluster settings",
    )

    add_box(
        ax,
        (0.68, 0.46),
        (0.28, 0.17),
        "Two-Hop Configuration",
        "num_hops = 2 for Phase 3.7/3.8 scaling\nInput representation: [x^(0) || x^(1) || x^(2)]\nClassifier trained on cached propagated features",
        edge=ACCENT,
        face="#EEF6FD",
        lw=1.2,
    )

    # Row 3: Metrics and outputs
    add_box(
        ax,
        (0.04, 0.18),
        (0.20, 0.16),
        "Quality Metrics",
        "Node accuracy\nLink prediction AUC\nBoundary recovery",
    )

    add_box(
        ax,
        (0.29, 0.18),
        (0.20, 0.16),
        "Systems Metrics",
        "Propagation time\nTraining time\nEnd-to-end runtime",
    )

    add_box(
        ax,
        (0.54, 0.18),
        (0.20, 0.16),
        "Scalability Metrics",
        "Speedup vs 16 executors\nParallel efficiency\nCoverage consistency",
    )

    add_box(
        ax,
        (0.79, 0.18),
        (0.17, 0.16),
        "Artifacts",
        "Tables 3/4/9/12/13\nFigure 1 architecture\nPhase 3.7 scaling plots",
    )

    # Flow arrows top->middle->bottom
    for x in [0.18, 0.50, 0.82]:
        add_arrow(ax, (x, 0.72), (x, 0.63), color="#475569", style="->", scale=9)
        add_arrow(ax, (x, 0.46), (x, 0.34), color="#475569", style="->", scale=9)

    # Horizontal relation arrows
    add_arrow(ax, (0.32, 0.545), (0.36, 0.545), color="#64748B", lw=1.0)
    add_arrow(ax, (0.64, 0.545), (0.68, 0.545), color="#64748B", lw=1.0)

    # Legend and note
    ax.text(0.04, 0.09, "Blue boxes: fixed controls used in validation and fairness comparisons.", fontsize=7.5, color=ACCENT)
    ax.text(0.04, 0.065, "Gray boxes: evaluated dimensions and reported outcomes.", fontsize=7.5, color="#64748B")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
