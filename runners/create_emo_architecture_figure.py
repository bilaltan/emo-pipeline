#!/usr/bin/env python3
"""Generate a publication-style EMO system architecture overview figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("results/figures")
PNG_PATH = OUT_DIR / "emo_system_architecture_overview.png"
PDF_PATH = OUT_DIR / "emo_system_architecture_overview.pdf"


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#1F2937", lw=1.2):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center", fontsize=9.6, weight="bold")
    ax.text(x + width / 2, y + height * 0.38, body, ha="center", va="center", fontsize=8.2)


def add_arrow(ax, start, end, color="#334155", style="-|>", lw=1.4):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(13.6, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Lane labels
    ax.text(0.02, 0.86, "Primary path in this paper", fontsize=9.8, weight="bold", color="#1D4ED8")
    ax.text(0.02, 0.60, "Optional community branch (ablations / variants)", fontsize=9.8, weight="bold", color="#475569")

    # Phase blocks common start and end
    add_box(
        ax,
        (0.06, 0.74),
        0.18,
        0.17,
        "Phase 0: Ingestion",
        "OGB/DGL inputs\nDelta nodes, edges, masks",
        "#DBEAFE",
        edgecolor="#1D4ED8",
        lw=1.6,
    )
    add_box(
        ax,
        (0.78, 0.74),
        0.18,
        0.17,
        "Phase 5: Reporting",
        "tables, figures,\nLaTeX artifacts",
        "#EDE9FE",
        edgecolor="#5B21B6",
        lw=1.4,
    )

    # Full-graph branch (highlighted)
    add_box(
        ax,
        (0.31, 0.77),
        0.19,
        0.14,
        "Phase 3.7",
        "cached hop propagation\n(full graph)",
        "#D1FAE5",
        edgecolor="#059669",
        lw=1.7,
    )
    add_box(
        ax,
        (0.55, 0.77),
        0.19,
        0.14,
        "Phase 3.8",
        "global classifier\n(edge-free training)",
        "#CCFBF1",
        edgecolor="#0F766E",
        lw=1.7,
    )

    # Community branch (optional)
    add_box(
        ax,
        (0.22, 0.50),
        0.16,
        0.14,
        "Phase 1",
        "LPA/Louvain\ncommunities",
        "#FEF3C7",
        edgecolor="#A16207",
    )
    add_box(
        ax,
        (0.41, 0.50),
        0.16,
        0.14,
        "Phase 2",
        "subgraphs +\nboundary tags",
        "#FFEDD5",
        edgecolor="#C2410C",
    )
    add_box(
        ax,
        (0.60, 0.50),
        0.16,
        0.14,
        "Phase 3 / 3b",
        "community GNN or\nCAAN variants",
        "#FCE7F3",
        edgecolor="#BE185D",
    )

    # Arrows, full-graph path
    add_arrow(ax, (0.24, 0.82), (0.31, 0.84), color="#1D4ED8", lw=1.9)
    add_arrow(ax, (0.50, 0.84), (0.55, 0.84), color="#1D4ED8", lw=1.9)
    add_arrow(ax, (0.74, 0.84), (0.78, 0.82), color="#1D4ED8", lw=1.9)

    # Arrows, community path
    add_arrow(ax, (0.24, 0.79), (0.24, 0.58), color="#64748B", lw=1.2)
    add_arrow(ax, (0.38, 0.57), (0.41, 0.57), color="#64748B", lw=1.2)
    add_arrow(ax, (0.57, 0.57), (0.60, 0.57), color="#64748B", lw=1.2)
    add_arrow(ax, (0.76, 0.57), (0.82, 0.74), color="#64748B", lw=1.2)

    # Branch legend callouts
    add_box(
        ax,
        (0.02, 0.46),
        0.17,
        0.12,
        "Interpretation",
        "This paper's scaling\nresults use Phase 3.7/3.8.",
        "#EFF6FF",
        edgecolor="#1E40AF",
        lw=1.1,
    )

    # Transaction/storage plane shared by both branches
    storage = FancyBboxPatch(
        (0.04, 0.12),
        0.92,
        0.24,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.3,
        edgecolor="#0F172A",
        facecolor="#F8FAFC",
    )
    ax.add_patch(storage)
    ax.text(0.50, 0.31, "Delta Lake Transactional Storage Plane", ha="center", va="center", fontsize=10.2, weight="bold")
    ax.text(
        0.50,
        0.22,
        "ACID isolation by experiment_id | checkpoint reuse via _delta_log | S3-backed immutable artifacts",
        ha="center",
        va="center",
        fontsize=8.4,
    )

    # Vertical connectors from phase blocks to storage plane
    for x, y in [
        (0.15, 0.74),
        (0.395, 0.77),
        (0.645, 0.77),
        (0.30, 0.50),
        (0.49, 0.50),
        (0.68, 0.50),
        (0.87, 0.74),
    ]:
        add_arrow(ax, (x, y), (x, 0.36), color="#334155", lw=1.1)

    fig.suptitle("EMO System Architecture: Full-Graph and Community Branches", fontsize=13.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
