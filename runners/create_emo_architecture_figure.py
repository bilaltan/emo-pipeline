#!/usr/bin/env python3
"""Generate a publication-grade EMO architecture + contribution mapping figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("results/figures")
PNG_PATH = OUT_DIR / "emo_system_architecture_overview.png"
PDF_PATH = OUT_DIR / "emo_system_architecture_overview.pdf"

INK = "#1F2937"
ACCENT = "#1E6091"
LIGHT = "#F8FAFC"
MID = "#E2E8F0"


def add_box(ax, xy, wh, title, body, face=LIGHT, edge=INK, lw=1.1, title_size=9.0, body_size=7.9):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=title_size, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=body_size, color=INK)


def add_arrow(ax, start, end, color=INK, lw=1.2, style="-|>", scale=12):
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

    fig, ax = plt.subplots(figsize=(13.8, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer frame for a cleaner academic look.
    add_box(
        ax,
        (0.015, 0.04),
        (0.97, 0.92),
        "",
        "",
        face="#FFFFFF",
        edge="#CBD5E1",
        lw=1.0,
        title_size=1,
        body_size=1,
    )

    ax.text(0.035, 0.94, "EMO System Architecture and Contribution Mapping", fontsize=13, weight="bold", color=INK)
    ax.text(0.035, 0.905, "Primary scaling path (Phase 3.7/3.8) and optional community branch under a shared transactional storage contract", fontsize=8.6, color="#475569")

    # Branch labels
    ax.text(0.04, 0.79, "Branch A: Primary evaluation path", fontsize=8.7, color=ACCENT, weight="bold")
    ax.text(0.04, 0.56, "Branch B: Ablation and variant path", fontsize=8.5, color="#64748B", weight="bold")

    # Branch A blocks
    add_box(ax, (0.05, 0.66), (0.16, 0.11), "Phase 0", "Ingestion\n(nodes, edges, masks)", face="#F1F7FC", edge=ACCENT, lw=1.2)
    add_box(ax, (0.26, 0.66), (0.16, 0.11), "Phase 3.7", "Two-hop propagation\nfeature caching", face="#F1F7FC", edge=ACCENT, lw=1.2)
    add_box(ax, (0.47, 0.66), (0.16, 0.11), "Phase 3.8", "Global classifier\n(edge-free)", face="#F1F7FC", edge=ACCENT, lw=1.2)
    add_box(ax, (0.68, 0.66), (0.16, 0.11), "Phase 5", "Reporting\n(tables, figures)", face="#F1F7FC", edge=ACCENT, lw=1.2)
    add_arrow(ax, (0.21, 0.715), (0.26, 0.715), color=ACCENT, lw=1.4)
    add_arrow(ax, (0.42, 0.715), (0.47, 0.715), color=ACCENT, lw=1.4)
    add_arrow(ax, (0.63, 0.715), (0.68, 0.715), color=ACCENT, lw=1.4)

    # Branch B blocks
    add_box(ax, (0.10, 0.45), (0.15, 0.10), "Phase 1", "LPA / Louvain", face="#F8FAFC", edge="#64748B")
    add_box(ax, (0.30, 0.45), (0.15, 0.10), "Phase 2", "Boundary +\nsubgraph isolation", face="#F8FAFC", edge="#64748B")
    add_box(ax, (0.50, 0.45), (0.15, 0.10), "Phase 3 / 3b", "Community models\n(+ CAAN variants)", face="#F8FAFC", edge="#64748B")
    add_arrow(ax, (0.25, 0.50), (0.30, 0.50), color="#64748B", lw=1.1)
    add_arrow(ax, (0.45, 0.50), (0.50, 0.50), color="#64748B", lw=1.1)

    # Shared transactional plane
    add_box(
        ax,
        (0.05, 0.19),
        (0.79, 0.16),
        "Shared Delta Lake Transactional Plane",
        "ACID experiment isolation | checkpoint reuse via _delta_log | versioned artifacts on S3",
        face="#F8FAFC",
        edge=INK,
        lw=1.2,
        title_size=9.4,
    )

    # Connectors to shared plane
    for x in [0.13, 0.34, 0.55, 0.76, 0.175, 0.375, 0.575]:
        add_arrow(ax, (x, 0.45 if x in [0.175, 0.375, 0.575] else 0.66), (x, 0.35), color="#475569", lw=0.95, style="->", scale=9)

    # Contribution mapping sidebar
    add_box(ax, (0.86, 0.19), (0.11, 0.58), "Contributions", "", face="#FFFFFF", edge="#94A3B8", lw=1.1, title_size=9.0, body_size=1)

    add_box(ax, (0.872, 0.60), (0.086, 0.12), "C1", "Transactional\nreproducibility", face="#EEF6FD", edge=ACCENT, lw=1.0, title_size=9.2)
    add_box(ax, (0.872, 0.44), (0.086, 0.12), "C2", "Relational graph\noperators", face="#EEF6FD", edge=ACCENT, lw=1.0, title_size=9.2)
    add_box(ax, (0.872, 0.28), (0.086, 0.12), "C3", "Communication-efficient\nscaling path", face="#EEF6FD", edge=ACCENT, lw=1.0, title_size=9.2)

    # Mapping arrows from components to contributions
    add_arrow(ax, (0.84, 0.27), (0.872, 0.66), color=ACCENT, lw=1.0, style="->", scale=9)
    add_arrow(ax, (0.64, 0.50), (0.872, 0.50), color=ACCENT, lw=1.0, style="->", scale=9)
    add_arrow(ax, (0.70, 0.72), (0.872, 0.34), color=ACCENT, lw=1.0, style="->", scale=9)

    # Legend
    ax.text(0.05, 0.125, "Line style: solid arrows = execution flow, thin arrows = storage contract linkage", fontsize=7.3, color="#64748B")
    ax.text(0.05, 0.105, "Primary branch in blue; optional branch in neutral gray.", fontsize=7.3, color="#64748B")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
