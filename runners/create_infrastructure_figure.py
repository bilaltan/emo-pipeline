#!/usr/bin/env python3
"""
create_infrastructure_figure.py
Generates a clean, streamlined, publication-grade infrastructure architecture diagram (Figure 2)
with minimal clutter, clear hierarchy, and elegant data-flow arrows.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path("results/figures")
OVERLEAF_DIR = Path("overleaf/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

CLR_BG          = "#FFFFFF"
CLR_DARK        = "#0F172A"   # Slate 900
CLR_MUTED       = "#64748B"   # Slate 500
CLR_BORDER      = "#CBD5E1"   # Slate 300

CLR_EMR_BG      = "#F1F5F9"
CLR_EMR_HDR     = "#1E293B"

CLR_DRIVER_BG   = "#EFF6FF"
CLR_DRIVER_HDR  = "#1D4ED8"
CLR_DRIVER_BORD = "#93C5FD"

CLR_WORKER_BG   = "#F8FAFC"
CLR_WORKER_HDR  = "#0F766E"
CLR_WORKER_BORD = "#99F6E4"

CLR_EXEC_BG     = "#FFFFFF"
CLR_EXEC_BORD   = "#CBD5E1"

CLR_S3_BG       = "#F0FDF4"
CLR_S3_HDR      = "#15803D"
CLR_S3_BORD     = "#86EFAC"


def _draw_box(ax, xy, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.2, radius=0.010, zorder=3):
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def _draw_arrow(ax, start, end, color="#2563EB", lw=1.6, style="->",
                connectionstyle="arc3,rad=0.0", zorder=5):
    arr = FancyArrowPatch(start, end, arrowstyle=style,
                          connectionstyle=connectionstyle,
                          color=color, linewidth=lw,
                          mutation_scale=11, zorder=zorder)
    ax.add_patch(arr)
    return arr


def create_figure2_infrastructure():
    fig, ax = plt.subplots(figsize=(14.5, 5.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Master EMR Container (Top 65% of canvas)
    emr_x, emr_y, emr_w, emr_h = 0.02, 0.36, 0.96, 0.60
    _draw_box(ax, (emr_x, emr_y), emr_w, emr_h, fc=CLR_EMR_BG, ec="#94A3B8", lw=1.4, radius=0.012)
    ax.text(emr_x + 0.02, emr_y + emr_h - 0.045, "AWS EMR Elastic Compute Plane (Spark on YARN)",
            fontsize=11.5, fontweight="bold", color=CLR_EMR_HDR)
    ax.text(emr_x + 0.48, emr_y + emr_h - 0.045, "• r6id.8xlarge Nodes (32 vCPUs, 256 GB RAM, NVMe SSDs) • Zero-Copy Arrow",
            fontsize=7.8, color=CLR_MUTED)

    # ── Driver Node (Left Box) ────────────────────────────────────────────────
    dx, dy, dw, dh = 0.04, 0.40, 0.26, 0.48
    _draw_box(ax, (dx, dy), dw, dh, fc=CLR_DRIVER_BG, ec=CLR_DRIVER_BORD, lw=1.3, radius=0.008)
    _draw_box(ax, (dx + 0.015, dy + dh - 0.075), dw - 0.03, 0.055, fc=CLR_DRIVER_HDR, ec=CLR_DRIVER_HDR, radius=0.005)
    ax.text(dx + dw/2, dy + dh - 0.047, "Spark Master / Driver Node",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF")

    ax.text(dx + 0.025, dy + dh - 0.11,
            "• Orchestrates Pipeline Controller\n• Louvain Modularity Partitioning\n• CaaN Super-Node Compression\n• Broadcasts Auxiliary Context",
            ha="left", va="top", fontsize=7.6, color=CLR_DARK, linespacing=1.6)

    # ── Worker Nodes Container (Right Box) ────────────────────────────────────
    wx, wy, ww, wh = 0.33, 0.40, 0.63, 0.48
    _draw_box(ax, (wx, wy), ww, wh, fc=CLR_WORKER_BG, ec=CLR_WORKER_BORD, lw=1.3, radius=0.008)
    _draw_box(ax, (wx + 0.015, wy + wh - 0.075), 0.36, 0.055, fc=CLR_WORKER_HDR, ec=CLR_WORKER_HDR, radius=0.005)
    ax.text(wx + 0.015 + 0.18, wy + wh - 0.047, "Worker Instances (1 to N Workers)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF")

    # 2 Clean Executor Cards inside Worker container
    ew, eh = 0.28, 0.35
    e1_x = wx + 0.02
    e2_x = wx + 0.32
    ey = wy + 0.035

    # Executor 1
    _draw_box(ax, (e1_x, ey), ew, eh, fc=CLR_EXEC_BG, ec=CLR_BORDER, lw=1.0, radius=0.006)
    ax.text(e1_x + ew/2, ey + eh - 0.040, "Spark Executor Container 1",
            ha="center", fontsize=7.8, fontweight="bold", color=CLR_DARK)
    ax.text(e1_x + 0.02, ey + eh - 0.080,
            "• 28 GB JVM Heap (Spark SQL)\n• 12 GB Off-Heap (PyTorch)\n• PyArrow Zero-Copy Sharing\n• NVMe Storage (/mnt/tmp)",
            ha="left", va="top", fontsize=7.0, color=CLR_MUTED, linespacing=1.45)

    # Executor 2
    _draw_box(ax, (e2_x, ey), ew, eh, fc=CLR_EXEC_BG, ec=CLR_BORDER, lw=1.0, radius=0.006)
    ax.text(e2_x + ew/2, ey + eh - 0.040, "Spark Executor Container 2",
            ha="center", fontsize=7.8, fontweight="bold", color=CLR_DARK)
    ax.text(e2_x + 0.02, ey + eh - 0.080,
            "• 28 GB JVM Heap (Spark SQL)\n• 12 GB Off-Heap (PyTorch)\n• PyArrow Zero-Copy Sharing\n• NVMe Storage (/mnt/tmp)",
            ha="left", va="top", fontsize=7.0, color=CLR_MUTED, linespacing=1.45)

    # Dispatch arrow from Driver to Workers
    _draw_arrow(ax, (dx + dw + 0.005, dy + dh/2), (wx - 0.005, dy + dh/2),
                color=CLR_DRIVER_HDR, lw=2.0)
    ax.text((dx + dw + wx)/2, dy + dh/2 + 0.035, "Broadcast\nTasks", ha="center", fontsize=6.8, fontweight="bold", color=CLR_DRIVER_HDR)

    # ── Amazon S3 Storage Plane (Bottom Container) ───────────────────────────
    s3_x, s3_y, s3_w, s3_h = 0.02, 0.05, 0.96, 0.25
    _draw_box(ax, (s3_x, s3_y), s3_w, s3_h, fc=CLR_S3_BG, ec=CLR_S3_BORD, lw=1.4, radius=0.012)
    _draw_box(ax, (s3_x + 0.015, s3_y + s3_h - 0.065), 0.38, 0.050, fc=CLR_S3_HDR, ec=CLR_S3_HDR, radius=0.005)
    ax.text(s3_x + 0.015 + 0.19, s3_y + s3_h - 0.040, "Amazon S3 Transactional Storage Plane",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#FFFFFF")

    s3_cards = [
        ("Raw Tables", "nodes/, edges/, masks/"),
        ("Community Map", "communities/{alg}/"),
        ("Subgraphs", "phase2_nodes, edges"),
        ("CaaN Macro-Graph", "supernodes, minornodes"),
        ("ACID Commit Log", "_delta_log/ (Checkpoints)")
    ]
    sw = 0.170
    s_xs = [s3_x + 0.015 + i * 0.186 for i in range(5)]
    for i, (stitle, sdesc) in enumerate(s3_cards):
        _draw_box(ax, (s_xs[i], s3_y + 0.025), sw, 0.13, fc="#FFFFFF", ec=CLR_S3_BORD, lw=0.9, radius=0.006)
        ax.text(s_xs[i] + sw/2, s3_y + 0.105, stitle, ha="center", fontsize=7.4, fontweight="bold", color=CLR_S3_HDR)
        ax.text(s_xs[i] + sw/2, s3_y + 0.055, sdesc, ha="center", fontsize=6.3, color=CLR_MUTED)

    # Clean bi-directional arrows between Compute and S3 Storage
    _draw_arrow(ax, (dx + dw/2, dy), (s_xs[0] + sw/2, s3_y + s3_h), color=CLR_DRIVER_HDR, lw=1.5, style="<->")
    _draw_arrow(ax, (wx + ww/4, wy), (s_xs[2] + sw/2, s3_y + s3_h), color=CLR_WORKER_HDR, lw=1.5, style="<->")
    _draw_arrow(ax, (wx + 3*ww/4, wy), (s_xs[3] + sw/2, s3_y + s3_h), color=CLR_WORKER_HDR, lw=1.5, style="<->")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "emo_infrastructure_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "emo_infrastructure_architecture.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_infrastructure_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_infrastructure_architecture.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Figure 2 (Infrastructure Architecture) generated cleanly with reduced detail.")


if __name__ == "__main__":
    create_figure2_infrastructure()
