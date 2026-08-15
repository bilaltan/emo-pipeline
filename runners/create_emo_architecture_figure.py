#!/usr/bin/env python3
"""
create_emo_architecture_figure.py
Generates clean, aesthetic, publication-grade conceptual and architectural figures:
  - Figure 1: Problem setting & tension (Distributed Bottleneck vs Naive Partition vs EMO CaaN Solution)
  - Figure 2: EMO End-to-End System Pipeline (Phase 0 -> 1 -> 2 -> 3 -> 3b + Bonus 3.7/3.8 Scaling)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT_DIR = Path("results/figures")
OVERLEAF_DIR = Path("overleaf/results/figures")

# ─── Refined Color Palette ───────────────────────────────────────────────────
CLR_CANVAS_BG   = "#F8FAFC"   # ultra-clean background
CLR_PRIMARY_HDR = "#0F172A"   # deep slate
CLR_CARD_FILL   = "#FFFFFF"   # crisp white
CLR_CARD_BORDER = "#CBD5E1"   # subtle slate border

CLR_RED_BG      = "#FEF2F2"   # light red tint (bottleneck)
CLR_RED_HDR     = "#991B1B"   # deep red
CLR_RED_BORDER  = "#F87171"   # red border

CLR_AMBER_BG    = "#FFFBEB"   # light amber tint (naive partition)
CLR_AMBER_HDR   = "#92400E"   # deep amber
CLR_AMBER_BORDER= "#FCD34D"   # amber border

CLR_BLUE_BG     = "#EFF6FF"   # light blue tint (EMO core pipeline)
CLR_BLUE_HDR    = "#1E40AF"   # royal blue
CLR_BLUE_BORDER = "#60A5FA"   # blue border

CLR_PURPLE_BG   = "#FAF5FF"   # light purple tint (CaaN recovery)
CLR_PURPLE_HDR  = "#6B21A8"   # deep purple
CLR_PURPLE_BORDER = "#A855F7" # purple border

CLR_GREEN_BG    = "#F0FDF4"   # light green tint (Delta Lake storage)
CLR_GREEN_HDR   = "#166534"   # forest green
CLR_GREEN_BORDER= "#4ADE80"   # green border

CLR_ARROW_FLOW  = "#2563EB"   # vibrant blue flow
CLR_ARROW_SEC   = "#9333EA"   # purple flow
CLR_TEXT        = "#0F172A"   # near black
CLR_MUTED       = "#475569"   # slate gray


def _draw_box(ax, xy, w, h, fc="#FFFFFF", ec="#333333", lw=1.2, radius=0.012, zorder=3):
    """Draws a smooth rounded rectangle."""
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def _draw_arrow(ax, start, end, color=CLR_ARROW_FLOW, lw=1.8, style="->",
                connectionstyle="arc3,rad=0.0", zorder=5):
    """Draws a vector arrow."""
    arr = FancyArrowPatch(start, end, arrowstyle=style,
                          connectionstyle=connectionstyle,
                          color=color, linewidth=lw,
                          mutation_scale=12, zorder=zorder)
    ax.add_patch(arr)
    return arr


def create_figure1_problem_and_solution():
    """Generates Figure 1: Problem Tension & Solution Overview."""
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title Banner
    ax.text(0.5, 0.96, "The Distributed GRL Dilemma and the EMO Relational Solution",
            ha="center", va="center", fontsize=13, fontweight="bold", color=CLR_PRIMARY_HDR)
    ax.text(0.5, 0.92, "Overcoming the communication bottleneck and boundary accuracy degradation via Lakehouse-native CaaN abstractions",
            ha="center", va="center", fontsize=8.5, color=CLR_MUTED)

    # Panel A: Synchronized Distributed GRL (Bottleneck)
    pa_x, pa_y, pa_w, pa_h = 0.03, 0.06, 0.28, 0.81
    _draw_box(ax, (pa_x, pa_y), pa_w, pa_h, fc=CLR_RED_BG, ec=CLR_RED_BORDER, lw=1.5, radius=0.015)
    _draw_box(ax, (pa_x + 0.015, pa_y + pa_h - 0.05), pa_w - 0.03, 0.038, fc=CLR_RED_HDR, ec=CLR_RED_HDR, radius=0.006)
    ax.text(pa_x + pa_w/2, pa_y + pa_h - 0.031, "(a) Synchronized Distributed GRL", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#FFFFFF")
    
    # Panel A Content Card
    _draw_box(ax, (pa_x + 0.015, pa_y + 0.02), pa_w - 0.03, pa_h - 0.09, fc=CLR_CARD_FILL, ec=CLR_RED_BORDER, lw=1.0)
    ax.text(pa_x + 0.03, pa_y + pa_h - 0.09, "• Full-graph state split across nodes\n• Heavy inter-worker RPC communication\n• Remote neighbor fetching every epoch\n• High cluster cost & complex GPU sync",
            ha="left", va="top", fontsize=8.0, color=CLR_TEXT, linespacing=1.6)
    
    # Graphic illustration inside Panel A
    ax.text(pa_x + pa_w/2, pa_y + 0.28, "Remote Parameter Servers", ha="center", fontsize=7.5, fontweight="bold", color=CLR_RED_HDR)
    _draw_box(ax, (pa_x + 0.04, pa_y + 0.12), 0.09, 0.10, fc="#FEE2E2", ec=CLR_RED_HDR, lw=1.0)
    _draw_box(ax, (pa_x + 0.15, pa_y + 0.12), 0.09, 0.10, fc="#FEE2E2", ec=CLR_RED_HDR, lw=1.0)
    ax.text(pa_x + 0.085, pa_y + 0.17, "Worker 1\n(Local)", ha="center", va="center", fontsize=6.8)
    ax.text(pa_x + 0.195, pa_y + 0.17, "Worker 2\n(Remote)", ha="center", va="center", fontsize=6.8)
    _draw_arrow(ax, (pa_x + 0.13, pa_y + 0.18), (pa_x + 0.15, pa_y + 0.18), color=CLR_RED_HDR, lw=1.8)
    _draw_arrow(ax, (pa_x + 0.15, pa_y + 0.15), (pa_x + 0.13, pa_y + 0.15), color=CLR_RED_HDR, lw=1.8)
    ax.text(pa_x + pa_w/2, pa_y + 0.05, "⚠️ Heavy Network RPCs: 28-46 GB", ha="center", fontsize=7.2, fontweight="bold", color=CLR_RED_HDR)

    # Panel B: Naive Partitioned GRL (Boundary Loss)
    pb_x, pb_y, pb_w, pb_h = 0.35, 0.06, 0.28, 0.81
    _draw_box(ax, (pb_x, pb_y), pb_w, pb_h, fc=CLR_AMBER_BG, ec=CLR_AMBER_BORDER, lw=1.5, radius=0.015)
    _draw_box(ax, (pb_x + 0.015, pb_y + pb_h - 0.05), pb_w - 0.03, 0.038, fc=CLR_AMBER_HDR, ec=CLR_AMBER_HDR, radius=0.006)
    ax.text(pb_x + pb_w/2, pb_y + pb_h - 0.031, "(b) Naive Community Partitioning", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#FFFFFF")
    
    # Panel B Content Card
    _draw_box(ax, (pb_x + 0.015, pb_y + 0.02), pb_w - 0.03, pb_h - 0.09, fc=CLR_CARD_FILL, ec=CLR_AMBER_BORDER, lw=1.0)
    ax.text(pb_x + 0.03, pb_y + pb_h - 0.09, "• Complete graph partitioning\n• 0.0 GB inter-worker network traffic\n• BUT severed boundary edges\n• Severe Boundary Accuracy Loss (-40%)\n• Minor community degeneracy",
            ha="left", va="top", fontsize=8.0, color=CLR_TEXT, linespacing=1.6)
    
    # Graphic illustration inside Panel B
    ax.text(pb_x + pb_w/2, pb_y + 0.28, "Severed Cross-Community Edges", ha="center", fontsize=7.5, fontweight="bold", color=CLR_AMBER_HDR)
    _draw_box(ax, (pb_x + 0.04, pb_y + 0.12), 0.09, 0.10, fc="#FEF3C7", ec=CLR_AMBER_HDR, lw=1.0)
    _draw_box(ax, (pb_x + 0.15, pb_y + 0.12), 0.09, 0.10, fc="#FEF3C7", ec=CLR_AMBER_HDR, lw=1.0)
    ax.text(pb_x + 0.085, pb_y + 0.17, "Comm A\n(Isolated)", ha="center", va="center", fontsize=6.8)
    ax.text(pb_x + 0.195, pb_y + 0.17, "Comm B\n(Isolated)", ha="center", va="center", fontsize=6.8)
    ax.text(pb_x + 0.14, pb_y + 0.17, "✂️", ha="center", va="center", fontsize=11)
    ax.text(pb_x + pb_w/2, pb_y + 0.05, "⚠️ Accuracy Drops from 94.5% → 52.9%", ha="center", fontsize=7.2, fontweight="bold", color=CLR_AMBER_HDR)

    # Panel C: EMO Relational CaaN Solution
    pc_x, pc_y, pc_w, pc_h = 0.67, 0.06, 0.30, 0.81
    _draw_box(ax, (pc_x, pc_y), pc_w, pc_h, fc=CLR_BLUE_BG, ec=CLR_BLUE_BORDER, lw=1.8, radius=0.015)
    _draw_box(ax, (pc_x + 0.015, pc_y + pc_h - 0.05), pc_w - 0.03, 0.038, fc=CLR_BLUE_HDR, ec=CLR_BLUE_HDR, radius=0.006)
    ax.text(pc_x + pc_w/2, pc_y + pc_h - 0.031, "(c) EMO: Partition-Centric + CaaN Recovery", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#FFFFFF")
    
    # Panel C Content Card
    _draw_box(ax, (pc_x + 0.015, pc_y + 0.02), pc_w - 0.03, pc_h - 0.09, fc=CLR_CARD_FILL, ec=CLR_BLUE_BORDER, lw=1.0)
    ax.text(pc_x + 0.03, pc_y + pc_h - 0.09, "• Transactional Delta Lake tables (ACID)\n• Decoupled PySpark UDF training\n• 0.0 GB inter-worker training traffic\n• Relational CaaN super-node broadcast\n• Restores 97.8% full-graph accuracy",
            ha="left", va="top", fontsize=8.0, color=CLR_TEXT, linespacing=1.6)
    
    # Graphic illustration inside Panel C
    ax.text(pc_x + pc_w/2, pc_y + 0.28, "Super-Node Context Broadcast", ha="center", fontsize=7.5, fontweight="bold", color=CLR_BLUE_HDR)
    _draw_box(ax, (pc_x + 0.035, pc_y + 0.12), 0.08, 0.10, fc=CLR_BLUE_BG, ec=CLR_BLUE_HDR, lw=1.0)
    _draw_box(ax, (pc_x + 0.185, pc_y + 0.12), 0.08, 0.10, fc=CLR_BLUE_BG, ec=CLR_BLUE_HDR, lw=1.0)
    _draw_box(ax, (pc_x + 0.118, pc_y + 0.135), 0.064, 0.07, fc=CLR_PURPLE_BG, ec=CLR_PURPLE_HDR, lw=1.2)
    ax.text(pc_x + 0.075, pa_y + 0.17, "Comm A", ha="center", va="center", fontsize=6.8)
    ax.text(pc_x + 0.225, pa_y + 0.17, "Comm B", ha="center", va="center", fontsize=6.8)
    ax.text(pc_x + 0.150, pa_y + 0.17, "Super\nNodes", ha="center", va="center", fontsize=6.0, fontweight="bold", color=CLR_PURPLE_HDR)
    _draw_arrow(ax, (pc_x + 0.15, pc_y + 0.21), (pc_x + 0.075, pc_y + 0.22), color=CLR_PURPLE_HDR, lw=1.2, connectionstyle="arc3,rad=-0.3")
    _draw_arrow(ax, (pc_x + 0.15, pc_y + 0.21), (pc_x + 0.225, pc_y + 0.22), color=CLR_PURPLE_HDR, lw=1.2, connectionstyle="arc3,rad=0.3")
    ax.text(pc_x + pc_w/2, pc_y + 0.05, "✓ 0.0 GB Network | Recovers 92.7% Acc", ha="center", fontsize=7.2, fontweight="bold", color=CLR_BLUE_HDR)

    fig.tight_layout()
    png_path = OUT_DIR / "emo_problem_and_solution_overview.png"
    pdf_path = OUT_DIR / "emo_problem_and_solution_overview.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    
    # Sync to overleaf
    OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OVERLEAF_DIR / "emo_problem_and_solution_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_problem_and_solution_overview.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Figure 1 (Problem & Solution Overview) generated successfully.")


def create_figure2_pipeline_architecture():
    """Generates Figure 2: EMO End-to-End System Pipeline Architecture."""
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Master Container
    _draw_box(ax, (0.02, 0.02), 0.96, 0.96, fc=CLR_CANVAS_BG, ec="#94A3B8", lw=1.5, radius=0.015)
    
    # Master Header
    ax.text(0.04, 0.945, "EMO Distributed Systems Architecture & Relational GRL Pipeline",
            fontsize=13, fontweight="bold", color=CLR_PRIMARY_HDR)
    ax.text(0.60, 0.945, "• Transactional Data Lake • Zero-Communication GNN Training • Relational CaaN Recovery",
            fontsize=8.0, color=CLR_MUTED)

    # -------------------------------------------------------------------------
    # TOP SECTION: CORE PARTITION-CENTRIC GRL PIPELINE (Phases 0 -> 1 -> 2 -> 3 + 3b)
    # -------------------------------------------------------------------------
    core_x, core_y, core_w, core_h = 0.035, 0.44, 0.93, 0.47
    _draw_box(ax, (core_x, core_y), core_w, core_h, fc="#FFFFFF", ec=CLR_BLUE_BORDER, lw=1.5, radius=0.012)
    _draw_box(ax, (core_x + 0.015, core_y + core_h - 0.035), 0.42, 0.026, fc=CLR_BLUE_HDR, ec=CLR_BLUE_HDR, radius=0.005)
    ax.text(core_x + 0.015 + 0.21, core_y + core_h - 0.022,
            "Core Pipeline: Partition-Centric Community GRL on Data Lakes (Phases 0–3b)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="#FFFFFF")

    # 5 Phase Cards in Core Pipeline
    card_w, card_h = 0.165, 0.36
    card_y = core_y + 0.03
    xs = [core_x + 0.015 + i * 0.183 for i in range(5)]

    # Card 0: Phase 0 Ingestion
    _draw_box(ax, (xs[0], card_y), card_w, card_h, fc=CLR_BLUE_BG, ec=CLR_BLUE_BORDER, lw=1.0)
    _draw_box(ax, (xs[0] + 0.008, card_y + card_h - 0.035), card_w - 0.016, 0.026, fc=CLR_BLUE_HDR, ec=CLR_BLUE_HDR, radius=0.004)
    ax.text(xs[0] + card_w/2, card_y + card_h - 0.022, "Phase 0: Delta Ingestion", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#FFFFFF")
    ax.text(xs[0] + 0.012, card_y + card_h - 0.055,
            "• Raw Graph Streaming\n• Symmetrization & Dedup\n• Bit-packed Hash Check\n• ACID Transaction Tables\n  nodes/, edges/, masks/",
            ha="left", va="top", fontsize=6.8, color=CLR_TEXT, linespacing=1.5)

    # Card 1: Phase 1 Community Detection
    _draw_box(ax, (xs[1], card_y), card_w, card_h, fc=CLR_BLUE_BG, ec=CLR_BLUE_BORDER, lw=1.0)
    _draw_box(ax, (xs[1] + 0.008, card_y + card_h - 0.035), card_w - 0.016, 0.026, fc=CLR_BLUE_HDR, ec=CLR_BLUE_HDR, radius=0.004)
    ax.text(xs[1] + card_w/2, card_y + card_h - 0.022, "Phase 1: Partitioning", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#FFFFFF")
    ax.text(xs[1] + 0.012, card_y + card_h - 0.055,
            "• Louvain Modularity (Default)\n  (Driver fast single-machine)\n• Distributed LPA (Fallback)\n• Minor Cluster Filter ($<K$)\n• Materialize to Delta\n  communities/{alg}/",
            ha="left", va="top", fontsize=6.8, color=CLR_TEXT, linespacing=1.5)

    # Card 2: Phase 2 Relational Boundary Extraction
    _draw_box(ax, (xs[2], card_y), card_w, card_h, fc=CLR_BLUE_BG, ec=CLR_BLUE_BORDER, lw=1.0)
    _draw_box(ax, (xs[2] + 0.008, card_y + card_h - 0.035), card_w - 0.016, 0.026, fc=CLR_BLUE_HDR, ec=CLR_BLUE_HDR, radius=0.004)
    ax.text(xs[2] + card_w/2, card_y + card_h - 0.022, "Phase 2: Relational SQL", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#FFFFFF")
    ax.text(xs[2] + 0.012, card_y + card_h - 0.055,
            "• Intra-Community Subgraphs\n• Relational Edge Joins\n• Degree Comparison:\n  deg_orig(v) > deg_intra(v)\n• Write Clustered Delta:\n  phase2_nodes, phase2_edges",
            ha="left", va="top", fontsize=6.8, color=CLR_TEXT, linespacing=1.5)

    # Card 3: Phase 3 Decoupled GNN Training
    _draw_box(ax, (xs[3], card_y), card_w, card_h, fc=CLR_BLUE_BG, ec=CLR_BLUE_BORDER, lw=1.0)
    _draw_box(ax, (xs[3] + 0.008, card_y + card_h - 0.035), card_w - 0.016, 0.026, fc=CLR_BLUE_HDR, ec=CLR_BLUE_HDR, radius=0.004)
    ax.text(xs[3] + card_w/2, card_y + card_h - 0.022, "Phase 3: Decoupled GNN", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#FFFFFF")
    ax.text(xs[3] + 0.012, card_y + card_h - 0.055,
            "• PySpark applyInPandas UDF\n• Local PyTorch Geometric\n• SAGE, GAT, GATv2, ARMA\n• ZERO Network Traffic\n• Node & Link Evaluation\n• High Link AUC (+18%)",
            ha="left", va="top", fontsize=6.8, color=CLR_TEXT, linespacing=1.5)

    # Card 4: Phase 3b CaaN Boundary Recovery
    _draw_box(ax, (xs[4], card_y), card_w, card_h, fc=CLR_PURPLE_BG, ec=CLR_PURPLE_BORDER, lw=1.3)
    _draw_box(ax, (xs[4] + 0.008, card_y + card_h - 0.035), card_w - 0.016, 0.026, fc=CLR_PURPLE_HDR, ec=CLR_PURPLE_HDR, radius=0.004)
    ax.text(xs[4] + card_w/2, card_y + card_h - 0.022, "Phase 3b: CaaN Recovery", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#FFFFFF")
    ax.text(xs[4] + 0.012, card_y + card_h - 0.055,
            "• Major Comm Centroid Contraction\n• Super-Node Feature Pooling\n• Broadcast Auxiliary Graph\n• Boundary Context Recovery\n• Restores 97.8% Node Acc\n• Zero Inter-worker Comm",
            ha="left", va="top", fontsize=6.8, color=CLR_TEXT, linespacing=1.5)

    # Flow arrows between Core Phase Cards
    for i in range(4):
        _draw_arrow(ax, (xs[i] + card_w + 0.002, card_y + card_h/2), (xs[i+1] - 0.002, card_y + card_h/2),
                    color=CLR_ARROW_FLOW if i < 3 else CLR_ARROW_SEC, lw=2.0)

    # -------------------------------------------------------------------------
    # MIDDLE SECTION: TRANSACTIONAL DATA LAKE PLANE (y in [0.25, 0.41])
    # -------------------------------------------------------------------------
    dl_x, dl_y, dl_w, dl_h = 0.035, 0.25, 0.93, 0.165
    _draw_box(ax, (dl_x, dl_y), dl_w, dl_h, fc=CLR_GREEN_BG, ec=CLR_GREEN_BORDER, lw=1.5, radius=0.012)
    _draw_box(ax, (dl_x + 0.015, dl_y + dl_h - 0.032), 0.38, 0.024, fc=CLR_GREEN_HDR, ec=CLR_GREEN_HDR, radius=0.005)
    ax.text(dl_x + 0.015 + 0.19, dl_y + dl_h - 0.020,
            "Transactional Delta Lake Storage Plane (Amazon S3 & ACID Logs)",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#FFFFFF")

    # 3 Storage Blocks
    sb_w, sb_h = 0.28, 0.095
    sb_y = dl_y + 0.02
    _draw_box(ax, (dl_x + 0.02, sb_y), sb_w, sb_h, fc="#FFFFFF", ec=CLR_GREEN_BORDER, lw=1.0)
    ax.text(dl_x + 0.03, sb_y + sb_h - 0.02, "ACID Isolation & Checkpoints", fontsize=7.2, fontweight="bold", color=CLR_GREEN_HDR)
    ax.text(dl_x + 0.03, sb_y + 0.015, "• Transaction logs (_delta_log/)\n• Phase-level deterministic checkpoint reuse\n• Strict namespace experiment isolation",
            fontsize=6.5, color=CLR_TEXT, linespacing=1.4)

    _draw_box(ax, (dl_x + 0.325, sb_y), sb_w, sb_h, fc="#FFFFFF", ec=CLR_GREEN_BORDER, lw=1.0)
    ax.text(dl_x + 0.335, sb_y + sb_h - 0.02, "Zero-Copy Arrow Serialization", fontsize=7.2, fontweight="bold", color=CLR_GREEN_HDR)
    ax.text(dl_x + 0.335, sb_y + 0.015, "• PyArrow RecordBatches in UDF\n• Bypasses slow Python pickle transfer\n• Direct JVM-to-PyTorch memory bridge",
            fontsize=6.5, color=CLR_TEXT, linespacing=1.4)

    _draw_box(ax, (dl_x + 0.63, sb_y), sb_w, sb_h, fc="#FFFFFF", ec=CLR_GREEN_BORDER, lw=1.0)
    ax.text(dl_x + 0.64, sb_y + sb_h - 0.02, "Cluster Stability Hardening", fontsize=7.2, fontweight="bold", color=CLR_GREEN_HDR)
    ax.text(dl_x + 0.64, sb_y + 0.015, "• 12GB off-heap memory headroom\n• Dynamic /mnt/tmp disk redirection\n• Automated worker bootstrap package sync",
            fontsize=6.5, color=CLR_TEXT, linespacing=1.4)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION: BONUS SCALING EXTENSION (Phases 3.7 & 3.8) (y in [0.04, 0.22])
    # -------------------------------------------------------------------------
    scale_x, scale_y, scale_w, scale_h = 0.035, 0.04, 0.93, 0.185
    _draw_box(ax, (scale_x, scale_y), scale_w, scale_h, fc="#F1F5F9", ec="#94A3B8", lw=1.2, radius=0.012)
    _draw_box(ax, (scale_x + 0.015, scale_y + scale_h - 0.032), 0.44, 0.024, fc="#334155", ec="#334155", radius=0.005)
    ax.text(scale_x + 0.015 + 0.22, scale_y + scale_h - 0.020,
            "Bonus Capability: Distributed Multi-Hop Feature Scaling (Phases 3.7 & 3.8)",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#FFFFFF")

    # 3 Sub-cards for Phase 3.7, Phase 3.8, and Phase 5 Reporting
    sc_w, sc_h = 0.28, 0.115
    sc_y = scale_y + 0.02
    _draw_box(ax, (scale_x + 0.02, sc_y), sc_w, sc_h, fc="#FFFFFF", ec="#94A3B8", lw=1.0)
    ax.text(scale_x + 0.03, sc_y + sc_h - 0.02, "Phase 3.7: SIGN Feature Propagation", fontsize=7.2, fontweight="bold", color="#334155")
    ax.text(scale_x + 0.03, sc_y + 0.015, "• Distributed K-hop sparse vector aggregation\n• Caches intermediate hops in Delta tables\n• Scales to 3.2B edges on CPU clusters",
            fontsize=6.5, color=CLR_TEXT, linespacing=1.4)

    _draw_box(ax, (scale_x + 0.325, sc_y), sc_w, sc_h, fc="#FFFFFF", ec="#94A3B8", lw=1.0)
    ax.text(scale_x + 0.335, sc_y + sc_h - 0.02, "Phase 3.8: Edge-Free Linear Probe", fontsize=7.2, fontweight="bold", color="#334155")
    ax.text(scale_x + 0.335, sc_y + 0.015, "• Distributed Spark ML LogisticRegression\n• Zero edge dependencies during model fit\n• Evaluates global feature representations",
            fontsize=6.5, color=CLR_TEXT, linespacing=1.4)

    _draw_box(ax, (scale_x + 0.63, sc_y), sc_w, sc_h, fc="#FFFFFF", ec="#94A3B8", lw=1.0)
    ax.text(scale_x + 0.64, sc_y + sc_h - 0.02, "Phase 5: Automated LaTeX Reporting", fontsize=7.2, fontweight="bold", color="#334155")
    ax.text(scale_x + 0.64, sc_y + 0.015, "• Multi-class Accuracy & ROC-AUC Metrics\n• Phase-by-phase execution timeline logs\n• Direct S3 sync and LaTeX table generation",
            fontsize=6.5, color=CLR_TEXT, linespacing=1.4)

    _draw_arrow(ax, (scale_x + 0.02 + sc_w + 0.005, sc_y + sc_h/2), (scale_x + 0.325 - 0.005, sc_y + sc_h/2), color="#475569", lw=1.5)
    _draw_arrow(ax, (scale_x + 0.325 + sc_w + 0.005, sc_y + sc_h/2), (scale_x + 0.63 - 0.005, sc_y + sc_h/2), color="#475569", lw=1.5)

    fig.tight_layout()
    png_path = OUT_DIR / "emo_system_architecture_overview.png"
    pdf_path = OUT_DIR / "emo_system_architecture_overview.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    # Sync to overleaf
    fig.savefig(OVERLEAF_DIR / "emo_system_architecture_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_system_architecture_overview.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Figure 2 (System Architecture Overview) generated successfully.")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    create_figure1_problem_and_solution()
    create_figure2_pipeline_architecture()
