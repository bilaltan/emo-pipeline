#!/usr/bin/env bash
#!/usr/bin/env python3
"""
create_emo_architecture_figure.py
Generates clean, elegant, publication-grade conceptual and architectural figures:
  - Figure 1: Problem Tension & Solution (DistDGL vs Naive Partition vs EMO)
  - Figure 3: EMO End-to-End System Pipeline (Simplified, elegant, minimal text)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

OUT_DIR = Path("results/figures")
OVERLEAF_DIR = Path("overleaf/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

# ─── Elegant Academic Palette ────────────────────────────────────────────────
CLR_BG          = "#FFFFFF"
CLR_DARK        = "#0F172A"   # Slate 900
CLR_MUTED       = "#64748B"   # Slate 500
CLR_BORDER      = "#CBD5E1"   # Slate 300

# Panel colors
CLR_RED_BG      = "#FEF2F2"
CLR_RED_HDR     = "#B91C1C"
CLR_RED_BORDER  = "#FCA5A5"

CLR_AMBER_BG    = "#FFFBEB"
CLR_AMBER_HDR   = "#B45309"
CLR_AMBER_BORDER= "#FCD34D"

CLR_BLUE_BG     = "#F0F7FF"
CLR_BLUE_HDR    = "#1D4ED8"
CLR_BLUE_BORDER = "#93C5FD"

CLR_PURPLE_BG   = "#FAF5FF"
CLR_PURPLE_HDR  = "#7E22CE"
CLR_PURPLE_BORDER = "#D8B4FE"

CLR_GREEN_BG    = "#F0FDF4"
CLR_GREEN_HDR   = "#15803D"
CLR_GREEN_BORDER= "#86EFAC"


def _draw_box(ax, xy, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.2, radius=0.012, zorder=3):
    """Draws a smooth rounded rectangle."""
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def _draw_arrow(ax, start, end, color="#2563EB", lw=1.6, style="->",
                connectionstyle="arc3,rad=0.0", zorder=5):
    """Draws a crisp vector arrow."""
    arr = FancyArrowPatch(start, end, arrowstyle=style,
                          connectionstyle=connectionstyle,
                          color=color, linewidth=lw,
                          mutation_scale=11, zorder=zorder)
    ax.add_patch(arr)
    return arr


def _draw_graph_cluster(ax, center, num_nodes=4, radius=0.035, color="#3B82F6", zorder=4):
    """Draws a mini graph cluster with nodes and connecting edges."""
    import numpy as np
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
    pts = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
    
    # Edges
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=color, lw=1.0, alpha=0.6, zorder=zorder)
    # Nodes
    for px, py in pts:
        c = Circle((px, py), 0.007, facecolor=color, edgecolor="#FFFFFF", lw=0.8, zorder=zorder+1)
        ax.add_patch(c)
    return pts


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Redesigned Problem Dilemma & EMO Relational Solution
# ══════════════════════════════════════════════════════════════════════════════
def create_figure1_problem_and_solution():
    fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pw, ph = 0.30, 0.90
    y_pos = 0.05

    # ── Panel A: State of the Art (Synchronized GNN) ──────────────────────────
    xa = 0.02
    _draw_box(ax, (xa, y_pos), pw, ph, fc=CLR_RED_BG, ec=CLR_RED_BORDER, lw=1.4, radius=0.015)
    _draw_box(ax, (xa + 0.015, y_pos + ph - 0.075), pw - 0.03, 0.055, fc=CLR_RED_HDR, ec=CLR_RED_HDR, radius=0.006)
    ax.text(xa + pw/2, y_pos + ph - 0.047, "(a) Synchronized Distributed GRL",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#FFFFFF")

    # Panel A Graphic: Two workers exchanging heavy RPCs
    _draw_box(ax, (xa + 0.025, y_pos + 0.38), 0.11, 0.38, fc="#FFFFFF", ec=CLR_RED_BORDER, lw=1.0)
    _draw_box(ax, (xa + 0.165, y_pos + 0.38), 0.11, 0.38, fc="#FFFFFF", ec=CLR_RED_BORDER, lw=1.0)
    ax.text(xa + 0.08, y_pos + 0.71, "Worker 1\n(Partition A)", ha="center", fontsize=7.5, fontweight="bold", color=CLR_DARK)
    ax.text(xa + 0.22, y_pos + 0.71, "Worker 2\n(Partition B)", ha="center", fontsize=7.5, fontweight="bold", color=CLR_DARK)

    _draw_graph_cluster(ax, (xa + 0.08, y_pos + 0.50), num_nodes=4, color="#EF4444")
    _draw_graph_cluster(ax, (xa + 0.22, y_pos + 0.50), num_nodes=4, color="#EF4444")

    # Cross-worker communication arrows
    _draw_arrow(ax, (xa + 0.135, y_pos + 0.53), (xa + 0.165, y_pos + 0.53), color=CLR_RED_HDR, lw=2.0)
    _draw_arrow(ax, (xa + 0.165, y_pos + 0.47), (xa + 0.135, y_pos + 0.47), color=CLR_RED_HDR, lw=2.0)
    ax.text(xa + 0.15, y_pos + 0.57, "Remote RPCs\n(Every Epoch)", ha="center", fontsize=6.8, fontweight="bold", color=CLR_RED_HDR)

    # Panel A Summary Card
    _draw_box(ax, (xa + 0.02, y_pos + 0.03), pw - 0.04, 0.30, fc="#FFFFFF", ec=CLR_BORDER, lw=0.8)
    ax.text(xa + 0.035, y_pos + 0.27, "• Full-graph state split across cluster\n• Heavy inter-worker RPC sync (28-46 GB)\n• High network communication overhead\n• Costly GPU / parameter server clusters",
            ha="left", va="top", fontsize=7.4, color=CLR_DARK, linespacing=1.45)

    # ── Panel B: Naive Partitioning (Boundary Collapse) ──────────────────────
    xb = 0.35
    _draw_box(ax, (xb, y_pos), pw, ph, fc=CLR_AMBER_BG, ec=CLR_AMBER_BORDER, lw=1.4, radius=0.015)
    _draw_box(ax, (xb + 0.015, y_pos + ph - 0.075), pw - 0.03, 0.055, fc=CLR_AMBER_HDR, ec=CLR_AMBER_HDR, radius=0.006)
    ax.text(xb + pw/2, y_pos + ph - 0.047, "(b) Naive Community Partitioning",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#FFFFFF")

    # Panel B Graphic: Severed boundary edges
    _draw_box(ax, (xb + 0.025, y_pos + 0.38), 0.11, 0.38, fc="#FFFFFF", ec=CLR_AMBER_BORDER, lw=1.0)
    _draw_box(ax, (xb + 0.165, y_pos + 0.38), 0.11, 0.38, fc="#FFFFFF", ec=CLR_AMBER_BORDER, lw=1.0)
    ax.text(xb + 0.08, y_pos + 0.71, "Community A\n(Isolated)", ha="center", fontsize=7.5, fontweight="bold", color=CLR_DARK)
    ax.text(xb + 0.22, y_pos + 0.71, "Community B\n(Isolated)", ha="center", fontsize=7.5, fontweight="bold", color=CLR_DARK)

    _draw_graph_cluster(ax, (xb + 0.08, y_pos + 0.50), num_nodes=4, color="#F59E0B")
    _draw_graph_cluster(ax, (xb + 0.22, y_pos + 0.50), num_nodes=4, color="#F59E0B")

    # Severed barrier
    ax.plot([xb + 0.148, xb + 0.148], [y_pos + 0.42, y_pos + 0.62], color=CLR_AMBER_HDR, lw=2.0, linestyle="--")
    ax.text(xb + 0.148, y_pos + 0.50, "✕", ha="center", va="center", fontsize=11, color=CLR_AMBER_HDR, fontweight="bold")
    ax.text(xb + 0.148, y_pos + 0.58, "Severed\nEdges", ha="center", fontsize=6.8, fontweight="bold", color=CLR_AMBER_HDR)

    # Panel B Summary Card
    _draw_box(ax, (xb + 0.02, y_pos + 0.03), pw - 0.04, 0.30, fc="#FFFFFF", ec=CLR_BORDER, lw=0.8)
    ax.text(xb + 0.035, y_pos + 0.27, "• Isolated training on subgraphs\n• 0.0 GB inter-worker network traffic\n• Boundary nodes lose external context\n• Severe accuracy drop (95% → 74%)",
            ha="left", va="top", fontsize=7.4, color=CLR_DARK, linespacing=1.45)

    # ── Panel C: EMO Relational Solution (CaaN Broadcast) ────────────────────
    xc = 0.68
    _draw_box(ax, (xc, y_pos), pw, ph, fc=CLR_BLUE_BG, ec=CLR_BLUE_BORDER, lw=1.6, radius=0.015)
    _draw_box(ax, (xc + 0.015, y_pos + ph - 0.075), pw - 0.03, 0.055, fc=CLR_BLUE_HDR, ec=CLR_BLUE_HDR, radius=0.006)
    ax.text(xc + pw/2, y_pos + ph - 0.047, "(c) EMO: Relational CaaN Solution",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#FFFFFF")

    # Panel C Graphic: Super-node broadcast
    _draw_box(ax, (xc + 0.025, y_pos + 0.38), 0.105, 0.38, fc="#FFFFFF", ec=CLR_BLUE_BORDER, lw=1.0)
    _draw_box(ax, (xc + 0.170, y_pos + 0.38), 0.105, 0.38, fc="#FFFFFF", ec=CLR_BLUE_BORDER, lw=1.0)
    _draw_box(ax, (xc + 0.110, y_pos + 0.44), 0.080, 0.22, fc=CLR_PURPLE_BG, ec=CLR_PURPLE_BORDER, lw=1.2)

    ax.text(xc + 0.077, y_pos + 0.71, "Worker 1\n(Local)", ha="center", fontsize=7.2, fontweight="bold", color=CLR_DARK)
    ax.text(xc + 0.223, y_pos + 0.71, "Worker 2\n(Local)", ha="center", fontsize=7.2, fontweight="bold", color=CLR_DARK)
    ax.text(xc + 0.150, y_pos + 0.61, "CaaN Super\nNodes (S, M)", ha="center", fontsize=6.8, fontweight="bold", color=CLR_PURPLE_HDR)

    _draw_graph_cluster(ax, (xc + 0.077, y_pos + 0.48), num_nodes=4, color="#3B82F6")
    _draw_graph_cluster(ax, (xc + 0.223, y_pos + 0.48), num_nodes=4, color="#3B82F6")
    
    # Broadcast arrows from super-nodes
    _draw_arrow(ax, (xc + 0.12, y_pos + 0.50), (xc + 0.09, y_pos + 0.54), color=CLR_PURPLE_HDR, lw=1.5, connectionstyle="arc3,rad=0.2")
    _draw_arrow(ax, (xc + 0.18, y_pos + 0.50), (xc + 0.21, y_pos + 0.54), color=CLR_PURPLE_HDR, lw=1.5, connectionstyle="arc3,rad=-0.2")

    # Panel C Summary Card
    _draw_box(ax, (xc + 0.02, y_pos + 0.03), pw - 0.04, 0.30, fc="#FFFFFF", ec=CLR_BORDER, lw=0.8)
    ax.text(xc + 0.035, y_pos + 0.27, "• Transactional Delta Lake on Amazon S3\n• 0.0 GB training network traffic\n• Relational CaaN super-node broadcast\n• Restores 98%+ full-graph accuracy",
            ha="left", va="top", fontsize=7.4, color=CLR_DARK, linespacing=1.45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "emo_problem_and_solution_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "emo_problem_and_solution_overview.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_problem_and_solution_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_problem_and_solution_overview.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Figure 1 (Problem & Solution Overview) generated cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Streamlined EMO System Pipeline (Less detail, clean flow)
# ══════════════════════════════════════════════════════════════════════════════
def create_figure3_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(14.5, 4.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Master Box
    _draw_box(ax, (0.01, 0.02), 0.98, 0.96, fc="#F8FAFC", ec="#CBD5E1", lw=1.2, radius=0.012)
    ax.text(0.03, 0.93, "EMO End-to-End System Pipeline Architecture", fontsize=11, fontweight="bold", color=CLR_DARK)
    ax.text(0.40, 0.93, "Declarative relational graph representation learning over transactional cloud data lakes", fontsize=8.0, color=CLR_MUTED)

    # ── Top Row: 5 Clean Phase Cards ──────────────────────────────────────────
    card_w, card_h = 0.176, 0.50
    card_y = 0.36
    xs = [0.03 + i * 0.192 for i in range(5)]

    phases_info = [
        ("Phase 0: Delta Ingestion", CLR_BLUE_BG, CLR_BLUE_HDR, CLR_BLUE_BORDER,
         "• Symmetrize & dedup\n• Checkpoint to S3\n• ACID table commit:\n  nodes/, edges/"),
        
        ("Phase 1: Partitioning", CLR_BLUE_BG, CLR_BLUE_HDR, CLR_BLUE_BORDER,
         "• Louvain modularity\n• Single-driver execution\n• Balanced clusters\n  communities/{alg}/"),
        
        ("Phase 2: Relational SQL", CLR_BLUE_BG, CLR_BLUE_HDR, CLR_BLUE_BORDER,
         "• Spark SQL joins\n• Boundary extraction\n• Clustered subgraphs:\n  phase2_nodes, edges"),
        
        ("Phase 3: Decoupled GNN", CLR_BLUE_BG, CLR_BLUE_HDR, CLR_BLUE_BORDER,
         "• PySpark applyInPandas\n• Local PyTorch training\n• 0.0 GB network traffic\n• Node & link evaluation"),
        
        ("Phase 3b: CaaN Recovery", CLR_PURPLE_BG, CLR_PURPLE_HDR, CLR_PURPLE_BORDER,
         "• Centroid super-nodes\n• Broadcast auxiliary set\n• Boundary GNN update\n• Recovers 98%+ accuracy")
    ]

    for i, (title, bg, hdr, border, desc) in enumerate(phases_info):
        _draw_box(ax, (xs[i], card_y), card_w, card_h, fc=bg, ec=border, lw=1.2, radius=0.010)
        _draw_box(ax, (xs[i] + 0.008, card_y + card_h - 0.075), card_w - 0.016, 0.055, fc=hdr, ec=hdr, radius=0.005)
        ax.text(xs[i] + card_w/2, card_y + card_h - 0.047, title, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color="#FFFFFF")
        ax.text(xs[i] + 0.015, card_y + card_h - 0.11, desc, ha="left", va="top",
                fontsize=7.3, color=CLR_DARK, linespacing=1.5)

    # Connecting Flow Arrows
    for i in range(4):
        _draw_arrow(ax, (xs[i] + card_w + 0.002, card_y + card_h/2), (xs[i+1] - 0.002, card_y + card_h/2),
                    color="#2563EB" if i < 3 else "#7E22CE", lw=2.0)

    # ── Bottom Row: Transactional Storage Plane (Delta Lake on Amazon S3) ────
    dl_x, dl_y, dl_w, dl_h = 0.03, 0.06, 0.94, 0.24
    _draw_box(ax, (dl_x, dl_y), dl_w, dl_h, fc=CLR_GREEN_BG, ec=CLR_GREEN_BORDER, lw=1.4, radius=0.010)
    _draw_box(ax, (dl_x + 0.015, dl_y + dl_h - 0.06), 0.32, 0.045, fc=CLR_GREEN_HDR, ec=CLR_GREEN_HDR, radius=0.005)
    ax.text(dl_x + 0.015 + 0.16, dl_y + dl_h - 0.037, "Transactional Data Lake Plane (Delta Lake on Amazon S3)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#FFFFFF")

    tables = [
        ("Raw Graph Tables", "nodes/, edges/, masks/"),
        ("Community Mapping", "communities/{alg}/"),
        ("Partitioned Subgraphs", "phase2_nodes, phase2_edges"),
        ("Auxiliary Macro-Graph", "caan_supernodes, minornodes"),
        ("Scaling Feature Cache", "hop_1/, hop_2/, features_k2/")
    ]
    tw = 0.170
    txs = [dl_x + 0.015 + i * 0.185 for i in range(5)]
    for i, (tbl_title, tbl_sub) in enumerate(tables):
        _draw_box(ax, (txs[i], dl_y + 0.025), tw, 0.13, fc="#FFFFFF", ec=CLR_GREEN_BORDER, lw=0.9, radius=0.006)
        ax.text(txs[i] + tw/2, dl_y + 0.105, tbl_title, ha="center", fontsize=7.2, fontweight="bold", color=CLR_GREEN_HDR)
        ax.text(txs[i] + tw/2, dl_y + 0.055, tbl_sub, ha="center", fontsize=6.3, color=CLR_MUTED)
        
        # Bidirectional arrows to top phase cards
        _draw_arrow(ax, (xs[i] + card_w/2, card_y), (txs[i] + tw/2, dl_y + 0.16),
                    color=CLR_GREEN_HDR, lw=1.2, style="<->")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "emo_system_architecture_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "emo_system_architecture_overview.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_system_architecture_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OVERLEAF_DIR / "emo_system_architecture_overview.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Figure 3 (System Architecture Overview) generated cleanly.")


if __name__ == "__main__":
    create_figure1_problem_and_solution()
    create_figure3_pipeline_architecture()
