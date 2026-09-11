#!/usr/bin/env python3
"""
create_caan_distributed_pipeline_figure.py
══════════════════════════════════════════════════════════════════════════════
Publication-Grade Architecture Diagram showing EXACTLY how the EMO / CaaN
codebase builds the Macro-Graph and distributes it across Spark workers:

  Tier 1: Delta Lake Storage & Driver Macro-Graph Construction (Phase 3b)
  Tier 2: Spark Context Broadcast Engine & Shuffless Community Dispatch
  Tier 3: Distributed Spark Executors (Local Augmented Subgraph Training)
  Tier 4: Driver-Side Embedding & Metric Aggregation
══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT_DIR = Path("results/figures")
OVERLEAF_DIR = Path("overleaf/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

# ── Color Tokens ─────────────────────────────────────────────────────────────
CLR_BG          = "#FFFFFF"
CLR_DARK        = "#0F172A"   # Slate 900
CLR_TEXT        = "#1E293B"   # Slate 800
CLR_MUTED       = "#64748B"   # Slate 500
CLR_BORDER      = "#CBD5E1"   # Slate 300
CLR_PANEL_BG    = "#F8FAFC"   # Slate 50

# Component Colors
CLR_S3_BG       = "#FFF7ED"   # Orange 50 (AWS S3)
CLR_S3_BORDER   = "#FDBA74"   # Orange 300
CLR_S3_HDR      = "#C2410C"   # Orange 700

CLR_DRIVER_BG   = "#EFF6FF"   # Blue 50
CLR_DRIVER_BD   = "#93C5FD"   # Blue 300
CLR_DRIVER_HDR  = "#1D4ED8"   # Blue 700

CLR_BC_BG       = "#FAF5FF"   # Purple 50
CLR_BC_BD       = "#D8B4FE"   # Purple 300
CLR_BC_HDR      = "#7E22CE"   # Purple 700

CLR_EXEC_BG     = "#F0FDF4"   # Emerald 50
CLR_EXEC_BD     = "#86EFAC"   # Emerald 300
CLR_EXEC_HDR    = "#15803D"   # Emerald 700

# Community & Super-Node Colors
CLR_C1_BG       = "#EFF6FF"   # Blue 50
CLR_C1_NODE     = "#2563EB"   # Blue 600
CLR_C1_SUPER    = "#1D4ED8"   # Blue 700
CLR_C1_BORDER   = "#93C5FD"   # Blue 300

CLR_C2_BG       = "#FAF5FF"   # Purple 50
CLR_C2_NODE     = "#9333EA"   # Purple 600
CLR_C2_SUPER    = "#7E22CE"   # Purple 700
CLR_C2_BORDER   = "#D8B4FE"   # Purple 300

CLR_MIN_BG      = "#FFFBEB"   # Amber 50
CLR_MIN_NODE    = "#D97706"   # Amber 600
CLR_MIN_BORDER  = "#FDE68A"   # Amber 200

CLR_BND_NODE    = "#059669"   # Emerald 600
CLR_BND_RING    = "#34D399"   # Emerald 400
CLR_CUT_LINE    = "#DC2626"   # Red 600


def _draw_box(ax, xy, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.3, radius=0.008, zorder=1):
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def _draw_arrow(ax, start, end, color="#0F172A", lw=1.5, style="-|>",
                connectionstyle="arc3,rad=0.0", ls="-", zorder=5):
    arr = FancyArrowPatch(start, end, arrowstyle=style,
                          connectionstyle=connectionstyle,
                          color=color, linewidth=lw, linestyle=ls,
                          mutation_scale=11, zorder=zorder)
    ax.add_patch(arr)
    return arr


def _draw_node(ax, pt, color, label="", zorder=6, r=0.011, text_color="#FFFFFF",
               ring_color=None, ring_r=0.015, font_size=7.0):
    if ring_color:
        c_ring = Circle(pt, ring_r, facecolor=ring_color, edgecolor="none", alpha=0.30, zorder=zorder-1)
        ax.add_patch(c_ring)
        c_ring_b = Circle(pt, ring_r, facecolor="none", edgecolor=ring_color, lw=1.1, ls="--", zorder=zorder)
        ax.add_patch(c_ring_b)
    c = Circle(pt, r, facecolor=color, edgecolor="#FFFFFF", lw=1.1, zorder=zorder)
    ax.add_patch(c)
    if label:
        ax.text(pt[0], pt[1], label, color=text_color, fontsize=font_size,
                fontweight="bold", ha="center", va="center", zorder=zorder+1)


def _draw_super_node(ax, pt, color, label="", zorder=6, r=0.018, text_color="#FFFFFF"):
    c_glow = Circle(pt, r + 0.006, facecolor=color, alpha=0.18, zorder=zorder-1)
    ax.add_patch(c_glow)
    c = Circle(pt, r, facecolor=color, edgecolor="#FFFFFF", lw=1.8, zorder=zorder)
    ax.add_patch(c)
    if label:
        ax.text(pt[0], pt[1], label, color=text_color, fontsize=8.2,
                fontweight="bold", ha="center", va="center", zorder=zorder+1)


def generate_distributed_pipeline_figure():
    fig, ax = plt.subplots(figsize=(21.0, 9.4), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main Header
    ax.text(0.50, 0.970, "Distributed CaaN Macro-Graph Construction & Spark Worker Dispatch Architecture",
            fontsize=15.5, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: STORAGE & DRIVER MACRO-GRAPH CONSTRUCTION (Left Column)
    # ══════════════════════════════════════════════════════════════════════════
    t1_x, t1_y, t1_w, t1_h = 0.015, 0.035, 0.285, 0.895
    _draw_box(ax, (t1_x, t1_y), t1_w, t1_h, fc=CLR_PANEL_BG, ec=CLR_DRIVER_BD, lw=1.8)
    ax.text(t1_x + t1_w/2, t1_y + t1_h - 0.024, "Tier 1: Storage & Driver Macro-Graph Engine",
            fontsize=11.2, fontweight="bold", color=CLR_DRIVER_HDR, ha="center")
    ax.text(t1_x + t1_w/2, t1_y + t1_h - 0.042, "Phase 3b: Delta Lake Ingestion & Driver-Side Aggregation",
            fontsize=7.2, color=CLR_MUTED, ha="center")

    # 1. Delta Lake Tables Box
    s3_bx, s3_by, s3_bw, s3_bh = t1_x + 0.012, t1_y + 0.610, t1_w - 0.024, 0.215
    _draw_box(ax, (s3_bx, s3_by), s3_bw, s3_bh, fc=CLR_S3_BG, ec=CLR_S3_BORDER, lw=1.2)
    ax.text(s3_bx + 0.010, s3_by + s3_bh - 0.018, "Amazon S3 / Delta Lake Storage Tables",
            fontsize=8.0, fontweight="bold", color=CLR_S3_HDR)
    
    tables_info = (
        r"$\bullet\;\mathbf{p2\_nodes:}\;(\mathrm{id}, \mathbf{X}, \mathbf{Y}, \mathrm{split}, \mathrm{community\_id}, \mathrm{is\_boundary})$" "\n"
        r"$\bullet\;\mathbf{p2\_edges:}\;(\mathrm{src}, \mathrm{dst}, \mathrm{community\_id})$ [Intra & Cut Edges]" "\n"
        r"$\bullet\;\mathbf{communities:}\;(\mathrm{id}, \mathrm{community\_id})$ [Louvain / LPA Partitions]"
    )
    ax.text(s3_bx + 0.010, s3_by + s3_bh - 0.065, tables_info, fontsize=6.5, color=CLR_TEXT, va="top")

    # Spark Ingestion Arrow
    _draw_arrow(ax, (s3_bx + s3_bw/2, s3_by), (s3_bx + s3_bw/2, s3_by - 0.025), color=CLR_DRIVER_HDR, lw=1.8)

    # 2. Driver Macro-Graph Assembly Engine
    dr_bx, dr_by, dr_bw, dr_bh = t1_x + 0.012, t1_y + 0.020, t1_w - 0.024, 0.540
    _draw_box(ax, (dr_bx, dr_by), dr_bw, dr_bh, fc="#FFFFFF", ec=CLR_DRIVER_BD, lw=1.3)
    ax.text(dr_bx + 0.010, dr_by + dr_bh - 0.018, "PySpark Driver Construction Steps",
            fontsize=8.2, fontweight="bold", color=CLR_DRIVER_HDR)

    # Step 1: Partition Filtering Card
    c1_y = dr_by + dr_bh - 0.090
    _draw_box(ax, (dr_bx + 0.008, c1_y), dr_bw - 0.016, 0.062, fc="#F1F5F9", ec=CLR_BORDER, lw=0.9)
    ax.text(dr_bx + 0.014, c1_y + 0.042, "1. Community Size Filtering (min_size = K)", fontsize=7.2, fontweight="bold", color=CLR_DARK)
    ax.text(dr_bx + 0.014, c1_y + 0.018, r"$\mathrm{cnt} \geq K \rightarrow \mathbf{Major\;Comms}\;C_i \quad\mid\quad \mathrm{cnt} < K \rightarrow \mathbf{Minor\;Comms}\;V_{\mathrm{minor}}$",
            fontsize=6.4, color=CLR_TEXT)

    # Step 2: Centroid Feature Aggregation
    c2_y = c1_y - 0.075
    _draw_box(ax, (dr_bx + 0.008, c2_y), dr_bw - 0.016, 0.065, fc="#F1F5F9", ec=CLR_BORDER, lw=0.9)
    ax.text(dr_bx + 0.014, c2_y + 0.045, "2. Centroid Super-Node Extraction", fontsize=7.2, fontweight="bold", color=CLR_DARK)
    ax.text(dr_bx + 0.014, c2_y + 0.018, r"$\mathbf{x}_{S_i} = \frac{1}{|C_i|}\sum_{u \in C_i} \mathbf{x}_u \rightarrow \mathbf{super\_nodes\_dict}[C_i]$",
            fontsize=6.4, color=CLR_C1_SUPER)

    # Step 3: Shuffless Map-Side Macro Edge Mapping
    c3_y = c2_y - 0.085
    _draw_box(ax, (dr_bx + 0.008, c3_y), dr_bw - 0.016, 0.075, fc="#F1F5F9", ec=CLR_BORDER, lw=0.9)
    ax.text(dr_bx + 0.014, c3_y + 0.054, "3. Map-Side Shuffless Edge Rewiring (RDD)", fontsize=7.2, fontweight="bold", color=CLR_DARK)
    ax.text(dr_bx + 0.014, c3_y + 0.022,
            r"$\mathrm{raw\_edges.mapPartitions(map\_edges).distinct()}$" "\n"
            r"$(u, v) \mapsto (-1000 - C_u, -1000 - C_v) \equiv (S_u, S_v) \rightarrow \mathbf{caan\_edges}$",
            fontsize=6.2, color=CLR_TEXT)

    # Step 4: Driver-Side Minor Global GNN Training
    c4_y = c3_y - 0.170
    _draw_box(ax, (dr_bx + 0.008, c4_y), dr_bw - 0.016, 0.160, fc=CLR_MIN_BG, ec=CLR_MIN_BORDER, lw=1.2)
    ax.text(dr_bx + 0.014, c4_y + 0.138, "4. Driver Global GNN on Macro-Graph", fontsize=7.4, fontweight="bold", color="#B45309")
    
    # Mini Macro-Graph Graphic
    mg_s1 = (dr_bx + 0.045, c4_y + 0.085)
    mg_s2 = (dr_bx + 0.125, c4_y + 0.085)
    mg_m1 = (dr_bx + 0.045, c4_y + 0.035)
    mg_m2 = (dr_bx + 0.125, c4_y + 0.035)
    
    ax.plot([mg_s1[0], mg_s2[0]], [mg_s1[1], mg_s2[1]], color=CLR_DARK, lw=1.8, zorder=3)
    ax.plot([mg_m1[0], mg_m2[0]], [mg_m1[1], mg_m2[1]], color=CLR_MIN_NODE, lw=1.2, zorder=3)
    ax.plot([mg_s1[0], mg_m1[0]], [mg_s1[1], mg_m1[1]], color=CLR_DARK, lw=1.0, ls="--", zorder=3)
    ax.plot([mg_s2[0], mg_m2[0]], [mg_s2[1], mg_m2[1]], color=CLR_DARK, lw=1.0, ls="--", zorder=3)
    
    _draw_super_node(ax, mg_s1, CLR_C1_SUPER, "$S_1$", r=0.013)
    _draw_super_node(ax, mg_s2, CLR_C2_SUPER, "$S_2$", r=0.013)
    _draw_node(ax, mg_m1, CLR_MIN_NODE, "$m_1$", r=0.008, font_size=5.5)
    _draw_node(ax, mg_m2, CLR_MIN_NODE, "$m_2$", r=0.008, font_size=5.5)

    ax.text(dr_bx + 0.155, c4_y + 0.060,
            r"$\mathbf{Global\;GNN}$" "\n"
            r"$\rightarrow \mathbf{h}_{V_{\mathrm{minor}}}$" "\n"
            r"$(S_i \text{ discarded})$",
            fontsize=6.4, fontweight="bold", color="#B45309", va="center")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2: SPARK BROADCAST ENGINE & DISPATCH (Center Column)
    # ══════════════════════════════════════════════════════════════════════════
    t2_x, t2_y, t2_w, t2_h = 0.315, 0.035, 0.285, 0.895
    _draw_box(ax, (t2_x, t2_y), t2_w, t2_h, fc=CLR_PANEL_BG, ec=CLR_BC_BD, lw=1.8)
    ax.text(t2_x + t2_w/2, t2_y + t2_h - 0.024, "Tier 2: Spark Broadcast Engine & Dispatch",
            fontsize=11.2, fontweight="bold", color=CLR_BC_HDR, ha="center")
    ax.text(t2_x + t2_w/2, t2_y + t2_h - 0.042, r"$\mathrm{sc.broadcast()}$ Payload & Shuffless Community Partitioning",
            fontsize=7.2, color=CLR_MUTED, ha="center")

    # 1. Broadcast Payload Container
    bc_bx, bc_by, bc_bw, bc_bh = t2_x + 0.012, t2_y + 0.440, t2_w - 0.024, 0.385
    _draw_box(ax, (bc_bx, bc_by), bc_bw, bc_bh, fc=CLR_BC_BG, ec=CLR_BC_BD, lw=1.3)
    ax.text(bc_bx + 0.010, bc_by + bc_bh - 0.018, "Immutable Broadcast Variables (sc.broadcast)",
            fontsize=8.0, fontweight="bold", color=CLR_BC_HDR)

    bc_items = [
        (r"$\mathbf{super\_nodes\_dict\_bc}$", r"$\{\mathrm{comm\_id} \mapsto \mathbf{x}_{S_i} \in \mathbb{R}^d\}$ (Centroids)", CLR_C1_SUPER),
        (r"$\mathbf{minor\_feats\_arr\_bc}$", r"$\mathbf{X}_{V_{\mathrm{minor}}} \in \mathbb{R}^{|V_{\mathrm{minor}}| \times d}$ (Pre-stacked)", "#B45309"),
        (r"$\mathbf{minor\_node\_to\_idx\_bc}$", r"$\{\mathrm{nid} \mapsto \mathrm{idx}\}$ (O(1) Vector Slice Mapping)", "#B45309"),
        (r"$\mathbf{caan\_adj\_bc}$", r"$\mathrm{defaultdict}(\mathrm{list})$ (Macro Topology)", CLR_DARK),
        (r"$\mathbf{node\_to\_comm\_bc}$", r"$\{\mathrm{node\_id} \mapsto \mathrm{comm\_id}\}$ (Global Partition Map)", CLR_DARK),
        (r"$\mathbf{major\_comms\_bc}$", r"$\mathrm{set}(\mathrm{major\_ids})$ (Fast Set Membership)", CLR_C2_SUPER)
    ]
    cur_y = bc_by + bc_bh - 0.045
    for tag, desc, clr_t in bc_items:
        _draw_box(ax, (bc_bx + 0.008, cur_y - 0.045), bc_bw - 0.016, 0.045, fc="#FFFFFF", ec=CLR_BORDER, lw=0.7, radius=0.004)
        ax.text(bc_bx + 0.014, cur_y - 0.016, tag, fontsize=6.8, fontweight="bold", color=clr_t)
        ax.text(bc_bx + 0.014, cur_y - 0.034, desc, fontsize=6.0, color=CLR_TEXT)
        cur_y -= 0.052

    # Connect Tier 1 to Tier 2
    _draw_arrow(ax, (t1_x + t1_w + 0.002, t1_y + t1_h * 0.55), (t2_x - 0.002, t2_y + t2_h * 0.55),
                color=CLR_BC_HDR, lw=2.0, style="-|>")

    # 2. Shuffless Community Binning & Dispatch Box
    bin_bx, bin_by, bin_bw, bin_bh = t2_x + 0.012, t2_y + 0.020, t2_w - 0.024, 0.395
    _draw_box(ax, (bin_bx, bin_by), bin_bw, bin_bh, fc="#FFFFFF", ec=CLR_BC_BD, lw=1.2)
    ax.text(bin_bx + 0.010, bin_by + bin_bh - 0.018, "Driver Community Binning & Manifest",
            fontsize=8.0, fontweight="bold", color=CLR_DARK)
    
    bin_text = (
        r"$\mathbf{1.\;Load\;Balancing\;Binning:}$" "\n"
        r"   $\mathrm{comms\_node\_counts.sort\_values('count')}$" "\n"
        r"   $\mathrm{bin\_id} = i \ \mathrm{mod} \ \mathrm{num\_bins}$" "\n\n"
        r"$\mathbf{2.\;Manifest\;DataFrame\;Generation:}$" "\n"
        r"   Attaches hyperparameters: $\mathrm{lr}, \mathrm{epochs}, \mathrm{hidden}$" "\n\n"
        r"$\mathbf{3.\;Shuffless\;Executor\;Dispatch:}$" "\n"
        r"   $\mathrm{manifest\_df.repartition(num\_bins, 'bin\_id')}$" "\n"
        r"   $\quad.\mathrm{groupBy('bin\_id').applyInPandas(caan\_udf)}$"
    )
    ax.text(bin_bx + 0.010, bin_by + 0.190, bin_text, fontsize=6.4, color=CLR_TEXT, va="center")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 3: DISTRIBUTED WORKERS (Right Column)
    # ══════════════════════════════════════════════════════════════════════════
    t3_x, t3_y, t3_w, t3_h = 0.615, 0.035, 0.370, 0.895
    _draw_box(ax, (t3_x, t3_y), t3_w, t3_h, fc=CLR_PANEL_BG, ec=CLR_EXEC_BD, lw=1.8)
    ax.text(t3_x + t3_w/2, t3_y + t3_h - 0.024, "Tier 3: Distributed Spark Executors (make_caan_udf)",
            fontsize=11.2, fontweight="bold", color=CLR_EXEC_HDR, ha="center")
    ax.text(t3_x + t3_w/2, t3_y + t3_h - 0.042, "Parallel Local Augmented Subgraph Training (0.0 GB Network Traffic)",
            fontsize=7.2, color=CLR_MUTED, ha="center")

    # Connect Tier 2 to Tier 3 (2 arrows into Worker 1 and Worker 2)
    _draw_arrow(ax, (t2_x + t2_w + 0.002, t2_y + t2_h * 0.68), (t3_x - 0.002, t3_y + t3_h * 0.68),
                color=CLR_C1_SUPER, lw=1.8, style="-|>")
    _draw_arrow(ax, (t2_x + t2_w + 0.002, t2_y + t2_h * 0.28), (t3_x - 0.002, t3_y + t3_h * 0.28),
                color=CLR_C2_SUPER, lw=1.8, style="-|>")

    # ── Executor Worker 1 (Community C1) ──────────────────────────────────────
    w1_bx, w1_by, w1_bw, w1_bh = t3_x + 0.012, t3_y + 0.450, t3_w - 0.024, 0.380
    _draw_box(ax, (w1_bx, w1_by), w1_bw, w1_bh, fc="#FFFFFF", ec=CLR_C1_BORDER, lw=1.4)
    ax.text(w1_bx + 0.010, w1_by + w1_bh - 0.018, "Spark Executor Worker 1 — Major Community $C_1$",
            fontsize=8.2, fontweight="bold", color=CLR_C1_SUPER)

    # Subgraph construction visual in Worker 1
    w1_sg_x, w1_sg_y = w1_bx + 0.010, w1_by + 0.020
    w1_sg_w, w1_sg_h = 0.170, w1_bh - 0.045
    _draw_box(ax, (w1_sg_x, w1_sg_y), w1_sg_w, w1_sg_h, fc=CLR_C1_BG, ec=CLR_C1_BORDER, lw=0.9, radius=0.006)
    ax.text(w1_sg_x + 0.008, w1_sg_y + w1_sg_h - 0.016, r"Augmented $(\mathcal{G}_{\mathrm{CaaN}} \setminus \{S_1\}) \cup C_1$",
            fontsize=6.4, fontweight="bold", color=CLR_C1_SUPER)
    
    # Internal exclusion badge
    _draw_box(ax, (w1_sg_x + 0.008, w1_sg_y + w1_sg_h - 0.040), 0.100, 0.020, fc="#FEF2F2", ec=CLR_CUT_LINE, lw=0.8, radius=0.003)
    ax.text(w1_sg_x + 0.058, w1_sg_y + w1_sg_h - 0.030, r"Internal $S_1 \notin$ Local", fontsize=5.8, fontweight="bold", color=CLR_CUT_LINE, ha="center", va="center")

    # Local nodes
    w1_u1, w1_u2, w1_v1 = (w1_sg_x + 0.030, w1_sg_y + 0.180), (w1_sg_x + 0.075, w1_sg_y + 0.190), (w1_sg_x + 0.060, w1_sg_y + 0.080)
    w1_s2 = (w1_sg_x + 0.135, w1_sg_y + 0.110)
    
    for pa, pb in [(w1_u1, w1_u2), (w1_u1, w1_v1), (w1_u2, w1_v1)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C1_NODE, lw=1.1, zorder=3)
    ax.plot([w1_v1[0], w1_s2[0]], [w1_v1[1], w1_s2[1]], color=CLR_C2_SUPER, lw=2.0, zorder=4)
    
    _draw_node(ax, w1_u1, CLR_C1_NODE, "$u_1$", r=0.009, font_size=6.0)
    _draw_node(ax, w1_u2, CLR_C1_NODE, "$u_2$", r=0.009, font_size=6.0)
    _draw_node(ax, w1_v1, CLR_BND_NODE, "$v_1$", r=0.009, font_size=6.0, ring_color=CLR_BND_RING, ring_r=0.013)
    _draw_super_node(ax, w1_s2, CLR_C2_SUPER, "$S_2$", r=0.014)
    
    ax.text(w1_v1[0] - 0.015, w1_v1[1] - 0.008, "Boundary $v_1$", fontsize=5.8, fontweight="bold", color=CLR_BND_NODE, ha="right")
    ax.text(w1_s2[0], w1_s2[1] + 0.020, "External $S_2$", fontsize=5.6, fontweight="bold", color=CLR_C2_SUPER, ha="center")
    ax.text(w1_sg_x + w1_sg_w/2, w1_sg_y + 0.015, "Boundary Link $(v_1, S_2)$ (From Broadcast)", fontsize=5.5, fontweight="bold", color=CLR_C2_SUPER, ha="center")

    # Local GNN Model Box
    _draw_arrow(ax, (w1_sg_x + w1_sg_w + 0.005, w1_by + w1_bh/2), (w1_bx + 0.235, w1_by + w1_bh/2), color=CLR_DARK, lw=1.6)
    _draw_box(ax, (w1_bx + 0.240, w1_by + w1_bh/2 - 0.055), 0.045, 0.110, fc="#E2E8F0", ec=CLR_DARK, lw=1.1, radius=0.006)
    ax.text(w1_bx + 0.2625, w1_by + w1_bh/2, "Local\nPyG\nModel", fontsize=7.2, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    # Local Output Embeddings
    _draw_arrow(ax, (w1_bx + 0.290, w1_by + w1_bh/2), (w1_bx + 0.305, w1_by + w1_bh/2), color=CLR_DARK, lw=1.6)
    _draw_box(ax, (w1_bx + 0.310, w1_by + 0.035), 0.030, w1_bh - 0.070, fc="#FFFFFF", ec=CLR_DARK, lw=1.1, radius=0.004)
    rect_e1 = Rectangle((w1_bx + 0.310, w1_by + 0.035 + (w1_bh - 0.070)*0.30), 0.030, (w1_bh - 0.070)*0.70, facecolor=CLR_C1_BG, edgecolor=CLR_C1_BORDER, lw=0.8)
    ax.add_patch(rect_e1)
    ax.text(w1_bx + 0.325, w1_by + w1_bh * 0.65, r"$\mathbf{h}_{V_{C_1}}$", fontsize=7.2, fontweight="bold", color=CLR_C1_SUPER, ha="center", va="center")
    ax.text(w1_bx + 0.325, w1_by + w1_bh * 0.20, "$S_2$\nDrop", fontsize=5.2, color=CLR_MUTED, ha="center", va="center")

    # ── Executor Worker 2 (Community C2) ──────────────────────────────────────
    w2_bx, w2_by, w2_bw, w2_bh = t3_x + 0.012, t3_y + 0.050, t3_w - 0.024, 0.380
    _draw_box(ax, (w2_bx, w2_by), w2_bw, w2_bh, fc="#FFFFFF", ec=CLR_C2_BORDER, lw=1.4)
    ax.text(w2_bx + 0.010, w2_by + w2_bh - 0.018, "Spark Executor Worker 2 — Major Community $C_2$",
            fontsize=8.2, fontweight="bold", color=CLR_C2_SUPER)

    # Subgraph construction visual in Worker 2
    w2_sg_x, w2_sg_y = w2_bx + 0.010, w2_by + 0.020
    w2_sg_w, w2_sg_h = 0.170, w2_bh - 0.045
    _draw_box(ax, (w2_sg_x, w2_sg_y), w2_sg_w, w2_sg_h, fc=CLR_C2_BG, ec=CLR_C2_BORDER, lw=0.9, radius=0.006)
    ax.text(w2_sg_x + 0.008, w2_sg_y + w2_sg_h - 0.016, r"Augmented $(\mathcal{G}_{\mathrm{CaaN}} \setminus \{S_2\}) \cup C_2$",
            fontsize=6.4, fontweight="bold", color=CLR_C2_SUPER)
    
    # Internal exclusion badge
    _draw_box(ax, (w2_sg_x + 0.008, w2_sg_y + w2_sg_h - 0.040), 0.100, 0.020, fc="#FEF2F2", ec=CLR_CUT_LINE, lw=0.8, radius=0.003)
    ax.text(w2_sg_x + 0.058, w2_sg_y + w2_sg_h - 0.030, r"Internal $S_2 \notin$ Local", fontsize=5.8, fontweight="bold", color=CLR_CUT_LINE, ha="center", va="center")

    # Local nodes
    w2_w1, w2_w2, w2_v2 = (w2_sg_x + 0.030, w2_sg_y + 0.180), (w2_sg_x + 0.075, w2_sg_y + 0.190), (w2_sg_x + 0.060, w2_sg_y + 0.080)
    w2_s1 = (w2_sg_x + 0.135, w2_sg_y + 0.110)
    
    for pa, pb in [(w2_w1, w2_w2), (w2_w1, w2_v2), (w2_w2, w2_v2)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C2_NODE, lw=1.1, zorder=3)
    ax.plot([w2_v2[0], w2_s1[0]], [w2_v2[1], w2_s1[1]], color=CLR_C1_SUPER, lw=2.0, zorder=4)
    
    _draw_node(ax, w2_w1, CLR_C2_NODE, "$w_1$", r=0.009, font_size=6.0)
    _draw_node(ax, w2_w2, CLR_C2_NODE, "$w_2$", r=0.009, font_size=6.0)
    _draw_node(ax, w2_v2, CLR_BND_NODE, "$v_2$", r=0.009, font_size=6.0, ring_color=CLR_BND_RING, ring_r=0.013)
    _draw_super_node(ax, w2_s1, CLR_C1_SUPER, "$S_1$", r=0.014)
    
    ax.text(w2_v2[0] - 0.015, w2_v2[1] - 0.008, "Boundary $v_2$", fontsize=5.8, fontweight="bold", color=CLR_BND_NODE, ha="right")
    ax.text(w2_s1[0], w2_s1[1] + 0.020, "External $S_1$", fontsize=5.6, fontweight="bold", color=CLR_C1_SUPER, ha="center")
    ax.text(w2_sg_x + w2_sg_w/2, w2_sg_y + 0.015, "Boundary Link $(v_2, S_1)$ (From Broadcast)", fontsize=5.5, fontweight="bold", color=CLR_C1_SUPER, ha="center")

    # Local GNN Model Box
    _draw_arrow(ax, (w2_sg_x + w2_sg_w + 0.005, w2_by + w2_bh/2), (w2_bx + 0.235, w2_by + w2_bh/2), color=CLR_DARK, lw=1.6)
    _draw_box(ax, (w2_bx + 0.240, w2_by + w2_bh/2 - 0.055), 0.045, 0.110, fc="#E2E8F0", ec=CLR_DARK, lw=1.1, radius=0.006)
    ax.text(w2_bx + 0.2625, w2_by + w2_bh/2, "Local\nPyG\nModel", fontsize=7.2, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    # Local Output Embeddings
    _draw_arrow(ax, (w2_bx + 0.290, w2_by + w2_bh/2), (w2_bx + 0.305, w2_by + w2_bh/2), color=CLR_DARK, lw=1.6)
    _draw_box(ax, (w2_bx + 0.310, w2_by + 0.035), 0.030, w2_bh - 0.070, fc="#FFFFFF", ec=CLR_DARK, lw=1.1, radius=0.004)
    rect_e2 = Rectangle((w2_bx + 0.310, w2_by + 0.035 + (w2_bh - 0.070)*0.30), 0.030, (w2_bh - 0.070)*0.70, facecolor=CLR_C2_BG, edgecolor=CLR_C2_BORDER, lw=0.8)
    ax.add_patch(rect_e2)
    ax.text(w2_bx + 0.325, w2_by + w2_bh * 0.65, r"$\mathbf{h}_{V_{C_2}}$", fontsize=7.2, fontweight="bold", color=CLR_C2_SUPER, ha="center", va="center")
    ax.text(w2_bx + 0.325, w2_by + w2_bh * 0.20, "$S_1$\nDrop", fontsize=5.2, color=CLR_MUTED, ha="center", va="center")

    # Final Driver Collection Banner at Bottom
    agg_bx, agg_by, agg_bw, agg_bh = t3_x + 0.012, t3_y + 0.008, t3_w - 0.024, 0.035
    _draw_box(ax, (agg_bx, agg_by), agg_bw, agg_bh, fc="#FEF3C7", ec="#F59E0B", lw=1.1, radius=0.004)
    ax.text(agg_bx + agg_bw/2, agg_by + agg_bh/2,
            r"$\mathbf{Driver\;Aggregation:}\;\mathbf{H} = [\mathbf{h}_{V_{\mathrm{minor}}} \mid \mathbf{h}_{V_{C_1}} \mid \mathbf{h}_{V_{C_2}} \mid \dots] \in \mathbb{R}^{|\mathcal{V}| \times d} \rightarrow \mathrm{Weighted\;Micro-F1\;/\;Accuracy}$",
            fontsize=6.8, fontweight="bold", color="#92400E", ha="center", va="center")

    out_pdf = OUT_DIR / "caan_distributed_pipeline.pdf"
    out_png = OUT_DIR / "caan_distributed_pipeline.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(OVERLEAF_DIR / "caan_distributed_pipeline.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OVERLEAF_DIR / "caan_distributed_pipeline.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Generated Distributed Pipeline Figure: {out_pdf}")


if __name__ == "__main__":
    generate_distributed_pipeline_figure()
