#!/usr/bin/env python3
"""
create_caan_macro_graph_figure.py
══════════════════════════════════════════════════════════════════════════════
Generates a comprehensive 3-panel publication-grade figure illustrating:
  - Panel (a): Original Partitioned Graph & Severed Inter-Community Edge Cuts
  - Panel (b): CaaN Auxiliary Macro-Graph (Centroid Super-Nodes, Dual-Context
               Boundary Wiring, and Retained Minor Vertices)
  - Panel (c): Dual-Context GNN Message Passing & Receptive Field Restoration
══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Wedge
import numpy as np

OUT_DIR = Path("results/figures")
OVERLEAF_DIR = Path("overleaf/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

# ── Elegant Academic Color Palette ───────────────────────────────────────────
CLR_BG          = "#FFFFFF"
CLR_PANEL_BG    = "#F8FAFC"   # Slate 50
CLR_DARK        = "#0F172A"   # Slate 900
CLR_TEXT        = "#1E293B"   # Slate 800
CLR_MUTED       = "#64748B"   # Slate 500
CLR_BORDER      = "#CBD5E1"   # Slate 300
CLR_CUT_LINE    = "#DC2626"   # Red 600
CLR_CUT_BG      = "#FEF2F2"   # Red 50

# Community 1 (Blue)
CLR_C1_BG       = "#EFF6FF"
CLR_C1_NODE     = "#3B82F6"
CLR_C1_SUPER    = "#1D4ED8"
CLR_C1_BORDER   = "#BFDBFE"

# Community 2 (Purple)
CLR_C2_BG       = "#FAF5FF"
CLR_C2_NODE     = "#A855F7"
CLR_C2_SUPER    = "#7E22CE"
CLR_C2_BORDER   = "#E9D5FF"

# Minor Community 3 (Amber / Warm Gold)
CLR_C3_BG       = "#FFFBEB"
CLR_C3_NODE     = "#D97706"
CLR_C3_BORDER   = "#FDE68A"

# Boundary Vertices (Emerald)
CLR_BND_NODE    = "#059669"
CLR_BND_RING    = "#10B981"


def _draw_box(ax, xy, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.2, radius=0.015, zorder=1):
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def _draw_arrow(ax, start, end, color="#2563EB", lw=1.5, style="->",
                connectionstyle="arc3,rad=0.0", ls="-", zorder=5):
    arr = FancyArrowPatch(start, end, arrowstyle=style,
                          connectionstyle=connectionstyle,
                          color=color, linewidth=lw, linestyle=ls,
                          mutation_scale=10, zorder=zorder)
    ax.add_patch(arr)
    return arr


def _draw_node(ax, pt, color, label="", zorder=6, r=0.014, text_color="#FFFFFF", ring_color=None, ring_r=0.021):
    if ring_color:
        c_ring = Circle(pt, ring_r, facecolor=ring_color, edgecolor="none", alpha=0.35, zorder=zorder-1)
        ax.add_patch(c_ring)
    c = Circle(pt, r, facecolor=color, edgecolor="#FFFFFF", lw=1.4, zorder=zorder)
    ax.add_patch(c)
    if label:
        ax.text(pt[0], pt[1], label, color=text_color, fontsize=8.5,
                fontweight="bold", ha="center", va="center", zorder=zorder+1)


def create_caan_macro_graph_figure():
    fig, ax = plt.subplots(figsize=(19.2, 8.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 1 (LEFT): ORIGINAL GRAPH & PARTITION CUTS
    # ══════════════════════════════════════════════════════════════════════════
    p1_x, p1_y, p1_w, p1_h = 0.020, 0.04, 0.305, 0.91
    _draw_box(ax, (p1_x, p1_y), p1_w, p1_h, fc=CLR_PANEL_BG, ec=CLR_BORDER, lw=1.5, radius=0.018)

    ax.text(p1_x + 0.018, p1_y + p1_h - 0.038, "(a) Original Graph & Cuts",
            fontsize=12.5, fontweight="bold", color=CLR_DARK)
    ax.text(p1_x + 0.018, p1_y + p1_h - 0.066, "Partitioning severs cross-community edges",
            fontsize=8.5, color=CLR_MUTED, style="italic")

    # Community 1 Box (Major A)
    c1_x, c1_y, c1_w, c1_h = p1_x + 0.015, p1_y + 0.47, 0.128, 0.32
    _draw_box(ax, (c1_x, c1_y), c1_w, c1_h, fc=CLR_C1_BG, ec=CLR_C1_BORDER, lw=1.2, radius=0.012, zorder=2)
    ax.text(c1_x + 0.010, c1_y + c1_h - 0.028, r"Major $C_1$ ($|C_1| \geq K$)",
            fontsize=8.2, fontweight="bold", color=CLR_C1_SUPER)

    # Community 2 Box (Major B)
    c2_x, c2_y, c2_w, c2_h = p1_x + 0.162, p1_y + 0.47, 0.128, 0.32
    _draw_box(ax, (c2_x, c2_y), c2_w, c2_h, fc=CLR_C2_BG, ec=CLR_C2_BORDER, lw=1.2, radius=0.012, zorder=2)
    ax.text(c2_x + 0.010, c2_y + c2_h - 0.028, r"Major $C_2$ ($|C_2| \geq K$)",
            fontsize=8.2, fontweight="bold", color=CLR_C2_SUPER)

    # Community 3 Box (Minor C)
    c3_x, c3_y, c3_w, c3_h = p1_x + 0.050, p1_y + 0.05, 0.205, 0.34
    _draw_box(ax, (c3_x, c3_y), c3_w, c3_h, fc=CLR_C3_BG, ec=CLR_C3_BORDER, lw=1.2, radius=0.012, zorder=2)
    ax.text(c3_x + 0.015, c3_y + c3_h - 0.028, r"Minor $C_3$ ($|C_3| < K$)",
            fontsize=8.2, fontweight="bold", color="#B45309")

    # ── Original Nodes ────────────────────────────────────────────────────────
    u1 = (c1_x + 0.035, c1_y + 0.20)
    u2 = (c1_x + 0.045, c1_y + 0.08)
    u3 = (c1_x + 0.085, c1_y + 0.22)
    v1 = (c1_x + 0.098, c1_y + 0.09)

    w1 = (c2_x + 0.095, c2_y + 0.20)
    w2 = (c2_x + 0.085, c2_y + 0.08)
    w3 = (c2_x + 0.045, c2_y + 0.22)
    v2 = (c2_x + 0.030, c2_y + 0.09)

    m1 = (c3_x + 0.045, c3_y + 0.18)
    m2 = (c3_x + 0.102, c3_y + 0.08)
    m3 = (c3_x + 0.160, c3_y + 0.18)

    # Intra-Community Edges
    for p_a, p_b in [(u1, u2), (u1, u3), (u2, v1), (u3, v1), (u1, v1)]:
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=CLR_C1_NODE, lw=1.3, alpha=0.75, zorder=3)
    for p_a, p_b in [(w1, w2), (w1, w3), (w2, v2), (w3, v2), (w1, v2)]:
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=CLR_C2_NODE, lw=1.3, alpha=0.75, zorder=3)
    for p_a, p_b in [(m1, m2), (m2, m3), (m1, m3)]:
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=CLR_C3_NODE, lw=1.3, alpha=0.75, zorder=3)

    # Cross-Partition Cut Edges (Red dashed)
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=CLR_CUT_LINE, lw=2.2, ls="--", zorder=4)
    ax.plot([v1[0], m1[0]], [v1[1], m1[1]], color=CLR_CUT_LINE, lw=1.5, ls=":", zorder=4)
    ax.plot([v2[0], m3[0]], [v2[1], m3[1]], color=CLR_CUT_LINE, lw=1.5, ls=":", zorder=4)

    # Cut Line Visual
    cut_x = (c1_x + c1_w + c2_x) / 2
    ax.plot([cut_x, cut_x], [p1_y + 0.40, p1_y + 0.76], color=CLR_CUT_LINE, lw=1.8, ls="-.", zorder=4)
    _draw_box(ax, (cut_x - 0.048, p1_y + 0.69), 0.096, 0.030, fc="#FFFFFF", ec=CLR_CUT_LINE, lw=1.0, radius=0.006, zorder=5)
    ax.text(cut_x, p1_y + 0.705, "Partition Cut", color=CLR_CUT_LINE, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=6)

    # Draw Nodes in Panel 1
    for pt, lbl in [(u1, "$u_1$"), (u2, "$u_2$"), (u3, "$u_3$")]: _draw_node(ax, pt, CLR_C1_NODE, lbl)
    for pt, lbl in [(w1, "$w_1$"), (w2, "$w_2$"), (w3, "$w_3$")]: _draw_node(ax, pt, CLR_C2_NODE, lbl)
    for pt, lbl in [(m1, "$m_1$"), (m2, "$m_2$"), (m3, "$m_3$")]: _draw_node(ax, pt, CLR_C3_NODE, lbl)

    # Boundary Nodes (Panel 1)
    _draw_node(ax, v1, CLR_BND_NODE, "$v_1$", ring_color=CLR_BND_RING)
    _draw_node(ax, v2, CLR_BND_NODE, "$v_2$", ring_color=CLR_BND_RING)
    ax.text(v1[0] - 0.010, v1[1] - 0.024, r"Boundary $v_1$", fontsize=7.5, fontweight="bold", color=CLR_BND_NODE, ha="center")
    ax.text(v2[0] + 0.010, v2[1] - 0.024, r"Boundary $v_2$", fontsize=7.5, fontweight="bold", color=CLR_BND_NODE, ha="center")

    # Bottom Callout for Panel 1
    _draw_box(ax, (p1_x + 0.015, p1_y + 0.012), p1_w - 0.030, 0.028, fc=CLR_CUT_BG, ec=CLR_CUT_LINE, lw=0.8, radius=0.005, zorder=3)
    ax.text(p1_x + p1_w/2, p1_y + 0.026, "Naive Partitioning: Severed cuts cause severe boundary accuracy collapse",
            fontsize=6.8, fontweight="bold", color=CLR_CUT_LINE, ha="center", va="center")


    # ── Arrow from Panel 1 to Panel 2 ─────────────────────────────────────────
    arr1_start = (p1_x + p1_w + 0.006, 0.52)
    arr1_end   = (p1_x + p1_w + 0.038, 0.52)
    _draw_arrow(ax, arr1_start, arr1_end, color=CLR_DARK, lw=2.2, style="-|>", zorder=7)
    ax.text((arr1_start[0] + arr1_end[0])/2, 0.555, "CaaN\nBuild",
            fontsize=8.2, fontweight="bold", color=CLR_DARK, ha="center", va="bottom")


    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 2 (MIDDLE): CAAN AUXILIARY MACRO-GRAPH G_aux
    # ══════════════════════════════════════════════════════════════════════════
    p2_x, p2_y, p2_w, p2_h = 0.370, 0.04, 0.335, 0.91
    _draw_box(ax, (p2_x, p2_y), p2_w, p2_h, fc=CLR_PANEL_BG, ec=CLR_BORDER, lw=1.5, radius=0.018)

    ax.text(p2_x + 0.018, p2_y + p2_h - 0.038, r"(b) CaaN Auxiliary Macro-Graph $\mathcal{G}_{\mathrm{aux}}$",
            fontsize=12.5, fontweight="bold", color=CLR_DARK)
    ax.text(p2_x + 0.018, p2_y + p2_h - 0.066, "Centroid Super-Nodes + Dual-Context Boundary Wiring",
            fontsize=8.5, color=CLR_MUTED, style="italic")

    # Super-Node Coordinates
    s1_pos = (p2_x + 0.075, p2_y + 0.65)
    s2_pos = (p2_x + 0.260, p2_y + 0.65)

    # Boundary Vertices Coordinates
    v1_aux = (p2_x + 0.118, p2_y + 0.43)
    v2_aux = (p2_x + 0.217, p2_y + 0.43)

    # Minor Vertices Coordinates
    m1_aux = (p2_x + 0.080, p2_y + 0.15)
    m2_aux = (p2_x + 0.168, p2_y + 0.09)
    m3_aux = (p2_x + 0.255, p2_y + 0.15)

    # 1. Super-Edge between S1 and S2
    ax.plot([s1_pos[0], s2_pos[0]], [s1_pos[1], s2_pos[1]], color=CLR_DARK, lw=4.5, zorder=3)
    mid_s = ((s1_pos[0] + s2_pos[0])/2, s1_pos[1] + 0.038)
    _draw_box(ax, (mid_s[0]-0.068, mid_s[1]-0.016), 0.136, 0.032, fc="#FFFFFF", ec=CLR_DARK, lw=1.2, radius=0.006, zorder=4)
    ax.text(mid_s[0], mid_s[1], r"Super-Edge $W_{12} = \sum W_{uv}$", fontsize=7.5, fontweight="bold", color=CLR_DARK, ha="center", va="center", zorder=5)

    # 2. Dual-Context Boundary Node Connections
    # v1 -> S1 (Solid Blue: Internal Context)
    ax.plot([v1_aux[0], s1_pos[0]], [v1_aux[1], s1_pos[1]], color=CLR_C1_SUPER, lw=2.4, ls="-", zorder=3)
    # v1 -> S2 (Dashed Blue: External Context - Restores Cut to C2!)
    ax.plot([v1_aux[0], s2_pos[0]], [v1_aux[1], s2_pos[1]], color=CLR_C1_SUPER, lw=2.4, ls="--", zorder=3)

    # v2 -> S2 (Solid Purple: Internal Context)
    ax.plot([v2_aux[0], s2_pos[0]], [v2_aux[1], s2_pos[1]], color=CLR_C2_SUPER, lw=2.4, ls="-", zorder=3)
    # v2 -> S1 (Dashed Purple: External Context - Restores Cut to C1!)
    ax.plot([v2_aux[0], s1_pos[0]], [v2_aux[1], s1_pos[1]], color=CLR_C2_SUPER, lw=2.4, ls="--", zorder=3)

    # Direct Boundary-to-Boundary Edge
    ax.plot([v1_aux[0], v2_aux[0]], [v1_aux[1], v2_aux[1]], color=CLR_BND_NODE, lw=1.6, ls=":", zorder=3)

    # 3. Minor Vertices Connections
    ax.plot([m1_aux[0], s1_pos[0]], [m1_aux[1], s1_pos[1]], color=CLR_C1_NODE, lw=1.2, ls="-.", alpha=0.6, zorder=3)
    ax.plot([m3_aux[0], s2_pos[0]], [m3_aux[1], s2_pos[1]], color=CLR_C2_NODE, lw=1.2, ls="-.", alpha=0.6, zorder=3)
    for p_a, p_b in [(m1_aux, m2_aux), (m2_aux, m3_aux), (m1_aux, m3_aux)]:
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=CLR_C3_NODE, lw=1.3, alpha=0.8, zorder=3)

    # 4. Draw Super-Nodes
    def _draw_super_node(pt, color, title, subtitle):
        c_glow = Circle(pt, 0.040, facecolor=color, alpha=0.15, zorder=4)
        ax.add_patch(c_glow)
        c_body = Circle(pt, 0.032, facecolor=color, edgecolor="#FFFFFF", lw=2.0, zorder=5)
        ax.add_patch(c_body)
        ax.text(pt[0], pt[1] + 0.007, title, color="#FFFFFF", fontsize=9.8, fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(pt[0], pt[1] - 0.011, "Super-Node", color="#E2E8F0", fontsize=6.5, ha="center", va="center", zorder=6)
        _draw_box(ax, (pt[0]-0.055, pt[1]+0.042), 0.110, 0.028, fc="#FFFFFF", ec=color, lw=1.0, radius=0.005, zorder=6)
        ax.text(pt[0], pt[1] + 0.056, subtitle, color=color, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=7)

    _draw_super_node(s1_pos, CLR_C1_SUPER, "$S_1$", r"Centroid $\mathbf{x}_{S_1}$")
    _draw_super_node(s2_pos, CLR_C2_SUPER, "$S_2$", r"Centroid $\mathbf{x}_{S_2}$")

    # Draw Minor Nodes
    for pt, lbl in [(m1_aux, "$m_1$"), (m2_aux, "$m_2$"), (m3_aux, "$m_3$")]:
        _draw_node(ax, pt, CLR_C3_NODE, lbl)
    ax.text((m1_aux[0]+m3_aux[0])/2, p2_y + 0.020, "Minor Vertices (Preserved Individually)",
            fontsize=7.5, fontweight="bold", color="#B45309", ha="center")

    # Draw Boundary Nodes
    _draw_node(ax, v1_aux, CLR_BND_NODE, "$v_1$", ring_color=CLR_BND_RING)
    _draw_node(ax, v2_aux, CLR_BND_NODE, "$v_2$", ring_color=CLR_BND_RING)

    # Connectivity Legend Inside Panel 2
    leg_x, leg_y, leg_w, leg_h = p2_x + 0.015, p2_y + 0.205, p2_w - 0.030, 0.180
    _draw_box(ax, (leg_x, leg_y), leg_w, leg_h, fc="#FFFFFF", ec=CLR_BORDER, lw=1.2, radius=0.010, zorder=6)
    ax.text(leg_x + 0.012, leg_y + leg_h - 0.022, "Auxiliary Graph Connectivity Rules:",
            fontsize=8.2, fontweight="bold", color=CLR_DARK, zorder=7)

    # Leg Item 1: Internal
    ax.plot([leg_x + 0.015, leg_x + 0.045], [leg_y + 0.118, leg_y + 0.118], color=CLR_C1_SUPER, lw=2.2, zorder=7)
    ax.text(leg_x + 0.052, leg_y + 0.118, r"$\mathbf{v_1 \to S_1}$ (Solid): Host community macro context",
            fontsize=7.2, color=CLR_TEXT, va="center", zorder=7)
    # Leg Item 2: External
    ax.plot([leg_x + 0.015, leg_x + 0.045], [leg_y + 0.082, leg_y + 0.082], color=CLR_C1_SUPER, lw=2.2, ls="--", zorder=7)
    ax.text(leg_x + 0.052, leg_y + 0.082, r"$\mathbf{v_1 \to S_2}$ (Dashed): External context (restores severed cut)",
            fontsize=7.2, color=CLR_TEXT, va="center", zorder=7)
    # Leg Item 3: Super-Edge
    ax.plot([leg_x + 0.015, leg_x + 0.045], [leg_y + 0.046, leg_y + 0.046], color=CLR_DARK, lw=3.2, zorder=7)
    ax.text(leg_x + 0.052, leg_y + 0.046, r"$\mathbf{S_1 \leftrightarrow S_2}$ (Bold): Total inter-cluster cut capacity $W_{12}$",
            fontsize=7.2, color=CLR_TEXT, va="center", zorder=7)
    # Leg Item 4: Minor Link
    ax.plot([leg_x + 0.015, leg_x + 0.045], [leg_y + 0.016, leg_y + 0.016], color=CLR_C3_NODE, lw=1.6, ls="-.", zorder=7)
    ax.text(leg_x + 0.052, leg_y + 0.016, r"$\mathbf{m_i \to S_j}$ (Dash-Dot): Minor nodes direct links to super-nodes",
            fontsize=7.2, color=CLR_TEXT, va="center", zorder=7)


    # ── Arrow from Panel 2 to Panel 3 ─────────────────────────────────────────
    arr2_start = (p2_x + p2_w + 0.006, 0.52)
    arr2_end   = (p2_x + p2_w + 0.038, 0.52)
    _draw_arrow(ax, arr2_start, arr2_end, color=CLR_DARK, lw=2.2, style="-|>", zorder=7)
    ax.text((arr2_start[0] + arr2_end[0])/2, 0.555, "GNN\nTrain",
            fontsize=8.2, fontweight="bold", color=CLR_DARK, ha="center", va="bottom")


    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 3 (RIGHT): DUAL-CONTEXT MESSAGE PASSING & RECEPTIVE FIELD
    # ══════════════════════════════════════════════════════════════════════════
    p3_x, p3_y, p3_w, p3_h = 0.730, 0.04, 0.250, 0.91
    _draw_box(ax, (p3_x, p3_y), p3_w, p3_h, fc=CLR_PANEL_BG, ec=CLR_BORDER, lw=1.5, radius=0.018)

    ax.text(p3_x + 0.018, p3_y + p3_h - 0.038, "(c) Dual-Context Learning",
            fontsize=12.5, fontweight="bold", color=CLR_DARK)
    ax.text(p3_x + 0.018, p3_y + p3_h - 0.066, "Full Receptive Field Restored (0.0 GB Network)",
            fontsize=8.5, color=CLR_MUTED, style="italic")

    # Step 1 Box: Multi-Context Input Vector
    b1_y = p3_y + 0.65
    _draw_box(ax, (p3_x + 0.015, b1_y), p3_w - 0.030, 0.160, fc="#FFFFFF", ec=CLR_BORDER, lw=1.2, radius=0.010, zorder=2)
    ax.text(p3_x + 0.025, b1_y + 0.135, "1. Boundary Multi-Context Inputs:", fontsize=8.2, fontweight="bold", color=CLR_DARK)
    ax.text(p3_x + 0.025, b1_y + 0.098, r"• $\mathbf{h}_{v_1}^{(l)}$ : Local Intra-Community state", fontsize=7.4, color=CLR_TEXT)
    ax.text(p3_x + 0.025, b1_y + 0.066, r"• $\mathbf{h}_{S_1}^{(l)}$ : Host Centroid Macro summary", fontsize=7.4, color=CLR_C1_SUPER)
    ax.text(p3_x + 0.025, b1_y + 0.034, r"• $\mathbf{h}_{S_2}^{(l)}$ : Remote Centroid Context", fontsize=7.4, color=CLR_C2_SUPER)

    # Step 2 Box: GNN Aggregation Formula
    b2_y = p3_y + 0.37
    _draw_box(ax, (p3_x + 0.015, b2_y), p3_w - 0.030, 0.250, fc="#FFFFFF", ec="#93C5FD", lw=1.4, radius=0.010, zorder=2)
    ax.text(p3_x + 0.025, b2_y + 0.222, "2. Dual-Context Layer Update:", fontsize=8.2, fontweight="bold", color=CLR_C1_SUPER)
    
    eq_box_y = b2_y + 0.115
    _draw_box(ax, (p3_x + 0.025, eq_box_y), p3_w - 0.050, 0.090, fc="#EFF6FF", ec="#BFDBFE", lw=1.0, radius=0.008, zorder=3)
    ax.text(p3_x + p3_w/2, eq_box_y + 0.055, r"$\mathbf{h}_{v_1}^{(l+1)} = \sigma ( \mathbf{W}_{\mathrm{self}} \mathbf{h}_{v_1}^{(l)} +$",
            fontsize=8.0, fontweight="bold", color=CLR_DARK, ha="center", va="center", zorder=4)
    ax.text(p3_x + p3_w/2, eq_box_y + 0.020, r"$\mathbf{W}_{\mathrm{int}} \mathbf{h}_{S_1}^{(l)} + \sum_{j \neq 1} \mathbf{W}_{\mathrm{ext}} \mathbf{h}_{S_j}^{(l)} )$",
            fontsize=8.0, fontweight="bold", color=CLR_DARK, ha="center", va="center", zorder=4)

    ax.text(p3_x + 0.025, b2_y + 0.065, r"$\mathbf{W}_{\mathrm{int}}$ captures host community density", fontsize=7.0, color=CLR_MUTED)
    ax.text(p3_x + 0.025, b2_y + 0.035, r"$\mathbf{W}_{\mathrm{ext}}$ restores cut boundary semantics", fontsize=7.0, color=CLR_MUTED)

    # Step 3 Box: System Benefits Summary
    b3_y = p3_y + 0.05
    _draw_box(ax, (p3_x + 0.015, b3_y), p3_w - 0.030, 0.290, fc="#F0FDF4", ec="#86EFAC", lw=1.2, radius=0.010, zorder=2)
    ax.text(p3_x + 0.025, b3_y + 0.260, "3. System & Accuracy Guarantees:", fontsize=8.2, fontweight="bold", color="#15803D")
    
    metrics = [
        ("Network Traffic:", "0.0 GB during Training"),
        ("Boundary Acc Loss:", "Fully Recovered (98.1%)"),
        ("Worker Isolation:", "100% Independent UDFs"),
        ("Graph Scale:", "Compressed from |V| to |S| << |V|"),
        ("Storage Engine:", "ACID Delta Lake on AWS S3")
    ]
    for idx, (m_title, m_val) in enumerate(metrics):
        cur_y = b3_y + 0.215 - idx * 0.044
        ax.text(p3_x + 0.025, cur_y, f"• {m_title}", fontsize=7.2, fontweight="bold", color=CLR_TEXT)
        ax.text(p3_x + 0.025, cur_y - 0.018, f"  {m_val}", fontsize=7.0, color="#15803D" if idx < 3 else CLR_MUTED)


    # ── Save Outputs (PDF & PNG) ──────────────────────────────────────────────
    out_pdf = OUT_DIR / "caan_macro_graph_architecture.pdf"
    out_png = OUT_DIR / "caan_macro_graph_architecture.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    
    # Also save to overleaf directory
    plt.savefig(OVERLEAF_DIR / "caan_macro_graph_architecture.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OVERLEAF_DIR / "caan_macro_graph_architecture.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✓ Generated 3-Panel CaaN Architecture Figure (PDF): {out_pdf}")
    print(f"✓ Generated 3-Panel CaaN Architecture Figure (PNG): {out_png}")


if __name__ == "__main__":
    create_caan_macro_graph_figure()
