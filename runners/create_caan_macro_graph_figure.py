#!/usr/bin/env python3
"""
create_caan_macro_graph_figure.py
══════════════════════════════════════════════════════════════════════════════
Publication-grade figures for Two-Level CaaN Macro-Graph Architecture:
  - Version 1: Detailed Architecture (with math subtitles).
  - Version 2: Minimalist & Clean (no formula subtitles under headers).

Engineered with:
  - Tight, compact, elegant canvas proportions with zero dead whitespace.
  - Symmetrically filled panels, rich GNN blocks, and full-width semantics card.
  - Zero text/line/ring collisions and pristine vector alignment.
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

# ── Academic Color Palette ───────────────────────────────────────────────────
CLR_BG          = "#FFFFFF"
CLR_DARK        = "#0F172A"   # Slate 900
CLR_TEXT        = "#1E293B"   # Slate 800
CLR_MUTED       = "#64748B"   # Slate 500
CLR_BORDER      = "#CBD5E1"   # Slate 300
CLR_PANEL_BG    = "#F8FAFC"   # Slate 50

# Community Colors
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


def _draw_box(ax, xy, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.3, radius=0.010, zorder=1):
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def _draw_arrow(ax, start, end, color="#0F172A", lw=1.6, style="-|>",
                connectionstyle="arc3,rad=0.0", ls="-", zorder=5):
    arr = FancyArrowPatch(start, end, arrowstyle=style,
                          connectionstyle=connectionstyle,
                          color=color, linewidth=lw, linestyle=ls,
                          mutation_scale=12, zorder=zorder)
    ax.add_patch(arr)
    return arr


def _draw_node(ax, pt, color, label="", zorder=6, r=0.013, text_color="#FFFFFF",
               ring_color=None, ring_r=0.0175, font_size=7.6):
    if ring_color:
        c_ring = Circle(pt, ring_r, facecolor=ring_color, edgecolor="none", alpha=0.30, zorder=zorder-1)
        ax.add_patch(c_ring)
        c_ring_b = Circle(pt, ring_r, facecolor="none", edgecolor=ring_color, lw=1.2, ls="--", zorder=zorder)
        ax.add_patch(c_ring_b)
    c = Circle(pt, r, facecolor=color, edgecolor="#FFFFFF", lw=1.2, zorder=zorder)
    ax.add_patch(c)
    if label:
        ax.text(pt[0], pt[1], label, color=text_color, fontsize=font_size,
                fontweight="bold", ha="center", va="center", zorder=zorder+1)


def _draw_super_node(ax, pt, color, label="", zorder=6, r=0.021, text_color="#FFFFFF"):
    c_glow = Circle(pt, r + 0.007, facecolor=color, alpha=0.18, zorder=zorder-1)
    ax.add_patch(c_glow)
    c = Circle(pt, r, facecolor=color, edgecolor="#FFFFFF", lw=2.0, zorder=zorder)
    ax.add_patch(c)
    if label:
        ax.text(pt[0], pt[1], label, color=text_color, fontsize=9.0,
                fontweight="bold", ha="center", va="center", zorder=zorder+1)


# ══════════════════════════════════════════════════════════════════════════════
# VERSION 1: DETAILED ARCHITECTURE FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def create_smooth_caan_figure():
    fig, ax = plt.subplots(figsize=(20.0, 9.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.970, "Two-Level Graph Representation Learning: CaaN Macro-Graph & Boundary Wiring Architecture",
            fontsize=15.5, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: INPUT GRAPH & PARTITIONING
    # ──────────────────────────────────────────────────────────────────────────
    p1_x, p1_y, p1_w, p1_h = 0.015, 0.035, 0.225, 0.895
    _draw_box(ax, (p1_x, p1_y), p1_w, p1_h, fc=CLR_PANEL_BG, ec=CLR_BORDER, lw=1.4)
    ax.text(p1_x + p1_w/2, p1_y + p1_h - 0.024, "(a) Input Graph & Partitioning", fontsize=11.5, fontweight="bold", color=CLR_DARK, ha="center")
    ax.text(p1_x + p1_w/2, p1_y + p1_h - 0.044, r"$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$ with 2 Major + Minor Clusters", fontsize=7.5, color=CLR_MUTED, ha="center")

    # C1 Box
    c1_bx, c1_by, c1_bw, c1_bh = p1_x + 0.010, p1_y + 0.540, p1_w - 0.020, 0.280
    _draw_box(ax, (c1_bx, c1_by), c1_bw, c1_bh, fc=CLR_C1_BG, ec=CLR_C1_BORDER, lw=1.2)
    ax.text(c1_bx + 0.010, c1_by + c1_bh - 0.016, r"Major Community $C_1$", fontsize=8.0, fontweight="bold", color=CLR_C1_SUPER)
    ax.text(c1_bx + 0.010, c1_by + c1_bh - 0.030, r"$(|C_1| \geq K)$", fontsize=6.8, color=CLR_MUTED)

    # C2 Box
    c2_bx, c2_by, c2_bw, c2_bh = p1_x + 0.010, p1_y + 0.235, p1_w - 0.020, 0.280
    _draw_box(ax, (c2_bx, c2_by), c2_bw, c2_bh, fc=CLR_C2_BG, ec=CLR_C2_BORDER, lw=1.2)
    ax.text(c2_bx + 0.010, c2_by + c2_bh - 0.016, r"Major Community $C_2$", fontsize=8.0, fontweight="bold", color=CLR_C2_SUPER)
    ax.text(c2_bx + 0.010, c2_by + c2_bh - 0.030, r"$(|C_2| \geq K)$", fontsize=6.8, color=CLR_MUTED)

    # Minor Box
    min_bx, min_by, min_bw, min_bh = p1_x + 0.010, p1_y + 0.015, p1_w - 0.020, 0.195
    _draw_box(ax, (min_bx, min_by), min_bw, min_bh, fc=CLR_MIN_BG, ec=CLR_MIN_BORDER, lw=1.2)
    ax.text(min_bx + min_bw/2, min_by + 0.020, r"Minor Communities $V_{\mathrm{minor}}$ ($|C_k| < K$)",
            fontsize=7.6, fontweight="bold", color="#B45309", ha="center")

    # C1 Nodes
    u1, u2, u3 = (c1_bx + 0.038, c1_by + 0.195), (c1_bx + 0.098, c1_by + 0.215), (c1_bx + 0.048, c1_by + 0.105)
    v1 = (c1_bx + 0.128, c1_by + 0.080)
    for pa, pb in [(u1, u2), (u1, u3), (u2, u3), (u3, v1), (u2, v1)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C1_NODE, lw=1.2, zorder=3)
    _draw_node(ax, u1, CLR_C1_NODE, "$u_1$")
    _draw_node(ax, u2, CLR_C1_NODE, "$u_2$")
    _draw_node(ax, u3, CLR_C1_NODE, "$u_3$")
    _draw_node(ax, v1, CLR_BND_NODE, "$v_1$", ring_color=CLR_BND_RING, ring_r=0.016)
    ax.text(v1[0] + 0.022, v1[1] + 0.005, r"Boundary $v_1$", fontsize=7.2, fontweight="bold", color=CLR_BND_NODE, ha="left", va="center")

    # C2 Nodes
    w1, w2, w3 = (c2_bx + 0.042, c2_by + 0.075), (c2_bx + 0.098, c2_by + 0.065), (c2_bx + 0.048, c2_by + 0.175)
    v2 = (c2_bx + 0.128, c2_by + 0.200)
    for pa, pb in [(w1, w2), (w1, w3), (w2, w3), (w3, v2), (w2, v2)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C2_NODE, lw=1.2, zorder=3)
    _draw_node(ax, w1, CLR_C2_NODE, "$w_1$")
    _draw_node(ax, w2, CLR_C2_NODE, "$w_2$")
    _draw_node(ax, w3, CLR_C2_NODE, "$w_3$")
    _draw_node(ax, v2, CLR_BND_NODE, "$v_2$", ring_color=CLR_BND_RING, ring_r=0.016)
    ax.text(v2[0] + 0.022, v2[1] - 0.005, r"Boundary $v_2$", fontsize=7.2, fontweight="bold", color=CLR_BND_NODE, ha="left", va="center")

    # Minor Nodes
    m1, m2, m3 = (min_bx + 0.040, min_by + 0.115), (min_bx + 0.102, min_by + 0.125), (min_bx + 0.160, min_by + 0.100)
    for pa, pb in [(m1, m2), (m2, m3)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_MIN_NODE, lw=1.2, zorder=3)
    _draw_node(ax, m1, CLR_MIN_NODE, "$m_1$")
    _draw_node(ax, m2, CLR_MIN_NODE, "$m_2$")
    _draw_node(ax, m3, CLR_MIN_NODE, "$m_3$")

    # Cut Edges
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=CLR_CUT_LINE, lw=1.8, ls="--", zorder=4)
    ax.text(v1[0] - 0.016, (v1[1] + v2[1])/2, "Severed Cut $(v_1, v_2)$",
            fontsize=6.8, fontweight="bold", color=CLR_CUT_LINE, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec=CLR_CUT_LINE, lw=0.8, alpha=0.95), zorder=5)
    ax.plot([w1[0], m1[0]], [w1[1], m1[1]], color=CLR_CUT_LINE, lw=1.4, ls=":", zorder=4)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: LOCAL GRL (Balanced Width & Full Flow)
    # ──────────────────────────────────────────────────────────────────────────
    p2_x, p2_y, p2_w, p2_h = 0.252, 0.490, 0.496, 0.440
    _draw_box(ax, (p2_x, p2_y), p2_w, p2_h, fc=CLR_PANEL_BG, ec="#2563EB", lw=1.8)
    ax.text(p2_x + p2_w/2, p2_y + p2_h - 0.024, r"(b) Local GRL on Augmented Subgraph $(\mathcal{G}_{\mathrm{CaaN}} \setminus \{S_1\}) \cup C_1$",
            fontsize=11.5, fontweight="bold", color="#1D4ED8", ha="center")
    ax.text(p2_x + p2_w/2, p2_y + p2_h - 0.044, "Boundary Vertex Wiring: Connects to External Super-Node $S_2$; Internal Super-Node $S_1$ is strictly EXCLUDED",
            fontsize=7.8, color=CLR_TEXT, ha="center")

    sub_bx, sub_by, sub_bw, sub_bh = p2_x + 0.014, p2_y + 0.018, 0.245, p2_h - 0.072
    _draw_box(ax, (sub_bx, sub_by), sub_bw, sub_bh, fc="#FFFFFF", ec=CLR_C1_BORDER, lw=1.3, radius=0.010)

    _draw_box(ax, (sub_bx + 0.010, sub_by + sub_bh - 0.035), 0.155, 0.026, fc="#FEF2F2", ec=CLR_CUT_LINE, lw=1.0, radius=0.004)
    ax.text(sub_bx + 0.0875, sub_by + sub_bh - 0.022, r"Internal Super-Node $S_1 \notin$ Local",
            fontsize=6.8, fontweight="bold", color=CLR_CUT_LINE, ha="center", va="center")

    lv_u1, lv_u2, lv_u3 = (sub_bx + 0.040, sub_by + 0.250), (sub_bx + 0.100, sub_by + 0.270), (sub_bx + 0.048, sub_by + 0.145)
    lv_v1 = (sub_bx + 0.115, sub_by + 0.125)
    for pa, pb in [(lv_u1, lv_u2), (lv_u1, lv_u3), (lv_u2, lv_u3), (lv_u3, lv_v1), (lv_u2, lv_v1)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C1_NODE, lw=1.2, zorder=3)
    _draw_node(ax, lv_u1, CLR_C1_NODE, "$u_1$")
    _draw_node(ax, lv_u2, CLR_C1_NODE, "$u_2$")
    _draw_node(ax, lv_u3, CLR_C1_NODE, "$u_3$")
    _draw_node(ax, lv_v1, CLR_BND_NODE, "$v_1$", ring_color=CLR_BND_RING, ring_r=0.016)
    ax.text(lv_v1[0] - 0.020, lv_v1[1] - 0.012, r"Boundary $v_1$", fontsize=7.2, fontweight="bold", color=CLR_BND_NODE, ha="right", va="center")

    ext_s2 = (sub_bx + 0.195, sub_by + 0.165)
    _draw_super_node(ax, ext_s2, CLR_C2_SUPER, "$S_2$")
    ax.text(ext_s2[0], ext_s2[1] + 0.028, "External Super-Node", fontsize=6.5, fontweight="bold", color=CLR_C2_SUPER, ha="center")
    ax.text(ext_s2[0], ext_s2[1] - 0.028, r"$\mathbf{x}_{S_2} = \mathrm{avg}(C_2)$", fontsize=6.2, color=CLR_MUTED, ha="center")

    ax.plot([lv_v1[0], ext_s2[0]], [lv_v1[1], ext_s2[1]], color=CLR_C2_SUPER, lw=2.4, zorder=4)
    _draw_box(ax, (sub_bx + 0.095, sub_by + 0.025), 0.140, 0.038, fc="#FAF5FF", ec=CLR_C2_BORDER, lw=0.9, radius=0.004)
    ax.text(sub_bx + 0.165, sub_by + 0.044, "Boundary Link $(v_1, S_2)$\nRestores cut $(v_1, v_2)$",
            fontsize=6.2, fontweight="bold", color=CLR_C2_SUPER, ha="center", va="center")

    # Local GNN Module (Symmetrically spaced)
    _draw_arrow(ax, (sub_bx + sub_bw + 0.008, sub_by + sub_bh/2), (p2_x + 0.285, sub_by + sub_bh/2), color=CLR_DARK, lw=1.8)
    _draw_box(ax, (p2_x + 0.290, sub_by + sub_bh/2 - 0.060), 0.085, 0.120, fc="#E2E8F0", ec=CLR_DARK, lw=1.3, radius=0.008)
    ax.text(p2_x + 0.290 + 0.0425, sub_by + sub_bh/2, "Local\nGNN\nModel", fontsize=8.6, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    _draw_arrow(ax, (p2_x + 0.380, sub_by + sub_bh/2), (p2_x + 0.405, sub_by + sub_bh/2), color=CLR_DARK, lw=1.8)

    # Local Embedding Matrix
    lemb_x, lemb_y, lemb_w, lemb_h = p2_x + 0.410, sub_by + 0.025, 0.075, sub_bh - 0.050
    _draw_box(ax, (lemb_x, lemb_y), lemb_w, lemb_h, fc="#FFFFFF", ec=CLR_DARK, lw=1.3, radius=0.006)
    rect_c1_emb = Rectangle((lemb_x, lemb_y + lemb_h * 0.25), lemb_w, lemb_h * 0.75, facecolor=CLR_C1_BG, edgecolor=CLR_C1_BORDER, lw=0.9)
    ax.add_patch(rect_c1_emb)
    ax.text(lemb_x + lemb_w/2, lemb_y + lemb_h * 0.625, r"$\mathbf{h}_{V_{C_1}}$" "\n(C1 Embeddings)",
            fontsize=7.2, fontweight="bold", color=CLR_C1_SUPER, ha="center", va="center")
    ax.text(lemb_x + lemb_w/2, lemb_y + lemb_h * 0.125, r"$S_2$ Discarded", fontsize=6.2, color=CLR_MUTED, ha="center", va="center")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: GLOBAL GRL (Balanced Width & Full Flow)
    # ──────────────────────────────────────────────────────────────────────────
    p3_x, p3_y, p3_w, p3_h = 0.252, 0.035, 0.496, 0.440
    _draw_box(ax, (p3_x, p3_y), p3_w, p3_h, fc=CLR_PANEL_BG, ec="#059669", lw=1.8)
    ax.text(p3_x + p3_w/2, p3_y + p3_h - 0.024, r"(c) Global GRL on CaaN Macro-Graph $\mathcal{G}_{\mathrm{CaaN}} \cup V_{\mathrm{minor}}$",
            fontsize=11.5, fontweight="bold", color="#047857", ha="center")
    ax.text(p3_x + p3_w/2, p3_y + p3_h - 0.044, "Macro-Graph Topology: Super-Nodes ($S_1, S_2$) + Super-Edge + Explicit Minor Vertices ($m_1, m_2, m_3$)",
            fontsize=7.8, color=CLR_TEXT, ha="center")

    mg_bx, mg_by, mg_bw, mg_bh = p3_x + 0.014, p3_y + 0.018, 0.245, p3_h - 0.072
    _draw_box(ax, (mg_bx, mg_by), mg_bw, mg_bh, fc="#FFFFFF", ec="#059669", lw=1.3, radius=0.010)

    gs1, gs2 = (mg_bx + 0.075, mg_by + 0.215), (mg_bx + 0.170, mg_by + 0.215)
    gm1, gm2, gm3 = (mg_bx + 0.045, mg_by + 0.065), (mg_bx + 0.120, mg_by + 0.065), (mg_bx + 0.195, mg_by + 0.065)

    ax.plot([gs1[0], gs2[0]], [gs1[1], gs2[1]], color=CLR_DARK, lw=2.4, zorder=3)
    ax.text((gs1[0] + gs2[0])/2, gs1[1] + 0.024, r"Super-Edge $e(S_1, S_2)$", fontsize=6.8, fontweight="bold", color=CLR_DARK, ha="center")

    for pa, pb in [(gm1, gm2), (gm2, gm3)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_MIN_NODE, lw=1.4, zorder=3)
    ax.plot([gs1[0], gm1[0]], [gs1[1], gm1[1]], color=CLR_DARK, lw=1.2, ls="--", zorder=3)
    ax.plot([gs1[0], gm2[0]], [gs1[1], gm2[1]], color=CLR_DARK, lw=1.2, ls="--", zorder=3)
    ax.plot([gs2[0], gm3[0]], [gs2[1], gm3[1]], color=CLR_DARK, lw=1.2, ls="--", zorder=3)

    _draw_super_node(ax, gs1, CLR_C1_SUPER, "$S_1$")
    _draw_super_node(ax, gs2, CLR_C2_SUPER, "$S_2$")
    ax.text(gs1[0], gs1[1] - 0.028, r"$\mathbf{x}_{S_1} = \mathrm{avg}(C_1)$", fontsize=6.3, color=CLR_MUTED, ha="center")
    ax.text(gs2[0], gs2[1] - 0.028, r"$\mathbf{x}_{S_2} = \mathrm{avg}(C_2)$", fontsize=6.3, color=CLR_MUTED, ha="center")

    _draw_node(ax, gm1, CLR_MIN_NODE, "$m_1$")
    _draw_node(ax, gm2, CLR_MIN_NODE, "$m_2$")
    _draw_node(ax, gm3, CLR_MIN_NODE, "$m_3$")
    ax.text((gm1[0] + gm3[0])/2, gm1[1] - 0.022, "Retained Minor Vertices (No Over-Smoothing)", fontsize=6.6, fontweight="bold", color="#B45309", ha="center")

    # Global GNN Module (Symmetrically spaced)
    _draw_arrow(ax, (mg_bx + mg_bw + 0.008, mg_by + mg_bh/2), (p3_x + 0.285, mg_by + mg_bh/2), color=CLR_DARK, lw=1.8)
    _draw_box(ax, (p3_x + 0.290, mg_by + mg_bh/2 - 0.060), 0.085, 0.120, fc="#E2E8F0", ec=CLR_DARK, lw=1.3, radius=0.008)
    ax.text(p3_x + 0.290 + 0.0425, mg_by + mg_bh/2, "Global\nGNN\nModel", fontsize=8.6, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    _draw_arrow(ax, (p3_x + 0.380, mg_by + mg_bh/2), (p3_x + 0.405, mg_by + mg_bh/2), color=CLR_DARK, lw=1.8)

    # Global Embedding Matrix
    gemb_x, gemb_y, gemb_w, gemb_h = p3_x + 0.410, mg_by + 0.025, 0.075, mg_bh - 0.050
    _draw_box(ax, (gemb_x, gemb_y), gemb_w, gemb_h, fc="#FFFFFF", ec=CLR_DARK, lw=1.3, radius=0.006)
    rect_min_emb = Rectangle((gemb_x, gemb_y), gemb_w, gemb_h * 0.60, facecolor=CLR_MIN_BG, edgecolor=CLR_MIN_BORDER, lw=0.9)
    ax.add_patch(rect_min_emb)
    ax.text(gemb_x + gemb_w/2, gemb_y + gemb_h * 0.30, r"$\mathbf{h}_{V_{\mathrm{minor}}}$" "\n(Minor Embeddings)",
            fontsize=7.2, fontweight="bold", color="#B45309", ha="center", va="center")
    ax.text(gemb_x + gemb_w/2, gemb_y + gemb_h * 0.80, r"$S_1, S_2$ Discarded", fontsize=6.2, color=CLR_MUTED, ha="center", va="center")

    # Clean Connecting Arrows between Left and Center Panels
    _draw_arrow(ax, (p1_x + p1_w + 0.003, p2_y + p2_h * 0.50), (p2_x - 0.003, p2_y + p2_h * 0.50),
                color=CLR_C1_SUPER, lw=2.0, style="-|>", connectionstyle="arc3,rad=0.0")
    _draw_arrow(ax, (p1_x + p1_w + 0.003, p3_y + p3_h * 0.50), (p3_x - 0.003, p3_y + p3_h * 0.50),
                color="#047857", lw=2.0, style="-|>", connectionstyle="arc3,rad=0.0")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4: FINAL GRL ASSEMBLY (Filled, Neatly Proportioned)
    # ──────────────────────────────────────────────────────────────────────────
    p4_x, p4_y, p4_w, p4_h = 0.760, 0.035, 0.225, 0.895
    _draw_box(ax, (p4_x, p4_y), p4_w, p4_h, fc=CLR_PANEL_BG, ec=CLR_DARK, lw=1.8)
    ax.text(p4_x + p4_w/2, p4_y + p4_h - 0.024, "(d) Final GRL Embedding Assembly", fontsize=11.5, fontweight="bold", color=CLR_DARK, ha="center")
    ax.text(p4_x + p4_w/2, p4_y + p4_h - 0.044, r"$f(\mathcal{G}) = [ \mathbf{h}_{V_{\mathrm{minor}}} \mid \bigcup_{i=1}^m \mathbf{h}_{V_{C_i}} ] \in \mathbb{R}^{|\mathcal{V}| \times d}$",
            fontsize=7.6, color=CLR_MUTED, ha="center")

    mat_x, mat_y, mat_w, mat_h = p4_x + 0.015, p4_y + 0.455, 0.075, 0.365
    _draw_box(ax, (mat_x, mat_y), mat_w, mat_h, fc="#FFFFFF", ec=CLR_DARK, lw=1.5, radius=0.006)

    slices = [
        (CLR_C1_BG, CLR_C1_BORDER, 0.35, r"$\mathbf{h}_{V_{C_1}}$", "Major $C_1$ (Local GRL)", CLR_C1_SUPER),
        (CLR_C2_BG, CLR_C2_BORDER, 0.35, r"$\mathbf{h}_{V_{C_2}}$", "Major $C_2$ (Local GRL)", CLR_C2_SUPER),
        (CLR_MIN_BG, CLR_MIN_BORDER, 0.30, r"$\mathbf{h}_{V_{\mathrm{minor}}}$", "Minor Nodes (Global)", "#B45309")
    ]
    cur_y = mat_y + mat_h
    for clr_bg, clr_bd, frac, tag, desc, clr_txt in slices:
        sh = mat_h * frac
        cur_y -= sh
        rect = Rectangle((mat_x, cur_y), mat_w, sh, facecolor=clr_bg, edgecolor=clr_bd, lw=1.0, zorder=6)
        ax.add_patch(rect)
        ax.text(mat_x + mat_w/2, cur_y + sh/2, tag, fontsize=7.4, fontweight="bold", color=clr_txt, ha="center", va="center", zorder=7)
        ax.text(mat_x + mat_w + 0.008, cur_y + sh/2, desc, fontsize=7.0, fontweight="bold", color=CLR_TEXT, va="center", zorder=7)

    _draw_arrow(ax, (p2_x + p2_w + 0.003, p2_y + p2_h * 0.45), (p4_x - 0.003, mat_y + mat_h * 0.80),
                color=CLR_C1_SUPER, lw=1.8, style="-|>", connectionstyle="arc3,rad=-0.08", zorder=7)
    _draw_arrow(ax, (p3_x + p3_w + 0.003, p3_y + p3_h * 0.45), (p4_x - 0.003, mat_y + mat_h * 0.20),
                color="#B45309", lw=1.8, style="-|>", connectionstyle="arc3,rad=0.08", zorder=7)

    rule_bx, rule_by, rule_bw, rule_bh = p4_x + 0.010, p4_y + 0.015, p4_w - 0.020, 0.420
    _draw_box(ax, (rule_bx, rule_by), rule_bw, rule_bh, fc="#FFFFFF", ec=CLR_BORDER, lw=1.1, radius=0.008)
    ax.text(rule_bx + 0.012, rule_by + rule_bh - 0.022, "Boundary & Macro Semantics:", fontsize=7.8, fontweight="bold", color=CLR_DARK)
    
    rules_text = (
        r"$\mathbf{1.\;Boundary\;Vertices:}$" "\n"
        r"   $v \in C_i$ has cut edge $(v, w)$ to $w \in C_j$." "\n\n"
        r"$\mathbf{2.\;Internal\;Connectivity\;(Local):}$" "\n"
        r"   $v$ keeps all intra-community edges $(v, u)$." "\n"
        r"   Host super-node $S_i$ is strictly EXCLUDED." "\n\n"
        r"$\mathbf{3.\;External\;Connectivity\;(Local):}$" "\n"
        r"   Severed edge $(v, w)$ maps to $(v, S_j)$," "\n"
        r"   connecting $v$ directly to external $S_j$." "\n\n"
        r"$\mathbf{4.\;Minor\;Vertices\;(Global):}$" "\n"
        r"   $|C_k| < K$ retained explicitly in $\mathcal{G}_{\mathrm{CaaN}}$" "\n"
        r"   to completely prevent over-smoothing."
    )
    ax.text(rule_bx + 0.010, rule_by + 0.200, rules_text, fontsize=6.8, color=CLR_TEXT, va="center")

    out_pdf = OUT_DIR / "caan_macro_graph_architecture.pdf"
    out_png = OUT_DIR / "caan_macro_graph_architecture.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(OVERLEAF_DIR / "caan_macro_graph_architecture.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OVERLEAF_DIR / "caan_macro_graph_architecture.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Generated Version 1 (Detailed): {out_pdf}")


# ══════════════════════════════════════════════════════════════════════════════
# VERSION 2: CLEAN MINIMALIST ARCHITECTURE FIGURE (NO PANEL SUBTITLES/EQUATIONS)
# ══════════════════════════════════════════════════════════════════════════════
def create_smooth_caan_figure_v2():
    fig, ax = plt.subplots(figsize=(20.0, 9.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main Header
    ax.text(0.50, 0.970, "Two-Level Graph Representation Learning: CaaN Macro-Graph & Boundary Wiring Architecture",
            fontsize=15.5, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: INPUT GRAPH & PARTITIONING
    # ──────────────────────────────────────────────────────────────────────────
    p1_x, p1_y, p1_w, p1_h = 0.015, 0.035, 0.225, 0.895
    _draw_box(ax, (p1_x, p1_y), p1_w, p1_h, fc=CLR_PANEL_BG, ec=CLR_BORDER, lw=1.4)
    ax.text(p1_x + p1_w/2, p1_y + p1_h - 0.026, "(a) Input Graph & Partitioning", fontsize=12.0, fontweight="bold", color=CLR_DARK, ha="center")

    # Major C1 Box
    c1_bx, c1_by, c1_bw, c1_bh = p1_x + 0.010, p1_y + 0.550, p1_w - 0.020, 0.290
    _draw_box(ax, (c1_bx, c1_by), c1_bw, c1_bh, fc=CLR_C1_BG, ec=CLR_C1_BORDER, lw=1.2)
    ax.text(c1_bx + 0.010, c1_by + c1_bh - 0.016, r"Major Community $C_1$", fontsize=8.2, fontweight="bold", color=CLR_C1_SUPER)
    ax.text(c1_bx + 0.010, c1_by + c1_bh - 0.030, r"$(|C_1| \geq K)$", fontsize=7.0, color=CLR_MUTED)

    # Major C2 Box
    c2_bx, c2_by, c2_bw, c2_bh = p1_x + 0.010, p1_y + 0.235, p1_w - 0.020, 0.290
    _draw_box(ax, (c2_bx, c2_by), c2_bw, c2_bh, fc=CLR_C2_BG, ec=CLR_C2_BORDER, lw=1.2)
    ax.text(c2_bx + 0.010, c2_by + c2_bh - 0.016, r"Major Community $C_2$", fontsize=8.2, fontweight="bold", color=CLR_C2_SUPER)
    ax.text(c2_bx + 0.010, c2_by + c2_bh - 0.030, r"$(|C_2| \geq K)$", fontsize=7.0, color=CLR_MUTED)

    # Minor Communities Box
    min_bx, min_by, min_bw, min_bh = p1_x + 0.010, p1_y + 0.015, p1_w - 0.020, 0.195
    _draw_box(ax, (min_bx, min_by), min_bw, min_bh, fc=CLR_MIN_BG, ec=CLR_MIN_BORDER, lw=1.2)
    ax.text(min_bx + min_bw/2, min_by + 0.020, r"Minor Communities $V_{\mathrm{minor}}$ ($|C_k| < K$)",
            fontsize=7.8, fontweight="bold", color="#B45309", ha="center")

    # Nodes in C1
    u1, u2, u3 = (c1_bx + 0.038, c1_by + 0.200), (c1_bx + 0.098, c1_by + 0.220), (c1_bx + 0.048, c1_by + 0.110)
    v1 = (c1_bx + 0.128, c1_by + 0.080)
    for pa, pb in [(u1, u2), (u1, u3), (u2, u3), (u3, v1), (u2, v1)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C1_NODE, lw=1.2, zorder=3)
    _draw_node(ax, u1, CLR_C1_NODE, "$u_1$")
    _draw_node(ax, u2, CLR_C1_NODE, "$u_2$")
    _draw_node(ax, u3, CLR_C1_NODE, "$u_3$")
    _draw_node(ax, v1, CLR_BND_NODE, "$v_1$", ring_color=CLR_BND_RING, ring_r=0.016)
    ax.text(v1[0] + 0.022, v1[1] + 0.005, r"Boundary $v_1$", fontsize=7.2, fontweight="bold", color=CLR_BND_NODE, ha="left", va="center")

    # Nodes in C2
    w1, w2, w3 = (c2_bx + 0.042, c2_by + 0.075), (c2_bx + 0.098, c2_by + 0.065), (c2_bx + 0.048, c2_by + 0.175)
    v2 = (c2_bx + 0.128, c2_by + 0.205)
    for pa, pb in [(w1, w2), (w1, w3), (w2, w3), (w3, v2), (w2, v2)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C2_NODE, lw=1.2, zorder=3)
    _draw_node(ax, w1, CLR_C2_NODE, "$w_1$")
    _draw_node(ax, w2, CLR_C2_NODE, "$w_2$")
    _draw_node(ax, w3, CLR_C2_NODE, "$w_3$")
    _draw_node(ax, v2, CLR_BND_NODE, "$v_2$", ring_color=CLR_BND_RING, ring_r=0.016)
    ax.text(v2[0] + 0.022, v2[1] - 0.005, r"Boundary $v_2$", fontsize=7.2, fontweight="bold", color=CLR_BND_NODE, ha="left", va="center")

    # Minor Nodes
    m1, m2, m3 = (min_bx + 0.040, min_by + 0.115), (min_bx + 0.102, min_by + 0.125), (min_bx + 0.160, min_by + 0.100)
    for pa, pb in [(m1, m2), (m2, m3)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_MIN_NODE, lw=1.2, zorder=3)
    _draw_node(ax, m1, CLR_MIN_NODE, "$m_1$")
    _draw_node(ax, m2, CLR_MIN_NODE, "$m_2$")
    _draw_node(ax, m3, CLR_MIN_NODE, "$m_3$")

    # Cross-partition Severed Cut Edges
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=CLR_CUT_LINE, lw=1.8, ls="--", zorder=4)
    ax.text(v1[0] - 0.016, (v1[1] + v2[1])/2, "Severed Cut $(v_1, v_2)$",
            fontsize=6.8, fontweight="bold", color=CLR_CUT_LINE, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec=CLR_CUT_LINE, lw=0.8, alpha=0.95), zorder=5)
    ax.plot([w1[0], m1[0]], [w1[1], m1[1]], color=CLR_CUT_LINE, lw=1.4, ls=":", zorder=4)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: LOCAL GRL ON AUGMENTED SUBGRAPH
    # ──────────────────────────────────────────────────────────────────────────
    p2_x, p2_y, p2_w, p2_h = 0.252, 0.490, 0.496, 0.440
    _draw_box(ax, (p2_x, p2_y), p2_w, p2_h, fc=CLR_PANEL_BG, ec="#2563EB", lw=1.8)
    ax.text(p2_x + p2_w/2, p2_y + p2_h - 0.026, r"(b) Local GRL on Augmented Subgraph $(\mathcal{G}_{\mathrm{CaaN}} \setminus \{S_1\}) \cup C_1$",
            fontsize=12.0, fontweight="bold", color="#1D4ED8", ha="center")

    sub_bx, sub_by, sub_bw, sub_bh = p2_x + 0.014, p2_y + 0.018, 0.245, p2_h - 0.062
    _draw_box(ax, (sub_bx, sub_by), sub_bw, sub_bh, fc="#FFFFFF", ec=CLR_C1_BORDER, lw=1.3, radius=0.010)

    _draw_box(ax, (sub_bx + 0.010, sub_by + sub_bh - 0.035), 0.155, 0.026, fc="#FEF2F2", ec=CLR_CUT_LINE, lw=1.0, radius=0.004)
    ax.text(sub_bx + 0.0875, sub_by + sub_bh - 0.022, r"Internal Super-Node $S_1 \notin$ Local",
            fontsize=6.8, fontweight="bold", color=CLR_CUT_LINE, ha="center", va="center")

    lv_u1, lv_u2, lv_u3 = (sub_bx + 0.040, sub_by + 0.250), (sub_bx + 0.100, sub_by + 0.270), (sub_bx + 0.048, sub_by + 0.145)
    lv_v1 = (sub_bx + 0.115, sub_by + 0.125)
    for pa, pb in [(lv_u1, lv_u2), (lv_u1, lv_u3), (lv_u2, lv_u3), (lv_u3, lv_v1), (lv_u2, lv_v1)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_C1_NODE, lw=1.2, zorder=3)
    _draw_node(ax, lv_u1, CLR_C1_NODE, "$u_1$")
    _draw_node(ax, lv_u2, CLR_C1_NODE, "$u_2$")
    _draw_node(ax, lv_u3, CLR_C1_NODE, "$u_3$")
    _draw_node(ax, lv_v1, CLR_BND_NODE, "$v_1$", ring_color=CLR_BND_RING, ring_r=0.016)
    ax.text(lv_v1[0] - 0.020, lv_v1[1] - 0.012, r"Boundary $v_1$", fontsize=7.2, fontweight="bold", color=CLR_BND_NODE, ha="right", va="center")

    ext_s2 = (sub_bx + 0.195, sub_by + 0.165)
    _draw_super_node(ax, ext_s2, CLR_C2_SUPER, "$S_2$")
    ax.text(ext_s2[0], ext_s2[1] + 0.028, "External Super-Node", fontsize=6.5, fontweight="bold", color=CLR_C2_SUPER, ha="center")
    ax.text(ext_s2[0], ext_s2[1] - 0.028, r"$\mathbf{x}_{S_2} = \mathrm{avg}(C_2)$", fontsize=6.2, color=CLR_MUTED, ha="center")

    ax.plot([lv_v1[0], ext_s2[0]], [lv_v1[1], ext_s2[1]], color=CLR_C2_SUPER, lw=2.4, zorder=4)
    _draw_box(ax, (sub_bx + 0.095, sub_by + 0.025), 0.140, 0.038, fc="#FAF5FF", ec=CLR_C2_BORDER, lw=0.9, radius=0.004)
    ax.text(sub_bx + 0.165, sub_by + 0.044, "Boundary Link $(v_1, S_2)$\nRestores cut $(v_1, v_2)$",
            fontsize=6.2, fontweight="bold", color=CLR_C2_SUPER, ha="center", va="center")

    # Local GNN Module (Symmetrically spaced)
    _draw_arrow(ax, (sub_bx + sub_bw + 0.008, sub_by + sub_bh/2), (p2_x + 0.285, sub_by + sub_bh/2), color=CLR_DARK, lw=1.8)
    _draw_box(ax, (p2_x + 0.290, sub_by + sub_bh/2 - 0.060), 0.085, 0.120, fc="#E2E8F0", ec=CLR_DARK, lw=1.3, radius=0.008)
    ax.text(p2_x + 0.290 + 0.0425, sub_by + sub_bh/2, "Local\nGNN\nModel", fontsize=8.6, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    _draw_arrow(ax, (p2_x + 0.380, sub_by + sub_bh/2), (p2_x + 0.405, sub_by + sub_bh/2), color=CLR_DARK, lw=1.8)

    # Local Embedding Matrix
    lemb_x, lemb_y, lemb_w, lemb_h = p2_x + 0.410, sub_by + 0.025, 0.075, sub_bh - 0.050
    _draw_box(ax, (lemb_x, lemb_y), lemb_w, lemb_h, fc="#FFFFFF", ec=CLR_DARK, lw=1.3, radius=0.006)
    rect_c1_emb = Rectangle((lemb_x, lemb_y + lemb_h * 0.25), lemb_w, lemb_h * 0.75, facecolor=CLR_C1_BG, edgecolor=CLR_C1_BORDER, lw=0.9)
    ax.add_patch(rect_c1_emb)
    ax.text(lemb_x + lemb_w/2, lemb_y + lemb_h * 0.625, r"$\mathbf{h}_{V_{C_1}}$" "\n(C1 Embeddings)",
            fontsize=7.2, fontweight="bold", color=CLR_C1_SUPER, ha="center", va="center")
    ax.text(lemb_x + lemb_w/2, lemb_y + lemb_h * 0.125, r"$S_2$ Discarded", fontsize=6.2, color=CLR_MUTED, ha="center", va="center")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: GLOBAL GRL ON MACRO-GRAPH
    # ──────────────────────────────────────────────────────────────────────────
    p3_x, p3_y, p3_w, p3_h = 0.252, 0.035, 0.496, 0.440
    _draw_box(ax, (p3_x, p3_y), p3_w, p3_h, fc=CLR_PANEL_BG, ec="#059669", lw=1.8)
    ax.text(p3_x + p3_w/2, p3_y + p3_h - 0.026, r"(c) Global GRL on CaaN Macro-Graph $\mathcal{G}_{\mathrm{CaaN}} \cup V_{\mathrm{minor}}$",
            fontsize=12.0, fontweight="bold", color="#047857", ha="center")

    mg_bx, mg_by, mg_bw, mg_bh = p3_x + 0.014, p3_y + 0.018, 0.245, p3_h - 0.062
    _draw_box(ax, (mg_bx, mg_by), mg_bw, mg_bh, fc="#FFFFFF", ec="#059669", lw=1.3, radius=0.010)

    gs1, gs2 = (mg_bx + 0.075, mg_by + 0.215), (mg_bx + 0.170, mg_by + 0.215)
    gm1, gm2, gm3 = (mg_bx + 0.045, mg_by + 0.065), (mg_bx + 0.120, mg_by + 0.065), (mg_bx + 0.195, mg_by + 0.065)

    ax.plot([gs1[0], gs2[0]], [gs1[1], gs2[1]], color=CLR_DARK, lw=2.4, zorder=3)
    ax.text((gs1[0] + gs2[0])/2, gs1[1] + 0.024, r"Super-Edge $e(S_1, S_2)$", fontsize=6.8, fontweight="bold", color=CLR_DARK, ha="center")

    for pa, pb in [(gm1, gm2), (gm2, gm3)]:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=CLR_MIN_NODE, lw=1.4, zorder=3)
    ax.plot([gs1[0], gm1[0]], [gs1[1], gm1[1]], color=CLR_DARK, lw=1.2, ls="--", zorder=3)
    ax.plot([gs1[0], gm2[0]], [gs1[1], gm2[1]], color=CLR_DARK, lw=1.2, ls="--", zorder=3)
    ax.plot([gs2[0], gm3[0]], [gs2[1], gm3[1]], color=CLR_DARK, lw=1.2, ls="--", zorder=3)

    _draw_super_node(ax, gs1, CLR_C1_SUPER, "$S_1$")
    _draw_super_node(ax, gs2, CLR_C2_SUPER, "$S_2$")
    ax.text(gs1[0], gs1[1] - 0.028, r"$\mathbf{x}_{S_1} = \mathrm{avg}(C_1)$", fontsize=6.3, color=CLR_MUTED, ha="center")
    ax.text(gs2[0], gs2[1] - 0.028, r"$\mathbf{x}_{S_2} = \mathrm{avg}(C_2)$", fontsize=6.3, color=CLR_MUTED, ha="center")

    _draw_node(ax, gm1, CLR_MIN_NODE, "$m_1$")
    _draw_node(ax, gm2, CLR_MIN_NODE, "$m_2$")
    _draw_node(ax, gm3, CLR_MIN_NODE, "$m_3$")
    ax.text((gm1[0] + gm3[0])/2, gm1[1] - 0.022, "Retained Minor Vertices (No Over-Smoothing)", fontsize=6.6, fontweight="bold", color="#B45309", ha="center")

    # Global GNN Module (Symmetrically spaced)
    _draw_arrow(ax, (mg_bx + mg_bw + 0.008, mg_by + mg_bh/2), (p3_x + 0.285, mg_by + mg_bh/2), color=CLR_DARK, lw=1.8)
    _draw_box(ax, (p3_x + 0.290, mg_by + mg_bh/2 - 0.060), 0.085, 0.120, fc="#E2E8F0", ec=CLR_DARK, lw=1.3, radius=0.008)
    ax.text(p3_x + 0.290 + 0.0425, mg_by + mg_bh/2, "Global\nGNN\nModel", fontsize=8.6, fontweight="bold", color=CLR_DARK, ha="center", va="center")

    _draw_arrow(ax, (p3_x + 0.380, mg_by + mg_bh/2), (p3_x + 0.405, mg_by + mg_bh/2), color=CLR_DARK, lw=1.8)

    # Global Embedding Matrix
    gemb_x, gemb_y, gemb_w, gemb_h = p3_x + 0.410, mg_by + 0.025, 0.075, mg_bh - 0.050
    _draw_box(ax, (gemb_x, gemb_y), gemb_w, gemb_h, fc="#FFFFFF", ec=CLR_DARK, lw=1.3, radius=0.006)
    rect_min_emb = Rectangle((gemb_x, gemb_y), gemb_w, gemb_h * 0.60, facecolor=CLR_MIN_BG, edgecolor=CLR_MIN_BORDER, lw=0.9)
    ax.add_patch(rect_min_emb)
    ax.text(gemb_x + gemb_w/2, gemb_y + gemb_h * 0.30, r"$\mathbf{h}_{V_{\mathrm{minor}}}$" "\n(Minor Embeddings)",
            fontsize=7.2, fontweight="bold", color="#B45309", ha="center", va="center")
    ax.text(gemb_x + gemb_w/2, gemb_y + gemb_h * 0.80, r"$S_1, S_2$ Discarded", fontsize=6.2, color=CLR_MUTED, ha="center", va="center")

    # Clean Connecting Arrows between Left and Center Panels
    _draw_arrow(ax, (p1_x + p1_w + 0.003, p2_y + p2_h * 0.50), (p2_x - 0.003, p2_y + p2_h * 0.50),
                color=CLR_C1_SUPER, lw=2.0, style="-|>", connectionstyle="arc3,rad=0.0")
    _draw_arrow(ax, (p1_x + p1_w + 0.003, p3_y + p3_h * 0.50), (p3_x - 0.003, p3_y + p3_h * 0.50),
                color="#047857", lw=2.0, style="-|>", connectionstyle="arc3,rad=0.0")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4: FINAL GRL ASSEMBLY
    # ──────────────────────────────────────────────────────────────────────────
    p4_x, p4_y, p4_w, p4_h = 0.760, 0.035, 0.225, 0.895
    _draw_box(ax, (p4_x, p4_y), p4_w, p4_h, fc=CLR_PANEL_BG, ec=CLR_DARK, lw=1.8)
    ax.text(p4_x + p4_w/2, p4_y + p4_h - 0.026, "(d) Final Embedding Assembly", fontsize=12.0, fontweight="bold", color=CLR_DARK, ha="center")

    # Stacked Matrix (Symmetrically balanced)
    mat_x, mat_y, mat_w, mat_h = p4_x + 0.015, p4_y + 0.465, 0.075, 0.365
    _draw_box(ax, (mat_x, mat_y), mat_w, mat_h, fc="#FFFFFF", ec=CLR_DARK, lw=1.5, radius=0.006)

    slices = [
        (CLR_C1_BG, CLR_C1_BORDER, 0.35, r"$\mathbf{h}_{V_{C_1}}$", "Major $C_1$ (Local GRL)", CLR_C1_SUPER),
        (CLR_C2_BG, CLR_C2_BORDER, 0.35, r"$\mathbf{h}_{V_{C_2}}$", "Major $C_2$ (Local GRL)", CLR_C2_SUPER),
        (CLR_MIN_BG, CLR_MIN_BORDER, 0.30, r"$\mathbf{h}_{V_{\mathrm{minor}}}$", "Minor Nodes (Global)", "#B45309")
    ]
    cur_y = mat_y + mat_h
    for clr_bg, clr_bd, frac, tag, desc, clr_txt in slices:
        sh = mat_h * frac
        cur_y -= sh
        rect = Rectangle((mat_x, cur_y), mat_w, sh, facecolor=clr_bg, edgecolor=clr_bd, lw=1.0, zorder=6)
        ax.add_patch(rect)
        ax.text(mat_x + mat_w/2, cur_y + sh/2, tag, fontsize=7.4, fontweight="bold", color=clr_txt, ha="center", va="center", zorder=7)
        ax.text(mat_x + mat_w + 0.008, cur_y + sh/2, desc, fontsize=7.0, fontweight="bold", color=CLR_TEXT, va="center", zorder=7)

    _draw_arrow(ax, (p2_x + p2_w + 0.003, p2_y + p2_h * 0.45), (p4_x - 0.003, mat_y + mat_h * 0.80),
                color=CLR_C1_SUPER, lw=1.8, style="-|>", connectionstyle="arc3,rad=-0.08", zorder=7)
    _draw_arrow(ax, (p3_x + p3_w + 0.003, p3_y + p3_h * 0.45), (p4_x - 0.003, mat_y + mat_h * 0.20),
                color="#B45309", lw=1.8, style="-|>", connectionstyle="arc3,rad=0.08", zorder=7)

    # Summary Rules Box
    rule_bx, rule_by, rule_bw, rule_bh = p4_x + 0.010, p4_y + 0.015, p4_w - 0.020, 0.430
    _draw_box(ax, (rule_bx, rule_by), rule_bw, rule_bh, fc="#FFFFFF", ec=CLR_BORDER, lw=1.1, radius=0.008)
    ax.text(rule_bx + 0.012, rule_by + rule_bh - 0.022, "Boundary & Macro Semantics:", fontsize=7.8, fontweight="bold", color=CLR_DARK)
    
    rules_text = (
        r"$\mathbf{1.\;Boundary\;Vertices:}$" "\n"
        r"   $v \in C_i$ has cut edge $(v, w)$ to $w \in C_j$." "\n\n"
        r"$\mathbf{2.\;Internal\;Connectivity\;(Local):}$" "\n"
        r"   $v$ keeps all intra-community edges $(v, u)$." "\n"
        r"   Host super-node $S_i$ is strictly EXCLUDED." "\n\n"
        r"$\mathbf{3.\;External\;Connectivity\;(Local):}$" "\n"
        r"   Severed edge $(v, w)$ maps to $(v, S_j)$," "\n"
        r"   connecting $v$ directly to external $S_j$." "\n\n"
        r"$\mathbf{4.\;Minor\;Vertices\;(Global):}$" "\n"
        r"   $|C_k| < K$ retained explicitly in $\mathcal{G}_{\mathrm{CaaN}}$" "\n"
        r"   to completely prevent over-smoothing."
    )
    ax.text(rule_bx + 0.010, rule_by + 0.205, rules_text, fontsize=6.8, color=CLR_TEXT, va="center")

    out_pdf = OUT_DIR / "caan_macro_graph_architecture_v2.pdf"
    out_png = OUT_DIR / "caan_macro_graph_architecture_v2.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(OVERLEAF_DIR / "caan_macro_graph_architecture_v2.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OVERLEAF_DIR / "caan_macro_graph_architecture_v2.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Generated Version 2 (Minimalist & Clean): {out_pdf}")


if __name__ == "__main__":
    create_smooth_caan_figure()
    create_smooth_caan_figure_v2()
