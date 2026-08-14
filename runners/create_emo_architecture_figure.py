#!/usr/bin/env python3
"""
create_emo_architecture_figure.py
Generates a publication-grade logical architecture + execution flow diagram (Figure 2)
for the EMO paper, matching the exact visual style, typography, and color palette of Figure 1.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path("results/figures")
PNG_PATH = OUT_DIR / "emo_system_architecture_overview.png"
PDF_PATH = OUT_DIR / "emo_system_architecture_overview.pdf"

# ─── Matching Color Palette (Identical to Figure 1) ───────────────────────────
CLR_CANVAS_BG   = "#F0F4F8"   # soft gray-blue container
CLR_BRANCHA_HDR = "#1565C0"   # dark blue (Branch A primary)
CLR_BRANCHA_BG  = "#E3F2FD"   # light blue tint
CLR_BRANCHB_HDR = "#6A1B9A"   # rich purple (Branch B CAAN)
CLR_BRANCHB_BG  = "#F3E5F5"   # light purple tint
CLR_CARD_FILL   = "#FFFFFF"   # crisp white cards
CLR_CARD_BORDER = "#90CAF9"   # light blue border
CLR_S3_BG       = "#E8F5E9"   # light green container
CLR_S3_HDR      = "#1B5E20"   # forest green
CLR_DELTA_FILL  = "#FFFFFF"   # crisp white
CLR_DELTA_BORD  = "#2E7D32"   # rich green border
CLR_ARROW_FLOW  = "#E65100"   # rich orange flow arrows
CLR_ARROW_SEC   = "#8E24AA"   # purple branch flow
CLR_TEXT        = "#1A202C"   # near black
CLR_MUTED       = "#4A5568"   # slate gray
CLR_CONTRIB_BG  = "#FFF8E1"   # warm gold/amber
CLR_CONTRIB_EC  = "#F57F17"   # deep amber border


def _draw_box(ax, xy, w, h, fc="#FFFFFF", ec="#333333", lw=1.2, radius=0.012, zorder=3):
    """Draw a smooth rounded rectangle."""
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def _draw_arrow(ax, start, end, color=CLR_ARROW_FLOW, lw=1.8, style="->",
                connectionstyle="arc3,rad=0.0", zorder=5):
    """Draw a crisp vector arrow."""
    arr = FancyArrowPatch(start, end, arrowstyle=style,
                          connectionstyle=connectionstyle,
                          color=color, linewidth=lw,
                          mutation_scale=12, zorder=zorder)
    ax.add_patch(arr)
    return arr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
    })

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # =========================================================================
    # 1. TOP SECTION: EMO EXECUTION PIPELINE (y in [0.43, 0.98])
    # =========================================================================
    pipe_x, pipe_y, pipe_w, pipe_h = 0.02, 0.43, 0.96, 0.55
    _draw_box(ax, (pipe_x, pipe_y), pipe_w, pipe_h,
              fc=CLR_CANVAS_BG, ec="#78909C", lw=1.8, radius=0.015, zorder=1)

    # Master Header
    ax.text(pipe_x + 0.02, pipe_y + pipe_h - 0.032,
            "EMO Logical Pipeline Architecture & Two-Branch Execution Model",
            fontsize=12.5, fontweight="bold", color=CLR_BRANCHA_HDR, zorder=2)
    ax.text(pipe_x + 0.48, pipe_y + pipe_h - 0.032,
            "•  Decoupled GRL  •  Relational Graph Algebra  •  Zero Cross-Worker Training Traffic",
            fontsize=8.0, color=CLR_MUTED, zorder=2)

    # -------------------------------------------------------------------------
    # BRANCH A: PRIMARY EVALUATION PATH (Top sub-container: y in [0.715, 0.925])
    # -------------------------------------------------------------------------
    ba_x, ba_y, ba_w, ba_h = 0.035, 0.715, 0.93, 0.21
    _draw_box(ax, (ba_x, ba_y), ba_w, ba_h,
              fc=CLR_BRANCHA_BG, ec=CLR_BRANCHA_HDR, lw=1.4, radius=0.010, zorder=2)

    # Branch A Header Badge
    _draw_box(ax, (ba_x + 0.012, ba_y + ba_h - 0.028), 0.32, 0.022,
              fc=CLR_BRANCHA_HDR, ec=CLR_BRANCHA_HDR, lw=0.5, radius=0.004, zorder=3)
    ax.text(ba_x + 0.012 + 0.16, ba_y + ba_h - 0.017,
            "Branch A: Full-Graph Propagation & Classification (Primary Path)",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#FFFFFF", zorder=4)

    # Branch A 4 Stage Cards
    cards_a = [
        ("Phase 0", "Graph Ingestion",
         "• OGB / DGL raw data load\n• Undirected edge symmetrization\n• Materialize nodes, edges, masks",
         "#E1F5FE", "#0288D1"),
        ("Phase 3.7", "SIGN Feature Propagation",
         "• K-hop mean-pooling via Spark SQL\n• Summarizer.mean JVM vector ops\n• Materialize cached hop tables [x⁰‖..‖xᴷ]",
         "#E1F5FE", "#0288D1"),
        ("Phase 3.8", "Global Edge-Free Classifier",
         "• LogisticRegression / MLP probe\n• Fits on concatenated features\n• 100% full-graph node coverage",
         "#E1F5FE", "#0288D1"),
        ("Phase 5", "Evaluation & Reporting",
         "• Multi-dataset metric aggregation\n• Test accuracy / AUC evaluation\n• Automated LaTeX & plot generation",
         "#E1F5FE", "#0288D1"),
    ]

    card_w = 0.205
    card_h = 0.145
    card_gap = (ba_w - 0.04 - len(cards_a) * card_w) / (len(cards_a) - 1)
    card_y = ba_y + 0.015

    for idx, (pnum, ptitle, pbody, badge_bg, badge_ec) in enumerate(cards_a):
        cx = ba_x + 0.02 + idx * (card_w + card_gap)
        _draw_box(ax, (cx, card_y), card_w, card_h,
                  fc=CLR_CARD_FILL, ec=CLR_CARD_BORDER, lw=1.0, radius=0.006, zorder=3)

        # Stage Badge
        _draw_box(ax, (cx + 0.008, card_y + card_h - 0.024), card_w - 0.016, 0.018,
                  fc=badge_bg, ec=badge_ec, lw=0.6, radius=0.003, zorder=4)
        ax.text(cx + card_w/2, card_y + card_h - 0.015, f"{pnum} : {ptitle}",
                ha="center", va="center", fontsize=7.2, fontweight="bold", color=CLR_BRANCHA_HDR, zorder=5)

        # Stage Details
        ax.text(cx + 0.012, card_y + card_h/2 - 0.005, pbody,
                ha="left", va="center", fontsize=6.2, color=CLR_TEXT, zorder=4)

        # Connecting Flow Arrow
        if idx < len(cards_a) - 1:
            arrow_start_x = cx + card_w + 0.002
            arrow_end_x = arrow_start_x + card_gap - 0.004
            arrow_y = card_y + card_h/2
            _draw_arrow(ax, (arrow_start_x, arrow_y), (arrow_end_x, arrow_y),
                        color=CLR_ARROW_FLOW, lw=2.0, style="->")

    # -------------------------------------------------------------------------
    # BRANCH B: COMMUNITY-DECOUPLED & CAAN PATH (Bottom sub-container: y in [0.455, 0.695])
    # -------------------------------------------------------------------------
    bb_x, bb_y, bb_w, bb_h = 0.035, 0.455, 0.93, 0.24
    _draw_box(ax, (bb_x, bb_y), bb_w, bb_h,
              fc=CLR_BRANCHB_BG, ec=CLR_BRANCHB_HDR, lw=1.4, radius=0.010, zorder=2)

    # Branch B Header Badge
    _draw_box(ax, (bb_x + 0.012, bb_y + bb_h - 0.028), 0.35, 0.022,
              fc=CLR_BRANCHB_HDR, ec=CLR_BRANCHB_HDR, lw=0.5, radius=0.004, zorder=3)
    ax.text(bb_x + 0.012 + 0.175, bb_y + bb_h - 0.017,
            "Branch B: Community-Decoupled GNNs & CAAN Super-Node Path",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#FFFFFF", zorder=4)

    # Branch B 4 Stage Cards
    cards_b = [
        ("Phase 1", "Community Detection",
         "• Spark native LPA (scalable)\n• igraph Louvain (driver memory)\n• Materialize communities/{alg}/",
         "#EDE7F6", "#7E57C2"),
        ("Phase 2", "Relational Subgraph Extraction",
         "• Boundary tag: deg_orig > deg_intra\n• Intra-community edge filtering\n• repartition('community_id') S3 locality",
         "#EDE7F6", "#7E57C2"),
        ("Phase 3", "Decoupled Community GNNs",
         "• Independent GraphSAGE / GATv2\n• Trained in executor worker memory\n• Zero cross-worker network traffic",
         "#EDE7F6", "#7E57C2"),
        ("Phase 3b", "CAAN Super-Node Recovery",
         "• Major community centroid super-nodes\n• Broadcast (S, M) auxiliary context\n• Recovers 97.8% boundary accuracy",
         "#EDE7F6", "#7E57C2"),
    ]

    card_b_y = bb_y + 0.015

    for idx, (pnum, ptitle, pbody, badge_bg, badge_ec) in enumerate(cards_b):
        cx = bb_x + 0.02 + idx * (card_w + card_gap)
        _draw_box(ax, (cx, card_b_y), card_w, card_h + 0.025,
                  fc=CLR_CARD_FILL, ec="#D1C4E9", lw=1.0, radius=0.006, zorder=3)

        # Stage Badge
        _draw_box(ax, (cx + 0.008, card_b_y + card_h + 0.025 - 0.024), card_w - 0.016, 0.018,
                  fc=badge_bg, ec=badge_ec, lw=0.6, radius=0.003, zorder=4)
        ax.text(cx + card_w/2, card_b_y + card_h + 0.025 - 0.015, f"{pnum} : {ptitle}",
                ha="center", va="center", fontsize=7.2, fontweight="bold", color=CLR_BRANCHB_HDR, zorder=5)

        # Stage Details
        ax.text(cx + 0.012, card_b_y + (card_h + 0.025)/2 - 0.005, pbody,
                ha="left", va="center", fontsize=6.2, color=CLR_TEXT, zorder=4)

        # Connecting Flow Arrow
        if idx < len(cards_b) - 1:
            arrow_start_x = cx + card_w + 0.002
            arrow_end_x = arrow_start_x + card_gap - 0.004
            arrow_y = card_b_y + (card_h + 0.025)/2
            _draw_arrow(ax, (arrow_start_x, arrow_y), (arrow_end_x, arrow_y),
                        color=CLR_ARROW_SEC, lw=2.0, style="->")

    # =========================================================================
    # 2. MIDDLE PROTOCOL & STORAGE BRIDGE (y in [0.355, 0.43])
    # =========================================================================
    # Downward connecting arrows from Branches to Storage
    for ax_pos in [0.15, 0.38, 0.62, 0.85]:
        _draw_arrow(ax, (ax_pos, pipe_y), (ax_pos, 0.33),
                    color=CLR_DELTA_BORD, lw=2.0, style="<->")

    # Central Delta Lake Protocol Banner
    proto_x, proto_y, proto_w, proto_h = 0.28, 0.355, 0.44, 0.058
    _draw_box(ax, (proto_x, proto_y), proto_w, proto_h,
              fc="#FFFFFF", ec=CLR_DELTA_BORD, lw=1.5, radius=0.008, zorder=6)
    ax.text(proto_x + proto_w/2, proto_y + proto_h/2 + 0.010,
            "Shared Delta Lake Transactional Storage Plane on Amazon S3",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=CLR_DELTA_BORD, zorder=7)
    ax.text(proto_x + proto_w/2, proto_y + proto_h/2 - 0.012,
            "ACID Isolation • Checkpoint Reuse via _delta_log • Schema Enforcement • Parquet Format",
            ha="center", va="center", fontsize=6.5, color=CLR_MUTED, zorder=7)

    # =========================================================================
    # 3. BOTTOM SECTION: STORAGE CONTRACTS & CONTRIBUTION MAPPING (y in [0.03, 0.33])
    # =========================================================================
    s3_x, s3_y, s3_w, s3_h = 0.02, 0.03, 0.96, 0.30
    _draw_box(ax, (s3_x, s3_y), s3_w, s3_h,
              fc=CLR_S3_BG, ec=CLR_S3_HDR, lw=1.8, radius=0.015, zorder=1)

    # Left: S3 Delta Lake Tables Sub-Container
    tbl_sub_x, tbl_sub_y, tbl_sub_w, tbl_sub_h = s3_x + 0.015, s3_y + 0.015, 0.64, s3_h - 0.03
    _draw_box(ax, (tbl_sub_x, tbl_sub_y), tbl_sub_w, tbl_sub_h,
              fc="#FFFFFF", ec=CLR_DELTA_BORD, lw=1.2, radius=0.010, zorder=2)

    ax.text(tbl_sub_x + 0.015, tbl_sub_y + tbl_sub_h - 0.022,
            "Delta Lake Tables & Partition Contracts (S3)",
            fontsize=10.0, fontweight="bold", color=CLR_S3_HDR, zorder=3)
    ax.text(tbl_sub_x + 0.32, tbl_sub_y + tbl_sub_h - 0.022,
            "URI: s3://us-east-1-s3-gnn/delta-data/{dataset}/",
            fontsize=7.2, fontfamily="monospace", color=CLR_MUTED, zorder=3)

    # 4 Table Category Cards inside Storage
    delta_cards = [
        ("Base Graph (P0)", "nodes/\nedges/\nmasks/", "id, label, features\nsrc, dst\nid, split"),
        ("Communities (P1)", "communities/\n{alg}/", "id: LONG\ncommunity_id: LONG"),
        ("Subgraphs (P2)", "phase2_nodes/\nphase2_edges/", "id, features, is_boundary\nsrc, dst, comm_id"),
        ("Propagated (P3.7)", "hop_k/\nfeatures_kK/", "hop 1..K mean cache\n[x⁰‖x¹‖...‖xᴷ]"),
    ]

    dt_w = 0.142
    dt_h = 0.170
    dt_gap = (tbl_sub_w - 0.03 - len(delta_cards) * dt_w) / (len(delta_cards) - 1)
    dt_y = tbl_sub_y + 0.020

    for idx, (dtitle, dpath, dcols) in enumerate(delta_cards):
        dx = tbl_sub_x + 0.015 + idx * (dt_w + dt_gap)
        _draw_box(ax, (dx, dt_y), dt_w, dt_h,
                  fc="#F1F8E9", ec=CLR_DELTA_BORD, lw=0.8, radius=0.005, zorder=3)

        # Card Title
        ax.text(dx + dt_w/2, dt_y + dt_h - 0.018, dtitle,
                ha="center", va="center", fontsize=7.2, fontweight="bold", color=CLR_DELTA_BORD, zorder=4)
        # Table Path
        ax.text(dx + dt_w/2, dt_y + dt_h - 0.048, dpath,
                ha="center", va="center", fontsize=6.2, fontfamily="monospace", fontweight="bold", color=CLR_TEXT, zorder=4)
        # Schema / Columns
        ax.text(dx + dt_w/2, dt_y + 0.045, dcols,
                ha="center", va="center", fontsize=5.5, fontfamily="monospace", color=CLR_MUTED, zorder=4)
        # Delta badge
        ax.text(dx + dt_w/2, dt_y + 0.012, "Δ Delta Parquet",
                ha="center", va="center", fontsize=5.5, color=CLR_DELTA_BORD, style="italic", zorder=4)

    # Right: Technical Contributions Mapping Sub-Container
    cb_x, cb_y, cb_w, cb_h = s3_x + 0.67, s3_y + 0.015, 0.275, s3_h - 0.03
    _draw_box(ax, (cb_x, cb_y), cb_w, cb_h,
              fc=CLR_CONTRIB_BG, ec=CLR_CONTRIB_EC, lw=1.2, radius=0.010, zorder=2)

    # Centered Header with ample padding
    ax.text(cb_x + cb_w/2, cb_y + cb_h - 0.024,
            "Core Technical Contributions",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#B78103", zorder=3)

    contrib_items = [
        ("C1: Transactional Lakehouse Storage",
         "ACID experiment isolation, phase checkpoint reuse via _delta_log, schema enforcement."),
        ("C2: Relational Graph Operations",
         "Boundary detection & CAAN auxiliary super-node construction via Spark SQL joins/aggregations."),
        ("C3: Cloud-Native Systems Hardening",
         "Dynamic EBS redirection (F2), YARN overhead sizing (F1), and Arrow zero-copy serialization."),
    ]

    for idx, (ctitle, cdesc) in enumerate(contrib_items):
        item_y = cb_y + cb_h - 0.066 - idx * 0.066
        _draw_box(ax, (cb_x + 0.010, item_y - 0.022), cb_w - 0.020, 0.058,
                  fc="#FFFFFF", ec=CLR_CONTRIB_EC, lw=0.7, radius=0.004, zorder=3)
        ax.text(cb_x + 0.016, item_y + 0.020, ctitle,
                ha="left", va="center", fontsize=6.6, fontweight="bold", color="#E65100", zorder=4)
        ax.text(cb_x + 0.016, item_y - 0.004, cdesc,
                ha="left", va="center", fontsize=5.5, color=CLR_TEXT, zorder=4)

    # Save High-Res Outputs
    fig.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    print(f"✓ Publication-grade Figure 2 successfully generated:")
    print(f"  - {PNG_PATH}")
    print(f"  - {PDF_PATH}")


if __name__ == "__main__":
    main()
