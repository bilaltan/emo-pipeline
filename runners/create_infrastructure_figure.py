#!/usr/bin/env python3
"""
create_infrastructure_figure.py
Generates a publication-grade Dorylus-style infrastructure architecture diagram
for the EMO paper, showing EMR instances, Delta Lake, and S3 with data-flow arrows.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path("results/figures")
PNG_PATH = OUT_DIR / "emo_infrastructure_architecture.png"
PDF_PATH = OUT_DIR / "emo_infrastructure_architecture.pdf"

# ─── Professional Color Palette ──────────────────────────────────────────────
CLR_EMR_BG      = "#F0F4F8"   # soft gray-blue container
CLR_DRIVER_HDR  = "#1565C0"   # dark blue
CLR_DRIVER_FILL = "#E3F2FD"   # light blue
CLR_WORKER_HDR  = "#0D47A1"   # navy
CLR_WORKER_FILL = "#E8EEF5"   # clean light slate
CLR_EXEC_FILL   = "#FFFFFF"   # crisp white cards
CLR_EXEC_BORDER = "#90CAF9"   # light blue border
CLR_S3_BG       = "#E8F5E9"   # light green container
CLR_S3_HDR      = "#1B5E20"   # forest green
CLR_DELTA_FILL  = "#FFFFFF"   # crisp white
CLR_DELTA_BORD  = "#2E7D32"   # rich green border
CLR_ARROW_FLOW  = "#E65100"   # rich orange
CLR_YARN_HDR    = "#6A1B9A"   # purple
CLR_YARN_FILL   = "#F3E5F5"
CLR_TEXT        = "#1A202C"   # near black
CLR_MUTED       = "#4A5568"   # dark gray
CLR_EBS_FILL    = "#FFF8E1"   # warm amber
CLR_EBS_BORDER  = "#F57F17"


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
    # 1. AWS EMR CLUSTER CONTAINER (Top Section: y in [0.40, 0.98])
    # =========================================================================
    emr_x, emr_y, emr_w, emr_h = 0.02, 0.40, 0.96, 0.58
    _draw_box(ax, (emr_x, emr_y), emr_w, emr_h,
              fc=CLR_EMR_BG, ec="#78909C", lw=1.8, radius=0.015, zorder=1)

    # EMR Title Banner
    ax.text(emr_x + 0.02, emr_y + emr_h - 0.032,
            "AWS EMR Cluster (Compute Plane)",
            fontsize=12.5, fontweight="bold", color=CLR_DRIVER_HDR, zorder=2)
    ax.text(emr_x + 0.30, emr_y + emr_h - 0.032,
            "•  emr-7.x  •  Apache Spark 3.5  •  Delta Lake 3.2  •  Apache YARN Resource Manager",
            fontsize=8.0, color=CLR_MUTED, zorder=2)

    # -------------------------------------------------------------------------
    # Driver Node (Left: x in [0.04, 0.24], y in [0.42, 0.90])
    # -------------------------------------------------------------------------
    drv_x, drv_y, drv_w, drv_h = 0.04, 0.42, 0.20, 0.49
    _draw_box(ax, (drv_x, drv_y), drv_w, drv_h,
              fc=CLR_DRIVER_FILL, ec=CLR_DRIVER_HDR, lw=1.6, radius=0.010, zorder=2)

    ax.text(drv_x + drv_w/2, drv_y + drv_h - 0.022,
            "Master / Driver Node", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=CLR_DRIVER_HDR, zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + drv_h - 0.044,
            "(r6id.8xlarge  •  32 vCPUs  •  256 GB RAM)", ha="center", va="center",
            fontsize=6.8, color=CLR_MUTED, zorder=4)

    # YARN RM inside Driver / Master
    _draw_box(ax, (drv_x + 0.012, drv_y + 0.345), drv_w - 0.024, 0.075,
              fc=CLR_YARN_FILL, ec=CLR_YARN_HDR, lw=1.2, radius=0.006, zorder=3)
    ax.text(drv_x + drv_w/2, drv_y + 0.395, "YARN Resource Manager",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=CLR_YARN_HDR, zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + 0.367, "Executor scheduling & container lifecycle",
            ha="center", va="center", fontsize=6.8, color=CLR_MUTED, style="italic", zorder=4)

    # Spark Driver Orchestrator
    _draw_box(ax, (drv_x + 0.012, drv_y + 0.235), drv_w - 0.024, 0.095,
              fc="#FFFFFF", ec=CLR_DRIVER_HDR, lw=1.0, radius=0.006, zorder=3)
    ax.text(drv_x + drv_w/2, drv_y + 0.300, "Spark Driver (PySpark)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=CLR_TEXT, zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + 0.275, "• Global Phase Orchestration (Phases 0-5)",
            ha="center", va="center", fontsize=6.8, color=CLR_MUTED, zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + 0.253, "• Broadcast centroid / super-node tables",
            ha="center", va="center", fontsize=6.8, color=CLR_MUTED, zorder=4)

    # igraph Louvain
    _draw_box(ax, (drv_x + 0.012, drv_y + 0.125), drv_w - 0.024, 0.095,
              fc="#FFFFFF", ec=CLR_DRIVER_HDR, lw=1.0, radius=0.006, zorder=3)
    ax.text(drv_x + drv_w/2, drv_y + 0.190, "igraph / Local Detection",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=CLR_TEXT, zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + 0.165, "Phase 1: Louvain community partitioning",
            ha="center", va="center", fontsize=6.8, color=CLR_MUTED, zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + 0.145, "(collected graph topology in driver memory)",
            ha="center", va="center", fontsize=6.2, color=CLR_MUTED, style="italic", zorder=4)

    # Driver Local Disk
    _draw_box(ax, (drv_x + 0.012, drv_y + 0.015), drv_w - 0.024, 0.095,
              fc=CLR_EBS_FILL, ec=CLR_EBS_BORDER, lw=1.0, radius=0.006, zorder=3)
    ax.text(drv_x + drv_w/2, drv_y + 0.082, "Driver Local Storage",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#B78103", zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + 0.058, "• /mnt/tmp (Dynamic redirection)",
            ha="center", va="center", fontsize=6.8, color=CLR_MUTED, zorder=4)
    ax.text(drv_x + drv_w/2, drv_y + 0.038, "• Matplotlib / LaTeX report output",
            ha="center", va="center", fontsize=6.8, color=CLR_MUTED, zorder=4)

    # -------------------------------------------------------------------------
    # Arrow Serialization Bridge (Center: x in [0.255, 0.38])
    # -------------------------------------------------------------------------
    bridge_x, bridge_y = 0.255, 0.65
    _draw_box(ax, (bridge_x, bridge_y), 0.125, 0.14,
              fc="#FFF3E0", ec=CLR_ARROW_FLOW, lw=1.4, radius=0.008, zorder=3)
    ax.text(bridge_x + 0.0625, bridge_y + 0.110, "Apache Arrow",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=CLR_ARROW_FLOW, zorder=4)
    ax.text(bridge_x + 0.0625, bridge_y + 0.082, "Zero-Copy Data Transfer",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=CLR_TEXT, zorder=4)
    ax.text(bridge_x + 0.0625, bridge_y + 0.055, "PyArrow RecordBatches",
            ha="center", va="center", fontsize=6.8, color=CLR_MUTED, zorder=4)
    ax.text(bridge_x + 0.0625, bridge_y + 0.030, "JVM Heap  ⇄  PyTorch",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color=CLR_ARROW_FLOW, zorder=4)

    # Connecting arrows Driver -> Bridge -> Workers
    _draw_arrow(ax, (drv_x + drv_w, bridge_y + 0.07), (bridge_x, bridge_y + 0.07),
                color=CLR_ARROW_FLOW, lw=2.0, style="<->")
    _draw_arrow(ax, (bridge_x + 0.125, bridge_y + 0.07), (0.40, bridge_y + 0.07),
                color=CLR_ARROW_FLOW, lw=2.0, style="<->")

    # YARN Schedule Arrow
    _draw_arrow(ax, (drv_x + drv_w - 0.02, drv_y + 0.380), (0.40, 0.83),
                color=CLR_YARN_HDR, lw=1.5, style="->", connectionstyle="arc3,rad=-0.12")
    ax.text(0.315, 0.845, "YARN Executor Launch & Heartbeats",
            ha="center", va="center", fontsize=7.2, color=CLR_YARN_HDR, fontweight="bold", zorder=5)

    # -------------------------------------------------------------------------
    # Worker Nodes (Right: x in [0.40, 0.96], y in [0.42, 0.90])
    # -------------------------------------------------------------------------
    workers_x = 0.40
    wk_w = 0.125
    wk_h = 0.47
    wk_gap = 0.015
    wk_y = 0.42

    workers_meta = [
        ("Worker 1", "r6id.8xlarge", "Exec 1, 2"),
        ("Worker 2", "r6id.8xlarge", "Exec 3, 4"),
        ("Worker 3", "r6id.8xlarge", "Exec 5, 6"),
        ("Worker 8", "r6id.8xlarge", "Exec 15, 16"),
    ]

    for idx, (wname, winst, wexec) in enumerate(workers_meta):
        if idx == 3:
            wx = workers_x + 3 * (wk_w + wk_gap) + 0.015
            dots_x = workers_x + 3 * (wk_w + wk_gap) - 0.007
            ax.text(dots_x, wk_y + wk_h/2, "• • •", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=CLR_WORKER_HDR, zorder=4)
        else:
            wx = workers_x + idx * (wk_w + wk_gap)

        _draw_box(ax, (wx, wk_y), wk_w, wk_h,
                  fc=CLR_WORKER_FILL, ec=CLR_WORKER_HDR, lw=1.4, radius=0.008, zorder=2)
        ax.text(wx + wk_w/2, wk_y + wk_h - 0.022, wname,
                ha="center", va="center", fontsize=9.5, fontweight="bold", color=CLR_WORKER_HDR, zorder=4)
        ax.text(wx + wk_w/2, wk_y + wk_h - 0.044, f"({winst})",
                ha="center", va="center", fontsize=7.0, color=CLR_MUTED, zorder=4)

        # Executors inside worker
        exec_y_starts = [wk_y + 0.255, wk_y + 0.130]
        exec_labels = wexec.split(", ")
        for e_i, ey in enumerate(exec_y_starts):
            eh = 0.115
            _draw_box(ax, (wx + 0.006, ey), wk_w - 0.012, eh,
                      fc="#FFFFFF", ec=CLR_EXEC_BORDER, lw=0.9, radius=0.005, zorder=3)
            ax.text(wx + wk_w/2, ey + eh - 0.014, f"Spark Executor {exec_labels[e_i].split()[-1]}",
                    ha="center", va="center", fontsize=7.2, fontweight="bold", color=CLR_TEXT, zorder=4)

            # Inside executor: JVM + Off-Heap
            box_half_w = (wk_w - 0.032) / 2
            # JVM side
            _draw_box(ax, (wx + 0.010, ey + 0.012), box_half_w, eh - 0.032,
                      fc="#E1F5FE", ec="#81D4FA", lw=0.6, radius=0.003, zorder=4)
            ax.text(wx + 0.010 + box_half_w/2, ey + eh - 0.032, "JVM Heap",
                    ha="center", va="center", fontsize=6.5, fontweight="bold", color="#01579B", zorder=5)
            ax.text(wx + 0.010 + box_half_w/2, ey + eh - 0.052, "28 GB",
                    ha="center", va="center", fontsize=6.0, color="#0288D1", zorder=5)
            ax.text(wx + 0.010 + box_half_w/2, ey + eh - 0.075, "Spark SQL\nJoins/Filter",
                    ha="center", va="center", fontsize=5.2, color=CLR_TEXT, zorder=5)

            # Off-Heap PyTorch side
            _draw_box(ax, (wx + 0.010 + box_half_w + 0.004, ey + 0.012), box_half_w, eh - 0.032,
                      fc="#FCE4EC", ec="#F48FB1", lw=0.6, radius=0.003, zorder=4)
            ax.text(wx + 0.010 + 1.5*box_half_w + 0.004, ey + eh - 0.032, "Off-Heap",
                    ha="center", va="center", fontsize=6.5, fontweight="bold", color="#880E4F", zorder=5)
            ax.text(wx + 0.010 + 1.5*box_half_w + 0.004, ey + eh - 0.052, "12 GB Ovhd",
                    ha="center", va="center", fontsize=6.0, color="#C2185B", zorder=5)
            ax.text(wx + 0.010 + 1.5*box_half_w + 0.004, ey + eh - 0.075, "PyTorch/DGL\nTensors/Train",
                    ha="center", va="center", fontsize=5.2, color=CLR_TEXT, zorder=5)

        # Worker Local Storage
        ebs_y = wk_y + 0.012
        ebs_h = 0.105
        _draw_box(ax, (wx + 0.007, ebs_y), wk_w - 0.014, ebs_h,
                  fc=CLR_EBS_FILL, ec=CLR_EBS_BORDER, lw=0.9, radius=0.005, zorder=3)
        ax.text(wx + wk_w/2, ebs_y + ebs_h - 0.018, "Local Attached Storage",
                ha="center", va="center", fontsize=6.5, fontweight="bold", color="#E65100", zorder=4)
        ax.text(wx + wk_w/2, ebs_y + ebs_h - 0.038, "HOME & TMPDIR Redirect",
                ha="center", va="center", fontsize=5.8, fontweight="bold", color=CLR_TEXT, zorder=4)
        ax.text(wx + wk_w/2, ebs_y + ebs_h - 0.065, "• PyTorch ~/.cache & wheels\n• Prevents root 15GB OOM (F2)",
                ha="center", va="center", fontsize=5.0, color=CLR_MUTED, zorder=4)

    # Worker summary badge
    ax.text(0.70, 0.912, "Cluster Total: 8 Worker Nodes • 16 Executors • 256 vCPUs • 2048 GB RAM",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=CLR_WORKER_HDR, zorder=4,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec=CLR_WORKER_HDR, lw=1.0, alpha=0.95))

    # =========================================================================
    # 2. AMAZON S3 & DELTA LAKE STORAGE (Bottom Section: y in [0.03, 0.35])
    # =========================================================================
    s3_x, s3_y, s3_w, s3_h = 0.02, 0.03, 0.96, 0.33
    _draw_box(ax, (s3_x, s3_y), s3_w, s3_h,
              fc=CLR_S3_BG, ec=CLR_S3_HDR, lw=1.8, radius=0.015, zorder=1)

    # S3 Headers (Stacked cleanly)
    ax.text(s3_x + 0.02, s3_y + s3_h - 0.025,
            "Amazon S3 (Transactional Data Lake Storage Plane)",
            fontsize=12.0, fontweight="bold", color=CLR_S3_HDR, zorder=2)
    ax.text(s3_x + 0.02, s3_y + s3_h - 0.048,
            "Root URI: s3://us-east-1-s3-gnn/delta-data/{dataset}/",
            fontsize=7.8, fontfamily="monospace", color=CLR_MUTED, zorder=2)

    # ACID Transaction Log Callout Box (Top Right of S3)
    acid_x, acid_y, acid_w, acid_h = s3_x + 0.68, s3_y + s3_h - 0.060, 0.26, 0.048
    _draw_box(ax, (acid_x, acid_y), acid_w, acid_h,
              fc="#FFFFFF", ec=CLR_DELTA_BORD, lw=1.0, radius=0.005, zorder=3)
    ax.text(acid_x + acid_w/2, acid_y + acid_h - 0.016, "_delta_log/ ACID Transaction Plane",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=CLR_DELTA_BORD, zorder=4)
    ax.text(acid_x + acid_w/2, acid_y + 0.014, "Per-experiment isolation • Zero write conflict • Checkpoint reuse",
            ha="center", va="center", fontsize=5.6, color=CLR_MUTED, zorder=4)

    # Delta Lake Table Definitions (6 Tables)
    tables = [
        ("nodes/", "Raw / Ingested Nodes", "id: LONG (PK)\nlabel: INT\nfeatures: ARRAY<FLOAT>", "Phase 0"),
        ("edges/", "Undirected Symmetrized Edges", "src: LONG (FK)\ndst: LONG (FK)", "Phase 0"),
        ("masks/", "Evaluation Splits", "id: LONG (FK)\nsplit: STRING\n('train','val','test')", "Phase 0"),
        ("communities/{alg}/", "Community Partitioning Map", "id: LONG (FK)\ncommunity_id: LONG\n(LPA / Louvain)", "Phase 1"),
        ("phase2_nodes/{tag}/", "Partitioned Subgraphs + Meta", "id, label, features\nsplit, community_id\nis_boundary: BOOL", "Phase 2"),
        ("phase37_prop./hop_k/", "Multi-Hop Propagated Features", "id: LONG (FK)\nfeatures: ARRAY<FLOAT>\n(Cached SIGN hops 1..K)", "Phase 3.7"),
    ]

    tbl_n = len(tables)
    tbl_w = 0.142
    tbl_h = 0.185
    tbl_gap = (s3_w - 0.04 - tbl_n * tbl_w) / (tbl_n - 1)
    tbl_y = s3_y + 0.025

    for idx, (tname, tdesc, tschema, tphase) in enumerate(tables):
        tx = s3_x + 0.02 + idx * (tbl_w + tbl_gap)
        _draw_box(ax, (tx, tbl_y), tbl_w, tbl_h,
                  fc=CLR_DELTA_FILL, ec=CLR_DELTA_BORD, lw=1.2, radius=0.007, zorder=3)

        # Phase tag badge
        _draw_box(ax, (tx + 0.008, tbl_y + tbl_h - 0.024), tbl_w - 0.016, 0.018,
                  fc="#E8F5E9", ec=CLR_DELTA_BORD, lw=0.6, radius=0.003, zorder=4)
        ax.text(tx + tbl_w/2, tbl_y + tbl_h - 0.015, tphase,
                ha="center", va="center", fontsize=6.8, fontweight="bold", color=CLR_DELTA_BORD, zorder=5)

        # Table Path Name
        ax.text(tx + tbl_w/2, tbl_y + tbl_h - 0.042, tname,
                ha="center", va="center", fontsize=7.2, fontweight="bold", fontfamily="monospace",
                color=CLR_TEXT, zorder=4)
        ax.text(tx + tbl_w/2, tbl_y + tbl_h - 0.060, tdesc,
                ha="center", va="center", fontsize=5.8, color=CLR_MUTED, style="italic", zorder=4)

        # Schema box
        _draw_box(ax, (tx + 0.008, tbl_y + 0.035), tbl_w - 0.016, 0.082,
                  fc="#FAFAFA", ec="#E0E0E0", lw=0.6, radius=0.003, zorder=4)
        ax.text(tx + tbl_w/2, tbl_y + 0.076, tschema,
                ha="center", va="center", fontsize=5.8, fontfamily="monospace", color=CLR_TEXT, zorder=5)

        # Delta Logo Badge
        ax.text(tx + tbl_w/2, tbl_y + 0.016, "Δ  Delta Lake (Parquet + Log)",
                ha="center", va="center", fontsize=6.0, fontweight="bold", color=CLR_DELTA_BORD, zorder=4)

    # =========================================================================
    # 3. BIDIRECTIONAL DATA-FLOW ARROWS (EMR <--> S3)
    # =========================================================================
    for ax_pos in [0.15, 0.48, 0.82]:
        _draw_arrow(ax, (ax_pos, emr_y), (ax_pos, s3_y + s3_h),
                    color=CLR_DELTA_BORD, lw=2.2, style="<->")

    # Central Delta Lake Protocol Banner
    proto_x, proto_y, proto_w, proto_h = 0.33, 0.365, 0.34, 0.060
    _draw_box(ax, (proto_x, proto_y), proto_w, proto_h,
              fc="#FFFFFF", ec=CLR_DELTA_BORD, lw=1.5, radius=0.008, zorder=6)
    ax.text(proto_x + proto_w/2, proto_y + proto_h/2 + 0.010,
            "Delta Lake Protocol (Spark SQL ⇄ S3 Object Store)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=CLR_DELTA_BORD, zorder=7)
    ax.text(proto_x + proto_w/2, proto_y + proto_h/2 - 0.012,
            "ACID Transactions • Schema Enforcement • Repartition Locality • Checkpoint Skip",
            ha="center", va="center", fontsize=6.5, color=CLR_MUTED, zorder=7)

    # Save High-Res Outputs
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    print(f"✓ Publication-grade architecture figure successfully generated:")
    print(f"  - {PNG_PATH}")
    print(f"  - {PDF_PATH}")


if __name__ == "__main__":
    main()
