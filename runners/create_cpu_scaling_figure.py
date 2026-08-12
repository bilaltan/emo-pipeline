#!/usr/bin/env python3
"""
create_cpu_scaling_figure.py
Generates an ML-GRL Figure 8 style publication plot showing running time and speedup
as the number of CPU cores / Spark executor count increases across datasets.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/figures")
PNG_PATH = OUT_DIR / "fig8_cpu_cores_scaling.png"
PDF_PATH = OUT_DIR / "fig8_cpu_cores_scaling.pdf"

# Publication style colors
COLOR_PAPERS = "#1E6091"
COLOR_PRODUCTS = "#D97706"
COLOR_WIKICS = "#10B981"
INK = "#1F2937"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })

    executors = np.array([16, 32, 64])
    cores = executors * 4  # 4 vCPUs per executor node

    # Runtime measurements in seconds (Phase 3.7 + 3.8)
    papers100m_time = np.array([2578.3, 1940.5, 1892.1])
    products_time = np.array([112.4, 84.5, 80.8])
    wikics_time = np.array([12.5, 14.8, 22.1])

    # Calculate Speedup relative to 16 executors baseline
    papers100m_speedup = papers100m_time[0] / papers100m_time
    products_speedup = products_time[0] / products_time
    wikics_speedup = wikics_time[0] / wikics_time

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- Panel A: Running Time vs CPU Cores ---
    ax1.plot(cores, papers100m_time, "o-", color=COLOR_PAPERS, lw=2, ms=6, label="ogbn-papers100M (111M nodes)")
    ax1.plot(cores, products_time, "s-", color=COLOR_PRODUCTS, lw=2, ms=6, label="ogbn-products (2.4M nodes)")
    ax1.plot(cores, wikics_time, "^-", color=COLOR_WIKICS, lw=2, ms=6, label="WikiCS (11K nodes)")
    
    ax1.set_yscale("log")
    ax1.set_xlabel("Total CPU Cores (Spark Executors)")
    ax1.set_ylabel("Execution Time (seconds, log scale)")
    ax1.set_title("(a) Total Running Time Scaling", pad=8)
    ax1.set_xticks(cores)
    ax1.set_xticklabels([f"64 cores\n(16 exec)", f"128 cores\n(32 exec)", f"256 cores\n(64 exec)"])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # --- Panel B: Speedup Factor vs Ideal Linear ---
    ideal_speedup = cores / cores[0]
    ax2.plot(cores, ideal_speedup, "--", color="#94A3B8", lw=1.8, label="Ideal Linear Speedup")
    ax2.plot(cores, papers100m_speedup, "o-", color=COLOR_PAPERS, lw=2, ms=6, label="ogbn-papers100M")
    ax2.plot(cores, products_speedup, "s-", color=COLOR_PRODUCTS, lw=2, ms=6, label="ogbn-products")
    ax2.plot(cores, wikics_speedup, "^-", color=COLOR_WIKICS, lw=2, ms=6, label="WikiCS")

    ax2.set_xlabel("Total CPU Cores (Spark Executors)")
    ax2.set_ylabel("Speedup Factor (relative to 16 exec)")
    ax2.set_title("(b) Speedup Efficiency Comparison", pad=8)
    ax2.set_xticks(cores)
    ax2.set_xticklabels([f"64 cores\n(16 exec)", f"128 cores\n(32 exec)", f"256 cores\n(64 exec)"])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(PDF_PATH, bbox_inches="tight")
    print(f"✓ Created CPU scaling figure (ML-GRL Fig. 8 style) at:")
    print(f"  - {PNG_PATH}")
    print(f"  - {PDF_PATH}")

if __name__ == "__main__":
    main()
