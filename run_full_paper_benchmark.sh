#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_full_paper_benchmark.sh
#  Smart Master Script for EMO Paper Full Benchmark Suite (IEEE Big Data 2026)
#
#  Workload Distribution Strategy:
#   1. Standard Scale Datasets (WikiCS, reddit, ogbn-arxiv, DeezerEurope):
#      -> Full Approach 1: Ingestion (P0) -> Partitioning (P1/P2) -> PyTorch GNN (P3) 
#         -> CAAN Context Recovery (P3b) -> DistDGL & PyG Baselines (P4).
#
#   2. Large Scale Datasets (ogbn-products, ogbn-papers100M):
#      -> Approach 2 (EMR/Spark Scaling): Zero-RAM Ingestion (P0) -> 2-Hop Delta Cached
#         Feature Propagation (P3.7) -> Global Spark ML Probe (P3.8).
#
#   3. System Ablation Study (reddit):
#      -> Full EMO vs. Decoupled (w/o CAAN) vs. Uncached S3 Reads.
#
#   4. Post-Processing & Automated Reporting:
#      -> Generates Excel Summary + LaTeX Tables (Tables 3, 4, 9, 12, 13) + Fig 8.
# ══════════════════════════════════════════════════════════════════════════════

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/full_benchmark_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================================================="
echo " EMO PIPELINE — FULL PAPER BENCHMARK EXECUTION"
echo " Timestamp: $TIMESTAMP"
echo " Log Directory: $LOG_DIR"
echo "=========================================================================="

# ── 1. Discover Environment & Large Storage Volumes ────────────────────────────
CANDIDATES=("/mnt/tmp" "/mnt1/tmp" "/mnt2/tmp" "/mnt/spark" "/mnt1/spark" "/mnt2/spark" "/tmp")
LARGE_TMP="/tmp"

for candidate in "${CANDIDATES[@]}"; do
    if [ ! -d "$candidate" ]; then
        mkdir -p "$candidate" 2>/dev/null || true
    fi
    if [ -w "$candidate" ]; then
        LARGE_TMP="$candidate"
        if [[ "$candidate" != "/tmp" ]]; then
            break
        fi
    fi
done

echo "► Using large storage volume: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VER/site-packages:$PYTHONPATH"

if [ -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_emr.py"
else
    RUNNER="python3 runners/run_local.py"
fi
echo "► Selected Execution Runner: $RUNNER"

# ── 2. STAGE 1: Standard Datasets — Approach 1 (Phases 0-3b) + Baselines (Phase 4) 
echo ""
echo "=========================================================================="
echo " [STAGE 1/4] Standard Scale Graphs (Approach 1 + Baselines)"
echo " Datasets: WikiCS, reddit, ogbn-arxiv, DeezerEurope"
echo "=========================================================================="

STAGE1_LOG="$LOG_DIR/stage1_standard_datasets.log"

$RUNNER \
  --experiment-name "paper_standard_scale" \
  --datasets "WikiCS,reddit,ogbn-arxiv,deezereurope" \
  --algorithms "lpa,louvain" \
  --run-phase0 \
  --run-phase3 \
  --run-phase3b \
  --run-phase4 \
  --global-mapping "true" \
  2>&1 | tee "$STAGE1_LOG"

echo "✓ Stage 1 Completed. Log: $STAGE1_LOG"

# ── 3. STAGE 2: Large Datasets — Approach 2 (Phases 3.7 & 3.8 Distributed Scale) ─
echo ""
echo "=========================================================================="
echo " [STAGE 2/4] Large Scale Graphs (Approach 2: 2-Hop Feature Propagation)"
echo " Datasets: ogbn-products (2.4M nodes), ogbn-papers100M (111M nodes)"
echo "=========================================================================="

STAGE2_LOG="$LOG_DIR/stage2_large_scale.log"

$RUNNER \
  --experiment-name "paper_large_scale" \
  --datasets "ogbn-products,ogbn-papers100M" \
  --run-phase0 \
  --run-phase37 \
  --run-phase38 \
  --no-phase3 \
  --no-phase3b \
  2>&1 | tee "$STAGE2_LOG"

echo "✓ Stage 2 Completed. Log: $STAGE2_LOG"

# ── 4. STAGE 3: System Ablation Matrix (Data for Table 9) ──────────────────────
echo ""
echo "=========================================================================="
echo " [STAGE 3/4] System & Component Ablation Study (reddit)"
echo " Evaluating: Full EMO vs. w/o CAAN vs. Uncached S3 Reads"
echo "=========================================================================="

STAGE3_LOG="$LOG_DIR/stage3_ablations.log"

# 3A: Decoupled Louvain (w/o CAAN)
$RUNNER \
  --experiment-name "ablation_louvain_nocaan" \
  --datasets "reddit" \
  --algorithms "louvain" \
  --run-phase0 \
  --run-phase3 \
  --no-phase3b \
  --global-mapping "true" \
  2>&1 | tee -a "$STAGE3_LOG"

# 3B: Decoupled LPA (w/o CAAN)
$RUNNER \
  --experiment-name "ablation_lpa_nocaan" \
  --datasets "reddit" \
  --algorithms "lpa" \
  --run-phase0 \
  --run-phase3 \
  --no-phase3b \
  --global-mapping "true" \
  2>&1 | tee -a "$STAGE3_LOG"

# 3C: Uncached S3 Reads (w/o Delta Caching)
$RUNNER \
  --experiment-name "ablation_uncached_s3" \
  --datasets "reddit" \
  --algorithms "louvain" \
  --run-phase0 \
  --force-reingest \
  --run-phase3 \
  --run-phase3b \
  2>&1 | tee -a "$STAGE3_LOG"

echo "✓ Stage 3 Completed. Log: $STAGE3_LOG"

# ── 5. STAGE 4: Automated Post-Processing, Table Export & Figure Rendering ─────
echo ""
echo "=========================================================================="
echo " [STAGE 4/4] Rendering LaTeX Tables & Figure 8 Scaling Plot"
echo "=========================================================================="

echo "► Exporting Excel summary and generating LaTeX tables..."
python3 -m phases.phase5_reporting 2>&1 | tee -a "$LOG_DIR/stage4_reporting.log"

echo "► Rendering ML-GRL Figure 8 CPU Cores Scaling plot..."
python3 runners/create_cpu_scaling_figure.py 2>&1 | tee -a "$LOG_DIR/stage4_figure8.log"

echo ""
echo "=========================================================================="
echo " FULL PAPER BENCHMARK COMPLETED SUCCESSFULLY!"
echo "=========================================================================="
echo " Results Excel: results/run-all_results-7.xlsx"
echo " LaTeX Tables:"
echo "   - Table 3 (Link Prediction):        results/table3_link_prediction_main.tex"
echo "   - Table 4 (Node Classification):   results/table4_node_classification_main.tex"
echo "   - Table 9 (Ablation Study):        results/table9_ablation_study.tex"
echo "   - Table 12 (DistDGL Comparison):   results/table12_distdgl_comparison.tex"
echo "   - Table 13 (Multi-Instance Matrix): results/table13_scalability_multi_instance.tex"
echo " Generated Figures:"
echo "   - Figure 8 (CPU Scaling):          results/figures/fig8_cpu_cores_scaling.pdf"
echo "=========================================================================="
