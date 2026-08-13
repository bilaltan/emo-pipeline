#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_missing_results.sh
#  Generates ALL missing experimental data for the EMO paper (IEEE Big Data 2026)
#
#  What this fills:
#    1. Table 12: [Pending CaaN] for ogbn-products  (CAAN on ogbn-products)
#    2. Table 9:  Delta Lake caching ablation row    (cold S3 vs Delta cached)
#    3. Table 9:  Community threshold K sweep        (K=100,500,1000,5000)
#    4. Table 9:  ogbn-products ablation variants    (multi-dataset ablation)
#
#  Usage:
#    ./run_missing_results.sh          # Runs all missing experiments
#    ./run_missing_results.sh 1        # Only CAAN on ogbn-products
#    ./run_missing_results.sh 2        # Only Delta Lake caching ablation
#    ./run_missing_results.sh 3        # Only K-threshold sweep
#    ./run_missing_results.sh 4        # Only ogbn-products ablation
# ══════════════════════════════════════════════════════════════════════════════

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/missing_results_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

TARGET="${1:-all}"

echo "=========================================================================="
echo " EMO Missing Results Generator"
echo " Target: $TARGET"
echo " Timestamp: $TIMESTAMP"
echo " Logs: $LOG_DIR"
echo "=========================================================================="

# ── Discover Environment & Storage ───────────────────────────────────────────
CANDIDATES=("/mnt/tmp" "/mnt1/tmp" "/mnt2/tmp" "/mnt/spark" "/mnt1/spark" "/tmp")
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

echo "► Large storage: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VER/site-packages:$PYTHONPATH"

# Determine runner
if [ -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_emr.py"
else
    RUNNER="python3 runners/run_local.py"
fi
echo "► Runner: $RUNNER"

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: CAAN on ogbn-products  (fills Table 12 [Pending CaaN])
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$TARGET" == "all" || "$TARGET" == "1" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [1/4] CAAN on ogbn-products — fills Table 12 [Pending CaaN]"
    echo "=========================================================================="

    LOG="$LOG_DIR/exp1_ogbn_products_caan.log"

    # 1A: ogbn-products with LPA + CAAN
    echo "► Running ogbn-products LPA + SAGE + CAAN..."
    $RUNNER \
      --experiment-name "missing_ogbn_products_lpa_caan" \
      --datasets "ogbn-products" \
      --algorithms "lpa" \
      --run-phase0 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      2>&1 | tee "$LOG"

    # 1B: ogbn-products with Louvain + CAAN
    echo "► Running ogbn-products Louvain + SAGE + CAAN..."
    $RUNNER \
      --experiment-name "missing_ogbn_products_louvain_caan" \
      --datasets "ogbn-products" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      2>&1 | tee -a "$LOG"

    echo "✓ Experiment 1 complete. Log: $LOG"
    echo "  → Use these Node Acc results to fill [Pending CaaN] in Table 12"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Delta Lake Caching Ablation  (new row for Table 9)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$TARGET" == "all" || "$TARGET" == "2" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [2/4] Delta Lake Caching Ablation — new Table 9 row"
    echo "=========================================================================="

    LOG="$LOG_DIR/exp2_delta_lake_ablation.log"

    # 2A: Force full re-ingestion (uncached cold read from S3)
    echo "► Running reddit Louvain + SAGE + CAAN with FORCED RE-INGESTION (cold S3)..."
    $RUNNER \
      --experiment-name "ablation_uncached_cold_s3" \
      --datasets "reddit" \
      --algorithms "louvain" \
      --run-phase0 \
      --force-reingest \
      --force-rerun \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      2>&1 | tee "$LOG"

    # 2B: Normal cached Delta run (same config, no force-reingest)
    echo "► Running reddit Louvain + SAGE + CAAN with DELTA LAKE CACHING (warm)..."
    $RUNNER \
      --experiment-name "ablation_cached_delta" \
      --datasets "reddit" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      2>&1 | tee -a "$LOG"

    echo "✓ Experiment 2 complete. Log: $LOG"
    echo "  → Compare Phase 0 ingestion time + total time between cold vs cached runs"
    echo "  → This creates 'w/o Delta Lake Optimizations' row in Table 9"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Community Threshold K Sweep  (parametric ablation for Table 9)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$TARGET" == "all" || "$TARGET" == "3" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [3/4] Community Threshold K Sweep — parametric ablation"
    echo "=========================================================================="

    LOG="$LOG_DIR/exp3_k_threshold_sweep.log"

    for K in 100 500 1000 5000; do
        echo "► Running reddit Louvain + SAGE + CAAN with min_community_size=$K ..."
        $RUNNER \
          --experiment-name "ablation_k${K}_louvain_caan" \
          --datasets "reddit" \
          --algorithms "louvain" \
          --min-community-size "$K" \
          --run-phase0 \
          --run-phase3 \
          --run-phase3b \
          --global-mapping "true" \
          2>&1 | tee -a "$LOG"

        echo "  ✓ K=$K done."
    done

    echo "✓ Experiment 3 complete. Log: $LOG"
    echo "  → Creates K-threshold sensitivity rows for Table 9"
    echo "  → Report: K value, #communities, node acc, link AUC, total time"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 4: ogbn-products Ablation Variants  (multi-dataset Table 9)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$TARGET" == "all" || "$TARGET" == "4" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [4/4] ogbn-products Ablation Variants — multi-dataset Table 9 coverage"
    echo "=========================================================================="

    LOG="$LOG_DIR/exp4_ogbn_products_ablation.log"

    # 4A: ogbn-products Decoupled LPA (w/o CAAN)
    echo "► Running ogbn-products LPA + SAGE (no CAAN)..."
    $RUNNER \
      --experiment-name "ablation_ogbn_products_lpa_nocaan" \
      --datasets "ogbn-products" \
      --algorithms "lpa" \
      --run-phase0 \
      --run-phase3 \
      --no-phase3b \
      --global-mapping "true" \
      2>&1 | tee "$LOG"

    # 4B: ogbn-products Decoupled Louvain (w/o CAAN)
    echo "► Running ogbn-products Louvain + SAGE (no CAAN)..."
    $RUNNER \
      --experiment-name "ablation_ogbn_products_louvain_nocaan" \
      --datasets "ogbn-products" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase3 \
      --no-phase3b \
      --global-mapping "true" \
      2>&1 | tee -a "$LOG"

    echo "✓ Experiment 4 complete. Log: $LOG"
    echo "  → Provides ogbn-products ablation data for a multi-dataset Table 9"
fi

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================================================="
echo " ALL MISSING RESULTS EXPERIMENTS COMPLETE"
echo " Logs directory: $LOG_DIR"
echo ""
echo " NEXT STEPS:"
echo "   1. Extract timing/accuracy from logs"
echo "   2. Update results/table12_distdgl_comparison.tex (fill [Pending CaaN])"
echo "   3. Update results/table9_ablation_study.tex (add new rows)"
echo "   4. Re-run Suite 5 to regenerate LaTeX: ./run_all_experiments.sh 5"
echo "=========================================================================="
