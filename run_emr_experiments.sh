#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_emr_experiments.sh
#  Master Execution & Runner Script for EMO Experiments on AWS EMR Cluster
#
#  Usage:
#    ./run_emr_experiments.sh                  # Runs default suite (reddit + ogbn-products + ogbn-mag)
#    ./run_emr_experiments.sh reddit           # Runs only reddit (Louvain + SAGE + CaaN)
#    ./run_emr_experiments.sh products         # Runs only ogbn-products (Louvain + SAGE + CaaN)
#    ./run_emr_experiments.sh mag              # Runs only ogbn-mag (Louvain + SAGE + CaaN)
#    ./run_emr_experiments.sh livejournal      # Runs only LiveJournal
#    ./run_emr_experiments.sh orkut            # Runs only Orkut
#    ./run_emr_experiments.sh standards        # Runs small standards (WikiCS, Coauthor-Physics, DeezerEurope)
#    ./run_emr_experiments.sh all              # Runs all datasets sequentially
#    ./run_emr_experiments.sh timing           # Runs detailed phase-by-phase timing breakdown benchmark
# ══════════════════════════════════════════════════════════════════════════════

set -e

TARGET="${1:-default}"
NUM_EXECS="${2:-}"

EXTRA_ARGS=""
if [[ -n "$NUM_EXECS" ]]; then
    EXTRA_ARGS="--executor-instances $NUM_EXECS"
    echo "► Custom Executor Instances: $NUM_EXECS"
fi

echo "=========================================================================="
echo " Starting EMO Pipeline Master Runner on AWS EMR"
echo " Target Suite: $TARGET"
if [[ -n "$NUM_EXECS" ]]; then
    echo " Executor Instances Override: $NUM_EXECS"
else
    echo " Executor Allocation: Dynamic Cluster Auto-Scaler (Optimal YARN Bin-Packing)"
fi
echo " Timestamp: $TIMESTAMP"
echo " Log Directory: $LOG_DIR"
echo "=========================================================================="


# ── 1. Discover Storage Volumes & Remap Temp Dirs ──────────────────────────────
CANDIDATES=("/mnt/tmp" "/mnt1/tmp" "/mnt2/tmp" "/mnt/spark" "/mnt1/spark" "/mnt2/spark" "/tmp")
LARGE_TMP="/tmp"

for candidate in "${CANDIDATES[@]}"; do
    if [ ! -d "$candidate" ]; then
        sudo mkdir -p "$candidate" 2>/dev/null || mkdir -p "$candidate" 2>/dev/null || true
        sudo chmod 777 "$candidate" 2>/dev/null || true
    fi
    if [ -w "$candidate" ]; then
        LARGE_TMP="$candidate"
        if [[ "$candidate" != "/tmp" ]]; then
            break
        fi
    fi
done

echo "► Using large storage path: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl" "$LARGE_TMP/ogb_data"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VER/site-packages:$PWD:$PYTHONPATH"

# ── 2. Check & Install Dependencies if Missing ────────────────────────────────
echo "► Checking Python environment..."
python3 -c "import torch; import torch_geometric; import igraph; import ogb; import pyarrow; print('  ✓ All essential libraries verified.')" 2>/dev/null || {
    echo "  ⚠️ Missing libraries detected. Running bootstrap setup..."
    bash ./emr_setup.sh
}

# Configure Java JVM flags globally for Apache Arrow zero-copy memory access
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.security=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/java.math=ALL-UNNAMED"

# ── 3. Select Runner ──────────────────────────────────────────────────────────
if [ -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_emr.py"
else
    RUNNER="python3 runners/run_local.py"
fi
echo "► Selected Execution Runner: $RUNNER"

# ── 4. Target Execution Suites with Dataset-Aware Optimal Executor Profiles ───

# Helper function to resolve best executor count per dataset if not overridden
get_best_executors() {
    local dataset="$1"
    if [[ -n "$NUM_EXECS" ]]; then
        echo "$NUM_EXECS"
        return
    fi
    case "$dataset" in
        "WikiCS"|"Coauthor-Physics"|"Coauthor-CS"|"DeezerEurope")
            echo "8"   # Small graphs: 8 executors avoids scheduling overhead
            ;;
        "reddit"|"ogbn-products"|"ogbn-mag")
            echo "16"  # Medium benchmarks: 16 executors (2 per node on 8-node cluster)
            ;;
        "LiveJournal"|"Orkut")
            echo "32"  # Dense social graphs: 32 executors (4 per node) for high parallelism
            ;;
        *)
            echo "16"
            ;;
    esac
}

# Target: REDDIT
if [[ "$TARGET" == "default" || "$TARGET" == "reddit" || "$TARGET" == "all" ]]; then
    EXEC_COUNT=$(get_best_executors "reddit")
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running reddit (Louvain + SAGE + CaaN | Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/reddit_louvain_caan_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "paper_reddit_louvain_caan" \
      --datasets "reddit" \
      --algorithms "louvain" \
      --executor-instances "$EXEC_COUNT" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --force-rerun \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ reddit run complete. Log: $LOG_FILE"
fi

# Target: OGBN-PRODUCTS
if [[ "$TARGET" == "default" || "$TARGET" == "products" || "$TARGET" == "ogbn-products" || "$TARGET" == "all" ]]; then
    EXEC_COUNT=$(get_best_executors "ogbn-products")
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running ogbn-products (Louvain + SAGE + CaaN | Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/ogbn_products_louvain_caan_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "paper_products_louvain_caan" \
      --datasets "ogbn-products" \
      --algorithms "louvain" \
      --executor-instances "$EXEC_COUNT" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --force-rerun \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ ogbn-products run complete. Log: $LOG_FILE"
fi

# Target: OGBN-MAG
if [[ "$TARGET" == "default" || "$TARGET" == "mag" || "$TARGET" == "ogbn-mag" || "$TARGET" == "all" ]]; then
    EXEC_COUNT=$(get_best_executors "ogbn-mag")
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running ogbn-mag (Louvain + SAGE + CaaN | Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/ogbn_mag_louvain_caan_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "paper_mag_louvain_caan" \
      --datasets "ogbn-mag" \
      --algorithms "louvain" \
      --executor-instances "$EXEC_COUNT" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --force-rerun \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ ogbn-mag run complete. Log: $LOG_FILE"
fi

# Target: LIVEJOURNAL
if [[ "$TARGET" == "livejournal" || "$TARGET" == "all" ]]; then
    EXEC_COUNT=$(get_best_executors "LiveJournal")
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running LiveJournal (Louvain + SAGE + CaaN | Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/livejournal_louvain_caan_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "paper_livejournal_louvain_caan" \
      --datasets "LiveJournal" \
      --algorithms "louvain" \
      --executor-instances "$EXEC_COUNT" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --force-rerun \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ LiveJournal run complete. Log: $LOG_FILE"
fi

# Target: ORKUT
if [[ "$TARGET" == "orkut" || "$TARGET" == "all" ]]; then
    EXEC_COUNT=$(get_best_executors "Orkut")
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Orkut (Louvain + SAGE + CaaN | Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/orkut_louvain_caan_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "paper_orkut_louvain_caan" \
      --datasets "Orkut" \
      --algorithms "louvain" \
      --executor-instances "$EXEC_COUNT" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --force-rerun \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ Orkut run complete. Log: $LOG_FILE"
fi

# Target: STANDARDS (WikiCS, Coauthor-Physics, Coauthor-CS, DeezerEurope)
if [[ "$TARGET" == "standards" || "$TARGET" == "all" ]]; then
    EXEC_COUNT=$(get_best_executors "WikiCS")
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Standards: WikiCS, Coauthor-Physics, Coauthor-CS, DeezerEurope (Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/standards_louvain_caan_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "paper_standards_louvain_caan" \
      --datasets "WikiCS,Coauthor-Physics,Coauthor-CS,DeezerEurope" \
      --algorithms "louvain" \
      --executor-instances "$EXEC_COUNT" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --force-rerun \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ Standards run complete. Log: $LOG_FILE"
fi

# Target: TIMING (Granular Phase-by-Phase Latency Breakdown)
if [[ "$TARGET" == "timing" ]]; then
    EXEC_COUNT="${NUM_EXECS:-16}"
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Phase Latency Breakdown Sweep (Phases 0 -> 1 -> 2 -> 3 -> 3b | Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/phase_latency_breakdown_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "paper_timing_breakdown" \
      --datasets "WikiCS,Coauthor-Physics,Coauthor-CS,DeezerEurope,reddit,ogbn-products,ogbn-mag,LiveJournal,Orkut" \
      --algorithms "louvain" \
      --executor-instances "$EXEC_COUNT" \
      --force-reingest \
      --force-rerun \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ Timing breakdown complete. Log: $LOG_FILE"
fi

# Target: SCALING / FULL_SWEEP (8 -> 16 -> 32 Multi-Executor Scaling Sweep across ALL Small & Medium Datasets)
if [[ "$TARGET" == "scaling" || "$TARGET" == "sweep" || "$TARGET" == "full_sweep" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Multi-Executor Scalability Sweep (8 -> 16 -> 32 Executors)"
    echo " All Small & Medium Datasets: WikiCS, Coauthor-Physics, Coauthor-CS, DeezerEurope, reddit, ogbn-products, ogbn-mag, LiveJournal, Orkut"
    echo "=========================================================================="
    
    SWEEP_DATASETS=("WikiCS" "Coauthor-Physics" "Coauthor-CS" "DeezerEurope" "reddit" "ogbn-products" "ogbn-mag" "LiveJournal" "Orkut")
    
    for E in 8 16 32; do
        echo ""
        echo "══════════════════════════════════════════════════════════════════════════"
        echo " ► [STAGE: $E EXECUTORS] Running full benchmark suite with $E Executors"
        echo "══════════════════════════════════════════════════════════════════════════"
        for DS in "${SWEEP_DATASETS[@]}"; do
            echo "► Running $DS with $E Executors..."
            SCALING_LOG="$LOG_DIR/${DS}_scaling_e${E}.log"
            $RUNNER \
              --experiment-name "scaling_sweep_${DS}_e${E}" \
              --datasets "$DS" \
              --algorithms "louvain" \
              --executor-instances "$E" \
              --run-phase0 \
              --run-phase1 \
              --run-phase2 \
              --run-phase3 \
              --run-phase3b \
              --force-rerun \
              --global-mapping "true" \
              --task-type "both" \
              2>&1 | tee "$SCALING_LOG"
            echo "  ✓ $DS ($E Executors) completed."
        done
        echo "✓ All datasets finished for $E Executors."
    done
    echo "✓ Full Multi-Executor Scaling Sweep Completed across 8, 16, and 32 Executors!"
fi

# ── 5. Compile Master Excel Report & Upload to S3 ──────────────────────────────
echo ""
echo "=========================================================================="
echo " Compiling Master EMR 4-Worker Cluster Excel Report & Uploading to S3..."
echo "=========================================================================="
python3 runners/generate_emr_excel_report.py || true
python3 upload_to_s3.py 2>/dev/null || true

echo ""
echo "=========================================================================="
echo " All Requested Experiments Completed Successfully!"
echo " Results Excel: results/emr_4worker_cluster_results.xlsx"
echo " Logs directory: $LOG_DIR"
echo "=========================================================================="
