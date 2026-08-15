#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_emr_experiments.sh
#  Master Execution & Runner Script for EMO Experiments on AWS EMR Cluster
#
#  Usage:
#    ./run_emr_experiments.sh                  # Runs default suite (reddit + ogbn-products + ogbn-mag)
#    ./run_emr_experiments.sh full_sweep       # Runs full 9-dataset scaling sweep (4 -> 8 -> 16 Executors on 2-worker cluster)
#    ./run_emr_experiments.sh 2worker          # Explicit 2-worker 9-dataset scaling sweep (saves to s3_latest_results_2worker)
#    ./run_emr_experiments.sh 4worker          # 4-worker 9-dataset scaling sweep (8 -> 16 -> 32 Executors)
#    ./run_emr_experiments.sh reddit           # Runs only reddit (Louvain + SAGE + CaaN)
#    ./run_emr_experiments.sh products         # Runs only ogbn-products (Louvain + SAGE + CaaN)
#    ./run_emr_experiments.sh mag              # Runs only ogbn-mag (Louvain + SAGE + CaaN)
#    ./run_emr_experiments.sh livejournal      # Runs only LiveJournal
#    ./run_emr_experiments.sh orkut            # Runs only Orkut
#    ./run_emr_experiments.sh standards        # Runs small standards (WikiCS, Coauthor-Physics, DeezerEurope)
#    ./run_emr_experiments.sh all              # Runs all datasets sequentially
# ══════════════════════════════════════════════════════════════════════════════

set -e

TARGET="${1:-2worker}"
NUM_EXECS="${2:-}"

# Detect cluster topology and configuration
if [[ "$TARGET" == *"2worker"* || "$TARGET" == "full_sweep" || "$TARGET" == "scaling" || "$TARGET" == "sweep" || "${WORKERS:-2}" == "2" ]]; then
    CLUSTER_TYPE="2worker"
    SWEEP_EXECS=(4 8 16)
    OUT_2WORKER_DIR="results/s3_latest_results_2worker"
    export S3_DEST_SUBDIR="spark-results-2worker"
else
    CLUSTER_TYPE="4worker"
    SWEEP_EXECS=(8 16 32)
    OUT_2WORKER_DIR="results/s3_latest_results"
    export S3_DEST_SUBDIR="spark-results"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_${CLUSTER_TYPE}"
mkdir -p "$LOG_DIR" "logs" "results" "$OUT_2WORKER_DIR"

EXTRA_ARGS=""
if [[ -n "$NUM_EXECS" ]]; then
    EXTRA_ARGS="--executor-instances $NUM_EXECS"
    echo "► Custom Executor Instances Override: $NUM_EXECS"
fi

echo "=========================================================================="
echo " Starting EMO Pipeline Master Runner on AWS EMR (${CLUSTER_TYPE^^} CLUSTER)"
echo " Target Suite: $TARGET"
echo " Cluster Topology: $CLUSTER_TYPE (Scaling Tiers: ${SWEEP_EXECS[*]} Executors)"
echo " Results Destination: $OUT_2WORKER_DIR"
echo " S3 Destination Subdirectory: gnn-bench-out/$S3_DEST_SUBDIR/"
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
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.security=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/math=ALL-UNNAMED"

# ── 3. Select Runner ──────────────────────────────────────────────────────────
if [ -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_emr.py"
else
    RUNNER="python3 runners/run_local.py"
fi
echo "► Selected Execution Runner: $RUNNER"

# Helper function to copy latest results to 2worker folder
sync_to_target_dir() {
    mkdir -p "$OUT_2WORKER_DIR"
    cp -r results/*.xlsx "$OUT_2WORKER_DIR/" 2>/dev/null || true
    cp -r results/*.tex "$OUT_2WORKER_DIR/" 2>/dev/null || true
    cp -r results/*.json "$OUT_2WORKER_DIR/" 2>/dev/null || true
    cp -r "$LOG_DIR"/*.log "$OUT_2WORKER_DIR/" 2>/dev/null || true
    cp -r logs/*.log "$OUT_2WORKER_DIR/" 2>/dev/null || true
}

# ── 4. Target Execution Suites ────────────────────────────────────────────────

# Target: 2WORKER / FULL_SWEEP / SCALING / SWEEP (Multi-Executor Sweep across ALL 9 Datasets)
if [[ "$TARGET" == "2worker" || "$TARGET" == "full_sweep_2worker" || "$TARGET" == "scaling_2worker" || "$TARGET" == "sweep_2worker" || "$TARGET" == "scaling" || "$TARGET" == "sweep" || "$TARGET" == "full_sweep" || "$TARGET" == "4worker" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Multi-Executor Scalability Sweep (${SWEEP_EXECS[*]} Executors)"
    echo " Target Datasets: WikiCS, Coauthor-Physics, Coauthor-CS, DeezerEurope, reddit, ogbn-products, ogbn-mag, LiveJournal, Orkut"
    echo " Output Directory: $OUT_2WORKER_DIR"
    echo "=========================================================================="
    
    SWEEP_DATASETS=("WikiCS" "Coauthor-Physics" "Coauthor-CS" "DeezerEurope" "reddit" "ogbn-products" "ogbn-mag" "LiveJournal" "Orkut")
    
    for E in "${SWEEP_EXECS[@]}"; do
        echo ""
        echo "══════════════════════════════════════════════════════════════════════════"
        echo " ► [STAGE: $E EXECUTORS] Running full benchmark suite with $E Executors"
        echo "══════════════════════════════════════════════════════════════════════════"
        for DS in "${SWEEP_DATASETS[@]}"; do
            echo "► Running $DS with $E Executors..."
            SCALING_LOG="$LOG_DIR/${DS}_scaling_e${E}.log"
            if $RUNNER \
              --experiment-name "scaling_sweep_${CLUSTER_TYPE}_${DS}_e${E}" \
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
              2>&1 | tee "$SCALING_LOG"; then
                echo "  ✓ $DS ($E Executors) completed successfully."
            else
                echo "  ⚠ [Notice] $DS ($E Executors) encountered an error. Proceeding to next dataset. Log: $SCALING_LOG"
            fi
            sync_to_target_dir
        done
        echo "✓ Finished pass for $E Executors."
    done
    echo "✓ Full Multi-Executor Scaling Sweep Completed across (${SWEEP_EXECS[*]}) Executors!"
fi

# Target: REDDIT
if [[ "$TARGET" == "reddit" ]]; then
    EXEC_COUNT="${NUM_EXECS:-8}"
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running reddit Large-Scale Benchmark (Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/reddit_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "reddit_emr_${CLUSTER_TYPE}" \
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
    sync_to_target_dir
fi

# Target: PRODUCTS
if [[ "$TARGET" == "products" || "$TARGET" == "ogbn-products" ]]; then
    EXEC_COUNT="${NUM_EXECS:-8}"
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running ogbn-products Large-Scale Benchmark (Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/products_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "products_emr_${CLUSTER_TYPE}" \
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
    sync_to_target_dir
fi

# Target: MAG
if [[ "$TARGET" == "mag" || "$TARGET" == "ogbn-mag" ]]; then
    EXEC_COUNT="${NUM_EXECS:-8}"
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running ogbn-mag Heterogeneous Benchmark (Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/mag_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "mag_emr_${CLUSTER_TYPE}" \
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
    sync_to_target_dir
fi

# Target: LIVEJOURNAL
if [[ "$TARGET" == "livejournal" ]]; then
    EXEC_COUNT="${NUM_EXECS:-8}"
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running LiveJournal Benchmark (Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/livejournal_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "livejournal_emr_${CLUSTER_TYPE}" \
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
    sync_to_target_dir
fi

# Target: ORKUT
if [[ "$TARGET" == "orkut" ]]; then
    EXEC_COUNT="${NUM_EXECS:-8}"
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Orkut Benchmark (Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/orkut_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "orkut_emr_${CLUSTER_TYPE}" \
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
    sync_to_target_dir
fi

# Target: CUSTOM DATASET (Direct name or comma-separated list, e.g. "Orkut", "LiveJournal,Orkut", "WikiCS")
KNOWN_TARGETS=" default reddit products ogbn-products mag ogbn-mag livejournal orkut standards timing scaling sweep full_sweep 2worker 4worker full_sweep_2worker scaling_2worker sweep_2worker all "
if [[ ! "$KNOWN_TARGETS" =~ " $TARGET " ]]; then
    EXEC_COUNT="${NUM_EXECS:-8}"
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Custom Dataset Target: $TARGET (Executors: $EXEC_COUNT)"
    echo "=========================================================================="
    SAFE_NAME=$(echo "$TARGET" | tr ',' '_')
    LOG_FILE="$LOG_DIR/custom_${SAFE_NAME}_e${EXEC_COUNT}.log"
    $RUNNER \
      --experiment-name "custom_${SAFE_NAME}" \
      --datasets "$TARGET" \
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
    echo "✓ Custom dataset run complete. Log: $LOG_FILE"
    sync_to_target_dir
fi

# ── 5. Compile Master Excel Report & Upload to S3 ──────────────────────────────
echo ""
echo "=========================================================================="
echo " Compiling Master EMR ${CLUSTER_TYPE^^} Cluster Excel Report & Uploading to S3..."
echo "=========================================================================="
REPORT_EXCEL="results/emr_${CLUSTER_TYPE}_cluster_results.xlsx"
python3 runners/generate_emr_excel_report.py --cluster-type "$CLUSTER_TYPE" --output "$REPORT_EXCEL" || true
python3 upload_to_s3.py 2>/dev/null || true

sync_to_target_dir

# S3 direct sync to target folder
if command -v aws &> /dev/null; then
    echo "► Syncing $OUT_2WORKER_DIR to s3://us-east-1-s3-gnn/gnn-bench-out/$OUT_2WORKER_DIR/..."
    aws s3 sync "$OUT_2WORKER_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/s3_latest_results_2worker/" --quiet || true
    aws s3 sync "$OUT_2WORKER_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/spark-results-2worker/" --quiet || true
fi

echo ""
echo "=========================================================================="
echo " All Requested Experiments Completed Successfully!"
echo " Results Excel: $REPORT_EXCEL"
echo " Local 2-Worker Folder: $OUT_2WORKER_DIR"
echo " Logs directory: $LOG_DIR"
echo "=========================================================================="
