#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_all_experiments_2worker.sh
#  Master Execution Script for 2-Worker AWS EMR Cluster Suite
#
#  Key Features:
#    1. Global graph experiments on OGBN-Products (Node Classification + Link Prediction) using Louvain
#    2. Re-runs all 9 benchmark datasets using self-collected cluster data
#    3. Sweeps 2-worker scaling tiers: 4 -> 8 -> 16 Executors
#    4. Collects granular Phase 0, 1, 2, 3 (Node + Link), and 3b timing breakdowns
#    5. Includes ~100M+ edge scale graphs: Orkut (117.2M edges) and LiveJournal (34.7M edges)
#    6. Saves all output directly to results/s3_latest_results_2worker/ and S3 bucket
#    7. Automatically compiles results/emr_2worker_cluster_results.xlsx
#
#  Usage:
#    ./run_all_experiments_2worker.sh              # Full 9-dataset scaling sweep (4 -> 8 -> 16 Executors)
#    ./run_all_experiments_2worker.sh ogbn-products# Run only OGBN-Products
#    ./run_all_experiments_2worker.sh reddit       # Run only Reddit
#    ./run_all_experiments_2worker.sh 8            # Run full sweep with fixed 8 executors
# ══════════════════════════════════════════════════════════════════════════════

set -e

ARG1="${1:-all}"
ARG2="${2:-}"

# Check if ARG1 is a number (executor override)
if [[ "$ARG1" =~ ^[0-9]+$ ]]; then
    CUSTOM_EXECS="$ARG1"
    TARGET="all"
elif [[ -n "$ARG2" && "$ARG2" =~ ^[0-9]+$ ]]; then
    TARGET="$ARG1"
    CUSTOM_EXECS="$ARG2"
else
    TARGET="$ARG1"
    CUSTOM_EXECS=""
fi

if [[ -n "$CUSTOM_EXECS" ]]; then
    SWEEP_EXECS=("$CUSTOM_EXECS")
    echo "► Fixed Executor Count Override: $CUSTOM_EXECS"
else
    SWEEP_EXECS=(4 8 16)
fi

CLUSTER_TYPE="2worker"
OUT_DIR="results/s3_latest_results_2worker"
LOG_DIR="logs_2worker"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$OUT_DIR" "$LOG_DIR" "logs" "results"

export S3_DEST_SUBDIR="spark-results-2worker"

echo "=========================================================================="
echo " Starting 2-Worker EMR Benchmark Suite (Louvain Default | Task: Node + Link)"
echo " Target Datasets: $TARGET"
echo " Executor Scaling Tiers: ${SWEEP_EXECS[*]}"
echo " Output Destination: $OUT_DIR"
echo " S3 Destination: s3://us-east-1-s3-gnn/gnn-bench-out/s3_latest_results_2worker/"
echo " Timestamp: $TIMESTAMP"
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

echo "► Using storage scratch path: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl" "$LARGE_TMP/ogb_data"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VER/site-packages:$PWD:$PYTHONPATH"

# JVM Flags for Apache Arrow memory sharing
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.security=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/math=ALL-UNNAMED"

RUNNER="python3 runners/run_emr.py"
if [ ! -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_local.py"
fi
echo "► Selected Runner: $RUNNER"

sync_artifacts() {
    mkdir -p "$OUT_DIR"
    cp -r results/*.xlsx "$OUT_DIR/" 2>/dev/null || true
    cp -r results/*.tex "$OUT_DIR/" 2>/dev/null || true
    cp -r results/*.json "$OUT_DIR/" 2>/dev/null || true
    cp -r "$LOG_DIR"/*.log "$OUT_DIR/" 2>/dev/null || true
    cp -r logs/*.log "$OUT_DIR/" 2>/dev/null || true
}

# ── 2. Determine Datasets to Run ──────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "sweep" || "$TARGET" == "full_sweep" || "$TARGET" == "2worker" ]]; then
    DATASET_LIST=("WikiCS" "Coauthor-Physics" "Coauthor-CS" "DeezerEurope" "reddit" "ogbn-products" "ogbn-mag" "LiveJournal" "Orkut")
else
    IFS=',' read -ra DATASET_LIST <<< "$TARGET"
fi

echo "► Target Dataset Queue: ${DATASET_LIST[*]}"

# ── 3. Run Benchmark Suite ────────────────────────────────────────────────────
for E in "${SWEEP_EXECS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════"
    echo " ► [STAGE: $E EXECUTORS] Running Benchmark Suite with $E Executors"
    echo "══════════════════════════════════════════════════════════════════════════"
    for DS in "${DATASET_LIST[@]}"; do
        echo ""
        echo "► [Running] Dataset: $DS | Partitioner: Louvain | Tasks: Node + Link | Executors: $E"
        RUN_LOG="$LOG_DIR/${DS}_2worker_e${E}.log"
        if $RUNNER \
          --experiment-name "scaling_sweep_2worker_${DS}_e${E}" \
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
          2>&1 | tee "$RUN_LOG"; then
            echo "  ✓ $DS ($E Executors) completed successfully."
        else
            echo "  ⚠ [Notice] $DS ($E Executors) encountered an error. Check log: $RUN_LOG"
        fi
        sync_artifacts
    done
    echo "✓ Finished executor pass: $E Executors."
done

# ── 4. Compile 2-Worker Master Excel Report & Upload ───────────────────────────
echo ""
echo "=========================================================================="
echo " Compiling 2-Worker Master Excel Report (results/emr_2worker_cluster_results.xlsx)..."
echo "=========================================================================="
REPORT_PATH="results/emr_2worker_cluster_results.xlsx"
python3 runners/generate_emr_excel_report.py --cluster-type 2worker --output "$REPORT_PATH" || true
python3 upload_to_s3.py 2>/dev/null || true
sync_artifacts

# S3 direct sync
if command -v aws &> /dev/null; then
    echo "► Uploading $OUT_DIR to S3..."
    aws s3 sync "$OUT_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/s3_latest_results_2worker/" --quiet || true
    aws s3 sync "$OUT_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/spark-results-2worker/" --quiet || true
    if [ -f "$REPORT_PATH" ]; then
        aws s3 cp "$REPORT_PATH" "s3://us-east-1-s3-gnn/gnn-bench-out/emr_2worker_cluster_results.xlsx" --quiet || true
    fi
fi

echo ""
echo "=========================================================================="
echo " 2-Worker EMR Benchmark Suite Completed!"
echo " Master Excel Report: $REPORT_PATH"
echo " Local Artifacts Directory: $OUT_DIR"
echo " S3 Destination: s3://us-east-1-s3-gnn/gnn-bench-out/s3_latest_results_2worker/"
echo "=========================================================================="
