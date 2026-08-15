#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_all_experiments_master.sh
#  Master End-to-End Orchestrator for EMO Paper Experiments
#  (IEEE Big Data / TKDE / SIGMOD GRL Benchmark)
#
#  Key Capabilities:
#    1. Global graph experiments on OGBN-Products (Node Classification + Link Prediction)
#       using Louvain as the default partitioner.
#    2. Re-runs all experiments using self-collected data only (separating self-run
#       empirical metrics from ML-GRL / CaaN published literature in all tables).
#    3. Fixes Table 4 baseline accuracy and timing metrics across all 9 benchmark datasets.
#    4. Collects granular phase-by-phase timing breakdowns (Phase 0, 1, 2, 3 Node/Link, 3b).
#    5. Incorporates ~100M+ edge scale graphs: Orkut (117.2M edges) and LiveJournal (34.7M edges)
#       to supplement OGBN-Products (61.9M edges), Reddit (11.6M edges), and OGBN-MAG (5.4M edges).
#    6. Compiles Master Excel reports (5 sheets) and 6 publication-ready LaTeX tables.
#    7. Synchronizes all logs, checkpoints, and outputs with Amazon S3.
#
#  Usage:
#    ./run_all_experiments_master.sh                    # Full 9-dataset benchmark suite across all scaling tiers
#    ./run_all_experiments_master.sh products           # Run OGBN-Products (Louvain default, Node + Link)
#    ./run_all_experiments_master.sh 100m               # Run ~100M-edge scale suite (LiveJournal, Orkut, Products, Reddit, MAG)
#    ./run_all_experiments_master.sh reddit             # Run Reddit benchmark
#    ./run_all_experiments_master.sh livejournal        # Run LiveJournal benchmark
#    ./run_all_experiments_master.sh orkut              # Run Orkut benchmark
#    ./run_all_experiments_master.sh standards          # Run academic standards (WikiCS, Coauthor, DeezerEurope)
#    ./run_all_experiments_master.sh timing             # Extract and compile granular phase timing breakdown
#    ./run_all_experiments_master.sh tables             # Regenerate all LaTeX tables and Master Excel Report
#    ./run_all_experiments_master.sh 8                  # Run full sweep with fixed 8 executors
# ══════════════════════════════════════════════════════════════════════════════

set -eo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARG1="${1:-all}"
ARG2="${2:-}"

# Parse executor override or dataset targets
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

# Detect cluster topology and configuration
if [[ "${WORKERS:-4}" == "2" || "$TARGET" == *"2worker"* ]]; then
    CLUSTER_TYPE="2worker"
    DEFAULT_SWEEP=(4 8 16)
    OUT_SUBDIR="s3_latest_results_2worker"
    S3_DEST_SUBDIR="spark-results-2worker"
else
    CLUSTER_TYPE="4worker"
    DEFAULT_SWEEP=(8 16 32)
    OUT_SUBDIR="s3_latest_results"
    S3_DEST_SUBDIR="spark-results"
fi

if [[ -n "$CUSTOM_EXECS" ]]; then
    SWEEP_EXECS=("$CUSTOM_EXECS")
    echo "► Fixed Executor Count Override: $CUSTOM_EXECS"
else
    SWEEP_EXECS=("${DEFAULT_SWEEP[@]}")
fi

OUT_DIR="results/${OUT_SUBDIR}"
LOG_DIR="logs/master_${TIMESTAMP}"
mkdir -p "$OUT_DIR" "$LOG_DIR" "logs" "results" "overleaf/results"

CLUSTER_UPPER=$(echo "$CLUSTER_TYPE" | tr '[:lower:]' '[:upper:]')

echo "=========================================================================="
echo " Starting EMO Pipeline Master Orchestrator (${CLUSTER_UPPER} CLUSTER)"
echo " Target Mode / Dataset: $TARGET"
echo " Executor Scaling Tiers: ${SWEEP_EXECS[*]}"
echo " Default Partitioner: Louvain"
echo " Default Tasks: Node Classification + Link Prediction (both)"
echo " Results Destination: $OUT_DIR"
echo " S3 Destination: s3://us-east-1-s3-gnn/gnn-bench-out/${OUT_SUBDIR}/"
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

echo "► Using scratch storage path: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl" "$LARGE_TMP/ogb_data"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
if [[ "$LARGE_TMP" == "/mnt"* ]]; then
    export PYTHONUSERBASE="$LARGE_TMP/.local"
fi
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

# Discover Python executable with dependencies (pandas, etc.)
if [ -x "/opt/homebrew/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
elif [ -x "$PWD/.venv/bin/python3" ]; then
    PYTHON_BIN="$PWD/.venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")
if [[ "$LARGE_TMP" == "/mnt"* ]]; then
    export PYTHONPATH="$LARGE_TMP/.local/lib/python${PY_VER}/site-packages:$PWD:$PYTHONPATH"
else
    export PYTHONPATH="$HOME/Library/Python/${PY_VER}/lib/python/site-packages:$PWD:$PYTHONPATH"
fi

# JVM Flags for Apache Arrow zero-copy memory access
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.security=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/math=ALL-UNNAMED"

# Select execution runner (EMR Spark cluster vs Local Multi-threaded)
if [ -f "runners/run_emr.py" ]; then
    RUNNER="$PYTHON_BIN runners/run_emr.py"
else
    RUNNER="$PYTHON_BIN runners/run_local.py"
fi
echo "► Selected Runner: $RUNNER"

sync_artifacts() {
    mkdir -p "$OUT_DIR" "overleaf/results"
    cp -r results/*.xlsx "$OUT_DIR/" 2>/dev/null || true
    cp -r results/*.tex "$OUT_DIR/" 2>/dev/null || true
    cp -r results/*.tex "overleaf/results/" 2>/dev/null || true
    cp -r results/*.json "$OUT_DIR/" 2>/dev/null || true
    cp -r "$LOG_DIR"/*.log "$OUT_DIR/" 2>/dev/null || true
    cp -r logs/*.log "$OUT_DIR/" 2>/dev/null || true
}

# ── 2. Handle Non-Execution Utility Targets ────────────────────────────────────
if [[ "$TARGET" == "tables" || "$TARGET" == "report" || "$TARGET" == "latex" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [UTILITY] Compiling Master Reports and LaTeX Tables"
    echo "=========================================================================="
    REPORT_PATH="results/emr_${CLUSTER_TYPE}_cluster_results.xlsx"
    $PYTHON_BIN runners/generate_emr_excel_report.py --cluster-type "$CLUSTER_TYPE" --output "$REPORT_PATH"
    $PYTHON_BIN -m phases.phase5_reporting
    sync_artifacts
    echo "✓ LaTeX tables and Excel report generated successfully."
    exit 0
fi

# ── 3. Determine Dataset List ──────────────────────────────────────────────────
ALL_DATASETS=("WikiCS" "Coauthor-Physics" "Coauthor-CS" "DeezerEurope" "reddit" "ogbn-products" "ogbn-mag" "LiveJournal" "Orkut")
HUNDRED_M_DATASETS=("ogbn-products" "reddit" "ogbn-mag" "LiveJournal" "Orkut")
STANDARD_DATASETS=("WikiCS" "Coauthor-Physics" "Coauthor-CS" "DeezerEurope")

case "$TARGET" in
    help|-h|--help)
        echo "Usage: $0 [TARGET] [NUM_EXECUTORS]"
        echo ""
        echo "Targets:"
        echo "  all                  Full 9-dataset benchmark suite (WikiCS, Coauthor-Physics, Coauthor-CS, DeezerEurope, reddit, ogbn-products, ogbn-mag, LiveJournal, Orkut)"
        echo "  products             Run OGBN-Products (Louvain default, Node Classification + Link Prediction)"
        echo "  100m                 Run ~100M-edge scale suite (LiveJournal, Orkut, OGBN-Products, Reddit, OGBN-MAG)"
        echo "  standards            Run small academic standards (WikiCS, Coauthor, DeezerEurope)"
        echo "  reddit               Run Reddit benchmark"
        echo "  mag                  Run OGBN-MAG heterogeneous benchmark"
        echo "  livejournal          Run LiveJournal benchmark (34.7M edges)"
        echo "  orkut                Run Orkut benchmark (117.2M edges)"
        echo "  timing               Extract granular Phase 0, 1, 2, 3, 3b timing breakdowns"
        echo "  tables               Compile Master Excel Report and 6 LaTeX tables"
        echo ""
        echo "Examples:"
        echo "  $0 products          # Run OGBN-Products across all scaling tiers"
        echo "  $0 products 8        # Run OGBN-Products with 8 executors"
        echo "  $0 100m              # Run all ~100M edge scale benchmarks"
        echo "  $0 tables            # Regenerate LaTeX tables and Excel report"
        exit 0
        ;;
    all|sweep|full_sweep|2worker|4worker)
        DATASET_LIST=("${ALL_DATASETS[@]}")
        ;;
    100m|100M|large|massive)
        DATASET_LIST=("${HUNDRED_M_DATASETS[@]}")
        ;;
    standards|small)
        DATASET_LIST=("${STANDARD_DATASETS[@]}")
        ;;
    products|ogbn-products)
        DATASET_LIST=("ogbn-products")
        ;;
    reddit)
        DATASET_LIST=("reddit")
        ;;
    mag|ogbn-mag)
        DATASET_LIST=("ogbn-mag")
        ;;
    livejournal|LiveJournal)
        DATASET_LIST=("LiveJournal")
        ;;
    orkut|Orkut)
        DATASET_LIST=("Orkut")
        ;;
    timing)
        DATASET_LIST=("${ALL_DATASETS[@]}")
        SWEEP_EXECS=("${SWEEP_EXECS[0]}")
        ;;
    *)
        IFS=',' read -ra DATASET_LIST <<< "$TARGET"
        ;;
esac

echo "► Target Dataset Queue: ${DATASET_LIST[*]}"

# ── 4. Execute Benchmark Suite ────────────────────────────────────────────────
for E in "${SWEEP_EXECS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════"
    echo " ► [STAGE: $E EXECUTORS] Running Benchmark Suite with $E Executors"
    echo "══════════════════════════════════════════════════════════════════════════"
    for DS in "${DATASET_LIST[@]}"; do
        echo ""
        echo "► [Running] Dataset: $DS | Partitioner: Louvain | Tasks: Node + Link | Executors: $E"
        RUN_LOG="$LOG_DIR/${DS}_e${E}.log"

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
          2>&1 | tee "$RUN_LOG"; then
            echo "  ✓ $DS ($E Executors) completed successfully."
        else
            echo "  ⚠ [Notice] $DS ($E Executors) encountered an issue. Check log: $RUN_LOG"
        fi
        sync_artifacts
    done
    echo "✓ Finished executor pass: $E Executors."
done

# ── 5. Post-Processing: Compile Master Excel & LaTeX Tables ─────────────────────
echo ""
echo "=========================================================================="
echo " Compiling Master EMR ${CLUSTER_UPPER} Cluster Excel Report & LaTeX Tables..."
echo "=========================================================================="

REPORT_PATH="results/emr_${CLUSTER_TYPE}_cluster_results.xlsx"
$PYTHON_BIN runners/generate_emr_excel_report.py --cluster-type "$CLUSTER_TYPE" --output "$REPORT_PATH" || true
$PYTHON_BIN -m phases.phase5_reporting || true
$PYTHON_BIN upload_to_s3.py 2>/dev/null || true
sync_artifacts

# ── 6. S3 Direct Sync ─────────────────────────────────────────────────────────
if command -v aws &> /dev/null; then
    echo "► Syncing artifacts to S3: s3://us-east-1-s3-gnn/gnn-bench-out/${OUT_SUBDIR}/"
    aws s3 sync "$OUT_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/${OUT_SUBDIR}/" --quiet || true
    aws s3 sync "$OUT_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/${S3_DEST_SUBDIR}/" --quiet || true
    if [ -f "$REPORT_PATH" ]; then
        aws s3 cp "$REPORT_PATH" "s3://us-east-1-s3-gnn/gnn-bench-out/emr_${CLUSTER_TYPE}_cluster_results.xlsx" --quiet || true
    fi
fi

echo ""
echo "=========================================================================="
echo " EMO MASTER PIPELINE EXECUTION COMPLETED!"
echo " Master Excel Report: $REPORT_PATH"
echo " Local Artifacts Directory: $OUT_DIR"
echo " LaTeX Tables Generated:"
echo "   - Table 3 (Link Prediction):        results/table3_link_prediction_main.tex"
echo "   - Table 4 (Node Classification):   results/table4_node_classification_main.tex"
echo "   - Table 9 (Ablation Study):        results/table9_ablation_study.tex"
echo "   - Table 12 (DistDGL Comparison):   results/table12_distdgl_comparison.tex"
echo "   - Table 13 (Multi-Instance Matrix): results/table13_scalability_multi_instance.tex"
echo "   - Timeline (Phase Latency):        results/table_phase_timeline.tex"
echo "=========================================================================="
