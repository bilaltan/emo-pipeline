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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/emr_run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

TARGET="${1:-default}"

echo "=========================================================================="
echo " Starting EMO Pipeline Master Runner on AWS EMR"
echo " Target: $TARGET"
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

# ── 4. Target Execution Suites ─────────────────────────────────────────────────

# Target: REDDIT
if [[ "$TARGET" == "default" || "$TARGET" == "reddit" || "$TARGET" == "all" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running reddit (Louvain + SAGE + CaaN)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/reddit_louvain_caan.log"
    $RUNNER \
      --experiment-name "paper_reddit_louvain_caan" \
      --datasets "reddit" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ reddit run complete. Log: $LOG_FILE"
fi

# Target: OGBN-PRODUCTS
if [[ "$TARGET" == "default" || "$TARGET" == "products" || "$TARGET" == "ogbn-products" || "$TARGET" == "all" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running ogbn-products (Louvain + SAGE + CaaN)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/ogbn_products_louvain_caan.log"
    $RUNNER \
      --experiment-name "paper_products_louvain_caan" \
      --datasets "ogbn-products" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ ogbn-products run complete. Log: $LOG_FILE"
fi

# Target: OGBN-MAG
if [[ "$TARGET" == "default" || "$TARGET" == "mag" || "$TARGET" == "ogbn-mag" || "$TARGET" == "all" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running ogbn-mag (Louvain + SAGE + CaaN)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/ogbn_mag_louvain_caan.log"
    $RUNNER \
      --experiment-name "paper_mag_louvain_caan" \
      --datasets "ogbn-mag" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ ogbn-mag run complete. Log: $LOG_FILE"
fi

# Target: LIVEJOURNAL
if [[ "$TARGET" == "livejournal" || "$TARGET" == "all" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running LiveJournal (Louvain + SAGE + CaaN)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/livejournal_louvain_caan.log"
    $RUNNER \
      --experiment-name "paper_livejournal_louvain_caan" \
      --datasets "LiveJournal" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ LiveJournal run complete. Log: $LOG_FILE"
fi

# Target: ORKUT
if [[ "$TARGET" == "orkut" || "$TARGET" == "all" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Orkut (Louvain + SAGE + CaaN)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/orkut_louvain_caan.log"
    $RUNNER \
      --experiment-name "paper_orkut_louvain_caan" \
      --datasets "Orkut" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ Orkut run complete. Log: $LOG_FILE"
fi

# Target: STANDARDS (WikiCS, Coauthor-Physics, DeezerEurope)
if [[ "$TARGET" == "standards" || "$TARGET" == "all" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Standards: WikiCS, Coauthor-Physics, DeezerEurope"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/standards_louvain_caan.log"
    $RUNNER \
      --experiment-name "paper_standards_louvain_caan" \
      --datasets "WikiCS,Coauthor-Physics,DeezerEurope" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase1 \
      --run-phase2 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      --task-type "both" \
      2>&1 | tee "$LOG_FILE"
    echo "✓ Standards run complete. Log: $LOG_FILE"
fi

# Target: TIMING (Granular Phase-by-Phase Latency Breakdown)
if [[ "$TARGET" == "timing" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [EXECUTION] Running Phase Latency Breakdown Sweep (Phases 0 -> 1 -> 2 -> 3 -> 3b)"
    echo "=========================================================================="
    LOG_FILE="$LOG_DIR/phase_latency_breakdown.log"
    $RUNNER \
      --experiment-name "paper_timing_breakdown" \
      --datasets "WikiCS,Coauthor-Physics,reddit,ogbn-products" \
      --algorithms "louvain,lpa" \
      --force-reingest \
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

# ── 5. Upload Output Tables & Excel to S3 ──────────────────────────────────────
echo ""
echo "=========================================================================="
echo " Uploading Results & Logs to S3..."
echo "=========================================================================="
python3 upload_to_s3.py 2>/dev/null || true

echo ""
echo "=========================================================================="
echo " All Requested Experiments Completed Successfully!"
echo " Logs directory: $LOG_DIR"
echo "=========================================================================="
