#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_gatv2_sweep.sh
#  Executes a SINGLE, comprehensive GATv2 sweep across all configured datasets:
#  - Reads datasets & configuration directly from experiment_config.py
#  - Runs Phases 0 -> 1 -> 2 -> 3 (GATv2) -> 3b (GATv2 CAAN) -> 4 -> 5
#  - Produces ONE unified Excel report: results/<EXPERIMENT_NAME>_results.xlsx
# ══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/gatv2_sweep_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/execution.log"

echo "=========================================================================="
echo " Starting Single Unified GATv2 Full Sweep"
echo " Timestamp: $TIMESTAMP"
echo " Log Directory: $LOG_DIR"
echo "=========================================================================="

# ── 1. Storage & Environment Setup ─────────────────────────────────────────────
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

echo "► Using storage directory: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$REPO_DIR:$LARGE_TMP/.local/lib/$PY_VER/site-packages:$PYTHONPATH"

# ── 2. Determine Runner (EMR vs Local) ─────────────────────────────────────────
if [ -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_emr.py"
else
    RUNNER="python3 runners/run_local.py"
fi
echo "► Runner Engine: $RUNNER"

# ── 3. Upload Code and Config to S3 if AWS S3 tools are available ──────────────
if python3 -c "import boto3" 2>/dev/null; then
    echo "► Uploading pipeline and configuration to S3..."
    python3 upload_to_s3.py 2>&1 | tee "$LOG_DIR/s3_upload.log" || true
fi

# ── 4. Execute Full Unified Sweep ──────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo " Executing GATv2 Pipeline (Phases 0 -> 1 -> 2 -> 3 -> 3b -> 4 -> 5)..."
echo "=========================================================================="

$RUNNER \
  --global-mapping "true" \
  2>&1 | tee "$LOG_FILE"

# ── 5. Post-Processing & Master Excel Report ───────────────────────────────────
echo ""
echo "=========================================================================="
echo " Generating Master Excel Report & LaTeX Summaries..."
echo "=========================================================================="

python3 -m phases.phase5_reporting 2>&1 | tee -a "$LOG_DIR/phase5_reporting.log" || true

if [ -f "runners/generate_emr_excel_report.py" ]; then
    python3 runners/generate_emr_excel_report.py 2>&1 | tee -a "$LOG_DIR/master_excel_report.log" || true
fi

echo "=========================================================================="
echo " GATv2 FULL SWEEP COMPLETED SUCCESSFULLY!"
echo " Log file: $LOG_FILE"
echo "=========================================================================="
