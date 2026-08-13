#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_single_fast_run.sh
#  Executes a SINGLE, streamlined, fast end-to-end experiment run across ALL phases
#  (Phases 0, 1, 2, 3, 3b, 3.7, 3.8, 4, 5) on WikiCS in ~1 minute.
# ══════════════════════════════════════════════════════════════════════════════

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/single_fast_run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================================================="
echo " Starting Single Fast End-to-End Experiment Run"
echo " Dataset: WikiCS (Fast 1-minute execution)"
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

echo "► Using large storage path: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VER/site-packages:$PYTHONPATH"

# Determine execution engine runner
if [ -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_emr.py"
else
    RUNNER="python3 runners/run_local.py"
fi
echo "► Selected Runner: $RUNNER"

# ── 2. Run All Phases in One Single Pass ────────────────────────────────────────
LOG_FILE="$LOG_DIR/single_run.log"

echo ""
echo "=========================================================================="
echo " Executing All Phases (Phase 0 -> Phase 5) in a Single Run..."
echo "=========================================================================="

$RUNNER \
  --experiment-name "fast_single_pass" \
  --datasets "WikiCS" \
  --algorithms "lpa" \
  --run-phase0 \
  --run-phase3 \
  --run-phase3b \
  --run-phase37 \
  --run-phase38 \
  --run-phase4 \
  --global-mapping "true" \
  2>&1 | tee "$LOG_FILE"

# ── 3. Post-Processing & Figure Generation ─────────────────────────────────────
echo ""
echo "=========================================================================="
echo " Post-Processing & Rendering Tables/Figures"
echo "=========================================================================="

python3 -m phases.phase5_reporting 2>&1 | tee -a "$LOG_DIR/phase5_reporting.log"
python3 runners/create_cpu_scaling_figure.py 2>&1 | tee -a "$LOG_DIR/figure8_generation.log"

echo "=========================================================================="
echo " SINGLE FAST RUN COMPLETED SUCCESSFULLY!"
echo " Log file: $LOG_FILE"
echo " Generated Artifacts:"
echo "   - Excel Summary: results/run-all_results-7.xlsx"
echo "   - Table 3: results/table3_link_prediction_main.tex"
echo "   - Table 4: results/table4_node_classification_main.tex"
echo "   - Table 9: results/table9_ablation_study.tex"
echo "   - Table 12: results/table12_distdgl_comparison.tex"
echo "   - Table 13: results/table13_scalability_multi_instance.tex"
echo "   - Figure 8: results/figures/fig8_cpu_cores_scaling.pdf"
echo "=========================================================================="
