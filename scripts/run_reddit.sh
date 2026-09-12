#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Launch a Reddit pipeline run on EMR, safely.
#
#  Why this wrapper exists
#  -----------------------
#  Two settings in experiment_config.py are traps when invoked bare:
#
#    * DATASETS_TO_RUN lists six datasets, so `python3 runners/run_emr.py`
#      runs ogbn-products, ogbn-mag, LiveJournal and Orkut as well as Reddit.
#    * Phase 2 output paths are scoped by EXPERIMENT_NAME
#      (phase2_nodes/{EXPERIMENT_NAME}_{dataset}_{alg}/), so changing the name
#      silently recomputes Phase 2 instead of reusing it. Community detection
#      is NOT experiment-scoped and is always reusable.
#
#  It also refuses to start until every worker passes preflight. One
#  unprovisioned node aborts the whole job from inside a pandas UDF, typically
#  deep in Phase 3b after Phase 3 has already succeeded - that cost two runs,
#  48 and 27 minutes, on 2026-09-11.
#
#  Usage
#  -----
#    scripts/run_reddit.sh                      # reuse phases 0-2, full run
#    scripts/run_reddit.sh --fresh              # recompute phases 0-2
#    scripts/run_reddit.sh --quick              # epochs=10 smoke test
#    scripts/run_reddit.sh --name my_experiment
#    scripts/run_reddit.sh --skip-preflight     # not recommended
# ══════════════════════════════════════════════════════════════════════════════

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Phase 2 tables written on 2026-09-11 live under this name; keep it to reuse them.
EXPERIMENT="gatv2_reddit_edgecap"
REUSE="--no-phase0 --no-phase1 --no-phase2"
EXTRA=""
PREFLIGHT=1

while [ $# -gt 0 ]; do
    case "$1" in
        --fresh)           REUSE=""; shift ;;
        --quick)           EXTRA="$EXTRA --num-epochs 10"; shift ;;
        --name)            EXPERIMENT="$2"; shift 2 ;;
        --skip-preflight)  PREFLIGHT=0; shift ;;
        *)                 EXTRA="$EXTRA $1"; shift ;;
    esac
done

LARGE_TMP=/mnt/tmp
export TMPDIR="$LARGE_TMP" TEMP="$LARGE_TMP" TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"
export PYSPARK_PYTHON="$(which python3)"
export PYSPARK_DRIVER_PYTHON="$(which python3)"
PY_VERSION=$(python3 -c "import sys; print('python%d.%d' % (sys.version_info.major, sys.version_info.minor))")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VERSION/site-packages:$PYTHONPATH"

LOG="$LARGE_TMP/reddit_$(date -u +%Y%m%d_%H%M%S).log"

if [ "$PREFLIGHT" -eq 1 ]; then
    echo "=== preflight: verifying every worker node ==="
    if ! spark-submit --master yarn --deploy-mode client "$REPO/scripts/emr_preflight.py"; then
        echo ""
        echo "PREFLIGHT FAILED - not starting the run."
        echo "Provision the workers first:"
        echo "  spark-submit --master yarn --deploy-mode client scripts/emr_provision_workers.py"
        exit 1
    fi
    echo ""
fi

echo "=== launching ==="
echo "  experiment : $EXPERIMENT"
echo "  reuse      : ${REUSE:-<none: recomputing phases 0-2>}"
echo "  extra args : ${EXTRA:-<none>}"
echo "  commit     : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  log        : $LOG"
echo ""

{
    echo "=== launch $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "=================================="
    START=$(date +%s)
    # shellcheck disable=SC2086
    python3 -u runners/run_emr.py \
        --datasets reddit \
        --experiment-name "$EXPERIMENT" \
        $REUSE $EXTRA
    RC=$?
    END=$(date +%s)
    echo "=================================="
    echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$RC ==="
    echo "=== WALL CLOCK: $((END-START))s ($(( (END-START)/60 ))m $(( (END-START)%60 ))s) ==="
} 2>&1 | tee "$LOG"

echo ""
echo "Log saved to $LOG"
echo "Useful greps:"
echo "  grep -E 'Wall time|WALL CLOCK|Weighted comm' $LOG"
echo "  grep 'partition metric' $LOG      # confirms the dual link metric ran"
echo "  grep 'Bounded blocks' $LOG        # Phase 3b unit count"
echo "  grep 'train cap relaxed' $LOG     # slot-aware unit fitting"
