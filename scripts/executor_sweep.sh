#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Executor scalability sweep: same workload, varying parallelism.
#
#  Measures how Phase 3 and Phase 3b wall clock respond to executor count.
#  Only the executor count varies; everything else is held fixed.
#
#  Two things must be pinned or the result is not a speedup curve
#  --------------------------------------------------------------
#  1. PHASE3_MAX_TRAIN_PER_UNIT must be 0. The slot-aware relax in
#     phase3_training.py keys off sc.defaultParallelism, so with it enabled a
#     different executor count produces a different unit count -- a different
#     workload -- and the runs stop being comparable.
#  2. DGL must be installed on the workers. The Phase 3 UDF attempts
#     `import dgl` and pip-installs it inside the task on failure, before the
#     branch that decides DGL is unnecessary for PyG backbones. That install
#     runs once per Python worker process, and the worker-process count scales
#     with executor count, so a missing DGL adds overhead that grows as the
#     cluster shrinks -- exactly backwards from a speedup curve.
#
#  Epochs are deliberately low (10): this measures scaling, not accuracy.
#
#  Usage:
#    scripts/executor_sweep.sh                    # 32 16 8
#    scripts/executor_sweep.sh 32 24 16 8
#    scripts/executor_sweep.sh --cores 4 32 16 8
# ══════════════════════════════════════════════════════════════════════════════

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CORES=4
EPOCHS=10
EXPERIMENT="gatv2_reddit_edgecap"
COUNTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --cores)  CORES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --name)   EXPERIMENT="$2"; shift 2 ;;
        *)        COUNTS+=("$1"); shift ;;
    esac
done
[ ${#COUNTS[@]} -eq 0 ] && COUNTS=(32 16 8)

LARGE_TMP=/mnt/tmp
export TMPDIR="$LARGE_TMP" TEMP="$LARGE_TMP" TMP="$LARGE_TMP"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"
export PYSPARK_PYTHON="$(which python3)"
export PYSPARK_DRIVER_PYTHON="$(which python3)"
PY_VERSION=$(python3 -c "import sys; print('python%d.%d' % (sys.version_info.major, sys.version_info.minor))")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VERSION/site-packages:${PYTHONPATH:-}"

STAMP=$(date -u +%Y%m%d_%H%M%S)
OUTDIR="$LARGE_TMP/sweep_$STAMP"
mkdir -p "$OUTDIR"

PIN=$(grep -E "^PHASE3_MAX_TRAIN_PER_UNIT" experiment_config.py | head -1)
echo "=============================================================="
echo "  EXECUTOR SWEEP"
echo "  counts    : ${COUNTS[*]}   (x $CORES cores = slots)"
echo "  epochs    : $EPOCHS   (scaling, not accuracy)"
echo "  experiment: $EXPERIMENT"
echo "  commit    : $(git rev-parse --short HEAD)"
echo "  workload  : $PIN"
echo "  outdir    : $OUTDIR"
echo "=============================================================="

if ! echo "$PIN" | grep -q "= *0"; then
    echo ""
    echo "  WARNING: PHASE3_MAX_TRAIN_PER_UNIT is not 0."
    echo "  The slot-aware relax will change the unit count per executor count,"
    echo "  so these runs will NOT be measuring the same workload."
    echo ""
fi

for N in "${COUNTS[@]}"; do
    LOG="$OUTDIR/exec_${N}.log"
    echo ""
    echo "--- $N executors x $CORES cores = $((N * CORES)) slots ---"
    START=$(date +%s)
    python3 -u runners/run_emr.py \
        --datasets reddit \
        --experiment-name "$EXPERIMENT" \
        --no-phase0 --no-phase1 --no-phase2 \
        --executor-instances "$N" \
        --executor-cores "$CORES" \
        --num-epochs "$EPOCHS" > "$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    echo "  rc=$RC  wall=$((END-START))s  log=$LOG"
    echo "$N $CORES $((END-START)) $RC" >> "$OUTDIR/wall.txt"
    # Fail loudly but keep going: a later count may still succeed.
    if [ $RC -ne 0 ]; then
        echo "  !! run failed; last lines:"
        tr '\r' '\n' < "$LOG" | grep -v "^\[Stage" | tail -6 | sed 's/^/     /'
    fi
done

echo ""
echo "=============================================================="
echo "  SWEEP SUMMARY"
echo "=============================================================="
printf "  %-6s %-6s %-9s %-11s %-11s %-9s %s\n" \
       "execs" "slots" "wall(s)" "phase3(s)" "phase3b(s)" "units" "rc"
BASE=""
while read -r N C WALL RC; do
    LOG="$OUTDIR/exec_${N}.log"
    CLEAN=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -v "^\[Stage")
    P3=$(echo "$CLEAN" | grep -m1 "Wall time:" | grep -oE "[0-9.]+" | head -1)
    P3B=$(echo "$CLEAN" | grep "Wall time:" | sed -n 2p | grep -oE "[0-9.]+" | head -1)
    UNITS=$(echo "$CLEAN" | grep -m1 "bounded training units" | grep -oE "> [0-9,]+ bounded" | grep -oE "[0-9,]+")
    printf "  %-6s %-6s %-9s %-11s %-11s %-9s %s\n" \
           "$N" "$((N * C))" "$WALL" "${P3:-—}" "${P3B:-—}" "${UNITS:-—}" "$RC"
    [ -z "$BASE" ] && BASE="$WALL"
done < "$OUTDIR/wall.txt"

echo ""
echo "  speedup relative to the SMALLEST executor count:"
SMALL=$(sort -n "$OUTDIR/wall.txt" | head -1 | awk '{print $1}')
SMALLW=$(awk -v n="$SMALL" '$1==n {print $3}' "$OUTDIR/wall.txt" | head -1)
while read -r N C WALL RC; do
    if [ "$RC" = "0" ] && [ -n "$SMALLW" ] && [ "$WALL" -gt 0 ]; then
        awk -v n="$N" -v w="$WALL" -v sw="$SMALLW" -v sn="$SMALL" \
            'BEGIN{printf "    %-4s execs: %.2fx faster than %s execs (ideal %.2fx)\n", n, sw/w, sn, n/sn}'
    fi
done < "$OUTDIR/wall.txt"

echo ""
echo "  logs: $OUTDIR"
echo "  NOTE: restore PHASE3_MAX_TRAIN_PER_UNIT to 1500 before real runs."
echo "=============================================================="
