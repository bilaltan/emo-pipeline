#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Is the flat speedup curve caused by memory contention rather than skew?
#
#  Evidence that prompted this
#  ---------------------------
#  On ogbn-products, an IDENTICAL workload (575 units) got slower per task as
#  the cluster grew:
#
#      cores   sum(task)    median task   max task
#        32     10,688.9s      18.2s        82.0s
#        64     13,192.0s      21.9s        95.4s
#       128     21,054.8s      37.8s       155.4s
#
#  Median task time more than doubles for the same unit of work. Tasks do not
#  slow down because there are more of them; they slow down because 8 executors
#  per node are competing for memory. The autoscaler sizes each executor at
#  8g heap + 20g overhead = 28g, so 8/node needs 224g of a 236g node -- 95%
#  full, with nothing left for page cache or shuffle.
#
#  The 20g overhead exists to host Python workers at a budgeted 5.0 GB/task,
#  but the Phase 3 result frames report peak_mem_mb median 1,902 and max 2,555.
#  So the budget is roughly 2x larger than anything actually observed.
#
#  What this runs
#  --------------
#  Two configurations, both keeping 128 total cores, against the measured
#  baseline of Phase 3 = 231.4s (32 execs x 4 cores, 20g overhead):
#
#    A  32 execs x 4 cores, overhead 14g  -> 22g x 8/node = 176g (75% of node)
#    B  16 execs x 8 cores, overhead 20g  -> 28g x 4/node = 112g, fewer JVMs
#
#  Phase 3b is skipped so each configuration takes ~8 minutes instead of ~15.
#  If contention is the cause, median task time should fall back toward ~18-22s
#  and Phase 3 should drop well below 231.4s on identical hardware.
#
#  Usage: scripts/memory_contention_test.sh [dataset] [experiment]
# ══════════════════════════════════════════════════════════════════════════════

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATASET="${1:-ogbn-products}"
EXPERIMENT="${2:-gatv2_products_v1}"
EPOCHS=10

LARGE_TMP=/mnt/tmp
export TMPDIR="$LARGE_TMP" TEMP="$LARGE_TMP" TMP="$LARGE_TMP"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"
export PYSPARK_PYTHON="$(which python3)"
export PYSPARK_DRIVER_PYTHON="$(which python3)"
PY_VERSION=$(python3 -c "import sys; print('python%d.%d' % (sys.version_info.major, sys.version_info.minor))")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VERSION/site-packages:${PYTHONPATH:-}"

OUTDIR="$LARGE_TMP/memtest_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "=============================================================="
echo "  MEMORY CONTENTION TEST"
echo "  dataset   : $DATASET"
echo "  baseline  : Phase 3 = 231.4s  (32x4 cores, 20g overhead, 128 slots)"
echo "  node RAM  : ~236g usable per node, 4 nodes"
echo "  outdir    : $OUTDIR"
echo "=============================================================="

# label : execs : cores : overhead_gb
CONFIGS=(
  "A_overhead14:32:4:14"
  "B_fewer_jvm:16:8:20"
)

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r LABEL N C OH <<< "$CFG"
    PER_NODE=$(( N / 4 ))
    CONTAINER=$(( 8 + OH ))
    echo ""
    echo "--- $LABEL : $N execs x $C cores = $((N * C)) slots ---"
    echo "    container ${CONTAINER}g x ${PER_NODE}/node = $(( CONTAINER * PER_NODE ))g of ~236g"
    LOG="$OUTDIR/${LABEL}.log"
    START=$(date +%s)
    python3 -u runners/run_emr.py \
        --datasets "$DATASET" \
        --experiment-name "$EXPERIMENT" \
        --no-phase0 --no-phase1 --no-phase2 --no-phase3b \
        --executor-instances "$N" \
        --executor-cores "$C" \
        --executor-memory-overhead-gb "$OH" \
        --num-epochs "$EPOCHS" > "$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    P3=$(tr '\r' '\n' < "$LOG" | grep -v "^\[Stage" | grep -m1 "Wall time:" | grep -oE "[0-9.]+" | head -1)
    echo "    rc=$RC wall=$((END-START))s  PHASE 3 = ${P3:-failed}s   (baseline 231.4s)"
    echo "$LABEL $N $C $OH $((END-START)) ${P3:-0} $RC" >> "$OUTDIR/results.txt"
    if [ $RC -ne 0 ]; then
        echo "    !! failed; last lines:"
        tr '\r' '\n' < "$LOG" | grep -v "^\[Stage" | tail -5 | sed 's/^/       /'
    fi
done

echo ""
echo "=============================================================="
echo "  RESULTS  (baseline: 231.4s at 32x4 with 20g overhead)"
echo "=============================================================="
printf "  %-14s %-6s %-6s %-9s %-9s %-10s %s\n" label execs cores overhead slots phase3 vs_baseline
while read -r L N C OH W P RC; do
    printf "  %-14s %-6s %-6s %-9s %-9s %-10s " "$L" "$N" "$C" "${OH}g" "$((N*C))" "$P"
    awk -v p="$P" 'BEGIN{ if (p+0>0) printf "%.2fx\n", 231.4/p; else print "-" }'
done < "$OUTDIR/results.txt"
echo ""
echo "  A faster Phase 3 on identical core counts means the cluster was"
echo "  contending for memory, not saturated with work."
echo "  logs: $OUTDIR"
echo "=============================================================="
