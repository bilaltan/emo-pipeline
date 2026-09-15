#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Node scaling sweep on 8 nodes = 128 PHYSICAL cores.
#
#  Why this is the first meaningful scaling test
#  ---------------------------------------------
#  Every earlier sweep ran on 4 nodes = 64 physical cores. ogbn-products has
#  usable parallelism (sum of task time / longest task) of 305.8, so it could
#  always absorb far more cores than existed. The cluster, not the workload, was
#  the limit. With 8 nodes there are 128 physical cores, still below 305.8, so
#  products should now scale for real.
#
#  r6id.8xlarge reports 32 vCPUs but has 16 physical cores (Thread(s) per core
#  2). Measured on 4 nodes, crossing one task per physical core returned only
#  ~11%, while giving each task the second hyperthread via OMP=2 was worth more:
#
#      64 slots, OMP=1   255.9s
#     128 slots, OMP=1   231.4s
#      64 slots, OMP=2   224.8s   <- best, on HALF the executors
#      32 slots, OMP=4   271.1s   <- saturates
#
#  So every point here uses OMP=2 with one task per physical core, and the slot
#  count is what varies. 128 slots now means 8 nodes fully used.
#
#  Baselines to beat (4 nodes, products, Phase 3):
#      32 slots  381.4s | 64 slots 255.9s (OMP=1) / 224.8s (OMP=2) | 128 slots 231.4s
#
#  Phase 3b is included: it has its own ceiling (823 units) and the question of
#  whether it scales differently from Phase 3 matters for the paper.
#
#  Usage: scripts/node_scaling_sweep.sh [dataset] [experiment] [slots...]
# ══════════════════════════════════════════════════════════════════════════════

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATASET="${1:-ogbn-products}"
EXPERIMENT="${2:-gatv2_products_v1}"
shift 2 2>/dev/null || true
COUNTS=("$@")
[ ${#COUNTS[@]} -eq 0 ] && COUNTS=(32 16 8)   # executors, x4 cores = slots
CORES=4
OMP=2
EPOCHS=10

LARGE_TMP=/mnt/tmp
export TMPDIR="$LARGE_TMP" TEMP="$LARGE_TMP" TMP="$LARGE_TMP"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"
export PYSPARK_PYTHON="$(which python3)"
export PYSPARK_DRIVER_PYTHON="$(which python3)"
PY_V=$(python3 -c "import sys; print('python%d.%d' % (sys.version_info.major, sys.version_info.minor))")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_V/site-packages:${PYTHONPATH:-}"

NODES=$(yarn node -list 2>/dev/null | grep -c "RUNNING")
OUT="$LARGE_TMP/nodesweep_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "=============================================================="
echo "  NODE SCALING SWEEP"
echo "  dataset      : $DATASET"
echo "  nodes live   : $NODES  (=> $((NODES * 16)) physical cores)"
echo "  executors    : ${COUNTS[*]}  x $CORES cores, OMP=$OMP"
echo "  epochs       : $EPOCHS"
echo "  commit       : $(git rev-parse --short HEAD)"
echo "  outdir       : $OUT"
echo "=============================================================="
echo "  4-node references for products Phase 3:"
echo "     32 slots 381.4s | 64 slots 224.8s (OMP=2) | 128 slots 231.4s (OMP=1)"
echo "=============================================================="

for N in "${COUNTS[@]}"; do
    SLOTS=$((N * CORES))
    LOG="$OUT/exec_${N}.log"
    echo ""
    echo "--- $N execs x $CORES cores = $SLOTS slots, OMP=$OMP ---"
    echo "    $SLOTS tasks over $((NODES * 16)) physical cores"
    START=$(date +%s)
    python3 -u runners/run_emr.py \
        --datasets "$DATASET" \
        --experiment-name "$EXPERIMENT" \
        --no-phase0 --no-phase1 --no-phase2 \
        --executor-instances "$N" \
        --executor-cores "$CORES" \
        --omp-threads "$OMP" \
        --num-epochs "$EPOCHS" > "$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    C=$(tr '\r' '\n' < "$LOG" | grep -v "^\[Stage")
    P3=$(echo "$C" | grep -m1 "Wall time:" | grep -oE "[0-9.]+" | head -1)
    P3B=$(echo "$C" | grep "Wall time:" | sed -n 2p | grep -oE "[0-9.]+" | head -1)
    U=$(echo "$C" | grep -m1 "bounded training units" | grep -oE "> [0-9,]+ bounded" | grep -oE "[0-9,]+")
    echo "    rc=$RC wall=$((END-START))s  phase3=${P3:-FAILED}s  phase3b=${P3B:-—}s  units=${U:-—}"
    echo "$N $SLOTS $((END-START)) ${P3:-0} ${P3B:-0} ${U:-0} $RC" >> "$OUT/results.txt"
    if [ $RC -ne 0 ]; then
        echo "    !! failed; last lines:"
        echo "$C" | tail -6 | sed 's/^/       /'
    fi
done

echo ""
echo "=============================================================="
echo "  SUMMARY  ($NODES nodes, $((NODES * 16)) physical cores)"
echo "=============================================================="
printf "  %-7s %-7s %-9s %-11s %-11s %-7s %s\n" execs slots wall phase3 phase3b units rc
while read -r N S W P3 P3B U RC; do
    printf "  %-7s %-7s %-9s %-11s %-11s %-7s %s\n" "$N" "$S" "$W" "$P3" "$P3B" "$U" "$RC"
done < "$OUT/results.txt"

echo ""
echo "  speedup on Phase 3, relative to the smallest slot count:"
BASE_P3=$(sort -n "$OUT/results.txt" | head -1 | awk '{print $4}')
BASE_S=$(sort -n "$OUT/results.txt" | head -1 | awk '{print $2}')
while read -r N S W P3 P3B U RC; do
    [ "$RC" = "0" ] || continue
    awk -v s="$S" -v p="$P3" -v bp="$BASE_P3" -v bs="$BASE_S" \
        'BEGIN{ if (p+0>0 && bp+0>0) printf "    %-5s slots: %.2fx vs %s slots (ideal %.2fx)\n", s, bp/p, bs, s/bs }'
done < "$OUT/results.txt"

echo ""
echo "  A near-ideal curve here is the scalability result: products can absorb"
echo "  ~306 slots, so below that the cluster should be the only limit."
echo "  logs: $OUT"
echo "=============================================================="
