#!/bin/bash
# Can the idle hyperthreads be used by threads instead of by extra tasks?
#
# r6id.8xlarge: 32 vCPUs but 16 PHYSICAL cores. Measured products Phase 3:
#   32 slots (8 tasks/node, 0.5x)  381.4s
#   64 slots (16 tasks/node, 1.0x) 255.9s
#  128 slots (32 tasks/node, 2.0x) 231.4s   <- only 11% better than 64
#
# So the second hyperthread is nearly worthless when it runs a competing task.
# This asks whether it is worth more when it serves the SAME task's tensor ops:
# 64 slots x OMP=2 = 32 threads/node, same thread count as the 128-slot config.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
LARGE_TMP=/mnt/tmp
export TMPDIR="$LARGE_TMP" TEMP="$LARGE_TMP" TMP="$LARGE_TMP"
export PYTHONUSERBASE="$LARGE_TMP/.local" DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"
export PYSPARK_PYTHON="$(which python3)"; export PYSPARK_DRIVER_PYTHON="$(which python3)"
PY_V=$(python3 -c "import sys; print('python%d.%d' % sys.version_info[:2])")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_V/site-packages:${PYTHONPATH:-}"

OUT="$LARGE_TMP/omptest_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
echo "=============================================================="
echo "  HYPERTHREAD TEST  (16 phys cores/node, 4 nodes = 64 phys)"
echo "  baselines: 64 slots OMP=1 = 255.9s | 128 slots OMP=1 = 231.4s"
echo "  outdir: $OUT"
echo "=============================================================="

#  label : execs : cores : omp
for CFG in "C_64slots_omp2:16:4:2" "D_32slots_omp4:8:4:4"; do
    IFS=':' read -r L N C OMP <<< "$CFG"
    echo ""
    echo "--- $L : $N execs x $C cores = $((N*C)) slots, OMP=$OMP ---"
    echo "    $(( N/4 * C )) tasks/node x $OMP threads = $(( N/4 * C * OMP )) threads on 32 hyperthreads (16 cores)"
    S=$(date +%s)
    python3 -u runners/run_emr.py --datasets ogbn-products \
        --experiment-name gatv2_products_v1 \
        --no-phase0 --no-phase1 --no-phase2 --no-phase3b \
        --executor-instances "$N" --executor-cores "$C" \
        --omp-threads "$OMP" --num-epochs 10 > "$OUT/$L.log" 2>&1
    RC=$?; E=$(date +%s)
    P3=$(tr '\r' '\n' < "$OUT/$L.log" | grep -v "^\[Stage" | grep -m1 "Wall time:" | grep -oE "[0-9.]+" | head -1)
    echo "    rc=$RC wall=$((E-S))s  PHASE 3 = ${P3:-FAILED}s"
    echo "$L $N $C $OMP $((E-S)) ${P3:-0} $RC" >> "$OUT/results.txt"
    [ $RC -ne 0 ] && tr '\r' '\n' < "$OUT/$L.log" | grep -v "^\[Stage" | tail -5 | sed 's/^/       /'
done

echo ""
echo "=============================================================="
printf "  %-16s %-6s %-6s %-5s %-6s %-9s %s\n" label execs cores omp slots phase3 "vs 64slot/OMP1 (255.9s)"
while read -r L N C OMP W P RC; do
    printf "  %-16s %-6s %-6s %-5s %-6s %-9s " "$L" "$N" "$C" "$OMP" "$((N*C))" "$P"
    awk -v p="$P" 'BEGIN{ if (p+0>0) printf "%.2fx\n", 255.9/p; else print "-" }'
done < "$OUT/results.txt"
echo ""
echo "  >1.00x means threads beat extra tasks at using the hyperthreads."
echo "  logs: $OUT"
echo "=============================================================="
