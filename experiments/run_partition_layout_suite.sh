#!/usr/bin/env bash
# Materialized partition table: range vs partitionBy(community_id) vs merged small communities.
# Round 2: fixed 'misc' tiny handling, 1 MB row-group variants, loader grouped per unit / Phase 3b bin /
# storage partition. Run from the repo root. ~1 h on a laptop; peak extra disk ~6 GB (products).
#   JAVA_HOME=... LAKEGRL_SPARK_JARS=... bash experiments/run_partition_layout_suite.sh
set -eo pipefail
PY=${PY:-.venv-spark/bin/python}
B=experiments/partition_layout_bench.py
D=local_data/joinbench/delta-data
OUT=results/partition_layout
mkdir -p "$OUT"
LOG=$OUT/suite_$(date +%Y%m%d-%H%M).log
run() {
  echo -e "\n######## $*" | tee -a "$LOG"
  "$PY" "$B" "$@" 2>&1 | grep --line-buffered -vE "WARN|FutureWarning|^\s*$" | tee -a "$LOG"
}
"$PY" scripts/test_partition_layout.py 2>&1 | grep -E "PASS|FAIL|SKIP|ALL" | tee -a "$LOG"

run --data-root $D --dataset ogbn-arxiv    --tiny keep misc \
    --layouts range range_rg hive hive_merged hive_merged_rg hive_naive \
    --bench-dir local_data/joinbench/layoutbench --out $OUT/partition_layout_r2.jsonl
run --data-root $D --dataset ogbn-products --tiny keep misc \
    --layouts range range_rg hive hive_merged hive_merged_rg \
    --bench-dir local_data/joinbench/layoutbench --out $OUT/partition_layout_r2.jsonl
echo "suite done: $LOG" | tee -a "$LOG"
