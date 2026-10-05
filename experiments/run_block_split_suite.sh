#!/usr/bin/env bash
# Block-split study suite (experiments/block_split_study.py). Run from the repo root.
# About 1.5 h on an 8-core laptop; every result is appended to results/block_split/summary.jsonl.
#   JAVA_HOME=... LAKEGRL_SPARK_JARS=... bash experiments/run_block_split_suite.sh
set -eo pipefail
PY=${PY:-.venv-spark/bin/python}
S=experiments/block_split_study.py
mkdir -p results/block_split
LOG=results/block_split/suite_$(date +%Y%m%d-%H%M).log
run() {
  echo -e "\n######## $*" | tee -a "$LOG"
  "$PY" "$S" "$@" 2>&1 | grep --line-buffered -vE "FutureWarning|torch\.jit|WARN|^\s*$" | tee -a "$LOG"
}

# 1. Main table: cap 10k (Phase 3b default), 3 seeds, every split community, plus I/O
run --dataset ogbn-arxiv    --caps 10000 --strategies id_pow2 id_equal louvain --seeds 0 1 2 --tag main
run --dataset ogbn-products --caps 10000 --strategies id_pow2 id_equal louvain --seeds 0 1 2 --epochs 60 --tag main

# 2. Memory budget sweep (1 seed)
run --dataset ogbn-arxiv    --caps 5000 20000 --strategies id_pow2 louvain --seeds 0 --steps quality accuracy --tag capsweep
run --dataset ogbn-products --caps 5000 20000 --strategies id_pow2 louvain --seeds 0 --epochs 60 \
    --train-communities 20 --steps quality accuracy --tag capsweep

# 3. Which algorithm splits the oversized communities (1 seed)
run --dataset ogbn-arxiv    --caps 10000 --strategies louvain leiden lpa --seeds 0 --steps quality accuracy --tag splitters
run --dataset ogbn-products --caps 10000 --strategies louvain leiden lpa --seeds 0 --epochs 60 \
    --train-communities 20 --steps quality accuracy --tag splitters

echo "suite done: $LOG" | tee -a "$LOG"
