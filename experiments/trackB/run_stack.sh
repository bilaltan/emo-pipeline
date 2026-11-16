#!/usr/bin/env bash
# Track B stack: Phase 3b's block work items (cap 10,000) through the S3 emulator, adding one storage change at a time
# to Phase 2's default (128 MB row groups, Phase 2 row order, isin reader): pruned reader (unit_dataset), sqrt row
# group, block order. 128 MB + block order is also run, to see whether block order needs the small row group.
#   bash experiments/trackB/run_stack.sh <dataset> <alg> <sqrt_rg_mb> [cap=10000]
set -e
ds=$1; alg=$2; rg=$3; cap=${4:-10000}
cd "$(dirname "$0")/../.."
src=$HOME/lakegrl/store/$ds
tmp=$HOME/lakegrl/stack/$ds-$alg
for r in 128 $rg; do
  for order in plain blocked; do
    rm -rf $tmp
    bash scripts/wsl_run.sh python experiments/trackB/rg_resize.py $src/p2_nodes/$alg $src/p2_edges/$alg $tmp --rg-mb $r \
      $([ $order = blocked ] && echo --block-cap $cap)
    modes=blocks_pruned; [ $r = 128 ] && [ $order = plain ] && modes="blocks blocks_pruned"
    bash scripts/wsl_run.sh python experiments/trackB/s3_emu_bench.py --table-dir $tmp --label "$ds/$alg/stack-rg$r/$order" \
      --modes $modes --cap $cap --passes 1 --out results/trackB/stack.jsonl
  done
done
rm -rf $tmp
