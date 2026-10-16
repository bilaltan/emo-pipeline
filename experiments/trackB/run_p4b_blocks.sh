#!/usr/bin/env bash
# Track B P4b: Phase 3b's block work items (units above the cap split by id & (n-1)) read through the S3 emulator,
# from the store's Phase 2 tables copied at the row group rg (rg_resize.py), in Phase 2's row order and block-ordered.
#   bash experiments/trackB/run_p4b_blocks.sh <dataset> <alg> <rg_mb> [cap=10000] [latency_ms=20]
set -e
ds=$1; alg=$2; rg=$3; cap=${4:-10000}; lat=${5:-20}
cd "$(dirname "$0")/../.."
src=$HOME/lakegrl/store/$ds
tmp=$HOME/lakegrl/p4b/$ds-$alg
for order in plain blocked; do
  rm -rf $tmp
  SECONDS=0
  bash scripts/wsl_run.sh python experiments/trackB/rg_resize.py $src/p2_nodes/$alg $src/p2_edges/$alg $tmp --rg-mb $rg \
    $([ $order = blocked ] && echo --block-cap $cap)
  echo "copy ($order) took $SECONDS s"
  bash scripts/wsl_run.sh python experiments/trackB/s3_emu_bench.py --table-dir $tmp --label "$ds/$alg/rg$rg/$order" \
    --latency-ms $lat --modes blocks_pruned --cap $cap --out results/trackB/p4b_blocks.jsonl
done
rm -rf $tmp
