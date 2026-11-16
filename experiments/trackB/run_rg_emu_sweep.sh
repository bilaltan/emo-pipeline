#!/usr/bin/env bash
# Track B P2 / P7: does the best row group move with storage latency, as the sqrt rule predicts
# (s* = sqrt(beta * table / (2 * units)), beta = per-row-group cost in bytes, about latency x bandwidth on S3)?
# Per row-group size the store's Phase 2 tables are copied (rg_resize.py) and read through the S3 emulator at each
# latency, as Phase 3b's bins with the pruned reader.
#   bash experiments/trackB/run_rg_emu_sweep.sh <dataset> <alg>      (RGS="..." LATS="..." override the grids)
set -e
ds=$1; alg=$2
cd "$(dirname "$0")/../.."
src=$HOME/lakegrl/store/$ds
tmp=$HOME/lakegrl/rgsweep/$ds-$alg
for rg in ${RGS:-0.125 0.25 0.5 1 2 4 16}; do
  rm -rf $tmp
  bash scripts/wsl_run.sh python experiments/trackB/rg_resize.py $src/p2_nodes/$alg $src/p2_edges/$alg $tmp --rg-mb $rg
  for lat in ${LATS:-5 20 80}; do
    bash scripts/wsl_run.sh python experiments/trackB/s3_emu_bench.py --table-dir $tmp --label "$ds/$alg/rg$rg" \
      --latency-ms $lat --modes p3b_bins_pruned --out results/trackB/rg_emu_sweep.jsonl
  done
done
rm -rf $tmp
