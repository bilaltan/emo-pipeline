#!/usr/bin/env bash
# Track B P7: write one store's Phase 2 tables in several layouts (kept), then read each through the S3 emulator.
#   bash experiments/trackB/run_s3_emu.sh <dataset> <alg> <rg_mb (sqrt-rule pick)> [latency_ms ...]
# Layouts: range (Phase 2 default, 128 MB row groups), range_rg at the sqrt-rule row group, hive, hive_merged_rg.
set -e
ds=$1; alg=$2; rg=$3; shift 3
lats=${*:-20}
cd "$(dirname "$0")/../.."
bench=$HOME/lakegrl/emubench
bash scripts/wsl_run.sh python experiments/partition_layout_bench.py --data-root $HOME/lakegrl/store --dataset $ds \
  --alg $alg --tiny misc --layouts range range_rg hive hive_merged_rg --rg-mb $rg --bench-dir $bench --cores 4 \
  --keep-tables --skip-direct --repeats 1 --out results/trackB/emu_layouts.jsonl
for lay in range range_rg hive hive_merged_rg; do
  for lat in $lats; do
    bash scripts/wsl_run.sh python experiments/trackB/s3_emu_bench.py --table-dir $bench/$ds/misc/$lay \
      --label "$ds/$alg/$lay@$rg" --latency-ms $lat --out results/trackB/s3_emu.jsonl
  done
done
rm -rf $bench/$ds/misc
