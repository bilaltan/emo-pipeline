#!/usr/bin/env bash
# Track B P4b in Phase 2's Spark write path: write cost of block order (p4b_spark_write.py), then Phase 3b's block
# work items read from both Spark-written tables through the S3 emulator.
#   bash experiments/trackB/run_p4b_spark.sh <dataset> <rg_mb> [cap=10000]    (ORDERS="plain blocked blocked_min")
# READ_CAPS="2500 10000 40000" reads each table with those Phase 3b caps (blocks written for <cap>).
set -e
ds=$1; rg=$2; cap=${3:-10000}
cd "$(dirname "$0")/../.."
bash scripts/wsl_run.sh python experiments/trackB/p4b_spark_write.py --dataset $ds --rg-mb $rg --cap $cap --repeats 3 --orders ${ORDERS:-plain blocked}
for order in ${ORDERS:-plain blocked}; do
  for rc in ${READ_CAPS:-$cap}; do
    bash scripts/wsl_run.sh python experiments/trackB/s3_emu_bench.py --table-dir $HOME/lakegrl/p4b_spark/$ds-louvain/$order \
      --label "$ds/louvain/spark-rg$rg/$order" --modes blocks_pruned --cap $rc --write-cap $cap --out results/trackB/p4b_blocks.jsonl
  done
done
rm -rf $HOME/lakegrl/p4b_spark/$ds-louvain
