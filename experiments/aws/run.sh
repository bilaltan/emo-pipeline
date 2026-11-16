#!/usr/bin/env bash
# AWS validation, Section 4.4 (advisor 2026-11-11), on the EMR primary after the bootstrap (experiments/aws/README.md):
#   bash experiments/aws/run.sh smoke|expA|expB-read|expB-train|save|nodes N
# smoke   : Spark's Delta reader counts every uploaded table (= local counts); one row 8b read with 16 Spark readers
# expA    : Table A rows 8b, 2 MV, 3 partitionBy, 1 join with 16 readers (4 per core node), 5 Liquid with 3 readers on
#           every 10th bin (as Table A), 3 cold runs each; coalesced schedule as Table A
# expB-*  : Table B rows 5 (LakeGRL full) and 4 (id mod n), W = 16, 8, 4, 2, then 1 (advisor: W = 1 last).
#           read = the run's bins read cold by W Spark readers, 3 runs; train = Stage 3b alone on the uploaded tables,
#           W executors x 1 core x 4 GB, 1 torch thread, EPOCHS (10) epochs, seed 0
# nodes N : node scaling (advisor 2026-11-13) on N core nodes with 4 workers each, W = 4N: rows 5 and 4, the bins read
#           cold by W Spark readers (3 runs), then Stage 3b with W executors, seeds 0-2. yarn_workers' 5.5 GB containers
#           fit at most 4 per m5.2xlarge, so W = 4N puts exactly 4 on every node. Output: results/aws/nodes/.
# Every read counts GETs and bytes in the reader (--emu count / --s3-emu count), as Table A. Output: results/aws/.
set -eo pipefail
cd "$(dirname "$0")/../.."
P=s3://${BUCKET:-us-east-1-s3-gnn}/${PREFIX:-LakeGRL}
export SPARK_HOME=/usr/lib/spark PYSPARK_PYTHON=/usr/bin/python3.11 PYSPARK_DRIVER_PYTHON=/usr/bin/python3.11
export PYTHONPATH=$PWD:$SPARK_HOME/python:$(ls $SPARK_HOME/python/lib/py4j-*-src.zip | head -1)   # EMR also has py4j-src.zip
export LAKEGRL_GIT=$(cat GIT 2>/dev/null) PYTHONUNBUFFERED=1 OUTDATED_IGNORE=1
O=results/aws; [ "${1:-}" = nodes ] && O=results/aws/nodes   # its own folder: the 11-12 files stay as they are
mkdir -p $O/logs
exec > >(tee -a "$O/logs/${1:-none}${2:-}.log") 2>&1
trap 'aws s3 sync --only-show-errors $O $P/$O' EXIT   # results and logs reach S3 even if a step fails
echo "== $(date -u +%FT%TZ) $1 (code $LAKEGRL_GIT)"
PY=/usr/bin/python3.11
MAP="--map /home/btan4/lakegrl=$P"
READ_A="$PY experiments/tableA/read_layouts.py --dataset ogbn-products --out-dir $P/tableA/rleiden_c10000_v2
        --coalesce-minor --plan-from merged_blk --emu count $MAP"
declare -A ROW=([5]="--unit-split leiden" [4]="--unit-split idmod")

case "${1:-}" in
smoke)
    yarn node -list -showDetails
    $PY experiments/aws/check_upload.py --root $P --spark yarn --expect $O/counts_local.jsonl
    $READ_A --layouts merged_blk --runs 1 --spark 16 --out $O/smoke_read.jsonl
    ;;
expA)
    $READ_A --layouts merged_blk mv hive join --runs 3 --spark 16 --out $O/expA_read.jsonl
    $READ_A --layouts liquid --runs 3 --spark 3 --bin-sample 10 --out $O/expA_read.jsonl
    ;;
expB-read)
    for W in ${WS:-16 8 4 2 1}; do for r in 5 4; do
        $PY experiments/tableB/read_units.py --work results/tableB/work/ogbn-products_row$r.json --row $r \
            --dataset ogbn-products --runs 3 --spark $W --emu count $MAP --out $O/expB_read.jsonl
    done; done
    ;;
expB-train)
    for W in ${WS:-16 8 4 2 1}; do for r in 5 4; do
        echo "== $(date -u +%FT%TZ) train row $r W=$W"
        $PY scripts/run_local_pipeline.py --dataset ogbn-products --alg leiden ${ROW[$r]} --tiny-comm drop \
            --c-max 10000 --p2-layout hive_merged --p2-rg-floor-mb 2 --p2-one-file --stage3-only --store $P/store \
            --executors $W --epochs ${EPOCHS:-10} --s3-emu count --seed 0 --driver-mem 10g \
            --json-out $O/expB_train.jsonl
    done; done
    ;;
nodes)
    N=${2:?core nodes}; W=$((4 * N))
    yarn node -list   # the log shows the N RUNNING core nodes
    for r in 5 4; do
        $PY experiments/tableB/read_units.py --work results/tableB/work/ogbn-products_row$r.json --row $r \
            --dataset ogbn-products --runs 3 --spark $W --emu count $MAP --out $O/nodes_read.jsonl
    done
    for s in 0 1 2; do for r in 5 4; do
        echo "== $(date -u +%FT%TZ) train row $r N=$N W=$W seed $s"
        $PY scripts/run_local_pipeline.py --dataset ogbn-products --alg leiden ${ROW[$r]} --tiny-comm drop \
            --c-max 10000 --p2-layout hive_merged --p2-rg-floor-mb 2 --p2-one-file --stage3-only --store $P/store \
            --executors $W --epochs ${EPOCHS:-10} --s3-emu count --seed $s --driver-mem 10g \
            --json-out $O/nodes_train.jsonl
    done; done
    ;;
save)
    ;;   # the EXIT trap syncs
*)
    sed -n 2,14p "$0"; exit 1 ;;
esac
echo "== $(date -u +%FT%TZ) $1 done"
