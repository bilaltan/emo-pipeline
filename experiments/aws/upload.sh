#!/usr/bin/env bash
# AWS validation: put the verified local ogbn-products tables and the code on S3 BEFORE the
# cluster exists, so no EMR time goes to uploads. No Phase 1-2 runs on AWS. Run in WSL (the tables are in ~/lakegrl),
# with AWS CLI v2 and the Windows credentials:
#   bash experiments/aws/upload.sh tables    # ~16 GB, ~62,000 Parquet files
#   bash experiments/aws/upload.sh code      # after the commit: git archive + Table B work files
# Hadoop's hidden .crc sidecars (one per Parquet file, 58,373 in row 3 alone) stay local; _delta_log goes up whole.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=s3://${BUCKET:-us-east-1-s3-gnn}/${PREFIX:-LakeGRL}
L=$HOME/lakegrl
W=${WINHOME:-$HOME}   # WINHOME: a Windows home seen from WSL (/mnt/c/Users/<you>), to use its AWS credentials
export PATH=$HOME/.local/bin:$PATH   # AWS CLI v2, installed per user
export AWS_SHARED_CREDENTIALS_FILE=${AWS_SHARED_CREDENTIALS_FILE:-$W/.aws/credentials}
export AWS_CONFIG_FILE=${AWS_CONFIG_FILE:-$W/.aws/config}
up() { aws s3 sync --only-show-errors --exclude "*.crc" --include "*_delta_log/0*.crc" "$1" "$2"; echo "  $2"; }

case "${1:-}" in
tables)
    aws configure set default.s3.max_concurrent_requests 100   # 58k small PUTs in parallel
    A=tableA/rleiden_c10000_v2/ogbn-products
    aws s3 cp --only-show-errors "$L/$A/manifest.json" "$P/$A/manifest.json"
    for l in merged_blk mv join liquid zorder hive; do up "$L/$A/$l" "$P/$A/$l"; done
    for t in nodes edges masks; do up "$L/store/ogbn-products/$t" "$P/store/ogbn-products/$t"; done
    for t in p2_nodes p2_edges p2_macro_nodes p2_macro_edges communities p2_unit_plan; do
        for g in leiden_c10000 leiden; do up "$L/store/ogbn-products/$t/$g" "$P/store/ogbn-products/$t/$g"; done
    done
    ;;
code)
    [ -z "$(git -c core.autocrlf=true status --porcelain --untracked-files=no)" ] || { echo "commit first"; exit 1; }
    T=$(mktemp -d)
    git archive --format=tar -o "$T/lakegrl-code.tar" HEAD
    git rev-parse --short HEAD > "$T/GIT"
    tar -rf "$T/lakegrl-code.tar" results/tableB/work/ogbn-products_row4.json results/tableB/work/ogbn-products_row5.json \
        results/aws/counts_local.jsonl
    tar -rf "$T/lakegrl-code.tar" -C "$T" GIT
    gzip "$T/lakegrl-code.tar"
    aws s3 cp --only-show-errors "$T/lakegrl-code.tar.gz" "$P/code/lakegrl-code.tar.gz"
    aws s3 cp --only-show-errors experiments/aws/bootstrap.sh "$P/code/bootstrap.sh"
    echo "  $P/code/ ($(cat "$T/GIT"))"; rm -rf "$T"
    ;;
*)
    sed -n 2,8p "$0"; exit 1 ;;
esac
