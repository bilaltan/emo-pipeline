#!/usr/bin/env python3
"""
Stored size of the Phase 2 views of a Table B store, per C_max tag (C_max sweep, advisor 2026-11-19): Table A's
storage measurement (experiments/partition_layout_bench.storage_stats, the current files of each Delta table) of
MV_unit = p2_nodes + p2_edges and MV_macro = p2_macro_nodes + p2_macro_edges. Run it after run.sh's vacuum.

    bash scripts/wsl_run.sh python experiments/tableB/store_size.py --dataset ogbn-products --tags leiden_c5000 \
        --out results/cmax_sweep/storage.jsonl
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "trackA"))
from experiments.partition_layout_bench import build_spark, storage_stats  # noqa: E402
from harness import git_hash  # noqa: E402
from pipeline.utils.common import spark_resources  # noqa: E402

VIEWS = {"mv_unit": ("p2_nodes", "p2_edges"), "mv_macro": ("p2_macro_nodes", "p2_macro_edges")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tags", nargs="+", required=True, help="e.g. leiden_c5000 leiden_c10000")
    ap.add_argument("--store", default=os.environ.get("LAKEGRL_STORE", os.path.expanduser("~/lakegrl/store")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--mem", default="4g")
    ap.add_argument("--spark-tmp", default=None)
    args = ap.parse_args()
    spark = build_spark(args)
    for tag in args.tags:
        st = {t: storage_stats(spark, f"{args.store}/{args.dataset}/{t}/{tag}") for v in VIEWS.values() for t in v}
        rec = {"dataset": args.dataset, "tag": tag, "store": args.store, "storage": st,
               **{f"{k}_bytes": sum(st[t]["bytes"] for t in v) for k, v in VIEWS.items()},
               "git": git_hash(), **spark_resources(spark.sparkContext),
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps({k: rec[k] for k in ("dataset", "tag", "mv_unit_bytes", "mv_macro_bytes")}))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
