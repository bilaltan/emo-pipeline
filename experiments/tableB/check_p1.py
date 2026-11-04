#!/usr/bin/env python3
"""
Table B check (advisor 2026-11-04, the recursive split in a process pool): Phase 1's communities table at its current
version holds the same (id, community_id) rows as at an earlier version (default: the one before), e.g. a run with
the sequential split.

    bash scripts/wsl_run.sh python experiments/tableB/check_p1.py --dataset ogbn-products --tag leiden_c10000
"""
import argparse
import json
import os
import time

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="leiden_c10000")
    ap.add_argument("--store", default=os.environ.get("LAKEGRL_STORE", os.path.expanduser("~/lakegrl/store")))
    ap.add_argument("--version", type=int, default=None, help="earlier version (default: current - 1)")
    ap.add_argument("--current", type=int, default=None, help="default: the latest version")
    ap.add_argument("--out", default="results/tableB/v2/check_p1.jsonl")
    args = ap.parse_args()
    b = (SparkSession.builder.appName("check-p1").master("local[8]")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.driver.memory", "8g").config("spark.sql.shuffle.partitions", "16"))
    spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    from delta.tables import DeltaTable
    path = f"file://{args.store}/{args.dataset}/communities/{args.tag}/"
    hist = DeltaTable.forPath(spark, path).history().select("version", "timestamp").orderBy("version").collect()
    cur = args.current if args.current is not None else hist[-1]["version"]
    old = args.version if args.version is not None else cur - 1
    ts = {r["version"]: str(r["timestamp"]) for r in hist}
    a = spark.read.format("delta").option("versionAsOf", cur).load(path).select("id", "community_id")
    o = spark.read.format("delta").option("versionAsOf", old).load(path).select("id", "community_id")
    n_a, n_o = a.count(), o.count()
    only_a, only_o = a.exceptAll(o).count(), o.exceptAll(a).count()
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "tag": args.tag,
           "current": cur, "current_ts": ts.get(cur), "earlier": old, "earlier_ts": ts.get(old),
           "rows": n_a, "rows_earlier": n_o, "only_current": only_a, "only_earlier": only_o,
           "identical": n_a == n_o and only_a == 0 and only_o == 0}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    assert rec["identical"], rec


if __name__ == "__main__":
    main()
