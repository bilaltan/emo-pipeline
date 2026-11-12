#!/usr/bin/env python3
"""
AWS validation (advisor 2026-11-11): are the uploaded Delta tables the local ones, and does Delta read them on S3?

--local DIR (WSL, after the upload, no cluster): for every table, the files of its current snapshot (replayed from the
  local _delta_log) exist on S3 under --root with the same sizes, and _delta_log has the same commit files. S3 LIST
  only, no data read.
--spark (EMR primary, smoke test; or locally on --root ~/lakegrl for the reference): spark.read.format("delta")
  .load(table).count() for every table, one JSON line each; --expect compares with the local reference.

    bash scripts/wsl_run.sh python experiments/aws/check_upload.py --root ~/lakegrl --spark local[8] \\
        > results/aws/counts_local.jsonl
    bash scripts/wsl_run.sh python experiments/aws/check_upload.py --root s3://us-east-1-s3-gnn/LakeGRL --local ~/lakegrl
    python3.11 experiments/aws/check_upload.py --root s3://us-east-1-s3-gnn/LakeGRL --spark yarn \\
        --expect results/aws/counts_local.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

A = "tableA/rleiden_c10000_v2/ogbn-products"
S = "store/ogbn-products"
TABLES = ([f"{A}/join/p2_map", f"{A}/join/p2_edges"]
          + [f"{A}/{lay}/{t}" for lay in ("mv", "hive", "liquid", "zorder", "merged_blk") for t in ("p2_nodes", "p2_edges")]
          + [f"{S}/{t}" for t in ("nodes", "edges", "masks")]
          + [f"{S}/{t}/{g}" for t in ("p2_nodes", "p2_edges", "p2_macro_nodes", "p2_macro_edges", "communities",
                                      "p2_unit_plan") for g in ("leiden_c10000", "leiden")])


def listing(root):
    """{path under root: size} of every object (S3) or file (local) below root, Hadoop .crc sidecars left out."""
    import pyarrow.fs as pafs
    from pipeline.utils.store_io import filesystem, to_path
    base = to_path(root)
    infos = filesystem(root).get_file_info(pafs.FileSelector(base, recursive=True, allow_not_found=True))
    return {i.path[len(base) + 1:]: i.size for i in infos
            if i.type == pafs.FileType.File and not i.base_name.startswith(".")}


def compare(root, local):
    from pipeline.utils.store_io import snapshot, to_path
    bad = 0
    for t in TABLES:
        loc = os.path.join(local, t)
        _, files, ver = snapshot(loc)
        live = {f[len(to_path(loc)) + 1:]: os.path.getsize(f) for f in files}
        logs = {k: v for k, v in listing(loc).items() if k.startswith("_delta_log/")}
        remote = listing(f"{root}/{t}")
        miss = [k for k, v in {**live, **logs}.items() if remote.get(k) != v]
        bad += bool(miss)
        print(json.dumps({"table": t, "version": ver, "files": len(live), "gb": round(sum(live.values()) / 1e9, 3),
                          "log_files": len(logs), "ok": not miss, "missing_or_size_differs": miss[:5]}), flush=True)
    return bad


def counts(root, master, expect):
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName("lakegrl-check").master(master)
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
    if master.startswith("local"):
        from delta import configure_spark_with_delta_pip
        b = configure_spark_with_delta_pip(b)
    spark = b.getOrCreate()
    ref = {r["table"]: r["rows"] for r in map(json.loads, open(expect))} if expect else {}
    bad = 0
    for t in TABLES:
        url = f"{root}/{t}" if "://" in root else f"file://{os.path.abspath(os.path.expanduser(root))}/{t}"
        n = spark.read.format("delta").load(url).count()
        ok = ref.get(t, n) == n
        bad += not ok
        print(json.dumps({"table": t, "rows": n, **({"expected": ref.get(t), "ok": ok} if ref else {})}), flush=True)
    spark.stop()
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="s3://us-east-1-s3-gnn/LakeGRL, or the local ~/lakegrl")
    ap.add_argument("--local", help="local root to compare the upload against")
    ap.add_argument("--spark", metavar="MASTER", help="count rows with Spark's Delta reader (yarn, local[N])")
    ap.add_argument("--expect", help="with --spark: the local reference counts (JSON lines)")
    args = ap.parse_args()
    bad = compare(args.root, os.path.expanduser(args.local)) if args.local else 0
    if args.spark:
        bad += counts(args.root, args.spark, args.expect)
    print(f"== {len(TABLES)} tables, {bad} with problems", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
