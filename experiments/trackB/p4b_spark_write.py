#!/usr/bin/env python3
"""
P4b in Phase 2's own write path: what the block-ordered write costs in Spark, and the tables it gives.

The store's Phase 2 tables are rewritten as utils.partition_layout.write_layout writes the range layout
(repartitionByRange(n_bins, community_id) + sortWithinPartitions), alternately
  plain    sort by community_id (Phase 2 as it is)
  blocked  _blk added (a node's Phase 3b block id & (n-1); an edge's block if both endpoints share it, else n; n from
           the community's node rows and --cap, as run_phase3b computes it) and sort by (community_id, _blk, id | src, dst)
  blocked_min  the same _blk, sorted by (community_id, _blk) only: enough for block skipping, a cheaper sort
  blocked_hier as blocked_min, but a cross-block edge gets the bucket of the coarsest level where its endpoints still
           share a block: j = low bits src and dst share (< log2 n), _blk = n + 2^j - 1 + (src & (2^j - 1)). A read
           with a coarser cap (2^j blocks or fewer) then finds its cross edges in a few buckets instead of all of them
  blocked_tree the same buckets as blocked_hier, numbered in preorder of the binary trie on id's low bits (lowest bit
           first): _blk = sum over the bucket's d-th bit of (1 if 0 else 2^(k-d+1)), k = log2 n. Every block of a
           coarser or equal cap is then one contiguous _blk range (s3_emu_bench --tree-blk)
  zorder   baseline: plain, then Delta OPTIMIZE ZORDER BY (community_id, id | src, dst), the standard clustering knob
  liquid   baseline: a Delta table created with CLUSTER BY (community_id, id | src, dst), the plain write appended, then
           OPTIMIZE (liquid clustering, Delta's replacement for partitioning and Z-order)
with the row group from --rg-mb. The blocked time includes counting the community sizes. The last copy of each is kept
in --out-dir/<order>/p2_{nodes,edges} for s3_emu_bench.

    bash scripts/wsl_run.sh python experiments/trackB/p4b_spark_write.py --dataset reddit-lp --rg-mb 4
"""
import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.partition_layout_bench import build_spark  # noqa: E402
from experiments.trackB.s3_emu_bench import delta_files, n_blocks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=os.path.expanduser("~/lakegrl/store"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alg", default="louvain")
    ap.add_argument("--rg-mb", type=float, required=True)
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--orders", nargs="+", default=["plain", "blocked"], choices=["plain", "blocked", "blocked_min", "blocked_hier", "blocked_tree", "zorder", "liquid"])
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--mem", default="10g")
    ap.add_argument("--spark-tmp", default=None)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/lakegrl/p4b_spark"))
    ap.add_argument("--out", default="results/trackB/p4b_spark_write.jsonl")
    args = ap.parse_args()
    from pyspark.sql import functions as F
    spark = build_spark(args)
    spark.sparkContext._jsc.hadoopConfiguration().set("parquet.block.size", str(int(args.rg_mb * (1 << 20))))
    src = {k: os.path.join(args.store, args.dataset, f"p2_{k}", args.alg) for k in ("nodes", "edges")}
    wd = os.path.join(args.out_dir, f"{args.dataset}-{args.alg}")
    shutil.rmtree(wd, ignore_errors=True)

    def write(order):
        t0 = time.perf_counter()
        nodes, edges = (spark.read.format("delta").load(src[k]) for k in ("nodes", "edges"))
        n_comm = None
        if order.startswith("blocked"):
            cnt = nodes.groupBy("community_id").count().toPandas()
            cnt["_n"] = [n_blocks(int(k), args.cap) for k in cnt["count"]]
            n_comm = len(cnt)
            nb = F.broadcast(spark.createDataFrame(cnt[["community_id", "_n"]]))
            m = F.col("_n") - 1
            nodes = nodes.join(nb, "community_id").withColumn("_blk", F.col("id").bitwiseAND(m))
            bs, bd = F.col("src").bitwiseAND(m), F.col("dst").bitwiseAND(m)
            edges = edges.join(nb, "community_id").withColumn("_blk", F.when(bs == bd, bs).otherwise(m + 1))
            levels = range(1, int(cnt["_n"].max()).bit_length())
            if order in ("blocked_hier", "blocked_tree"):
                j = " + ".join(f"CASE WHEN ((src ^ dst) & {(1 << s) - 1}) = 0 AND {1 << s} <= _n THEN 1 ELSE 0 END"
                               for s in levels)   # low bits src and dst share, up to log2 n
                edges = edges.withColumn("_j", F.expr(j or "0")).withColumn("_blk", F.expr(
                    "CASE WHEN shiftleft(1, _j) = _n THEN _blk ELSE _n + shiftleft(1, _j) - 1 + (src & (shiftleft(1, _j) - 1)) END"))
            if order == "blocked_tree":   # preorder index of the trie node at depth _j (nodes: leaf, depth k) on key's low bits
                k = " + ".join(f"CASE WHEN {1 << s} <= _n THEN 1 ELSE 0 END" for s in levels) or "0"
                pre = lambda key, depth: " + ".join(
                    f"CASE WHEN {d} <= {depth} THEN (CASE WHEN (shiftright({key}, {d - 1}) & 1) = 0 THEN 1 "
                    f"ELSE shiftleft(1, _k - {d} + 1) END) ELSE 0 END" for d in levels) or "0"
                nodes = nodes.withColumn("_k", F.expr(k)).withColumn("_blk", F.expr(pre("id", "_k")))
                edges = edges.withColumn("_k", F.expr(k)).withColumn("_blk", F.expr(pre("src", "_j")))
            nodes, edges = nodes.drop("_n", "_k"), edges.drop("_n", "_j", "_k")
        n_bins = min(200, n_comm or nodes.select("community_id").distinct().count())   # as Phase 2: min(200, units)
        for name, df, keys in (("nodes", nodes, ["id"]), ("edges", edges, ["src", "dst"])):
            sort = ["community_id"] + (["_blk"] + keys if order == "blocked" else ["_blk"] if order.startswith("blocked") else [])
            out = os.path.join(wd, order, f"p2_{name}")
            if order == "liquid":
                from delta.tables import DeltaTable
                shutil.rmtree(out, ignore_errors=True)
                DeltaTable.create(spark).location(out).addColumns(df.schema).clusterBy("community_id", *keys).execute()
            (df.repartitionByRange(n_bins, "community_id").sortWithinPartitions(*sort)
             .write.format("delta").mode("append" if order == "liquid" else "overwrite")
             .option("overwriteSchema", "true").save(out))
            if order == "zorder":
                from delta.tables import DeltaTable
                DeltaTable.forPath(spark, out).optimize().executeZOrderBy("community_id", *keys)
            if order == "liquid":
                spark.sql(f"OPTIMIZE delta.`{out}`")
        return time.perf_counter() - t0, n_bins

    walls = {o: [] for o in args.orders}
    for _ in range(args.repeats):
        for order in walls:
            w, n_bins = write(order)
            walls[order].append(w)
            print(f"{order}: wrote in {w:.1f} s ({n_bins} range partitions)", flush=True)
    for order, w in walls.items():
        files = [f for k in ("nodes", "edges") for f in delta_files(os.path.join(wd, order, f"p2_{k}"))]
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "alg": args.alg, "order": order,
               "rg_mb": args.rg_mb, "cap": args.cap, "write_s": w, "table_dir": os.path.join(wd, order),
               "files": len(files), "bytes": sum(os.path.getsize(f) for f in files)}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    spark.stop()


if __name__ == "__main__":
    main()
