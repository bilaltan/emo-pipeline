#!/usr/bin/env python3
"""
Table B check (advisor 2026-11-04): the macro-graph Phase 2 now stores (utils.macro_graph) is the one Stage 3b built
at the start of every run before (phase3b_caan._build_macro, from p2_nodes and the raw node, mask and edge tables):
the same super-nodes with the same features (up to float32 summation order), the same minor nodes with the same
features, labels and splits, the same edges. Run on the store a Stage 3b run left (its Phase 2 wrote the tables).

    bash scripts/wsl_run.sh python experiments/tableB/check_macro.py --dataset ogbn-arxiv --tag leiden_c10000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from delta import configure_spark_with_delta_pip  # noqa: E402
from pyspark.sql import SparkSession, functions as F  # noqa: E402

from phases.phase3b_caan import _build_macro  # noqa: E402
from pipeline.utils import macro_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True, help="leiden_c10000 (row 5) or leiden (row 4)")
    ap.add_argument("--store", default=os.environ.get("LAKEGRL_STORE", os.path.expanduser("~/lakegrl/store")))
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--tiny-comm", default="drop")
    ap.add_argument("--out", default="results/tableB/v2/check_macro.jsonl")
    args = ap.parse_args()
    b = (SparkSession.builder.appName("check-macro").master("local[8]")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.driver.memory", "16g").config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.parquet.columnarReaderBatchSize", "512")
         .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500"))
    spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext
    base = f"file://{args.store}/{args.dataset}"
    p_alg = {t: f"{base}/{t}/" for t in ("nodes", "edges", "masks")}
    p_alg.update({t: f"{base}/{t}/{args.tag}/"
                  for t in ("communities", "p2_nodes", "p2_edges", *macro_graph.TABLES)})
    assert macro_graph.stored(spark, p_alg), "no macro-graph stored with the current p2_nodes"

    t0 = time.time()
    sup, minor, edges = macro_graph.load_macro(spark, p_alg)
    t_load = time.time() - t0

    t0 = time.time()
    comms_df = spark.read.format("delta").load(p_alg["communities"])
    major = {int(r["community_id"]) for r in
             comms_df.groupBy("community_id").count().filter(F.col("count") >= args.min_size).collect()}
    n2c = comms_df.toPandas()
    node_to_comm = {int(k): int(v) for k, v in zip(n2c["id"], n2c["community_id"])}
    sup0, minor0, edges0 = _build_macro(
        spark, spark.read.format("delta").load(p_alg["p2_nodes"]).drop("storage_part"), comms_df,
        spark.read.format("delta").load(p_alg["edges"]), p_alg, major, sc.broadcast(node_to_comm),
        sc.broadcast(major), args.tiny_comm)
    t_build = time.time() - t0

    assert set(sup) == set(sup0) == major, (len(sup), len(sup0), len(major))
    a = np.array([sup[c] for c in sorted(sup)], np.float32)
    a0 = np.array([sup0[c] for c in sorted(sup0)], np.float32)
    d = float(np.abs(a - a0).max()) if len(a) else 0.0
    exact = bool(np.array_equal(a, a0))   # both builds average in id order (utils.macro_graph.mean_features)
    assert np.allclose(a, a0, rtol=1e-4, atol=1e-5), d
    assert minor == minor0, (len(minor), len(minor0))
    assert len(edges) == len(set(edges)) and set(edges) == set(edges0), (len(edges), len(set(edges0)))
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "tag": args.tag, "ok": True,
           "super_nodes": len(sup), "minor_nodes": len(minor), "edges": len(edges), "super_feat_max_abs_diff": d, "super_feat_exact": exact,
           "load_s": t_load, "build_s": t_build}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
