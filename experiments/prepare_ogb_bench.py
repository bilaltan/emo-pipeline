#!/usr/bin/env python3
"""
Build the inputs of experiments/global_join_vs_mv_bench.py for an OGB node
dataset, from the raw files already in data/ogb/<name>/.

  Phase 0 equivalent  raw/*.csv.gz -> Delta tables under <out-root>/<dataset>/
      nodes  (id long, label int, features array<float>), written in id order
      edges  (src long, dst long): both directions, no self-loops, no duplicates
      masks  (id long, split string): the official OGB split
  Phase 1             the pipeline's own run_phase1 with Louvain (igraph on the
      driver), minor communities absorbed into their best-connected major one;
      same settings as scripts/run_local_pipeline.py -> communities/louvain

Everything is read with Spark, so node features never pass through the driver.
Phase 1 collects the edge list (one direction) to the driver for igraph: about
1 GB for ogbn-products, plus igraph's own copy (a 16 GB machine is enough).

  .venv-spark/bin/python experiments/prepare_ogb_bench.py --dataset ogbn-products
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + (os.pathsep + os.environ["PYTHONPATH"]
                                   if os.environ.get("PYTHONPATH") else "")


def total_ram_gb():
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return None


def auto_mem(ram):
    if ram is None:
        return "4g"
    return "10g" if ram >= 30 else "6g" if ram >= 15 else "4g"


def build_spark(args):
    if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/opt/openjdk@17"):
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName(f"prepare-{args.dataset}")
         .master(f"local[{args.cores}]")
         .config("spark.driver.host", "127.0.0.1")
         .config("spark.driver.memory", args.mem)
         .config("spark.driver.maxResultSize", "0")   # Phase 1 collects the edge list
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.sql.shuffle.partitions", str(args.cores * 4))
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.parquet.columnarReaderBatchSize", "512")
         .config("spark.ui.showConsoleProgress", "false"))
    if args.spark_tmp:
        b = b.config("spark.local.dir", args.spark_tmp)
    jars = os.environ.get("LAKEGRL_SPARK_JARS")
    if jars:
        spark = b.config("spark.jars", jars).getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def exists(spark, path):
    try:
        spark.read.format("delta").load(path).limit(1).count()
        return True
    except Exception:
        return False


def ingest(spark, args, p):
    from pyspark.sql import functions as F
    raw = os.path.join(args.ogb_root, args.dataset.replace("-", "_"))
    split_root = os.path.join(raw, "split")
    split_dir = os.path.join(split_root, sorted(os.listdir(split_root))[0])
    t0 = time.time()

    # gzip is not splittable: each file is one partition, so monotonically_increasing_id
    # numbers the rows 0..n-1 in file order, which is the OGB node index.
    feat = spark.read.csv(os.path.abspath(f"{raw}/raw/node-feat.csv.gz")).coalesce(1)
    feat = feat.select(F.monotonically_increasing_id().alias("id"),
                       F.array(*[F.col(c).cast("float") for c in feat.columns]).alias("features"))
    lab = (spark.read.csv(os.path.abspath(f"{raw}/raw/node-label.csv.gz")).coalesce(1)
           .select(F.monotonically_increasing_id().alias("id"),
                   F.col("_c0").cast("double").cast("int").alias("label")))
    nodes = feat.join(lab, "id").select("id", "label", "features")
    (nodes.repartitionByRange(args.node_files, "id").sortWithinPartitions("id")
          .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(p["nodes"]))
    t_nodes = time.time() - t0
    print(f"  nodes written  [{t_nodes:.0f}s]")

    e = spark.read.csv(os.path.abspath(f"{raw}/raw/edge.csv.gz")).select(
        F.col("_c0").cast("long").alias("src"), F.col("_c1").cast("long").alias("dst"))
    sym = (e.union(e.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
            .filter("src != dst").dropDuplicates(["src", "dst"]))
    sym.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(p["edges"])
    t_edges = time.time() - t0 - t_nodes
    print(f"  edges written  [{t_edges:.0f}s]")

    parts = [spark.read.csv(os.path.abspath(f"{split_dir}/{sp}.csv.gz"))
                  .select(F.col("_c0").cast("long").alias("id"), F.lit(sp).alias("split"))
             for sp in ("train", "valid", "test")]
    masks = parts[0].unionByName(parts[1]).unionByName(parts[2])
    masks.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(p["masks"])

    n = spark.read.format("delta").load(p["nodes"])
    stats = {"nodes": n.count(), "node_files": len(n.inputFiles()),
             "feat_dim": len(n.select("features").first()[0]),
             "edges": spark.read.format("delta").load(p["edges"]).count(),
             "masks": spark.read.format("delta").load(p["masks"]).groupBy("split").count()
                           .toPandas().set_index("split")["count"].to_dict(),
             "ingest_s": round(time.time() - t0, 1), "split": os.path.basename(split_dir)}
    print(f"  {stats}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ogbn-products", choices=["ogbn-products", "ogbn-arxiv"])
    ap.add_argument("--ogb-root", default="data/ogb")
    ap.add_argument("--out-root", default="local_data/joinbench/delta-data")
    ap.add_argument("--alg", default="louvain", choices=["louvain", "leiden"])
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--node-files", type=int, default=16)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--mem", default="auto")
    ap.add_argument("--spark-tmp", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild tables that already exist")
    ap.add_argument("--log", default=None, help="JSON summary (default results/partition_join/prepare_<dataset>.json)")
    args = ap.parse_args()

    ram = total_ram_gb()
    if args.mem == "auto":
        args.mem = auto_mem(ram)
    print(f"== prepare {args.dataset}: local[{args.cores}], driver {args.mem}, "
          f"machine RAM {ram:.1f} GB" if ram else f"== prepare {args.dataset}")
    if ram is not None and ram < 15 and args.dataset == "ogbn-products":
        print("  ⚠ less than 16 GB RAM: Phase 1 (igraph Louvain on the driver) may swap heavily")

    base = os.path.abspath(os.path.join(args.out_root, args.dataset))

    def get_paths(dataset, alg=None):
        p = {"nodes": f"{base}/nodes", "edges": f"{base}/edges", "masks": f"{base}/masks",
             "checkpoints": f"{base}/_checkpoints/"}
        if alg:
            p.update({"communities": f"{base}/communities/{alg}", "tag": f"prep_{dataset}_{alg}"})
        return p

    spark = build_spark(args)
    p = get_paths(args.dataset)
    summary = {"dataset": args.dataset, "cores": args.cores, "driver_mem": args.mem,
               "ram_gb": round(ram, 1) if ram else None}
    if args.force or not all(exists(spark, p[k]) for k in ("nodes", "edges", "masks")):
        print("\n-- Phase 0 equivalent: raw OGB files -> Delta")
        summary["ingest"] = ingest(spark, args, p)
    else:
        print("\n-- Phase 0: tables exist, skipped (use --force to rebuild)")

    pc = get_paths(args.dataset, args.alg)
    if args.force or not exists(spark, pc["communities"]):
        print(f"\n-- Phase 1: {args.alg} (pipeline run_phase1)")
        from phases.phase1_community import run_phase1
        n = spark.read.format("delta").load(p["nodes"])
        n_classes = int(n.agg({"label": "max"}).first()[0]) + 1
        feat_dim = len(n.select("features").first()[0])
        timing, res = {}, {}
        run_phase1(spark, spark.sparkContext, [args.dataset], [args.alg], lpa_max_iter=20,
                   resolution=1.0, random_seed=42, min_size=args.min_size,
                   dataset_cfg={args.dataset: {"in_feats": feat_dim, "num_classes": n_classes}},
                   get_paths_fn=get_paths, timing=timing, results=res, force_rerun=True,
                   lpa_tol=0.001, merge_minor_communities=True)
        summary["phase1"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                             for k, v in res.get((args.dataset, args.alg), {}).items()}
    else:
        print(f"\n-- Phase 1: communities/{args.alg} exists, skipped (use --force to rebuild)")

    from pyspark.sql import functions as F
    c = spark.read.format("delta").load(pc["communities"])
    sizes = c.groupBy("community_id").count()
    summary["communities"] = {
        "total": sizes.count(),
        "major": sizes.filter(F.col("count") >= args.min_size).count(),
        "largest": [int(r["count"]) for r in sizes.orderBy(F.desc("count")).limit(5).collect()],
        "nodes_assigned": c.count()}
    print(f"  communities: {summary['communities']}")
    spark.stop()

    log = args.log or os.path.join("results", "partition_join", f"prepare_{args.dataset}.json")
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  summary -> {log}")


if __name__ == "__main__":
    main()
