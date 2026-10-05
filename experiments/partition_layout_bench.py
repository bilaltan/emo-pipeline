#!/usr/bin/env python3
"""
Physical layout of the materialized partition table (Phase 2's p2_nodes / p2_edges):

  range        current default: no partition columns, repartitionByRange + sort
               by community_id (Delta min/max statistics skip other communities)
  hive         .partitionBy("community_id"): one directory per community
  hive_merged  .partitionBy("storage_part"): large communities keep their own
               directory, minor/small ones are packed into shared directories
               (utils/partition_layout.py)
  range_rg, hive_merged_rg  (optional) the same with small parquet row groups
               (--rg-mb), so a predicate can skip inside a file shared by
               several communities
  hive_merged_size  (optional) as hive_merged, but also packs communities whose
               data is below --merge-min-mb, not only minor ones
  hive_naive   (optional) .partitionBy("community_id") on input that is not
               clustered by community (hash-partitioned by id, as rows come out
               of Phase 2's joins) and without repartitioning by the key: every
               task writes one file into every directory it touches

For each --tiny mode (how Phase 1's small communities enter the table: keep =
each its own unit, misc = the pipeline's default hash bins, drop), Phase 2 runs
once; the result is persisted and written in every layout from the same rows,
so layouts differ only in physical organisation. Measured per layout:

  storage    write time, partition directories, files, bytes, file-size
             distribution, small files, Delta log size and add actions
  metadata   cold snapshot load (log replay; DeltaLog cache cleared) and a
             pyarrow directory crawl of the table
  direct     Phase 3b's per-unit loader (pyarrow worker processes, one task per
             community, snapshot file list + partition pruning): wall time,
             files and row-group bytes the predicates select
  scan       Spark full scan of p2_nodes and p2_edges (Phase 3's path)
  point      Spark read of single communities (largest / median / smallest):
             wall time and files read

Per-community checksums (rows, id sum, edges) must match across layouts.

  .venv-spark/bin/python experiments/partition_layout_bench.py \\
      --data-root local_data/joinbench/delta-data --dataset ogbn-arxiv \\
      --bench-dir local_data/joinbench/layoutbench
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

LAYOUT_CHOICES = ("range", "range_rg", "hive", "hive_merged", "hive_merged_rg", "hive_merged_size",
                  "hive_naive")


# ------------------------------------------------------------------ spark
def total_ram_gb():
    import subprocess
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return None


def build_spark(args):
    if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/opt/openjdk@17"):
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName("partition-layout-bench").master(f"local[{args.cores}]")
         .config("spark.driver.host", "127.0.0.1")
         .config("spark.driver.memory", args.mem)
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.sql.shuffle.partitions", str(args.cores * 2))
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


def clear_delta_cache(spark):
    try:
        spark._jvm.org.apache.spark.sql.delta.DeltaLog.clearCache()
    except Exception:
        pass


def local(p):
    from urllib.parse import urlparse, unquote
    return unquote(urlparse(p).path) if p.startswith("file:") else p


# ------------------------------------------------------------------ storage stats
def storage_stats(spark, path):
    from utils.partition_layout import current_files, partition_columns
    files = [local(f) for f in current_files(spark, path)]
    sizes = np.array([os.path.getsize(f) for f in files], dtype=np.int64)
    cols = partition_columns(spark, path)
    dirs = len({os.path.dirname(f) for f in files}) if cols else 0
    logdir = os.path.join(local(path), "_delta_log")
    log_bytes, adds = 0, 0
    for fn in os.listdir(logdir):
        fp = os.path.join(logdir, fn)
        if os.path.isfile(fp):
            log_bytes += os.path.getsize(fp)
            if fn.endswith(".json"):
                with open(fp) as fh:
                    adds += sum(1 for line in fh if line.startswith('{"add"'))
    pct = lambda q: int(np.percentile(sizes, q)) if len(sizes) else 0  # noqa: E731
    return {"partition_columns": cols, "dirs": dirs, "files": int(len(sizes)), "bytes": int(sizes.sum()),
            "file_bytes_min": int(sizes.min()) if len(sizes) else 0, "file_bytes_p10": pct(10),
            "file_bytes_median": pct(50), "file_bytes_max": int(sizes.max()) if len(sizes) else 0,
            "files_lt_1mb": int((sizes < 1 << 20).sum()), "files_lt_16mb": int((sizes < 16 << 20).sum()),
            "delta_log_bytes": int(log_bytes), "delta_log_add_actions": int(adds)}


def rowgroup_index(spec):
    """Per row group: file, partition value, community_id min/max, compressed bytes."""
    import pyarrow.parquet as pq
    out = {"file": [], "part": [], "cmin": [], "cmax": [], "bytes": []}
    col = spec.get("part_col")
    for fi, f in enumerate(spec["files"]):
        lf = local(f)
        pv = None
        if col:
            seg = [s for s in lf.split(os.sep) if s.startswith(col + "=")]
            pv = seg[-1].split("=", 1)[1] if seg else None
        md = pq.ParquetFile(lf).metadata
        names = [md.schema.column(i).name for i in range(md.num_columns)]
        ci = names.index("community_id") if "community_id" in names else None
        for r in range(md.num_row_groups):
            g = md.row_group(r)
            if ci is not None and g.column(ci).statistics is not None and g.column(ci).statistics.has_min_max:
                lo, hi = int(g.column(ci).statistics.min), int(g.column(ci).statistics.max)
            else:                               # hive: community_id is the directory
                lo = hi = int(pv)
            out["file"].append(fi)
            out["part"].append(pv)
            out["cmin"].append(lo)
            out["cmax"].append(hi)
            out["bytes"].append(sum(g.column(k).total_compressed_size for k in range(g.num_columns)))
    return {k: np.asarray(v) if k != "part" else np.asarray(v, dtype=object) for k, v in out.items()}


def planned_reads(idx, groups, spec, comm_to_part):
    """Files, row groups and bytes selected when each group of communities is read
    with one predicate (pyarrow keeps a row group if any community in the group
    falls in its [min, max]), summed over the groups."""
    col = spec.get("part_col")
    files = rgs = byt = 0
    for g in groups:
        m = np.zeros(len(idx["bytes"]), dtype=bool)
        for c in g:
            mc = (idx["cmin"] <= c) & (idx["cmax"] >= c)
            if col == "storage_part":
                mc &= idx["part"] == comm_to_part[c]
            elif col == "community_id":
                mc &= idx["part"] == str(c)
            m |= mc
        files += len(np.unique(idx["file"][m]))
        rgs += int(m.sum())
        byt += int(idx["bytes"][m].sum())
    return files, rgs, byt


# ------------------------------------------------------------------ direct loads
_WCACHE = {}


def load_group(args):
    """Read one group of communities with one predicate (as a Phase 3b worker reads
    its bin), then split per community: (community -> (rows, id sum, edges))."""
    nodes_spec, edges_spec, comm_to_part, cs = args
    from utils.partition_layout import open_dataset, unit_filter
    out = []
    for spec, cols in ((nodes_spec, ["id", "features", "community_id"]),
                       (edges_spec, ["src", "dst", "community_id"])):
        key = (spec["url"], len(spec["files"]), spec.get("part_col"))
        if key not in _WCACHE:
            _WCACHE[key] = open_dataset(spec)
        out.append(_WCACHE[key].to_table(columns=cols, filter=unit_filter(spec, cs, comm_to_part)))
    n, e = out
    res = {int(c): [0, 0, 0] for c in cs}
    if n.num_rows:
        n.column("features").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        cid = n.column("community_id").to_numpy()
        ids = n.column("id").to_numpy()
        for c in np.unique(cid):
            sel = cid == c
            res[int(c)][0] = int(sel.sum())
            res[int(c)][1] = int(ids[sel].sum())
    if e.num_rows:
        ec, cnt = np.unique(e.column("community_id").to_numpy(), return_counts=True)
        for c, k in zip(ec, cnt):
            res[int(c)][2] = int(k)
    return {c: tuple(v) for c, v in res.items()}


def direct_loads(pool, lay, groups, repeats):
    tasks = [(lay["nodes"], lay["edges"], lay["comm_to_part"], [int(c) for c in g]) for g in groups]
    walls, check = [], None
    for _ in range(repeats + 1):                  # first pass = warm-up
        t0 = time.perf_counter()
        res = pool.map(load_group, tasks, chunksize=max(1, len(tasks) // 400))
        walls.append(time.perf_counter() - t0)
        check = {}
        for r in res:
            check.update(r)
    return walls[1:], check


def phase3b_bins(stats, default_parallelism):
    """Phase 3b's bins (run_phase3b): communities by size desc, bin = i % num_bins."""
    order = stats.sort_values("rows", ascending=False)["community_id"].astype(int).tolist()
    n = len(order)
    nb = n if n <= 2000 else min(max(default_parallelism * 4, 1000), n)
    bins = [[] for _ in range(nb)]
    for i, c in enumerate(order):
        bins[i % nb].append(c)
    return bins


# ------------------------------------------------------------------ spark reads
def timed(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="dir with <dataset>/{nodes,edges,masks,communities/<alg>}")
    ap.add_argument("--dataset", default="ogbn-arxiv")
    ap.add_argument("--alg", default="louvain")
    ap.add_argument("--tiny", nargs="+", default=["keep", "misc"], choices=["keep", "misc", "drop"])
    ap.add_argument("--layouts", nargs="+", default=["range", "hive", "hive_merged"], choices=LAYOUT_CHOICES)
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--merge-minor-members", type=int, default=None, help="default: --min-size (the paper's K)")
    ap.add_argument("--merge-target-mb", type=float, default=64.0,
                    help="target size of a shared partition (converted to rows with the table's bytes/row)")
    ap.add_argument("--rg-mb", type=float, default=1.0, help="row-group size for the *_rg layouts")
    ap.add_argument("--merge-min-mb", type=float, default=16.0,
                    help="hive_merged_size: also merge communities whose data is below this size")
    ap.add_argument("--bench-dir", required=True, help="scratch dir (its tables are deleted and rewritten)")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--mem", default="auto")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--point-sample", type=int, default=5, help="communities per size class for point reads")
    ap.add_argument("--skip-phase2", action="store_true", help="reuse <bench-dir>/<dataset>/<tiny>/phase2")
    ap.add_argument("--skip-direct", action="store_true",
                    help="skip the pyarrow parts (direct loads, row-group index, crawl); for machines without pyarrow")
    ap.add_argument("--spark-tmp", default=None)
    ap.add_argument("--keep-tables", action="store_true",
                    help="keep every layout's tables (default: delete each after measuring, to bound disk use)")
    ap.add_argument("--out", default="results/partition_layout/partition_layout.jsonl")
    args = ap.parse_args()

    import multiprocessing as mp
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    from phases.phase2_subgraph import run_phase2
    from utils.partition_layout import (community_stats, merge_plan, table_layout, with_storage_part,
                                        write_layout)
    ram = total_ram_gb()
    if args.mem == "auto":
        args.mem = "4g" if ram is None or ram < 15 else "8g" if ram < 30 else "12g"
    minor = args.merge_minor_members if args.merge_minor_members is not None else args.min_size
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    def emit(rec):
        rec = {"run_id": run_id, "dataset": args.dataset, "alg": args.alg, "cores": args.cores, **rec}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    spark = build_spark(args)
    raw = os.path.abspath(os.path.join(args.data_root, args.dataset))
    pool = None if args.skip_direct else mp.get_context("spawn").Pool(args.cores)
    print(f"\n== {args.dataset}/{args.alg}: tiny modes {args.tiny}, layouts {args.layouts}, "
          f"local[{args.cores}], driver {args.mem}, run {run_id}", flush=True)

    for tiny in args.tiny:
        base = os.path.abspath(os.path.join(args.bench_dir, args.dataset, tiny))

        def get_paths(dataset, alg=None):
            p = {"nodes": f"{raw}/nodes", "edges": f"{raw}/edges", "masks": f"{raw}/masks",
                 "checkpoints": f"{base}/_ckpt/"}
            if alg:
                p.update({"communities": f"{raw}/communities/{args.alg}",
                          "p2_nodes": f"{base}/{alg}/p2_nodes", "p2_edges": f"{base}/{alg}/p2_edges",
                          "tag": f"layout_{dataset}_{tiny}_{alg}"})
            return p

        print(f"\n-- tiny_comm_handling={tiny}", flush=True)
        src = get_paths(args.dataset, "phase2")
        if not args.skip_phase2:
            for k in ("p2_nodes", "p2_edges"):
                shutil.rmtree(src[k], ignore_errors=True)
            timing = {}
            run_phase2(spark, spark.sparkContext, [args.dataset], ["phase2"], use_global_mapping=True,
                       min_size=args.min_size, get_paths_fn=get_paths, timing=timing, results={},
                       expand_boundary_nodes=True, tiny_comm_handling=tiny, force_rerun=True,
                       p2_layout="range")
            emit({"kind": "phase2", "tiny": tiny, "phase2_s": timing.get(("phase2", args.dataset, "phase2"))})

        nodes = spark.read.format("delta").load(src["p2_nodes"]).persist(StorageLevel.MEMORY_AND_DISK)
        edges = spark.read.format("delta").load(src["p2_edges"]).persist(StorageLevel.MEMORY_AND_DISK)
        n_rows, n_edges = nodes.count(), edges.count()
        stats = community_stats(nodes)
        n_comm = len(stats)
        n_major = int((stats["members"] >= args.min_size).sum())
        src_bytes = storage_stats(spark, src["p2_nodes"])["bytes"]
        bpr = src_bytes / max(n_rows, 1)
        target_rows = max(1, int(args.merge_target_mb * (1 << 20) / bpr))
        min_rows = int(args.merge_min_mb * (1 << 20) / bpr)
        plan = merge_plan(stats, minor, 0, target_rows)
        plan_size = merge_plan(stats, minor, min_rows, target_rows)
        n_shared = int(plan.loc[plan["small"], "storage_part"].nunique())
        n_shared_size = int(plan_size.loc[plan_size["small"], "storage_part"].nunique())
        size_q = np.percentile(stats["rows"], [0, 50, 90, 100]).astype(int).tolist()
        print(f"   {n_comm:,} communities ({n_major} with >= {args.min_size} members), {n_rows:,} node rows, "
              f"{n_edges:,} edge rows; rows/community min/median/p90/max {size_q}; {bpr:.0f} B/row", flush=True)
        print(f"   hive_merged: {int(plan['small'].sum()):,} minor communities (< {minor} members) -> "
              f"{n_shared} shared partitions of <= {target_rows:,} rows (~{args.merge_target_mb:g} MB); "
              f"hive_merged_size: {int(plan_size['small'].sum()):,} communities (< {minor} members or "
              f"< {min_rows:,} rows ~{args.merge_min_mb:g} MB) -> {n_shared_size} shared partitions", flush=True)
        emit({"kind": "units", "tiny": tiny, "communities": n_comm, "major": n_major, "node_rows": n_rows,
              "edge_rows": n_edges, "rows_quantiles_0_50_90_100": size_q, "bytes_per_row": bpr,
              "merge_minor_members": minor, "merge_min_rows": min_rows, "merge_target_rows": target_rows,
              "small_communities": int(plan["small"].sum()), "shared_partitions": n_shared,
              "size_small_communities": int(plan_size["small"].sum()), "size_shared_partitions": n_shared_size})

        bins = min(200, max(1, n_major))
        results = {}
        for layout in args.layouts:
            p = get_paths(args.dataset, layout)
            for k in ("p2_nodes", "p2_edges"):
                shutil.rmtree(p[k], ignore_errors=True)
            hconf = spark.sparkContext._jsc.hadoopConfiguration()
            old_rg = hconf.get("parquet.block.size")
            if layout.endswith("_rg"):
                hconf.set("parquet.block.size", str(int(args.rg_mb * (1 << 20))))
            t0 = time.perf_counter()
            if layout == "hive_naive":
                nparts = int(spark.conf.get("spark.sql.shuffle.partitions"))
                for df, k, key in ((nodes, "p2_nodes", "id"), (edges, "p2_edges", "src")):
                    (df.repartition(nparts, key).write.format("delta").mode("overwrite")
                       .partitionBy("community_id").save(p[k]))
            elif layout in ("hive_merged", "hive_merged_rg", "hive_merged_size"):
                pl = plan_size if layout == "hive_merged_size" else plan
                nb = min(200, max(1, pl["storage_part"].nunique()))
                write_layout(with_storage_part(spark, nodes, pl), p["p2_nodes"], "hive_merged", nb)
                write_layout(with_storage_part(spark, edges, pl), p["p2_edges"], "hive_merged", nb)
            else:
                base_layout = "range" if layout == "range_rg" else layout
                write_layout(nodes, p["p2_nodes"], base_layout, bins)
                write_layout(edges, p["p2_edges"], base_layout, bins)
            write_s = time.perf_counter() - t0
            if old_rg is None:
                hconf.unset("parquet.block.size")
            else:
                hconf.set("parquet.block.size", old_rg)
            st = {"nodes": storage_stats(spark, p["p2_nodes"]), "edges": storage_stats(spark, p["p2_edges"])}

            # metadata: cold snapshot (log replay) and a pyarrow crawl of the directory tree
            def cold():
                clear_delta_cache(spark)
                spark.read.format("delta").load(p["p2_nodes"]).inputFiles()
                spark.read.format("delta").load(p["p2_edges"]).inputFiles()
            snap_s = timed(cold, args.repeats)

            def crawl():
                import pyarrow.dataset as ds
                for k in ("p2_nodes", "p2_edges"):
                    ds.dataset(local(p[k]), format="parquet", partitioning="hive",
                               ignore_prefixes=["_delta_log", "."]).files

            lay = table_layout(spark, p)
            comms = [int(c) for c in stats["community_id"]]
            c2p = lay["comm_to_part"] or {}
            groupings = {"unit": [[c] for c in comms],
                         "p3b_bins": phase3b_bins(stats, spark.sparkContext.defaultParallelism)}
            if lay["comm_to_part"]:                  # merged layouts: one task per storage partition
                byp = {}
                for c in comms:
                    byp.setdefault(c2p[c], []).append(c)
                groupings["part"] = list(byp.values())
            direct, check = {}, None
            if args.skip_direct:
                crawl_s = [float("nan")]
            else:
                crawl_s = timed(crawl, args.repeats)
                idx_n, idx_e = rowgroup_index(lay["nodes"]), rowgroup_index(lay["edges"])
                for mode, groups in groupings.items():
                    fn_, rn_, bn_ = planned_reads(idx_n, groups, lay["nodes"], c2p)
                    fe_, re_, be_ = planned_reads(idx_e, groups, lay["edges"], c2p)
                    walls, chk = direct_loads(pool, lay, groups, args.repeats)
                    direct[mode] = {"tasks": len(groups), "wall_s": walls,
                                    "files_selected": fn_ + fe_, "row_groups_selected": rn_ + re_,
                                    "bytes_selected": bn_ + be_, "matches_spark": None}
                    direct[mode]["_check"] = chk
                    check = check or chk
            # Spark-side checksum per community (always): node rows, id sum, edge rows
            cn = (spark.read.format("delta").load(p["p2_nodes"]).groupBy("community_id")
                  .agg(F.count(F.lit(1)).alias("n"), F.sum("id").alias("s")).collect())
            ce = (spark.read.format("delta").load(p["p2_edges"]).groupBy("community_id")
                  .agg(F.count(F.lit(1)).alias("e")).collect())
            ce = {int(r[0]): int(r[1]) for r in ce}
            spark_check = {int(r[0]): (int(r[1]), int(r[2]), ce.get(int(r[0]), 0)) for r in cn}
            for d in direct.values():
                d["matches_spark"] = d.pop("_check") == spark_check

            full = lambda: (spark.read.format("delta").load(p["p2_nodes"]).drop("storage_part")  # noqa: E731
                            .write.format("noop").mode("overwrite").save(),
                            spark.read.format("delta").load(p["p2_edges"]).drop("storage_part")
                            .write.format("noop").mode("overwrite").save())
            scan_s = timed(full, args.repeats)

            srt = stats.sort_values("rows")
            k = args.point_sample
            sample = {"smallest": srt.head(k)["community_id"].tolist(),
                      "median": srt.iloc[len(srt) // 2 - k // 2: len(srt) // 2 - k // 2 + k]["community_id"].tolist(),
                      "largest": srt.tail(k)["community_id"].tolist()}
            point = {}
            for cls, cs in sample.items():
                ts, nfiles, nfiles_part = [], [], []
                for c in cs:
                    c = int(c)
                    df = spark.read.format("delta").load(p["p2_nodes"]).where(F.col("community_id") == c)
                    ts.append(statistics.median(timed(
                        lambda: df.select("id", "features").write.format("noop").mode("overwrite").save(),
                        args.repeats)))
                    nfiles.append(len(df.inputFiles()))
                    if layout.startswith("hive_merged"):
                        dfp = (spark.read.format("delta").load(p["p2_nodes"])
                               .where((F.col("storage_part") == c2p[c]) & (F.col("community_id") == c)))
                        nfiles_part.append(len(dfp.inputFiles()))
                point[cls] = {"median_s": statistics.median(ts), "files_mean": float(np.mean(nfiles)),
                              "files_mean_with_part_filter": float(np.mean(nfiles_part)) if nfiles_part else None}

            rec = emit({"kind": "layout", "tiny": tiny, "layout": layout, "write_s": write_s, "storage": st,
                        "snapshot_cold_s": snap_s, "pyarrow_crawl_s": crawl_s,
                        "direct": direct,
                        "units": len(comms),
                        "spark_full_scan_s": scan_s, "spark_point": point})
            results[layout] = (rec, all(d["matches_spark"] for d in direct.values()), spark_check)
            if not args.keep_tables:
                for k in ("p2_nodes", "p2_edges"):
                    shutil.rmtree(p[k], ignore_errors=True)
            s = st["nodes"]
            print(f"   {layout:12s} write {write_s:6.1f}s | nodes: {s['dirs']:5d} dirs {s['files']:6d} files "
                  f"({s['files_lt_1mb']} < 1 MB, median {s['file_bytes_median'] / 1e6:.2f} MB), log "
                  f"{s['delta_log_bytes'] / 1e6:.2f} MB | cold snapshot {statistics.median(snap_s):.2f}s | "
                  + "".join(f"direct[{m}] {statistics.median(d['wall_s']):6.2f}s {d['bytes_selected'] / 1e9:.2f} GB | "
                            for m, d in direct.items()) +
                  f"full scan {statistics.median(scan_s):6.2f}s | point small/large "
                  f"{point['smallest']['median_s']:.2f}/{point['largest']['median_s']:.2f}s", flush=True)

        ref = next(iter(results.values()))
        same = {lay: v[2] == ref[2] and (args.skip_direct or v[1]) for lay, v in results.items()}
        emit({"kind": "check", "tiny": tiny, "identical_units": same, "direct_checked": not args.skip_direct})
        print(f"   identical rows / id sums / edges per community across layouts"
              f"{'' if not args.skip_direct else ' (Spark only)'}: {same}", flush=True)
        nodes.unpersist()
        edges.unpersist()

    if pool is not None:
        pool.close()
    spark.stop()
    summarize(args.out, run_id)


def summarize(path, run_id):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    recs = [r for r in recs if r.get("run_id") == run_id and r["kind"] == "layout"]
    if not recs:
        return
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    print("\n== Summary (medians; files/<1MB/log = nodes + edges; direct = per-unit loader, wall s / GB selected)")
    print(f"  {'tiny':5s} {'layout':17s} {'write s':>7s} {'dirs':>5s} {'files':>6s} {'<1MB':>5s} {'log MB':>6s} "
          f"{'snap s':>6s} {'unit':>14s} {'p3b_bins':>14s} {'part':>14s} {'scan s':>6s} {'pt sm/lg s':>11s}")
    for r in recs:
        s, e = r["storage"]["nodes"], r["storage"]["edges"]
        d = r.get("direct", {})
        cell = lambda m: (f"{med(d[m]['wall_s']):6.1f}/{d[m]['bytes_selected'] / 1e9:6.2f}"  # noqa: E731
                          if m in d else f"{'-':>13s}")
        print(f"  {r['tiny']:5s} {r['layout']:17s} {r['write_s']:7.1f} {s['dirs']:5d} {s['files'] + e['files']:6d} "
              f"{s['files_lt_1mb'] + e['files_lt_1mb']:5d} {(s['delta_log_bytes'] + e['delta_log_bytes']) / 1e6:6.2f} "
              f"{med(r['snapshot_cold_s']):6.2f} {cell('unit'):>14s} {cell('p3b_bins'):>14s} {cell('part'):>14s} "
              f"{med(r['spark_full_scan_s']):6.2f} "
              f"{r['spark_point']['smallest']['median_s']:5.2f}/{r['spark_point']['largest']['median_s']:5.2f}")


if __name__ == "__main__":
    main()
