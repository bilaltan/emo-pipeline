#!/usr/bin/env python3
"""
How much of the raw nodes table does one training unit read once the partition
table is joined with it, and how does that depend on file size and layout?

After the partition-table change, Phase 2's p2_nodes holds only
(id, community_id, is_boundary, is_member). A unit (community) gets its
features by fetching its ids from the raw nodes table:

    phase3b_caan._load_communities_data_batch:  nodes_ds.to_table(filter=id.isin(unit_ids))

Delta skips data at FILE granularity using per-file [min,max] statistics in the
log. For an IN-list, Delta 3.2 builds an *imprecise range test* over the whole
list -- a file is kept when its [min(id), max(id)] overlaps
[min(unit ids), max(unit ids)] (DataFiltersBuilder.constructLiteralInListDataFilters).
So what a unit reads depends on (a) how many files the table has and (b) whether
the unit's ids are physically close together.

This harness rewrites a copy of the raw nodes table at several target file sizes
(maxRecordsPerFile -- OSS Delta has no delta.targetFileSize property) and in
several row orders, then reports for every unit:

  delta_range   files Delta's own IN-list skipping keeps (the range test)
  exact         files that actually contain at least one of the unit's ids
                (what a reader handed an exact file list would open)

The partition table, not the raw table, decides the units, so every layout is
evaluated against the same units. The raw table's schema never changes.

Usage
-----
    python experiments/join_skipping_bench.py --data-root local_data/delta-data \\
        --dataset ogbn-arxiv --p2 local_data/delta-data/ogbn-arxiv/phase2_nodes/<tag> \\
        --sizes-mb 88 16 4 1 0.25 --layouts id random community
Set LAKEGRL_SPARK_JARS to a comma-separated list of Delta jars to avoid Maven.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_spark(tmp_dir, mem):
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName("join-skipping-bench").master("local[*]")
         .config("spark.driver.host", "127.0.0.1")
         .config("spark.driver.memory", mem)
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.local.dir", tmp_dir))
    jars = os.environ.get("LAKEGRL_SPARK_JARS")
    if jars:
        b = b.config("spark.jars", jars)
    else:
        b = b.config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def write_variant(spark, nodes, member_comm, layout, rows_per_file, path, halo_target=None):
    """Rewrite the raw nodes rows in `layout` order, rows_per_file rows per file.

    Only row order and file boundaries change; columns are exactly the raw ones.
    """
    from pyspark.sql import functions as F
    n = nodes.count()
    nparts = max(1, -(-n // rows_per_file))
    cols = nodes.columns
    if layout == "id":
        keys, df = ["id"], nodes
    elif layout == "random":
        df, keys = nodes.withColumn("_k", F.xxhash64("id")), ["_k"]
    elif layout == "community":
        # rows of the same (member) community together, in id order inside it;
        # nodes outside every kept community go last
        df = (nodes.join(member_comm, "id", "left")
                   .withColumn("_c", F.coalesce("_c", F.lit(2**62))))
        keys = ["_c", "id"]
    elif layout == "community_halo":
        # as "community", but inside a community rows are grouped by the OTHER
        # community each node has most edges to. A unit's halo is exactly the
        # boundary nodes of its neighbours that point at it, so this puts each
        # halo in a contiguous run instead of scattering it over the neighbour.
        df = (nodes.join(member_comm, "id", "left")
                   .join(halo_target, "id", "left")
                   .withColumn("_c", F.coalesce("_c", F.lit(2**62)))
                   .withColumn("_t", F.coalesce("_t", F.lit(-1))))
        keys = ["_c", "_t", "id"]
    else:
        raise ValueError(layout)
    (df.repartitionByRange(nparts, *keys).sortWithinPartitions(*keys)
       .select(*cols)
       .write.format("delta").mode("overwrite")
       .option("maxRecordsPerFile", int(rows_per_file))
       .option("overwriteSchema", "true")
       .save(path))


def file_index(spark, path):
    """{file: (min_id, max_id, bytes)} and id -> file, from the table's snapshot."""
    import pandas as pd
    from pyspark.sql import functions as F
    from urllib.parse import urlparse, unquote
    df = spark.read.format("delta").load(path)
    live = df.inputFiles()
    m = (df.select("id", F.input_file_name().alias("f")).toPandas())
    m["f"] = m["f"].map(lambda u: unquote(urlparse(u).path) if u.startswith("file:") else u)
    sizes = {}
    for u in live:
        p = unquote(urlparse(u).path) if u.startswith("file:") else u
        sizes[p] = os.path.getsize(p) if os.path.exists(p) else 0
    g = m.groupby("f")["id"].agg(["min", "max"])
    files = {f: (int(r["min"]), int(r["max"]), sizes.get(f, 0)) for f, r in g.iterrows()}
    return files, dict(zip(m["id"].values.tolist(), m["f"].values.tolist()))


def evaluate(units, files, id2file):
    """Per-unit files/bytes kept by Delta's range test and by an exact file list."""
    import numpy as np
    fnames = list(files)
    fmin = np.array([files[f][0] for f in fnames])
    fmax = np.array([files[f][1] for f in fnames])
    fbytes = np.array([files[f][2] for f in fnames])
    fpos = {f: i for i, f in enumerate(fnames)}
    rows = []
    for cid, (ids, is_member) in units.items():
        lo, hi = int(ids.min()), int(ids.max())
        rng = (fmin <= hi) & (fmax >= lo)
        ex = np.zeros(len(fnames), bool)
        ex_m = np.zeros(len(fnames), bool)
        for i, m in zip(ids.tolist(), is_member.tolist()):
            k = fpos[id2file[i]]
            ex[k] = True
            if m:
                ex_m[k] = True
        rows.append(dict(community_id=int(cid), rows=int(len(ids)), members=int(is_member.sum()),
                         range_files=int(rng.sum()), range_bytes=int(fbytes[rng].sum()),
                         exact_files=int(ex.sum()), exact_bytes=int(fbytes[ex].sum()),
                         exact_member_files=int(ex_m.sum())))
    return rows, int(fbytes.sum()), len(fnames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--p2", required=True, help="partition table (phase2_nodes/<tag>)")
    ap.add_argument("--old-p2", default=None, help="optional pre-change p2_nodes (with features) for comparison")
    ap.add_argument("--sizes-mb", nargs="+", type=float, default=[88, 16, 4, 1, 0.25])
    ap.add_argument("--layouts", nargs="*", default=["id", "random", "community"])
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--validate", type=int, default=2, help="units to cross-check against Delta's own skipping")
    ap.add_argument("--mem", default="2g")
    ap.add_argument("--spark-tmp", default=None, help="Spark scratch dir (default: system temp)")
    ap.add_argument("--out", default=None, help="JSON lines file; one record per variant is appended")
    a = ap.parse_args()

    import numpy as np
    from pyspark.sql import functions as F
    from utils.partition_join import read_partition

    root = os.path.abspath(os.path.join(a.data_root, a.dataset))
    bench = a.bench_dir or os.path.join(root, "join_bench")
    import tempfile
    spark = build_spark(a.spark_tmp or os.environ.get("LAKEGRL_SPARK_TMP") or tempfile.gettempdir(), a.mem)

    part = read_partition(spark, {"p2_nodes": a.p2}).toPandas()
    units = {cid: (g["id"].values.astype(np.int64), g["is_member"].values.astype(bool))
             for cid, g in part.groupby("community_id")}
    member_comm = (read_partition(spark, {"p2_nodes": a.p2})
                   .filter("is_member").select("id", F.col("community_id").alias("_c"))
                   .dropDuplicates(["id"]))
    nodes = spark.read.format("delta").load(os.path.join(root, "nodes"))
    halo_target = None
    if "community_halo" in a.layouts:
        # For every node, the other community it has the most edges to.
        from pyspark.sql import Window
        e = spark.read.format("delta").load(os.path.join(root, "edges")).select("src", "dst")
        mc = member_comm
        x = (e.join(mc.withColumnRenamed("id", "src").withColumnRenamed("_c", "cs"), "src")
              .join(mc.withColumnRenamed("id", "dst").withColumnRenamed("_c", "cd"), "dst")
              .filter("cs != cd").groupBy("src", "cd").count())
        w = Window.partitionBy("src").orderBy(F.desc("count"), F.asc("cd"))
        halo_target = (x.withColumn("_r", F.row_number().over(w)).filter("_r = 1")
                        .select(F.col("src").alias("id"), F.col("cd").alias("_t")))
    n_rows = nodes.count()
    from delta.tables import DeltaTable
    raw_bytes = int(DeltaTable.forPath(spark, os.path.join(root, "nodes")).detail().first()["sizeInBytes"])
    row_bytes = raw_bytes / n_rows
    print(f"raw nodes: {n_rows:,} rows, {raw_bytes/1e6:.1f} MB ({row_bytes:.0f} B/row); "
          f"{len(units)} units, {len(part):,} unit rows ({(~part['is_member']).sum():,} halo)")

    out = {"dataset": a.dataset, "n_rows": n_rows, "raw_bytes": raw_bytes, "units": len(units),
           "unit_rows": int(len(part)), "halo_rows": int((~part["is_member"]).sum()), "variants": []}

    for layout in a.layouts:
        for mb in a.sizes_mb:
            rpf = max(1, int(mb * 1e6 / row_bytes))
            path = os.path.join(bench, f"{layout}_{mb:g}mb")
            t = time.time()
            write_variant(spark, nodes, member_comm, layout, rpf, path, halo_target)
            files, id2file = file_index(spark, path)
            rows, tbytes, nfiles = evaluate(units, files, id2file)
            v = dict(layout=layout, target_mb=mb, rows_per_file=rpf, files=nfiles, table_bytes=tbytes,
                     mean_range_files=float(np.mean([r["range_files"] for r in rows])),
                     mean_exact_files=float(np.mean([r["exact_files"] for r in rows])),
                     mean_exact_member_files=float(np.mean([r["exact_member_files"] for r in rows])),
                     range_bytes_all_units=int(sum(r["range_bytes"] for r in rows)),
                     exact_bytes_all_units=int(sum(r["exact_bytes"] for r in rows)),
                     per_unit=rows)
            # Cross-check the emulated range test against Delta's own file skipping.
            checks = []
            dfv = spark.read.format("delta").load(path)
            # smallest units: building a large IN-list through py4j is slow
            for r in sorted(rows, key=lambda r: r["rows"])[:a.validate]:
                ids = units[r["community_id"]][0].tolist()
                kept = len(dfv.where(F.col("id").isin(ids)).inputFiles())
                checks.append(dict(community_id=r["community_id"], delta_kept=kept, emulated=r["range_files"]))
            v["delta_crosscheck"] = checks
            out["variants"].append(v)
            _append(a.out, dict(kind="variant", dataset=a.dataset, n_rows=n_rows, raw_bytes=raw_bytes,
                                units=len(units), **v))
            print(f"{layout:>9} {mb:>6g} MB  files={nfiles:>4}  per unit: Delta range test "
                  f"{v['mean_range_files']:6.1f} files ({100*v['mean_range_files']/nfiles:5.1f}%), "
                  f"exact {v['mean_exact_files']:6.1f} ({100*v['mean_exact_files']/nfiles:5.1f}%), "
                  f"members-only exact {v['mean_exact_member_files']:5.1f}   "
                  f"all units read: range {v['range_bytes_all_units']/1e6:8.1f} MB, exact "
                  f"{v['exact_bytes_all_units']/1e6:8.1f} MB  "
                  f"check={[(c['delta_kept'], c['emulated']) for c in checks]}  [{time.time()-t:.0f}s]",
                  flush=True)

    if a.old_p2:
        # Pre-change design: features copied into p2_nodes, read with community_id = c.
        from urllib.parse import urlparse, unquote
        old = spark.read.format("delta").load(a.old_p2)
        m = old.select("community_id", F.input_file_name().alias("f")).groupBy("f") \
               .agg(F.min("community_id").alias("lo"), F.max("community_id").alias("hi")).toPandas()
        m["f"] = m["f"].map(lambda u: unquote(urlparse(u).path) if u.startswith("file:") else u)
        m["b"] = m["f"].map(lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
        per = [int(m[(m.lo <= c) & (m.hi >= c)]["b"].sum()) for c in units]
        out["old_design"] = {"table_bytes": int(m["b"].sum()), "files": int(len(m)),
                             "bytes_all_units": int(sum(per)), "mean_bytes_per_unit": float(np.mean(per))}
        _append(a.out, dict(kind="old_design", dataset=a.dataset, **out["old_design"]))
        print(f"old design (features copied into p2_nodes): table {m['b'].sum()/1e6:.1f} MB in {len(m)} files; "
              f"all units read {sum(per)/1e6:.1f} MB")



def _append(path, rec):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
