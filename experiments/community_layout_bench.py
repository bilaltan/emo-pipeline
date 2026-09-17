#!/usr/bin/env python3
"""
Does the Phase 2 storage layout let Phase 3's per-unit reads skip anything?

Phase 3 does not scan the partitioned tables once. It reads them once per
training unit, through pyarrow with a predicate on community_id:

    ds.field("community_id").isin(comm_ids)      phase3_training.py:70,77

Phase 2 writes those tables with repartition(200, 'community_id'), which is a
HASH shuffle (phase2_subgraph.py:168-176). Hash scatters neighbouring community
ids across every file, so each file's min/max community_id can span nearly the
whole id space even though rows are sorted inside the file. If so, statistics
pruning skips nothing and every per-unit read pays for the entire table - a cost
multiplied by the number of units (113 on reddit, 575 on ogbn-products).

This measures that, and compares layouts that keep a community's rows together:

    A hash        as submitted: repartition(200, community_id) + sort within
    B zorder      A, then OPTIMIZE ... ZORDER BY (community_id)
    C range       repartitionByRange(200, community_id) + sort within
    D partitioned Delta partitionBy(community_id): one directory per community

Metric
------
For each community, the bytes that must actually be read to answer its query,
computed from parquet row-group statistics on community_id: a row group whose
[min,max] range does not contain the community cannot be read, everything else
must be. Summed over every community, this is what Phase 3 pays per epoch of
unit loading. It is independent of the page cache.

Wall-clock time for the real pyarrow load is measured alongside, for the
communities in --time-sample.

Usage
-----
    python experiments/community_layout_bench.py --dataset ogbn-arxiv --alg louvain
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zorder_bench import build_spark, dir_stats, log_stats  # noqa: E402


def rowgroup_index(path):
    """[(file_bytes, [(min_cid, max_cid, rowgroup_bytes), ...]), ...]"""
    import pyarrow.parquet as pq
    out = []
    for root, _dirs, names in os.walk(path):
        if "_delta_log" in root:
            continue
        for n in sorted(names):
            if not n.endswith(".parquet"):
                continue
            fp = os.path.join(root, n)
            md = pq.ParquetFile(fp).metadata
            col = md.schema.names.index("community_id")
            groups = []
            for g in range(md.num_row_groups):
                rg = md.row_group(g)
                st = rg.column(col).statistics
                if st is None:
                    groups.append((None, None, rg.total_byte_size))
                else:
                    groups.append((st.min, st.max, rg.total_byte_size))
            out.append((os.path.getsize(fp), groups))
    return out


def bytes_for(index, cid):
    """Bytes unavoidably read for one community, and row groups touched."""
    total, touched = 0, 0
    for _fsize, groups in index:
        for lo, hi, size in groups:
            if lo is None or (lo <= cid <= hi):
                total += size
                touched += 1
    return total, touched


def partitioned_bytes_for(path, cid):
    """Same, for a Delta table physically partitioned by community_id."""
    d = os.path.join(path, "community_id=%d" % cid)
    if not os.path.isdir(d):
        return 0, 0
    total, n = 0, 0
    for f in os.listdir(d):
        if f.endswith(".parquet"):
            total += os.path.getsize(os.path.join(d, f))
            n += 1
    return total, n


def real_load(path, cid, hive=False):
    """The pipeline's own read path, verbatim, for one community.

    `hive=True` is needed only for the physically partitioned variant, where
    community_id lives in the directory name rather than in the file. Note that
    the submitted reader (phase3_training.py:47-57) does NOT pass this, so it
    cannot read a partitionBy'd table at all without a code change.
    """
    import pyarrow.dataset as ds
    dataset = ds.dataset(path, format="parquet",
                         partitioning="hive" if hive else None,
                         ignore_prefixes=["_delta_log", "."])
    t = time.time()
    tbl = dataset.to_table(filter=(ds.field("community_id").isin([cid])),
                           use_threads=True)
    return time.time() - t, tbl.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ogbn-arxiv")
    ap.add_argument("--alg", default="louvain")
    ap.add_argument("--repo", default="/Users/bilaltan/Desktop/emo-submitted")
    ap.add_argument("--local-data-dir", default=None)
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--table", default="p2_edges", choices=["p2_edges", "p2_nodes"])
    ap.add_argument("--bins", type=int, default=200,
                    help="shuffle bins, matching phase 2's p2_bins cap")
    ap.add_argument("--time-sample", type=int, default=20,
                    help="communities to time the real pyarrow load on")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    local_data_dir = args.local_data_dir or os.path.join(repo, "local_data")
    sys.path.insert(0, repo)
    spark = build_spark(repo, local_data_dir, args.master)
    from pyspark.sql import functions as F
    from pipeline.utils.paths import get_paths
    import experiment_config as config

    gp = lambda d, a=None: get_paths(d, a, local_data_dir=local_data_dir)
    p_alg = gp(args.dataset, args.alg)

    if not os.path.exists(p_alg["p2_edges"]):
        from pipeline.phases import run_phase1, run_phase2
        print("  building communities and partitions with the submitted code ...")
        run_phase1(spark, spark.sparkContext,
                   datasets=[args.dataset], algorithms=[args.alg],
                   lpa_max_iter=config.LPA_MAX_ITER,
                   resolution=getattr(config, "RESOLUTION", 1.0),
                   random_seed=config.RANDOM_SEED,
                   min_size=config.MIN_COMMUNITY_SIZE,
                   dataset_cfg=config.DATASET_CFG,
                   get_paths_fn=gp, timing={}, results={},
                   metis_k=getattr(config, "METIS_K", 100), force_rerun=False)
        run_phase2(spark, spark.sparkContext,
                   datasets=[args.dataset], algorithms=[args.alg],
                   use_global_mapping=config.USE_GLOBAL_MAPPING,
                   min_size=config.MIN_COMMUNITY_SIZE,
                   get_paths_fn=gp, timing={}, results={},
                   tiny_comm_handling=getattr(config, "TINY_COMM_HANDLING", "misc"),
                   expand_boundary_nodes=getattr(config, "EXPAND_BOUNDARY_NODES", True),
                   force_rerun=False)

    src = p_alg[args.table]
    base = spark.read.format("delta").load(src)
    comms = sorted(r["community_id"] for r in
                   base.select("community_id").distinct().collect())
    print("\n  %s: %d rows, %d communities" % (args.table, base.count(), len(comms)))

    root = os.path.join(local_data_dir, "layout_bench", args.dataset, args.table)
    variants = {}

    def write(name, df, partition_by=None):
        path = os.path.join(root, name)
        if os.path.exists(path):
            shutil.rmtree(path)
        w = df.write.format("delta").mode("overwrite")
        if partition_by:
            w = w.partitionBy(partition_by)
        t = time.time()
        w.save(path)
        variants[name] = {"path": path, "write_seconds": round(time.time() - t, 2)}

    bins = min(args.bins, max(1, len(comms)))
    # A reproduces phase 2's own write, so it is the real baseline, not a strawman.
    write("A_hash", base.repartition(bins, "community_id")
          .sortWithinPartitions("community_id"))
    write("C_range", base.repartitionByRange(bins, "community_id")
          .sortWithinPartitions("community_id"))
    write("D_partitioned", base, partition_by="community_id")

    zpath = os.path.join(root, "B_zorder")
    if os.path.exists(zpath):
        shutil.rmtree(zpath)
    shutil.copytree(variants["A_hash"]["path"], zpath)
    t = time.time()
    spark.sql("OPTIMIZE delta.`%s` ZORDER BY (community_id)" % zpath).collect()
    variants["B_zorder"] = {"path": zpath,
                            "write_seconds": round(time.time() - t, 2)}

    order = ["A_hash", "B_zorder", "C_range", "D_partitioned"]
    sample = comms[:: max(1, len(comms) // args.time_sample)][:args.time_sample]

    for name in order:
        v = variants[name]
        path = v["path"]
        v["files"], v["bytes_on_disk"] = dir_stats(path)
        v["files_active"], v["bytes_active"] = log_stats(spark, path)

        if name == "D_partitioned":
            per = [partitioned_bytes_for(path, c) for c in comms]
            # Directory pruning happens before any file is opened, so the
            # denominator is the same compressed total the files occupy.
            whole = v["bytes_on_disk"]
        else:
            idx = rowgroup_index(path)
            per = [bytes_for(idx, c) for c in comms]
            # Row-group sizes are UNCOMPRESSED, so the fraction skipped has to
            # be taken against the uncompressed total, not the file bytes.
            whole = sum(size for _f, groups in idx for _lo, _hi, size in groups)
        v["table_bytes_basis"] = whole

        v["read_bytes_total"] = sum(b for b, _ in per)
        v["read_bytes_median"] = int(statistics.median([b for b, _ in per]))
        v["rowgroups_median"] = int(statistics.median([g for _, g in per]))
        v["skip_fraction"] = round(1 - v["read_bytes_median"] / max(1, whole), 4)

        times, rows = [], 0
        for c in sample:
            dt, n = real_load(path, c, hive=(name == "D_partitioned"))
            times.append(dt)
            rows += n
        v["load_seconds_median"] = round(statistics.median(times), 4)
        v["load_seconds_total_sample"] = round(sum(times), 3)
        v["sample_rows"] = rows

        print("\n=== %s ===" % name)
        print("  files                 %d" % v["files_active"])
        print("  size (MB)             %.2f" % (v["bytes_active"] / 1e6))
        print("  bytes read / unit     %s  (median)" % f"{v['read_bytes_median']:,}")
        print("  bytes read, all units %s" % f"{v['read_bytes_total']:,}")
        print("  fraction skipped      %.1f%%" % (100 * v["skip_fraction"]))
        print("  pyarrow load / unit   %.4fs (median over %d communities)"
              % (v["load_seconds_median"], len(sample)))

    doc = {
        "dataset": args.dataset, "algorithm": args.alg, "table": args.table,
        "communities": len(comms), "rows": base.count(),
        "code_commit": os.popen("git -C %s rev-parse HEAD" % repo).read().strip(),
        "spark_version": spark.version, "delta_version": "3.2.0",
        "read_predicate": "community_id == <cid>, one read per training unit",
        "metric_note":
            "read_bytes_* is computed from parquet row-group statistics: a row "
            "group whose [min,max] community_id excludes the target cannot be "
            "read, every other row group must be. Cache-independent.",
        "caveats": [
            "Local NVMe; on S3 the per-file GET latency would add to the file "
            "count differences measured here.",
            "D_partitioned creates one directory per community, which is a "
            "small-file problem at high community counts - its file count is "
            "reported so that cost is visible.",
        ],
        "variants": {k: variants[k] for k in order},
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
        print("\nwrote %s" % args.out)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
