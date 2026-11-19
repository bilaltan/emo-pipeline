#!/usr/bin/env python3
"""
Table A, step 1 (plan: "LakeGRL Table A plan"): one dataset's Phase 2 rows written in the eight layouts of Table A.

Phase 2 (members + 1-hop halo, every community kept as a unit, tiny_comm_handling=keep) runs once and is persisted;
every layout is written from those rows, so Stage 3b's work items read identical rows from all of them.

  join        row 1  normalized: p2_map (community_id, id, is_member, is_boundary); features stay in the raw node
                     table and are fetched by id; p2_edges as in mv
  mv          row 2  denormalized p2_nodes / p2_edges, repartitionByRange + sort by community_id, 128 MB row groups
  hive        row 3  .partitionBy("community_id") of the same clustered rows
  zorder      row 4  mv, then OPTIMIZE ZORDER BY (community_id, id) / (community_id, src, dst)
  liquid      row 5  a table created CLUSTER BY the same keys, the rows appended, then OPTIMIZE
  merged      row 6  .partitionBy(storage_part): communities with < --min-size members share directories (merge_plan);
                     rows in node-id order inside a directory (id / src, dst), so units interleave in shared files
  merged_rg   row 7  merged with the bounded sqrt-rule row group: max(sqrt_rg_mb over mv's bytes and the written
                     blocks, --rg-floor-mb)
  merged_blk  row 8  merged_rg sorted by unit (community_id), then _blk (add_blk, bit-reversed block key; 0 for
                     every row when no unit is above --cap): every unit, and every --cap block, is a contiguous run
  blk_128     diagnostic: row 8's order with Spark's 128 MB row groups (unit order without row 7)
  zorder_2mb  zorder written and OPTIMIZEd with parquet.block.size = --zl-rg-mb (advisor 2026-11-18)
  liquid_2mb  liquid, the same
  hive_coarse .partitionBy(storage_part = community_id mod --coarse-buckets), one file per directory, sorted by
              community_id, Spark's 128 MB row groups (advisor 2026-11-18: B = 611, our directory count)
Rows 6-8 and the diagnostic write one file per storage_part directory (advisor 2026-10-31), so they hold the same rows
in the same files: 6 -> 7 changes only the row-group size, 7 -> 8 only the order inside a file; the diagnostic is
6 -> 8 without 7.

Each layout is written --repeats times (every write timed, table deleted first); the last copy stays in
<out-dir>/<dataset>/<layout> for read_layouts.py. Also written there: manifest.json (Stage 3b's work items, as
run_phase3b builds them) and per-layout meta.json (partitioning, written blocks, community -> storage_part).

    bash scripts/wsl_run.sh python experiments/tableA/write_layouts.py --dataset ogbn-arxiv
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.partition_layout_bench import build_spark, storage_stats  # noqa: E402

LAYOUTS = ("mv", "join", "hive", "zorder", "liquid", "merged", "merged_rg", "merged_blk", "blk_128",   # mv first: rows 7-8 size from it
           "zorder_2mb", "liquid_2mb", "hive_coarse")


def manifest(spark, nodes, stats, cap):
    """Stage 3b's work items as run_phase3b builds them: communities by rows desc, unit_blocks(rows, cap) blocks,
    halved while a block has no training label. Returns [(community, n_blocks, written_blocks)]."""
    from pyspark.sql import functions as F
    from pipeline.utils.common import unit_blocks
    st = stats.sort_values("rows", ascending=False, kind="mergesort")
    comms = st["community_id"].astype(int).tolist()
    written = [unit_blocks(int(r), cap) for r in st["rows"]]
    split = [c for c, n in zip(comms, written) if n > 1]
    tr = (nodes.filter(F.col("is_member") & (F.col("split") == "train") & F.col("community_id").isin(split))
          .select("community_id", "id").toPandas()) if split else None
    tr_ids = {int(c): g.to_numpy(np.int64) for c, g in tr.groupby("community_id")["id"]} if split else {}
    out = []
    for c, w in zip(comms, written):
        n, ids = w, tr_ids.get(c)
        while ids is not None and n > 1 and len(np.unique(ids & (n - 1))) < n:
            n //= 2
        out.append((c, n, w))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.expanduser("~/lakegrl/store"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alg", default="leiden")
    ap.add_argument("--min-size", type=int, default=100, help="minor community: fewer members (the paper's K)")
    ap.add_argument("--cap", type=int, default=10000, help="C_max: rows per unit, members + 1-hop halo")
    ap.add_argument("--unit-split", default="leiden", choices=["leiden", "idmod"],
                    help="leiden: the recursive Leiden units of run_local_pipeline --unit-split leiden at --cap "
                         "(communities/<alg>_c<cap>, hub halos capped); idmod: the unsplit communities, cut by Stage 3b")
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=LAYOUTS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--merge-target-mb", type=float, default=64.0, help="size of a shared directory (merge_plan)")
    ap.add_argument("--rg-beta-mb", type=float, default=0.6)
    ap.add_argument("--rg-floor-mb", type=float, default=0.0, help="rows 7-8: row groups at least this (MB); 0 = pure √ rule")
    ap.add_argument("--zl-rg-mb", type=float, default=2.0, help="zorder_2mb / liquid_2mb: parquet.block.size (MB)")
    ap.add_argument("--coarse-buckets", type=int, default=611, help="hive_coarse: B, directories = community_id mod B")
    ap.add_argument("--phase2-from", default=None,
                    help="read the Phase 2 rows (p2_nodes, p2_edges) from this directory instead of rerunning Phase 2; "
                         "the directory is only read (e.g. a frozen run set's <dataset>/phase2)")
    ap.add_argument("--cores", type=int, default=os.cpu_count())
    ap.add_argument("--mem", default="16g")
    ap.add_argument("--spark-tmp", default=None)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/lakegrl/tableA"))
    ap.add_argument("--out", default="results/tableA/write.jsonl")
    args = ap.parse_args()
    args.layouts = sorted(args.layouts, key=LAYOUTS.index)

    from pyspark import StorageLevel
    from phases.phase2_subgraph import run_phase2
    from pipeline.utils.common import spark_resources, unit_blocks
    from pipeline.utils.partition_layout import (BLOCK_CAP_PROP, add_blk, community_stats, merge_plan, sqrt_rg_mb,
                                                 with_storage_part, write_layout)
    from delta.tables import DeltaTable

    spark = build_spark(args)
    sc = spark.sparkContext
    raw = os.path.join(args.data_root, args.dataset)
    wd = os.path.join(args.out_dir, args.dataset)
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    res = spark_resources(sc)

    def emit(rec):
        rec = {"run_id": run_id, "dataset": args.dataset, "alg": args.alg, "cap": args.cap, "unit_split": args.unit_split,
               "min_size": args.min_size, "rg_floor_mb": args.rg_floor_mb,
               **res, **rec}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    tag = f"_c{args.cap}" if args.unit_split == "leiden" else ""

    def get_paths(dataset, alg=None):
        p = {"nodes": f"{raw}/nodes", "edges": f"{raw}/edges", "masks": f"{raw}/masks", "checkpoints": f"{wd}/_ckpt/"}
        if alg:
            p.update({"communities": f"{raw}/communities/{args.alg}{tag}", "p2_nodes": f"{wd}/{alg}/p2_nodes",
                      "p2_edges": f"{wd}/{alg}/p2_edges", "tag": f"tableA_{dataset}_{alg}"})
        return p

    timing = {}
    if args.phase2_from:   # the same rows as an earlier run set, read only: never cleared or rewritten here
        src = {k: os.path.join(args.phase2_from, k) for k in ("p2_nodes", "p2_edges")}
    else:
        src = get_paths(args.dataset, "phase2")
        for k in ("p2_nodes", "p2_edges"):
            shutil.rmtree(src[k], ignore_errors=True)
        run_phase2(spark, sc, [args.dataset], ["phase2"], use_global_mapping=True, min_size=args.min_size,
                   get_paths_fn=get_paths, timing=timing, results={}, expand_boundary_nodes=True,
                   tiny_comm_handling="keep", force_rerun=True, p2_layout="range",
                   halo_cap=args.cap if args.unit_split == "leiden" else 0)
    nodes = spark.read.format("delta").load(src["p2_nodes"]).persist(StorageLevel.MEMORY_AND_DISK)
    edges = spark.read.format("delta").load(src["p2_edges"]).persist(StorageLevel.MEMORY_AND_DISK)
    n_rows, n_edges = nodes.count(), edges.count()
    stats = community_stats(nodes)
    items = manifest(spark, nodes, stats, args.cap)
    json.dump({"cap": args.cap, "items": items}, open(os.path.join(wd, "manifest.json"), "w"))
    written_units = sum(w for _, _, w in items)
    assert args.unit_split == "idmod" or written_units == len(items), "a recursive Leiden unit is above --cap"
    bpr = storage_stats(spark, src["p2_nodes"])["bytes"] / max(1, n_rows)
    plan = merge_plan(stats, args.min_size, 0, max(1, int(args.merge_target_mb * (1 << 20) / bpr)))
    bins = min(200, len(stats))
    emit({"kind": "phase2", "phase2_s": timing.get(("phase2", args.dataset, "phase2")), "communities": len(stats),
          "minor": int((stats["members"] < args.min_size).sum()), "node_rows": n_rows, "edge_rows": n_edges,
          "work_items": sum(n for _, n, _ in items), "written_blocks": written_units,
          "split_communities": sum(w > 1 for _, _, w in items), "raw": raw, "phase2_from": args.phase2_from})
    print(f"== {args.dataset}/{args.alg}: {len(stats):,} communities ({int((stats['members'] < args.min_size).sum()):,}"
          f" minor), {n_rows:,} node rows, {n_edges:,} edge rows, {sum(n for _, n, _ in items):,} work items at cap "
          f"{args.cap:,}; Phase 2 {timing.get(('phase2', args.dataset, 'phase2'), 0):.0f} s; local[{args.cores}]",
          flush=True)

    hconf = sc._jsc.hadoopConfiguration()
    keys = {"nodes": ["id"], "edges": ["src", "dst"]}
    frames = {"nodes": nodes, "edges": edges}
    rg_mb = None

    def write(layout, d):
        nonlocal rg_mb
        meta = {"part_col": None, "written_blocks": False}
        if layout == "join":
            write_layout(nodes.select("community_id", "id", "is_member", "is_boundary"), f"{d}/p2_map", "range", bins)
            write_layout(edges, f"{d}/p2_edges", "range", bins)
            meta["features"] = {"nodes": f"{raw}/nodes", "masks": f"{raw}/masks"}
        elif layout in ("mv", "hive", "zorder", "zorder_2mb", "liquid", "liquid_2mb"):
            if layout.endswith("_2mb"):   # the initial write and OPTIMIZE's rewrite both take it
                hconf.set("parquet.block.size", str(int(args.zl_rg_mb * (1 << 20))))
            try:
                for name, df in frames.items():
                    out = f"{d}/p2_{name}"
                    if layout.startswith("liquid"):
                        (DeltaTable.create(spark).location(out).addColumns(df.schema)
                         .clusterBy("community_id", *keys[name]).execute())
                        (df.repartitionByRange(bins, "community_id").sortWithinPartitions("community_id")
                         .write.format("delta").mode("append").save(out))
                        spark.sql(f"OPTIMIZE delta.`{out}`")
                        continue
                    write_layout(df, out, "hive" if layout == "hive" else "range", bins)
                    if layout.startswith("zorder"):
                        DeltaTable.forPath(spark, out).optimize().executeZOrderBy("community_id", *keys[name])
            finally:
                hconf.unset("parquet.block.size")
            meta["part_col"] = "community_id" if layout == "hive" else None
        elif layout == "hive_coarse":
            coarse = plan[["community_id"]].assign(
                storage_part=["h%d" % (c % args.coarse_buckets) for c in plan["community_id"].astype(int)])
            for name, df in frames.items():
                write_layout(with_storage_part(spark, df, coarse), f"{d}/p2_{name}", "hive_merged",
                             min(200, args.coarse_buckets), one_file_per_dir=True)
            meta["part_col"] = "storage_part"
            meta["comm_to_part"] = {int(c): p for c, p in zip(coarse["community_id"], coarse["storage_part"])}
            meta["coalesce_plan"] = False   # read with the merged plan's work items, as rows 1-5
        else:   # merged, merged_rg, merged_blk, blk_128
            fr = frames
            if layout in ("merged_blk", "blk_128"):
                n_b, e_b, _ = add_blk(spark, nodes, edges, args.cap)
                fr = {"nodes": n_b, "edges": e_b}
                meta["written_blocks"] = True
            if layout in ("merged_rg", "merged_blk"):
                if rg_mb is None:   # mv written by an earlier run
                    rg_mb = sqrt_rg_mb(sum(storage_stats(spark, f"{wd}/mv/p2_{t}")["bytes"] for t in frames),
                                       written_units, args.rg_beta_mb, args.rg_floor_mb)
                hconf.set("parquet.block.size", str(int(rg_mb * (1 << 20))))
            nb = min(200, max(1, plan["storage_part"].nunique()))
            try:
                for name, df in fr.items():
                    write_layout(with_storage_part(spark, df, plan), f"{d}/p2_{name}", "hive_merged", nb,
                                 sort=None if meta["written_blocks"] else ["storage_part", *keys[name]],
                                 one_file_per_dir=True)
                    if layout in ("merged_blk", "blk_128"):
                        spark.sql(f"ALTER TABLE delta.`{d}/p2_{name}` SET TBLPROPERTIES ('{BLOCK_CAP_PROP}' = '{args.cap}')")
            finally:
                hconf.unset("parquet.block.size")
            meta["part_col"] = "storage_part"
            meta["comm_to_part"] = {int(c): p for c, p in zip(plan["community_id"], plan["storage_part"])}
        return meta

    split_files = {}   # rows 6-8: files per table, must agree
    for layout in args.layouts:
        d = os.path.join(wd, layout)
        walls = []
        for _ in range(args.repeats):
            shutil.rmtree(d, ignore_errors=True)
            t0 = time.perf_counter()
            meta = write(layout, d)
            walls.append(time.perf_counter() - t0)
        tables = sorted(t for t in os.listdir(d) if t.startswith("p2_"))
        st = {t: storage_stats(spark, f"{d}/{t}") for t in tables}
        if layout == "mv":
            rg_mb = sqrt_rg_mb(sum(s["bytes"] for s in st.values()), written_units, args.rg_beta_mb, args.rg_floor_mb)
        meta.update(layout=layout, rg_mb=rg_mb if layout in ("merged_rg", "merged_blk")
                    else args.zl_rg_mb if layout.endswith("_2mb") else None)
        with open(os.path.join(d, "meta.json"), "w") as fh:   # closed: an unflushed meta.json was lost on 10-27
            json.dump(meta, fh)
        emit({"kind": "layout", "layout": layout, "write_s": walls, "storage": st,
              "rg_mb": meta["rg_mb"], "table_dir": d})
        tot = {k: sum(s[k] for s in st.values()) for k in ("bytes", "files", "files_lt_1mb")}
        print(f"  {layout:10s} write {np.mean(walls):7.1f} +/- {np.std(walls, ddof=1) if len(walls) > 1 else 0:5.1f} s"
              f" | {tot['bytes'] / 1e9:6.2f} GB, {tot['files']:,} files ({tot['files_lt_1mb']:,} < 1 MB)"
              + (f" | rg {meta['rg_mb']:g} MB" if meta["rg_mb"] else ""), flush=True)
        if meta["part_col"] == "storage_part" and layout != "hive_coarse":
            split_files[layout] = {t: s["files"] for t, s in st.items()}
            assert len({json.dumps(v) for v in split_files.values()}) == 1, f"rows 6-8 differ in files: {split_files}"
    nodes.unpersist()
    edges.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
