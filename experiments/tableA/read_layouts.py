#!/usr/bin/env python3
"""
Table A, step 2: Stage 3b's storage reads of every work item, from each layout written by write_layouts.py, through
the S3 emulator (utils.store_io: every read call on a data file is one GET of --emu latency + bytes / bandwidth,
counted). Reads only: no model, no training.

Each work item is read by Stage 3b's own loader (phase3b_caan._load_communities_data_batch: the same columns,
community / block predicates, statistics pruning and _blk ranges). Row 1 (join) reads its unit's p2_map rows the same
way, then fetches those ids' features and split from the raw node and mask tables (row groups pruned by id statistics).

A run is cold: the page cache is dropped (as root) and fresh reader processes start (no cached footers or datasets).
--runs runs per layout (seeds 0..runs-1 label them; reads are deterministic). Per run: wall time, GETs, bytes, rows,
and an md5 of the per-item checksums (rows, id sum, edges), which must agree across layouts.

    bash scripts/wsl_run.sh python experiments/tableA/read_layouts.py --dataset ogbn-arxiv
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

LAYOUTS = ("join", "mv", "hive", "zorder", "liquid", "merged", "merged_rg", "merged_blk")


def drop_caches():
    """Empty the OS page cache, so every run reads from disk (cold). Needs root: sudo -n, else WSL's own wsl.exe."""
    cmd = "sync; echo 3 > /proc/sys/vm/drop_caches"
    for argv in (["sudo", "-n", "sh", "-c", cmd],
                 ["wsl.exe", "-d", os.environ.get("WSL_DISTRO_NAME", "Ubuntu-24.04"), "-u", "root", "-e", "sh", "-c", cmd]):
        try:
            if subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120).returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    raise RuntimeError("cannot drop the page cache (no passwordless sudo, no wsl.exe): reads would not be cold")


def spec(url, emu):
    """utils.partition_layout.table_layout()'s per-table spec, without Spark: the snapshot's files, the partition
    column from the paths."""
    from pipeline.utils.store_io import snapshot
    _, files, _ = snapshot(url)
    seg = files[0][len(url) + 1:].split("/")[0] if files else ""
    part = seg.split("=")[0] if "=" in seg else None
    return {"url": url, "files": files, "part_col": part,
            "part_type": None if part is None else "string" if part == "storage_part" else "int64", "emu": emu}


_W = {}


def _init(layout_spec, join_src, ready):
    _W["lay"], _W["join"] = layout_spec, join_src
    import pipeline.phases.phase3b_caan  # noqa: F401  (imports torch: start-up stays out of the timed reads)
    ready.wait()


def _by_ids(sp, ids, cols):
    """Rows of a raw table (sorted by id) for the given ids, reading only the row groups whose id range holds one."""
    import pyarrow.dataset as ds
    from pipeline.phases.phase3b_caan import _layout_dataset
    from pipeline.utils.partition_layout import unit_dataset
    d = unit_dataset(_layout_dataset(sp), ids, ds.scalar(True), col="id")
    return d.to_table(columns=cols, filter=ds.field("id").isin(ids))


def _read(item):
    """One work item as Stage 3b loads it: (key, (node rows, id sum, edges), GETs, bytes)."""
    from pipeline.phases.phase3b_caan import _load_communities_data_batch
    from pipeline.utils import store_io
    c, n, w = item
    g0, b0 = store_io.STATS["gets"], store_io.STATS["bytes"]
    out = []
    for b in range(n):
        nodes, edges = _load_communities_data_batch(None, None, [c], block=b, n_blocks=n, layout=_W["lay"],
                                                    n_wblocks=w)
        if _W["join"] is not None and len(nodes):     # row 1: the features and split of the unit's rows, by id
            ids = np.unique(nodes["id"].to_numpy(np.int64)).tolist()
            f = _by_ids(_W["join"]["nodes"], ids, ["id", "label", "features"])
            m = _by_ids(_W["join"]["masks"], ids, ["id", "split"])
            f.column("features").combine_chunks().flatten().to_numpy(zero_copy_only=False)
            assert f.num_rows == len(ids) and m.num_rows == len(ids), (c, b, f.num_rows, m.num_rows, len(ids))
        elif len(nodes):
            np.concatenate(nodes["features"].to_numpy())     # decoded, as training would use it
        out.append(((c, b), (len(nodes), int(nodes["id"].sum()) if len(nodes) else 0, len(edges))))
    return out, store_io.STATS["gets"] - g0, store_io.STATS["bytes"] - b0


def run(lay, join_src, items, procs):
    ctx = mp.get_context("spawn")
    ready = ctx.Barrier(procs + 1)
    with ctx.Pool(procs, initializer=_init, initargs=(lay, join_src, ready)) as pool:
        ready.wait()
        t0 = time.perf_counter()
        res = pool.map(_read, items, chunksize=1)
        wall = time.perf_counter() - t0
    check = sorted((k, v) for r, _, _ in res for k, v in r)
    return wall, sum(r[1] for r in res), sum(r[2] for r in res), check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=LAYOUTS)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--procs", type=int, default=os.cpu_count(), help="reader processes (Stage 3b's task slots)")
    ap.add_argument("--emu", default="20,80", help="utils.store_io.parse_emu: 'LAT_MS,MBPS' per GET, or 'count'")
    ap.add_argument("--no-drop-caches", action="store_true", help="warm runs (tests only)")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/lakegrl/tableA"))
    ap.add_argument("--out", default="results/tableA/read.jsonl")
    args = ap.parse_args()
    from pipeline.utils.store_io import parse_emu
    emu = parse_emu(args.emu)
    wd = os.path.join(args.out_dir, args.dataset)
    man = json.load(open(os.path.join(wd, "manifest.json")))
    items = [tuple(int(v) for v in it) for it in man["items"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"== {args.dataset}: {len(items):,} communities, {sum(n for _, n, _ in items):,} work items, "
          f"{args.procs} readers, GET = {args.emu}", flush=True)
    for layout in args.layouts:
        d = os.path.join(wd, layout)
        meta = json.load(open(os.path.join(d, "meta.json")))
        lay = {"nodes": spec(f"{d}/p2_map" if layout == "join" else f"{d}/p2_nodes", emu),
               "edges": spec(f"{d}/p2_edges", emu),
               "comm_to_part": {int(k): v for k, v in meta["comm_to_part"].items()} if meta.get("comm_to_part") else None}
        lay["layout"] = {None: "range", "community_id": "hive", "storage_part": "hive_merged"}[lay["nodes"]["part_col"]]
        join_src = {k: spec(u, emu) for k, u in meta["features"].items()} if layout == "join" else None
        its = [(c, n, w if meta["written_blocks"] else 0) for c, n, w in items]
        for seed in range(args.runs):
            if not args.no_drop_caches:
                drop_caches()
            wall, gets, nbytes, check = run(lay, join_src, its, args.procs)
            rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "layout": layout, "seed": seed,
                   "cold": not args.no_drop_caches, "emu": args.emu, "procs": args.procs, "wall_s": wall,
                   "gets": gets, "bytes": nbytes, "items": len(check),
                   "node_rows": sum(v[0] for _, v in check), "edge_rows": sum(v[2] for _, v in check),
                   "check_md5": hashlib.md5(json.dumps(check).encode()).hexdigest()}
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  {layout:10s} run {seed}: {wall:8.1f} s {gets:>9,} GETs {nbytes / 1e9:8.2f} GB | "
                  f"{rec['node_rows']:,} node rows, {rec['edge_rows']:,} edges | {rec['check_md5'][:8]}", flush=True)


if __name__ == "__main__":
    main()
