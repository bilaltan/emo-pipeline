#!/usr/bin/env python3
"""
Table B, rows 1-2 (full-graph GraphSAGE, Cluster-GCN): their storage read, one cold read of the whole raw node, edge
and mask tables through the S3 emulator (advisor 2026-11-02). They have no units, so every data file of the current
snapshots is one task, read whole by pyarrow (utils.store_io: one GET of --emu latency + bytes / bandwidth per read
call) with the features decoded, by --procs reader processes as Table A reads Stage 3b's bins. Reads only.
Rows 4-5 (--row, --tables): what Stage 3b reads outside the unit bins, read the same way. Since 2026-11-04 (advisor)
that is the macro-graph Phase 2 stores, the communities table (the units map cut edges to super-nodes by it) and
p2_nodes' community_id column (the unit plan); a table given as TABLE:COL,COL is read with those columns only.

    bash scripts/wsl_run.sh python experiments/tableB/read_full.py --dataset ogbn-products
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "tableA"))

from read_layouts import drop_caches  # noqa: E402

TABLES = ("nodes", "edges", "masks")
_W = {}


def _init(emu, ready):
    _W["emu"] = emu
    import pyarrow.parquet  # noqa: F401  (start-up stays out of the timed reads)
    ready.wait()


def _read(item):
    import pyarrow.parquet as pq
    from pipeline.utils import store_io
    path, columns = item
    g0, b0 = store_io.STATS["gets"], store_io.STATS["bytes"]
    t = pq.read_table(path, columns=columns, filesystem=store_io.filesystem(path, _W["emu"]))
    if "features" in t.column_names:
        t.column("features").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return t.num_rows, store_io.STATS["gets"] - g0, store_io.STATS["bytes"] - b0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--store", default=os.path.expanduser("~/lakegrl/store"))
    ap.add_argument("--tables", nargs="+", default=list(TABLES), help="Delta tables under STORE/DATASET[:COL,COL]")
    ap.add_argument("--row", default="1-2", help="Table B row label")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--procs", type=int, default=20, help="reader processes (Table A: Stage 3b's 20 task slots)")
    ap.add_argument("--emu", default="20,80", help="utils.store_io.parse_emu: 'LAT_MS,MBPS' per GET, or 'count'")
    ap.add_argument("--out", default="results/tableB/read_full.jsonl")
    args = ap.parse_args()
    from pipeline.utils.store_io import parse_emu, snapshot
    emu = parse_emu(args.emu)
    files, versions = [], {}
    for t in args.tables:
        t, _, cols = t.partition(":")
        _, paths, versions[t] = snapshot(os.path.join(args.store, args.dataset, t))
        files += [(f, cols.split(",") if cols else None) for f in paths]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"== {args.dataset}: {len(files)} files, {args.procs} readers, GET = {args.emu}", flush=True)
    ctx = mp.get_context("spawn")
    for seed in range(args.runs):
        drop_caches()
        ready = ctx.Barrier(args.procs + 1)
        with ctx.Pool(args.procs, initializer=_init, initargs=(emu, ready)) as pool:
            ready.wait()
            t0 = time.perf_counter()
            res = pool.map(_read, files, chunksize=1)
            wall = time.perf_counter() - t0
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "row": args.row,
               "tables": args.tables, "versions": versions, "seed": seed, "cold": True, "emu": args.emu, "procs": args.procs,
               "files": len(files), "wall_s": wall, "rows": sum(r[0] for r in res),
               "gets": sum(r[1] for r in res), "bytes": sum(r[2] for r in res)}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  run {seed}: {wall:8.1f} s {rec['gets']:>9,} GETs {rec['bytes'] / 1e9:8.2f} GB {rec['rows']:,} rows",
              flush=True)


if __name__ == "__main__":
    main()
