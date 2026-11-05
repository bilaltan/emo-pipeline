#!/usr/bin/env python3
"""
P3, step 2 (docs/trackB_storage_layout.md §6): measure the reuse-aware halo layout against Phase 2's full materialized
halo, with identical node rows per unit. Built from a Track A cache (x, comm, halo); pyarrow only, as Phase 3b's loader.

  full        one table of (id, unit, features), members + every halo row, sorted by (unit, id) (Phase 2 as written)
  hybrid(B)   the same table without the halo vertices of reuse > B; those go once into a hot table (id, features)
              plus a hot index (unit, id). Each of the B reader processes loads the hot table and index once.

Readers are B processes that take the units in Phase 3b's order (size descending), read each unit with a unit predicate
(row-group statistics skip the rest), add its hot rows from the cache, and return (rows, id sum, feature sum) per unit.
These must equal the full layout's. Bytes read are planned from the Parquet metadata: the row groups the predicate
selects, plus B times the hot table and index. Node rows only; edges are read the same way in both layouts.
With --latency-ms every read (unit rows, hot table, index) goes through the S3 emulator of s3_emu_bench (GET =
latency + bytes / --mbps), and the measured GETs and bytes are recorded too.

    bash scripts/wsl_run.sh python experiments/trackB/halo_hybrid_bench.py --dataset reddit-lp --readers 1 2 4
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import statistics
import sys
import threading
import time
import types

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.trackB.s3_emu_bench import emu_fs  # noqa: E402

NO_COMM = np.iinfo(np.int64).min
MB = 1 << 20


def unit_rows(cache):
    """(ids, units) of every unit row (members + halo) sorted by (unit, id), and the reuse of each row's vertex."""
    comm = np.load(os.path.join(cache, "comm.npy"))
    halo = np.load(os.path.join(cache, "halo.npy"))
    mem = np.flatnonzero(comm != NO_COMM)
    ids = np.r_[mem, halo[0]]
    units = np.r_[comm[mem], halo[1]]
    member = np.r_[np.ones(len(mem), bool), np.zeros(halo.shape[1], bool)]
    v, r = np.unique(halo[0], return_counts=True)
    reuse = np.zeros(len(comm), np.int64)
    reuse[v] = r
    o = np.lexsort((ids, units))
    return ids[o], units[o], member[o], reuse


def write_rows(path, x, ids, units, rg_rows, chunk_rgs=64, dictionary=True):
    """Parquet (id, unit, features) in uniform row groups of rg_rows; units=None writes (id, features).
    dictionary: pyarrow's use_dictionary (its default, on, stores float features 1.27-1.29x larger than off)."""
    f = x.shape[1]
    w = None
    for s in range(0, max(len(ids), 1), rg_rows * chunk_rgs):
        i = ids[s:s + rg_rows * chunk_rgs]
        cols = {"id": pa.array(i, pa.int64())}
        if units is not None:
            cols["unit"] = pa.array(units[s:s + rg_rows * chunk_rgs], pa.int64())
        cols["features"] = pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(x[i], np.float32).ravel()), f)
        t = pa.table(cols)
        w = w or pq.ParquetWriter(path, t.schema, use_dictionary=dictionary)
        w.write_table(t, row_group_size=rg_rows)
    w.close()
    return os.path.getsize(path)


def planned_bytes(path, units):
    """Bytes of the row groups whose unit [min, max] contains each unit, summed over units."""
    md = pq.ParquetFile(path).metadata
    ui = [md.schema.column(i).name for i in range(md.num_columns)].index("unit")
    lo, hi, b = [], [], []
    for r in range(md.num_row_groups):
        g = md.row_group(r)
        lo.append(g.column(ui).statistics.min)
        hi.append(g.column(ui).statistics.max)
        b.append(sum(g.column(k).total_compressed_size for k in range(g.num_columns)))
    lo, hi, b = map(np.asarray, (lo, hi, b))
    return int(sum(b[(lo <= u) & (hi >= u)].sum() for u in units))


_HOT, _DS, _IO = {}, {}, {}


def _load_hot(hot_path, idx_path, lat_ms=None, mbps=None):
    _HOT.clear()
    _IO["st"] = st = types.SimpleNamespace(lock=threading.Lock(), gets=0, bytes=0, seen=(0, 0))
    _IO["fs"] = emu_fs(lat_ms, mbps, st) if lat_ms is not None else None
    if hot_path:
        t = pq.read_table(hot_path, filesystem=_IO["fs"])
        _HOT["ids"] = t.column("id").to_numpy()
        _HOT["x"] = t.column("features").combine_chunks().flatten().to_numpy().reshape(len(t), -1)
        ix = pq.read_table(idx_path, filesystem=_IO["fs"])
        _HOT["idx_unit"], _HOT["idx_id"] = ix.column("unit").to_numpy(), ix.column("id").to_numpy()


def _read_unit(args):
    path, u = args
    if path not in _DS:                      # one dataset object per worker, as Phase 3b's loader caches it
        _DS[path] = pds.dataset(path, filesystem=_IO["fs"])
    t = _DS[path].to_table(filter=pds.field("unit") == u, columns=["id", "features"])
    ids = t.column("id").to_numpy()
    x = t.column("features").combine_chunks().flatten().to_numpy().reshape(len(t), -1)
    if _HOT:
        lo, hi = np.searchsorted(_HOT["idx_unit"], [u, u + 1])
        h = _HOT["idx_id"][lo:hi]
        ids = np.r_[ids, h]
        x = np.vstack([x, _HOT["x"][np.searchsorted(_HOT["ids"], h)]])
    o = np.argsort(ids, kind="stable")
    st = _IO["st"]
    io, st.seen = (st.gets - st.seen[0], st.bytes - st.seen[1]), (st.gets, st.bytes)   # first task carries the hot load
    return int(u), (len(ids), int(ids.sum()), float(x[o].sum(dtype=np.float64))), io


def read_all(path, order, B, hot=None, repeats=2, emu=(None, None)):
    """Wall per repeat (each a fresh, cold pool), checksums, and (GETs, bytes) of the last repeat."""
    walls, check = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        with mp.get_context("spawn").Pool(B, initializer=_load_hot, initargs=(*(hot or (None, None)), *emu)) as pool:
            res = pool.map(_read_unit, [(path, int(u)) for u in order], chunksize=1)
        walls.append(time.perf_counter() - t0)
        check = {u: c for u, c, _ in res}
        io = tuple(int(sum(r[2][k] for r in res)) for k in (0, 1))
    return walls, check, io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.expanduser("~/lakegrl/store"))
    ap.add_argument("--dataset", default="ogbn-arxiv-lp")
    ap.add_argument("--alg", default="louvain")
    ap.add_argument("--readers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--rg-mb", type=float, default=1.0, help="row-group size of the unit tables")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--work-dir", default=os.path.expanduser("~/lakegrl/halobench"))
    ap.add_argument("--latency-ms", type=float, default=None, help="read through the S3 emulator (default: local)")
    ap.add_argument("--mbps", type=float, default=80.0, help="emulated bandwidth of one GET")
    ap.add_argument("--out", default="results/trackB/halo_hybrid.jsonl")
    args = ap.parse_args()
    emu = (args.latency_ms, args.mbps)
    cache = os.path.join(args.store, args.dataset, f"trackA_{args.alg}")
    x = np.load(os.path.join(cache, "x.npy"), mmap_mode="r")
    ids, units, member, reuse = unit_rows(cache)
    rg_rows = max(1, int(args.rg_mb * MB / (16 + 4 * x.shape[1])))
    order = [u for u, _ in sorted(zip(*np.unique(units, return_counts=True)), key=lambda p: -p[1])]
    wd = os.path.join(args.work_dir, args.dataset)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    base = {"run_id": run_id, "dataset": args.dataset, "alg": args.alg, "rg_mb": args.rg_mb, "units": len(order),
            "members": int(member.sum()), "halo_rows": int((~member).sum()), "latency_ms": args.latency_ms,
            "mbps": args.mbps if args.latency_ms is not None else None}

    def emit(rec):
        rec = {**base, **rec}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k not in base}), flush=True)

    full = os.path.join(wd, "full.parquet")
    t0 = time.perf_counter()
    stored = write_rows(full, x, ids, units, rg_rows)
    write_s = time.perf_counter() - t0
    ref = None
    for B in args.readers:
        walls, chk, io = read_all(full, order, B, repeats=args.repeats, emu=emu)
        ref = ref or chk
        emit({"layout": "full", "readers": B, "rows_stored": len(ids), "bytes_stored": stored, "write_s": write_s,
              "bytes_read": planned_bytes(full, order), "wall_s": walls, "gets": io[0], "bytes_io": io[1],
              "identical": chk == ref})
    for B in args.readers:
        hot = (~member) & (reuse[ids] > B)
        hv = np.unique(ids[hot])
        p, hp, ip = (os.path.join(wd, f"{k}_B{B}.parquet") for k in ("units", "hot", "hot_index"))
        t0 = time.perf_counter()
        sb = write_rows(p, x, ids[~hot], units[~hot], rg_rows)
        hb = write_rows(hp, x, hv, None, rg_rows) if len(hv) else 0
        o = np.lexsort((ids[hot], units[hot]))
        pq.write_table(pa.table({"unit": units[hot][o], "id": ids[hot][o]}), ip)
        ib = os.path.getsize(ip)
        write_s = time.perf_counter() - t0
        walls, chk, io = read_all(p, order, B, (hp, ip) if len(hv) else None, args.repeats, emu)
        emit({"layout": "hybrid", "readers": B, "tau": B, "hot_vertices": len(hv),
              "rows_stored": int((~hot).sum()) + len(hv), "bytes_stored": sb + hb + ib, "write_s": write_s,
              "bytes_read": planned_bytes(p, order) + B * (hb + ib), "wall_s": walls, "gets": io[0], "bytes_io": io[1],
              "identical": chk == ref})
        for f in (p, hp, ip):
            if os.path.exists(f):
                os.remove(f)
    shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()
