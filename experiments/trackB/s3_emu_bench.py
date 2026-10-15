#!/usr/bin/env python3
"""
Phase 3b's per-unit reads against an emulated object store (Track B P7: what the read savings are worth off local disk).

Local disk plus the page cache hides bytes and requests (P3: 0.35x the bytes, same wall time). Here every read call
on a data file is one GET: it is counted and pays --latency-ms plus its bytes at --mbps, like a range GET to S3
from the same region. pyarrow issues the same calls it would issue to S3 (footer, then column-chunk ranges), so the
request counts are pyarrow's own; latency and bandwidth are parameters, not measurements of S3. No outside service
is used. A real S3 / MinIO check needs Bilal's OK.

Input: Phase 2 tables kept by the layout bench (--keep-tables): <table-dir>/p2_nodes, <table-dir>/p2_edges. Units and
Phase 3b bins are derived from the table. Per mode (unit, p3b_bins, and *_pruned = utils.partition_layout.unit_dataset)
it records the cold pass (every process starts without footers) and the warm passes (footers cached in each worker,
as Spark's reused Python workers keep them), GETs, bytes, and per-unit checksums against a plain local read.

    python experiments/trackB/s3_emu_bench.py --table-dir ~/lakegrl/layoutbench/ogbn-arxiv-lp/misc/range_rg \\
        --label arxiv/range_rg@0.44 --out results/trackB/s3_emu.jsonl
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from urllib.parse import unquote

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


# ------------------------------------------------------------------ emulated store
class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.gets = 0
        self.bytes = 0


class _SlowFile:
    """A local file whose every read is one GET: counted, and delayed by latency + bytes / bandwidth."""

    def __init__(self, path, lat_s, bps, st):
        self.f, self.lat, self.bps, self.st = open(path, "rb"), lat_s, bps, st
        self.closed = False

    def read(self, n=-1):
        data = self.f.read(n)
        with self.st.lock:
            self.st.gets += 1
            self.st.bytes += len(data)
        time.sleep(self.lat + len(data) / self.bps)       # releases the GIL: concurrent GETs overlap
        return data

    def seek(self, off, whence=0):
        return self.f.seek(off, whence)

    def tell(self):
        return self.f.tell()

    def close(self):
        self.f.close()
        self.closed = True

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False


def emu_fs(lat_ms, mbps, st):
    import pyarrow as pa
    import pyarrow.fs as pafs
    local = pafs.LocalFileSystem()

    class Handler(pafs.FileSystemHandler):
        def get_type_name(self):
            return "s3emu"

        def normalize_path(self, path):
            return path

        def equals(self, other):
            return self is other

        def get_file_info(self, paths):
            return local.get_file_info(paths)

        def get_file_info_selector(self, selector):
            return local.get_file_info(selector)

        def open_input_file(self, path):
            return pa.PythonFile(_SlowFile(path, lat_ms / 1e3, mbps * 1e6, st), mode="r")

        def open_input_stream(self, path):
            return self.open_input_file(path)

        def _ro(self, *a, **k):
            raise OSError("read-only emulated store")
        create_dir = delete_dir = delete_dir_contents = delete_root_dir_contents = delete_file = _ro
        move = copy_file = open_output_stream = open_append_stream = _ro

    return pafs.PyFileSystem(Handler())


# ------------------------------------------------------------------ table description (pyarrow only)
def delta_files(table):
    """Data files of the latest version, from the JSON commits (the bench writes each table once: no checkpoint)."""
    log = os.path.join(table, "_delta_log")
    assert not any(f.endswith(".checkpoint.parquet") for f in os.listdir(log)), "checkpointed table: not supported"
    live = {}
    for fn in sorted(f for f in os.listdir(log) if f.endswith(".json")):
        for line in open(os.path.join(log, fn)):
            a = json.loads(line)
            if "add" in a:
                live[a["add"]["path"]] = True
            elif "remove" in a:
                live.pop(a["remove"]["path"], None)
    return sorted(os.path.join(table, unquote(p)) for p in live)


def describe(table_dir):
    """{'nodes'|'edges': (files, part_col)}, community -> storage_part, unit sizes (rows)."""
    import pyarrow.dataset as ds
    out = {}
    for name in ("nodes", "edges"):
        files = delta_files(os.path.join(table_dir, f"p2_{name}"))
        rel = files[0][len(os.path.join(table_dir, f"p2_{name}")) + 1:]
        part = rel.split("=")[0] if "=" in rel.split("/")[0] else None
        out[name] = (files, part)
    files, part = out["nodes"]
    d = ds.dataset(files, format="parquet", partitioning="hive" if part else None,
                   partition_base_dir=os.path.join(table_dir, "p2_nodes") if part else None)
    cols = ["community_id"] + ([part] if part and part != "community_id" else [])
    t = d.to_table(columns=cols).to_pandas()
    rows = t.groupby("community_id").size()
    c2p = None
    if part and part != "community_id":
        c2p = {int(c): p for c, p in t.drop_duplicates("community_id")[["community_id", part]].itertuples(index=False)}
    return out, c2p, rows


def phase3b_bins(rows, parallelism):
    """As run_phase3b: communities by size desc, bin = i % num_bins."""
    order = rows.sort_values(ascending=False).index.astype(int).tolist()
    n = len(order)
    nb = n if n <= 2000 else min(max(parallelism * 4, 1000), n)
    bins = [[] for _ in range(nb)]
    for i, c in enumerate(order):
        bins[i % nb].append(c)
    return bins


# ------------------------------------------------------------------ worker
_W = {}


def _init(table_dir, desc, c2p, lat_ms, mbps):
    import pyarrow.dataset as ds
    _W["st"] = _Stats()
    fs = emu_fs(lat_ms, mbps, _W["st"]) if lat_ms is not None else None
    _W["c2p"] = c2p
    for name, (files, part) in desc.items():
        base = os.path.join(table_dir, f"p2_{name}")
        _W[name] = (ds.dataset(files, format="parquet", filesystem=fs, partitioning="hive" if part else None,
                               partition_base_dir=base if part else None), part)


def _load(args):
    import pyarrow.dataset as ds
    from utils.partition_layout import unit_dataset
    cs, prune = args
    st = _W["st"]
    g0, b0 = st.gets, st.bytes
    res = {int(c): [0, 0, 0] for c in cs}
    for name, cols in (("nodes", ["id", "features", "community_id"]), ("edges", ["src", "dst", "community_id"])):
        d, part = _W[name]
        f = ds.field("community_id").isin(cs)
        if part and part != "community_id":
            f = ds.field(part).isin(sorted({_W["c2p"][c] for c in cs})) & f
        if prune:
            d = unit_dataset(d, cs)
        t = d.to_table(columns=cols, filter=f)
        cid = t.column("community_id").to_numpy()
        if name == "nodes":
            t.column("features").combine_chunks().flatten().to_numpy(zero_copy_only=False)
            ids = t.column("id").to_numpy()
            for c in np.unique(cid):
                sel = cid == c
                res[int(c)][0], res[int(c)][1] = int(sel.sum()), int(ids[sel].sum())
        else:
            for c, k in zip(*np.unique(cid, return_counts=True)):
                res[int(c)][2] = int(k)
    return {c: tuple(v) for c, v in res.items()}, st.gets - g0, st.bytes - b0


def run_mode(table_dir, desc, c2p, groups, prune, lat_ms, mbps, procs, passes):
    """passes[0] is cold (fresh workers), the rest warm. Returns per pass (wall, gets, bytes) and the checksums."""
    ctx = mp.get_context("spawn")
    out, check = [], None
    with ctx.Pool(procs, initializer=_init, initargs=(table_dir, desc, c2p, lat_ms, mbps)) as pool:
        tasks = [([int(c) for c in g], prune) for g in groups]
        for _ in range(passes):
            t0 = time.perf_counter()
            res = pool.map(_load, tasks, chunksize=max(1, len(tasks) // (procs * 8)))
            wall = time.perf_counter() - t0
            check = {}
            for chk, _, _ in res:
                check.update(chk)
            out.append((wall, sum(r[1] for r in res), sum(r[2] for r in res)))
    return out, check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--latency-ms", type=float, default=20.0)
    ap.add_argument("--mbps", type=float, default=80.0, help="bandwidth of one GET")
    ap.add_argument("--procs", type=int, default=4, help="reader processes (Phase 3b workers)")
    ap.add_argument("--passes", type=int, default=2, help="1 cold + warm passes")
    ap.add_argument("--modes", nargs="+", default=["unit", "p3b_bins", "unit_pruned", "p3b_bins_pruned"])
    ap.add_argument("--out", default="results/trackB/s3_emu.jsonl")
    args = ap.parse_args()
    td = os.path.abspath(os.path.expanduser(args.table_dir))
    desc, c2p, rows = describe(td)
    groupings = {"unit": [[int(c)] for c in rows.index], "p3b_bins": phase3b_bins(rows, args.procs)}
    _, truth = run_mode(td, desc, c2p, groupings["unit"], False, None, None, args.procs, 1)   # plain local read
    nbytes = sum(os.path.getsize(f) for files, _ in desc.values() for f in files)
    print(f"== {args.label}: {len(rows)} units, {len(groupings['p3b_bins'])} bins, "
          f"{len(desc['nodes'][0])} + {len(desc['edges'][0])} files, {nbytes / 1e6:.1f} MB; "
          f"GET = {args.latency_ms:g} ms + bytes at {args.mbps:g} MB/s", flush=True)
    for mode in args.modes:
        base = mode.replace("_pruned", "")
        passes, check = run_mode(td, desc, c2p, groupings[base], mode.endswith("_pruned"), args.latency_ms, args.mbps,
                                 args.procs, args.passes)
        rec = {"label": args.label, "table_dir": td, "mode": mode, "units": len(rows), "tasks": len(groupings[base]),
               "files": len(desc["nodes"][0]) + len(desc["edges"][0]), "table_bytes": nbytes,
               "latency_ms": args.latency_ms, "mbps": args.mbps, "procs": args.procs,
               "cold": dict(zip(("wall_s", "gets", "bytes"), passes[0])),
               "warm": [dict(zip(("wall_s", "gets", "bytes"), p)) for p in passes[1:]],
               "identical": check == truth}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        w = rec["warm"][-1] if rec["warm"] else rec["cold"]
        print(f"  {mode:16s} cold {rec['cold']['wall_s']:7.2f}s {rec['cold']['gets']:>8,} GETs "
              f"{rec['cold']['bytes'] / 1e6:9.1f} MB | warm {w['wall_s']:7.2f}s {w['gets']:>8,} GETs "
              f"{w['bytes'] / 1e6:9.1f} MB | identical {rec['identical']}", flush=True)


if __name__ == "__main__":
    main()
