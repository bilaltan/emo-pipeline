#!/usr/bin/env python3
"""
Unit definitions under recursive Leiden at one C_max, measured without Spark: units and rows Phase 2 would store
(members + 1-hop halo) from a dataset's Leiden communities, for the advisor's choice before the Table A reruns.

    idmod         communities uncut, full halo (what the id mod n set stores; Stage 3b cuts at read time)
    rows          members + halo <= C_max (split_units, the current default); hubs' halo capped to C_max
    members-half  members <= C_max / 2, halo cut to C_max - members (most edges into the unit first)
    members       members <= C_max, halo uncapped

The member modes run split_units with an empty adjacency, so |N[S]| = members and the same Leiden recursion and
next-fit packing cap members instead of rows. One record per mode to --out.

    bash scripts/wsl_run.sh python experiments/tableA/probe_units.py reddit --c-max 10000 --out results/tableA/unit_probe.jsonl
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from phases.phase1_community import closed_nbhd, split_units  # noqa: E402
from pipeline.utils.store_io import snapshot  # noqa: E402

MODES = ["idmod", "rows", "members-half", "members"]


def load(url, cols):
    fs, files, _ = snapshot(url)
    t = [pq.read_table(f, filesystem=fs, columns=cols) for f in files]
    return {c: np.concatenate([x.column(c).to_numpy() for x in t]).astype(np.int64) for c in cols}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--c-max", type=int, default=10_000)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/tableA/unit_probe.jsonl")
    args = ap.parse_args()
    store = os.path.expanduser(f"~/lakegrl/store/{args.dataset}")
    t0 = time.time()
    e = load(f"{store}/edges", ["src", "dst"])
    cm = load(f"{store}/communities/leiden", ["id", "community_id"])
    m = e["src"] < e["dst"]   # edges are stored both ways (checked on arxiv and Reddit)
    nid = np.union1d(e["src"][m], e["dst"][m])
    u, v = np.searchsorted(nid, e["src"][m]), np.searchsorted(nid, e["dst"][m])
    del e, m
    n = len(nid)
    s, d = np.r_[u, v], np.r_[v, u]
    indices = d[np.argsort(s, kind="stable")]
    indptr = np.r_[0, np.cumsum(np.bincount(s, minlength=n))].astype(np.int64)
    del s, d
    import igraph as ig
    G = ig.Graph(n=n, edges=np.stack([u, v], 1), directed=False)
    del u, v
    o = np.argsort(cm["id"])
    _, memb0 = np.unique(cm["community_id"][o][np.searchsorted(cm["id"][o], nid)], return_inverse=True)
    print(f"{args.dataset}: {n:,} vertices, {len(indices) // 2:,} edges, {memb0.max() + 1:,} communities "
          f"[{time.time() - t0:.0f} s]", flush=True)
    no_adj = (np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64))   # |N[S]| = members
    C, base = args.c_max, None
    for mode in args.modes:
        t1 = time.time()
        st = {}
        if mode == "idmod":
            memb = memb0
        else:
            cap, adj = (C, (indptr, indices)) if mode == "rows" else (C // 2 if mode == "members-half" else C, no_adj)
            memb, st = split_units(G, *adj, memb0, cap, 1.0, args.seed)
        order, cnt = np.argsort(memb, kind="stable"), np.bincount(memb)
        b = np.r_[0, np.cumsum(cnt)]
        nz = np.nonzero(cnt)[0]
        full = np.array([len(closed_nbhd(indptr, indices, order[b[c]:b[c + 1]])) for c in nz])
        mem = cnt[nz]
        stored = full if mode in ("idmod", "members") else np.minimum(full, C)
        if mode == "rows":   # independent check of split_units: only single vertices may exceed C
            assert np.all(mem[full > C] == 1), "a unit of several vertices is above C_max"
        base = int(stored.sum()) if mode == "idmod" else base
        rec = dict(dataset=args.dataset, c_max=C, mode=mode, seed=args.seed, units=len(nz),
                   units_over_cap=int((full > C).sum()), rows_stored=int(stored.sum()),
                   rows_vs_idmod=round(stored.sum() / base, 3) if base else None,
                   halo_kept=round(float((stored - mem).sum() / max(1, (full - mem).sum())), 4),
                   members_p50=int(np.median(mem)), members_max=int(mem.max()), rows_p50=int(np.median(stored)),
                   rows_max=int(stored.max()), units_lt100_members=int((mem < 100).sum()),
                   seconds=round(time.time() - t1, 1),
                   peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                   spark="none", **{k: st[k] for k in ("split", "leiden_calls", "max_resolution") if k in st})
        print(json.dumps(rec), flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
