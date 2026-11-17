#!/usr/bin/env python3
"""
Units and the rows Phase 2 would store (members + 1-hop halo) from a dataset's Leiden communities, without Spark:

    idmod  communities uncut, full halo (what the id mod n set stores; Stage 3b cuts at read time)
    core   Phase 1's rule: recursive Leiden to <= C_max / 2 core members per unit
           (phase1_community.split_units), Phase 2's halo cap min(members, C_max - members), so <= C_max rows

One record per (C_max, mode) to --out.

    bash scripts/wsl_run.sh python experiments/tableA/probe_units.py reddit --c-max 10000 20000
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


def load(url, cols):
    fs, files, _ = snapshot(url)
    t = [pq.read_table(f, filesystem=fs, columns=cols) for f in files]
    return {c: np.concatenate([x.column(c).to_numpy() for x in t]).astype(np.int64) for c in cols}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--c-max", type=int, nargs="+", default=[10_000])
    ap.add_argument("--seed", type=int, default=42, help="Phase 1's random_seed (run_local_pipeline passes 42)")
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
    base = None
    for C in args.c_max:
        for mode in ("idmod", "core"):
            if mode == "idmod" and base is not None:
                continue
            t1 = time.time()
            memb, st = (memb0, {}) if mode == "idmod" else split_units(G, memb0, C // 2, 1.0, args.seed)
            order, cnt = np.argsort(memb, kind="stable"), np.bincount(memb)
            b = np.r_[0, np.cumsum(cnt)]
            nz = np.nonzero(cnt)[0]
            mem = cnt[nz]
            halo = np.array([len(closed_nbhd(indptr, indices, order[b[c]:b[c + 1]])) for c in nz]) - mem
            kept = halo if mode == "idmod" else np.minimum(halo, np.minimum(mem, C - mem))
            rows = mem + kept
            if mode == "core":
                assert mem.max() <= C // 2 and rows.max() <= C, (mem.max(), rows.max())
            else:
                base = int(rows.sum())
            rec = dict(dataset=args.dataset, c_max=C if mode == "core" else None, mode=mode, seed=args.seed,
                       units=len(nz), rows_stored=int(rows.sum()), rows_vs_idmod=round(rows.sum() / base, 3),
                       halo_kept=round(float(kept.sum() / max(1, halo.sum())), 4),
                       units_halo_capped=int((kept < halo).sum()),
                       core_p50=int(np.median(mem)), core_max=int(mem.max()), rows_p50=int(np.median(rows)),
                       rows_max=int(rows.max()), units_ge_1000_core=int((mem >= 1000).sum()),
                       units_lt100_core=int((mem < 100).sum()), seconds=round(time.time() - t1, 1),
                       peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                       spark="none", **{k: st[k] for k in ("split", "leiden_calls", "max_resolution") if k in st})
            print(json.dumps(rec), flush=True)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
