#!/usr/bin/env python3
"""
Table A's halo sub-table (plan: "LakeGRL Table A plan"): 1-hop uncapped vs capped halo on a link-prediction store.

The units and halos are Track A's own (harness.unit_layout: members + every 1-hop neighbour, or at most --budget x
members of them, those with the most edges into the unit), so the AUC runs (harness M2 with and without
--halo-budget) train on exactly these rows. Each variant is written as one (id, unit, features) table sorted by
(unit, id) with the sqrt-rule row group (bounded below by --rg-floor-mb), then every unit is read through the S3
emulator, cold (page cache dropped, fresh readers): storage bytes, bytes read, GETs.

First, the link-leakage rule: no held-out val/test pair (either direction) may be in the cache's training edges, the
store's edge table or its Phase 2 edge table, since Phases 1 and 2 must never see them. The run stops otherwise.

    bash scripts/wsl_run.sh python experiments/tableA/halo_table.py --dataset reddit-lp --alg leiden
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def leakage(data, store, alg):
    """Held-out pairs found in each edge source (must all be 0)."""
    import pyarrow.dataset as ds
    from pipeline.utils.store_io import snapshot
    held = np.unique(np.concatenate([data.edge_key(data.val[0], data.val[1]),
                                     data.edge_key(data.test[0], data.test[1])]))
    out = {"held_out_pairs": int(len(held)),
           "cache_edges": int(np.isin(data.edge_key(data.edges[0], data.edges[1]), held).sum())}
    for name, url in (("edges", f"{store}/edges"), ("p2_edges", f"{store}/p2_edges/{alg}")):
        if not os.path.isdir(url):
            out[name] = None
            continue
        fs, files, _ = snapshot(url)
        t = ds.dataset(files, filesystem=fs, format="parquet").to_table(columns=["src", "dst"])
        out[name] = int(np.isin(data.edge_key(t.column("src").to_numpy(), t.column("dst").to_numpy()), held).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="link-prediction store, e.g. reddit-lp")
    ap.add_argument("--alg", default="leiden")
    ap.add_argument("--budget", type=float, default=1.0, help="capped halo: at most budget x members per unit")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--emu", default="20,80")
    ap.add_argument("--rg-floor-mb", type=float, default=0.0, help="bounded sqrt rule: row group >= this many MB")
    ap.add_argument("--store", default=os.path.expanduser("~/lakegrl/store"))
    ap.add_argument("--work-dir", default=os.path.expanduser("~/lakegrl/tableA_halo"))
    ap.add_argument("--out", default="results/tableA/halo.jsonl")
    args = ap.parse_args()
    from experiments.tableA.read_layouts import drop_caches
    from experiments.trackA.harness import Data, unit_layout
    from experiments.trackB.halo_hybrid_bench import read_all, write_rows
    from pipeline.utils.partition_layout import sqrt_rg_mb
    store = os.path.join(args.store, args.dataset)
    data = Data(os.path.join(store, f"trackA_{args.alg}"))
    leak = leakage(data, store, args.alg)
    print(f"== {args.dataset}/{args.alg}: leakage check {leak}", flush=True)
    assert all(not v for k, v in leak.items() if k != "held_out_pairs"), f"held-out pairs leak into training: {leak}"
    x = np.load(os.path.join(store, f"trackA_{args.alg}", "x.npy"), mmap_mode="r")
    lat, mbps = (float(v) for v in args.emu.split(","))
    wd = os.path.join(args.work_dir, args.dataset)
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for name, budget in (("uncapped", 0.0), ("capped", args.budget)):
        lay = unit_layout(data, True, budget)
        ids = np.concatenate([np.r_[u["members"], u["halo"]] for u in lay.values()]).astype(np.int64)
        units = np.concatenate([np.full(len(u["members"]) + len(u["halo"]), c, np.int64) for c, u in lay.items()])
        o = np.lexsort((ids, units))
        ids, units = ids[o], units[o]
        row_b = 16 + 4 * x.shape[1]
        rg_rows = max(1, int(sqrt_rg_mb(len(ids) * row_b, len(lay), floor_mb=args.rg_floor_mb) * (1 << 20) / row_b))
        path = os.path.join(wd, f"{name}.parquet")
        t0 = time.perf_counter()
        # advisor 2026-11-05: no dictionary encoding, so stored GB compare with Table A (Spark's p2_nodes, ~1.0x raw)
        stored = write_rows(path, x, ids, units, rg_rows, dictionary=False)
        write_s = time.perf_counter() - t0
        order = [c for c, _ in sorted(lay.items(), key=lambda kv: -(len(kv[1]["members"]) + len(kv[1]["halo"])))]
        drop_caches()
        walls, _, (gets, nbytes) = read_all(path, order, args.procs, repeats=1, emu=(lat, mbps))
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "alg": args.alg, "halo": name,
               "budget": budget, "units": len(lay), "members": int(sum(len(u["members"]) for u in lay.values())),
               "halo_rows": int(sum(len(u["halo"]) for u in lay.values())), "rows_stored": int(len(ids)),
               "bytes_stored": stored, "dictionary": False, "rg_floor_mb": args.rg_floor_mb, "rg_rows": rg_rows, "write_s": write_s,
               "read_wall_s": walls[0], "gets": gets, "bytes_read": nbytes, "emu": args.emu, "procs": args.procs,
               "leakage": leak}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {name:8s} {rec['rows_stored']:,} rows ({rec['halo_rows']:,} halo) | stored {stored / 1e9:.2f} GB | "
              f"read {nbytes / 1e9:.2f} GB, {gets:,} GETs, {walls[0]:.1f} s", flush=True)
        os.remove(path)


if __name__ == "__main__":
    main()
