#!/usr/bin/env python3
"""
Table B (advisor 2026-11-05, CaaN Eq. 2): what each major unit's local graph holds in row 5 (Stage 3b, context
'macro') against row 3 (Stage 3a, context 'none'), counted from the stored tables without training. Read-only.

The counting follows phase3b_caan's unit function: local rows = the unit's p2_nodes rows (members + capped halo);
a stored edge's far end in another major community becomes that community's super-node (-1000 - c); every super-node
but the unit's own joins; minor nodes reached by an edge and not already local rows join as context; meta-edges
(p2_macro_edges) among the joined super-nodes and context minor nodes, none touching the own super-node. Row 3 keeps
member-member edges only. Per unit: nodes and directed edges by kind (the stage's n_edges) and message-passing edges.

    bash scripts/wsl_run.sh python experiments/tableB/count_context.py --dataset ogbn-products
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.dataset as pds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.utils.macro_graph import read_small  # noqa: E402
from pipeline.utils.store_io import snapshot  # noqa: E402


def table(uri, cols):
    fs, files, _ = snapshot(uri)
    return pds.dataset(files, filesystem=fs, format="parquet").to_table(columns=cols).to_pandas()


def upairs(a, b, off, m):
    """Distinct (a, b) pairs, as np.unique(axis=0) on one int64 key (a + off and b + off in [0, m))."""
    k = np.unique((a + off) * m + (b + off))
    return np.stack([k // m - off, k % m - off], 1)


def mp(pairs, off, m):
    """Message-passing edges: both directions, deduplicated (common.undirected)."""
    return len(upairs(np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]], off, m))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="leiden_c10000")
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--store", default=os.environ.get("LAKEGRL_STORE", os.path.expanduser("~/lakegrl/store")))
    ap.add_argument("--out", default="results/caan_context/units.jsonl")
    args = ap.parse_args()
    s = os.path.join(args.store, args.dataset)
    t0 = time.time()
    comm = read_small(f"{s}/communities/{args.tag}")[["id", "community_id"]]
    sizes = comm.groupby("community_id").size()
    major = np.sort(sizes.index[sizes >= args.min_size].to_numpy(np.int64))
    lut = np.full(int(comm["id"].max()) + 1, -1, np.int64)
    lut[comm["id"].to_numpy(np.int64)] = comm["community_id"].to_numpy(np.int64)
    mn = read_small(f"{s}/p2_macro_nodes/{args.tag}")[["macro_id"]]
    minor = np.sort(mn["macro_id"].to_numpy(np.int64)[mn["macro_id"].to_numpy() >= 0])
    me = read_small(f"{s}/p2_macro_edges/{args.tag}")
    msrc, mdst = me["src"].to_numpy(np.int64), me["dst"].to_numpy(np.int64)
    assert len(mn) - len(minor) == len(major), "stored super-nodes != major communities"
    nodes = table(f"{s}/p2_nodes/{args.tag}", ["id", "community_id", "is_member"])
    edges = table(f"{s}/p2_edges/{args.tag}", ["src", "dst", "community_id"])
    print(f"read in {time.time() - t0:.0f} s: {len(major)} major, {len(minor):,} minor nodes, {len(me):,} macro edges, "
          f"{len(nodes):,} p2_nodes rows, {len(edges):,} p2_edges rows", flush=True)
    nodes = nodes[nodes["community_id"].isin(major)]
    ng = {c: g for c, g in nodes.groupby("community_id")}
    eg = {c: g for c, g in edges[edges["community_id"].isin(major)].groupby("community_id")}
    sup_all = -1000 - major
    off = 1000 + int(major.max()) + 1          # super-node ids >= -off + 1
    m = len(lut) + off
    assert m * m < 2 ** 62

    recs = []
    for c in major:
        g = ng[c].drop_duplicates("id")   # members first is not needed: the unit is one community
        loc = np.sort(g["id"].to_numpy(np.int64))
        mem = np.sort(g.loc[g["is_member"].astype(bool), "id"].to_numpy(np.int64))
        e = eg.get(c)
        src = e["src"].to_numpy(np.int64) if e is not None else np.zeros(0, np.int64)
        dst = e["dst"].to_numpy(np.int64) if e is not None else np.zeros(0, np.int64)
        wc = np.where((dst >= 0) & (dst < len(lut)), lut[np.clip(dst, 0, len(lut) - 1)], -1)
        coll = (wc != c) & (wc >= 0) & np.isin(wc, major)
        mapped = np.where(coll, -1000 - wc, dst)
        u = np.unique(mapped[mapped >= 0])
        ctx_minor = u[np.isin(u, minor) & ~np.isin(u, loc)]
        sup = sup_all[sup_all != -1000 - c]
        valid = np.r_[sup, ctx_minor]
        known = np.r_[loc, valid]
        keep = np.isin(src, loc) & np.isin(mapped, known)
        # local-edge pairs, unique as the stage's np.unique over (src_idx, dst_idx)
        pe = upairs(src[keep], mapped[keep], off, m)
        sel = np.isin(msrc, valid) & np.isin(mdst, valid) & (mdst != -1000 - c)
        pm = upairs(msrc[sel], mdst[sel], off, m)
        pairs = upairs(np.r_[pe[:, 0], pm[:, 0]], np.r_[pe[:, 1], pm[:, 1]], off, m)
        d = pairs[:, 1]
        s_is_mem = np.isin(pairs[:, 0], mem)
        kind = {"member_member": int((s_is_mem & np.isin(d, mem)).sum()),
                "member_super": int((s_is_mem & (d <= -1000)).sum()),
                "member_minor_halo": int((s_is_mem & np.isin(d, loc) & ~np.isin(d, mem) & np.isin(d, minor)).sum()),
                "member_minor_ctx": int((s_is_mem & np.isin(d, ctx_minor)).sum()),
                "member_other_halo": int((s_is_mem & np.isin(d, loc) & ~np.isin(d, mem) & ~np.isin(d, minor)).sum()),
                "from_non_member": int((~s_is_mem & np.isin(pairs[:, 0], loc)).sum()),
                "meta_super_super": int(((pairs[:, 0] <= -1000) & (d <= -1000)).sum()),
                "meta_with_minor": int((np.isin(pairs[:, 0], valid) & np.isin(d, valid)
                                        & ((pairs[:, 0] >= 0) | (d >= 0))).sum())}
        assert sum(kind.values()) == len(pairs), (c, kind, len(pairs))
        if len(recs) % 100 == 0:
            print(f"  {len(recs)} units, {time.time() - t0:.0f} s", flush=True)
        halo = loc[~np.isin(loc, mem)]
        touched = np.unique(np.r_[pairs[:, 0], d])
        # row 3 (context none): edges into members only, both ends local rows
        k3 = np.isin(dst, mem) & np.isin(src, loc)
        r3 = upairs(src[k3], dst[k3], off, m)
        recs.append({"community_id": int(c), "members": len(mem), "halo_rows": len(halo),
                     "halo_minor": int(np.isin(halo, minor).sum()),
                     "halo_isolated": int((~np.isin(halo, touched)).sum()),
                     "super_nodes": len(sup), "ctx_minor": len(ctx_minor), "nodes_row5": len(loc) + len(valid),
                     "edges_row5": len(pairs), "mp_edges_row5": mp(pairs, off, m), **kind,
                     "nodes_row3": len(loc), "edges_row3": len(r3), "mp_edges_row3": mp(r3, off, m)})

    tot = {k: int(sum(r[k] for r in recs)) for k in recs[0] if k != "community_id"}
    mean = {k: round(v / len(recs), 1) for k, v in tot.items()}
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "tag": args.tag,
           "units": len(recs), "minor_nodes": len(minor), "macro_edges": len(me), "total": tot, "mean": mean,
           "units_with_ctx_minor": sum(r["ctx_minor"] > 0 for r in recs),
           "units_with_halo_minor": sum(r["halo_minor"] > 0 for r in recs),
           "max_halo_minor": max(r["halo_minor"] for r in recs), "s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    with open(args.out.replace(".jsonl", f"_{args.dataset}_per_unit.jsonl"), "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
