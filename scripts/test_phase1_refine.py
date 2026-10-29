#!/usr/bin/env python3
"""
Phase 1 refinement (phases/phase1_community.refine_communities) on a planted graph: one community of four dense
clusters (300 vertices each, a few edges between) and one small community. With a 400-member cap the large one
splits along its clusters, every piece is <= 400, and the small community keeps its id.
split_units (recursive Leiden, cap on members + 1-hop halo) on the same graph plus a hub of 500 neighbours: every unit
of two or more vertices has <= 400 rows, the hub is the only unit above, hub-free clusters stay whole.

    bash scripts/wsl_run.sh python scripts/test_phase1_refine.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phases.phase1_community import closed_nbhd, refine_communities, split_units  # noqa: E402


def main():
    import igraph as ig
    ig.set_random_number_generator(random.Random(0))
    rng = np.random.default_rng(0)
    k, size, small = 4, 300, 50
    n = k * size + small
    edges = set()
    for c in range(k):
        base = c * size
        for _ in range(size * 8):   # dense inside a cluster
            a, b = rng.integers(0, size, 2)
            if a != b:
                edges.add((base + int(min(a, b)), base + int(max(a, b))))
    for _ in range(20):             # sparse between clusters
        a, b = rng.integers(0, k * size, 2)
        if a // size != b // size:
            edges.add((int(min(a, b)), int(max(a, b))))
    for _ in range(small * 3):
        a, b = rng.integers(k * size, n, 2)
        if a != b:
            edges.add((int(min(a, b)), int(max(a, b))))
    G = ig.Graph(n=n, edges=sorted(edges))
    memb = np.array([0] * (k * size) + [1] * small)

    out = refine_communities(G, memb, 400)
    sizes = np.bincount(out)
    assert sizes.max() <= 400, sizes
    assert sizes.sum() == n
    assert (out[k * size:] == 1).all(), "the small community must keep its id"
    for c in range(k):              # each planted cluster lands in one piece
        ids = np.unique(out[c * size:(c + 1) * size])
        assert len(ids) == 1, (c, ids)
    print(f"ok: 1 community of {k * size} -> {len(np.unique(out[:k * size]))} pieces of <= 400, "
          f"clusters intact, small community unchanged")

    hub = 0   # in cluster 0, 500 neighbours anywhere
    edges |= {(hub, int(v)) for v in rng.choice(np.arange(1, n), 500, replace=False)}
    G = ig.Graph(n=n, edges=sorted(edges))
    el = np.asarray(G.get_edgelist())
    s, d = np.r_[el[:, 0], el[:, 1]], np.r_[el[:, 1], el[:, 0]]
    indices = d[np.argsort(s, kind="stable")]
    indptr = np.r_[0, np.cumsum(np.bincount(s, minlength=n))]
    out, st = split_units(G, indptr, indices, memb, 400, 1.0, 0)
    assert (out[k * size:] == 1).all(), "the small community must keep its id"
    for u in np.unique(out):
        vs = np.nonzero(out == u)[0]
        rows = len(closed_nbhd(indptr, indices, vs))
        assert rows <= 400 or (len(vs) == 1 and vs[0] == hub), (u, len(vs), rows)
    assert st["hub_units"] == 1 and st["max_unit_rows"] <= 400, st
    for c in range(1, k):
        assert len(np.unique(out[c * size:(c + 1) * size])) == 1, c
    print(f"ok: split_units, cap 400 rows: {st['units_before']} -> {st['units']} units, max {st['max_unit_rows']} "
          f"rows, 1 hub unit ({st['max_hub_rows']} rows), hub-free clusters intact")


if __name__ == "__main__":
    main()
