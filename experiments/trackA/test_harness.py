#!/usr/bin/env python3
"""
Checks for the Track A harness on a toy graph (CPU, seconds):

  1. SAGE.isolated(x) equals the full forward pass for a vertex with no edges;
  2. unit_layout puts every intra edge in exactly one unit and every cut edge in both units,
     and its halo equals the Phase 2 halo rows;
  3. every method scores every positive and negative pair (no NaN) and the evaluation has
     intra / cross / all strata for both negative schemes;
  4. eval negatives never hit a training edge, a held-out positive or a self-pair.

    bash scripts/wsl_run.sh python experiments/trackA/test_harness.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402


def toy_cache(root, seed=0):
    """Two dense communities of 40 vertices, a few cut edges, held-out intra and cross links."""
    rng = np.random.default_rng(seed)
    n = 80
    comm = np.r_[np.zeros(40), np.ones(40)].astype(np.int64)
    pairs = set()
    for c in (0, 1):
        base = 40 * c
        while len([p for p in pairs if comm[p[0]] == c and comm[p[1]] == c]) < 200:
            a, b = sorted(rng.choice(40, 2, replace=False) + base)
            pairs.add((int(a), int(b)))
    while len([p for p in pairs if comm[p[0]] != comm[p[1]]]) < 20:
        pairs.add((int(rng.integers(0, 40)), int(rng.integers(40, 80))))
    allp = np.array(sorted(pairs)).T
    is_cross = comm[allp[0]] != comm[allp[1]]
    intra_i = rng.permutation(np.where(~is_cross)[0])
    cross_i = rng.permutation(np.where(is_cross)[0])
    # val = 10 intra; test = 12 intra + 8 cross; the rest (incl. 12 cut edges) is the training graph
    held_i = np.r_[intra_i[:22], cross_i[:8]]
    held = allp[:, held_i]
    train = allp[:, np.setdiff1d(np.arange(allp.shape[1]), held_i)]
    halo = set()
    for a, b in train.T:
        if comm[a] != comm[b]:
            halo.add((int(b), int(comm[a]))); halo.add((int(a), int(comm[b])))
    halo = np.array(sorted(halo)).T
    x = rng.standard_normal((n, 8)).astype(np.float32) + comm[:, None]
    np.save(os.path.join(root, "x.npy"), x)
    np.save(os.path.join(root, "y.npy"), comm)
    np.save(os.path.join(root, "edges.npy"), train)
    np.save(os.path.join(root, "comm.npy"), comm)
    np.save(os.path.join(root, "halo.npy"), halo)
    np.save(os.path.join(root, "links.npy"), np.vstack([held, np.r_[np.zeros(10), np.ones(20)]]).astype(np.int64))
    json.dump({"toy": True}, open(os.path.join(root, "meta.json"), "w"))


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        sys.exit(1)


def main():
    root = tempfile.mkdtemp(prefix="trackA_toy_")
    try:
        toy_cache(root)
        data = H.Data(root)
        dev = torch.device("cpu")

        enc = H.SAGE(8, 16, 0.0).eval()
        A = H.mean_adj(3, np.array([0]), np.array([1]), dev)   # vertex 2 has no edges
        x = data.x[:3]
        with torch.no_grad():
            check(torch.allclose(enc(x, A)[2], enc.isolated(x[2:3])[0], atol=1e-6),
                  "isolated() equals the forward pass for a vertex without edges")

        lay = H.unit_layout(data, halo=True)
        cu, cv = data.comm[data.edges[0]], data.comm[data.edges[1]]
        n_intra = sum(u["intra"].shape[1] for u in lay.values())
        n_cut_sides = sum(u["cut"].shape[1] for u in lay.values())
        check(n_intra == int((cu == cv).sum()), f"each intra edge in one unit ({n_intra})")
        check(n_cut_sides == 2 * int((cu != cv).sum()), f"each cut edge in both units ({n_cut_sides})")
        mine = set((int(v), c) for c, u in lay.items() for v in u["halo"])
        theirs = set(zip(data.halo_rows[0].tolist(), data.halo_rows[1].tolist()))
        check(mine == theirs, f"layout halo equals the Phase 2 halo rows ({len(mine)})")
        for budget in (0.05, 0.2):
            ok = True
            for c, u in H.unit_layout(data, halo=True, budget=budget).items():
                hv, cnt = np.unique(lay[c]["cut"][1], return_counts=True)
                k = min(len(hv), int(budget * len(u["members"])))
                kth = np.sort(cnt)[::-1][k - 1] if k else np.inf       # the k-th largest edge count into the unit
                n_into = dict(zip(hv.tolist(), cnt.tolist()))
                ok &= len(u["halo"]) == k and all(n_into[v] >= kth for v in u["halo"].tolist())
                ok &= bool(np.isin(u["cut"][1], u["halo"]).all())
            check(ok, f"halo budget {budget}: keeps the floor({budget} x members) halo vertices with the most edges "
                      "into the unit, and only their cut edges")

        pos = data.test
        negs, _ = H.eval_negatives(data, pos, seed=1)
        known = set(data.known_keys().tolist())
        for k, p in negs.items():
            keys = data.edge_key(p[0], p[1])
            check(not any(int(t) in known for t in keys) and not (p[0] == p[1]).any(),
                  f"{k} negatives avoid true edges, held-out positives and self-pairs")
        m = negs["matched"]
        check(bool((data.comm[m[1]] == data.comm[pos[1]]).all()), "matched negatives come from v's unit")

        cfg = {"epochs": 30, "batch": 4096, "hidden": 16, "lr": 0.01, "dropout": 0.0}
        pairs = {"pos": pos, **negs}
        for name, fn in H.METHODS.items():
            scores, stats = fn(data, pairs, cfg, dev, torch.Generator().manual_seed(0))
            ok = all(np.isfinite(s).all() and len(s) == pos.shape[1] for s in scores.values())
            res = H.evaluate(data, pos, negs, scores)
            keys = {f"auc_{s}_{t}" for s in ("global", "matched") for t in ("intra", "cross", "all")}
            check(ok and keys <= set(res), f"{name}: every pair scored, all strata reported "
                                           f"(all/global {res['auc_global_all']}, cross/matched {res['auc_matched_cross']})")
        _, s1a = H.METHODS["M1a"](data, pairs, cfg, dev, torch.Generator().manual_seed(0))
        _, s5 = H.METHODS["M5"](data, pairs, cfg, dev, torch.Generator().manual_seed(0))
        check(s5["subgraph_rows"] == s1a["subgraph_rows"] and s5["halo_rows"] == 0
              and s5["cut_edges"] == int((cu != cv).sum()),
              f"M5 reads Stage 3a's rows ({s5['subgraph_rows']}) plus the {s5['cut_edges']} cut edges, no halo")
        scores, stats = H.METHODS["M2"](data, pairs, {**cfg, "halo_budget": 0.05}, dev, torch.Generator().manual_seed(0))
        check(all(np.isfinite(s).all() for s in scores.values()) and stats["halo_rows"] == 2 * int(0.05 * 40),
              f"M2 with halo budget 0.05: every pair scored, {stats['halo_rows']} halo rows")
        import report as R
        base = {"epochs": 100, "batch": 65536, "hidden": 256, "lr": 0.005, "dropout": 0.5}
        fake = os.path.join(root, "runs.jsonl")
        with open(fake, "w") as fh:
            for m, g, mt, rows, cfg in (("M0", 0.9, 0.8, None, base), ("M1a", 0.7, 0.85, 100, base),
                                        ("M2", 0.85, 0.9, 150, base), ("M2", 0.8, 0.9, 120, {**base, "halo_budget": 0.2})):
                rec = {"dataset": "toy", "method": m, "split": "test", "git": "x", "cfg": cfg,
                       "auc_global_all": g, "auc_matched_all": mt}
                fh.write(json.dumps({**rec, **({"subgraph_rows": rows} if rows else {})}) + "\n")
        gate = R.gate_table(fake)
        check("| toy | M2 | 1 | 1.50x | +75% | M0 ≤ M1a" in gate and "| toy | M2@R=0.2 | 1 | 1.20x | +50% |" in gate,
              "gate table: gap share per scheme, M0 <= M1a flagged, halo-budget variants as their own rows")
        print("\n  ALL PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
