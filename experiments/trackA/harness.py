#!/usr/bin/env python3
"""
Track A harness: score link prediction for every method on one shared held-out split.

Protocol and methods: docs/trackA_link_prediction.md. Reads the cache from export_cache.py.

  M0       full-graph GraphSAGE (upper bound; sees every training edge)
  M1a      per-unit models, members only, intra-unit supervision (pure decoupling)
  M1b      per-unit models with Phase 2's 1-hop halo as context, intra supervision
           (what Phase 3 does today with EXPAND_BOUNDARY_NODES = True)
  M2       M1b + cut-edge supervision: member-halo training edges are positives too

Pair scoring for unit methods. A pair (u, w) is scored in the unit of u and in the unit of w, and the two scores
are averaged. In unit c, a vertex that is in c's subgraph (member or halo) uses its subgraph embedding. Any other
vertex is embedded by c's model as an isolated vertex (features only, empty neighbourhood). This is what a
deployment can do without other units' data, and it treats positives and negatives the same way.

The encoder and decoder copy Phase 3's link path: L2-normalised features, 2-layer mean SAGE (hidden 256),
dropout, Siamese decoder [h_u, h_v, |h_u - h_v|, h_u * h_v] -> MLP, BCE, Adam with weight decay 5e-4.

    bash scripts/wsl_run.sh python experiments/trackA/harness.py --dataset reddit-lp --methods M0,M1a,M1b,M2
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NO_COMM = np.iinfo(np.int64).min


# ----------------------------------------------------------------------------- data

class Data:
    def __init__(self, cache_dir):
        L = lambda f: np.load(os.path.join(cache_dir, f))
        self.dir = cache_dir
        self.meta = json.load(open(os.path.join(cache_dir, "meta.json")))
        x = L("x.npy")
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        self.x = torch.from_numpy((x / np.where(nrm > 0, nrm, 1.0)).astype(np.float32))
        self.n = self.x.shape[0]
        self.edges = L("edges.npy")                      # [2, E] undirected, src < dst
        self.comm = L("comm.npy")
        self.halo_rows = L("halo.npy")                   # [2, H] (vertex, unit)
        links = L("links.npy")
        self.val = links[:2, links[2] == 0]
        self.test = links[:2, links[2] == 1]
        self.units = np.unique(self.comm[self.comm != NO_COMM])

    def edge_key(self, a, b):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return lo.astype(np.int64) * self.n + hi

    def known_keys(self):
        """Training edges plus every held-out positive: never usable as a negative."""
        if not hasattr(self, "_known"):
            ks = [self.edge_key(self.edges[0], self.edges[1]),
                  self.edge_key(self.val[0], self.val[1]), self.edge_key(self.test[0], self.test[1])]
            self._known = np.unique(np.concatenate(ks))
        return self._known


def eval_negatives(data, pos, seed):
    """One `global` and one `matched` negative per positive (u, v), fixed per split seed.

    global:  (u, w), w uniform over all vertices.
    matched: (u, w), w uniform over the members of v's unit, so a scorer cannot win by
             recognising "same unit" alone. Falls back to global when v has no unit.
    True edges, held-out positives and self-pairs are rejected.
    """
    rng = np.random.default_rng(seed)
    known = data.known_keys()
    members = {}
    order = np.argsort(data.comm, kind="stable")
    cs = data.comm[order]
    bounds = np.searchsorted(cs, data.units), np.searchsorted(cs, data.units, side="right")
    for c, a, b in zip(data.units, *bounds):
        members[int(c)] = order[a:b]
    u, v = pos
    out, info = {}, {"n_matched_fallback": 0}
    for scheme in ("global", "matched"):
        w = np.empty_like(u)
        todo = np.arange(len(u))
        for attempt in range(250):
            if len(todo) == 0:
                break
            # matched: after 200 tries a pair whose u already links to (nearly) all of v's unit gets a global
            # negative; the count is reported (n_matched_fallback)
            if scheme == "global" or attempt >= 200:
                if scheme == "matched" and attempt == 200:
                    info["n_matched_fallback"] = int(len(todo))
                cand = rng.integers(0, data.n, len(todo))
            else:
                cand = np.empty(len(todo), dtype=np.int64)
                for i, t in enumerate(todo):
                    m = members.get(int(data.comm[v[t]]))
                    cand[i] = m[rng.integers(0, len(m))] if m is not None else rng.integers(0, data.n)
            bad = (cand == u[todo]) | np.isin(data.edge_key(u[todo], cand), known)
            w[todo[~bad]] = cand[~bad]
            todo = todo[bad]
        assert len(todo) == 0, f"could not draw {scheme} negatives for {len(todo)} pairs"
        out[scheme] = np.stack([u, w])
    return out, info


# ----------------------------------------------------------------------------- model

def bytes_est(data, rows, edges):
    """Bytes a method reads, estimated from uncompressed rows (id + float32 features) and edges (two int64 ids).
    Parquet compression changes the absolute numbers; ratios between methods are what the gate uses."""
    return int(rows * (8 + 4 * data.x.shape[1]) + edges * 16)


def mean_adj(n, src, dst, device):
    """Row-normalised sparse adjacency for mean aggregation over both directions."""
    s = np.concatenate([src, dst]); d = np.concatenate([dst, src])
    deg = np.bincount(s, minlength=n).astype(np.float32)
    val = (1.0 / deg[s]).astype(np.float32)
    idx = torch.from_numpy(np.stack([s, d]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(val), (n, n)).coalesce().to_sparse_csr().to(device)


class SAGE(nn.Module):
    """2-layer mean SAGE, same parametrisation as PyG SAGEConv: lin_l(mean_neigh) + lin_r(x)."""
    def __init__(self, f, h, dropout):
        super().__init__()
        self.l1, self.r1 = nn.Linear(f, h), nn.Linear(f, h, bias=False)
        self.l2, self.r2 = nn.Linear(h, h), nn.Linear(h, h, bias=False)
        self.dr = nn.Dropout(dropout)

    def forward(self, x, A):
        h = F.relu(self.l1(torch.sparse.mm(A, x)) + self.r1(x))
        h = self.dr(h)
        return self.l2(torch.sparse.mm(A, h)) + self.r2(h)

    def isolated(self, x):
        """Embedding of a vertex with no neighbours: the mean aggregate is 0."""
        h = F.relu(self.l1.bias + self.r1(x))
        return self.l2.bias + self.r2(h)


class Decoder(nn.Module):
    def __init__(self, h, dropout):
        super().__init__()
        self.fc1, self.fc2, self.dr = nn.Linear(4 * h, h), nn.Linear(h, 1), nn.Dropout(dropout)

    def forward(self, a, b):
        z = torch.cat([a, b, (a - b).abs(), a * b], dim=-1)
        return self.fc2(self.dr(F.relu(self.fc1(z)))).squeeze(-1)

    def sym(self, a, b):
        return 0.5 * (self(a, b) + self(b, a))


def train_link(x, A, pos, neg_pool_src, neg_pool_dst, cfg, device, gen):
    """Full-batch training on one (sub)graph. pos: [2, P] local positives (both endpoints in the subgraph).
    Negatives per step: sources from neg_pool_src, destinations from neg_pool_dst (local ids)."""
    enc = SAGE(x.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    dec = Decoder(cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg["lr"], weight_decay=5e-4)
    pos_t = torch.from_numpy(pos).to(device)
    ps = torch.from_numpy(neg_pool_src).to(device)
    pd_ = torch.from_numpy(neg_pool_dst).to(device)
    enc.train(); dec.train()
    for _ in range(cfg["epochs"]):
        b = pos_t[:, torch.randint(0, pos_t.shape[1], (min(cfg["batch"], pos_t.shape[1]),),
                                   generator=gen, device="cpu").to(device)]
        ns = ps[torch.randint(0, len(ps), (b.shape[1],), generator=gen).to(device)]
        nd = pd_[torch.randint(0, len(pd_), (b.shape[1],), generator=gen).to(device)]
        h = enc(x, A)
        logit = torch.cat([dec(h[b[0]], h[b[1]]), dec(h[ns], h[nd])])
        y = torch.cat([torch.ones(b.shape[1], device=device), torch.zeros(b.shape[1], device=device)])
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); dec.eval()
    return enc, dec


# ----------------------------------------------------------------------------- methods

def run_full_graph(data, pairs, cfg, device, gen):
    t0 = time.time()
    x = data.x.to(device)
    A = mean_adj(data.n, data.edges[0], data.edges[1], device)
    allv = np.arange(data.n)
    enc, dec = train_link(x, A, data.edges, allv, allv, cfg, device, gen)
    with torch.no_grad():
        h = enc(x, A)
        scores = {k: dec.sym(h[torch.from_numpy(p[0]).to(device)], h[torch.from_numpy(p[1]).to(device)]).cpu().numpy()
                  for k, p in pairs.items()}
    return scores, {"train_seconds": round(time.time() - t0, 1), "units": 1,
                    "read_bytes_est": bytes_est(data, data.n, data.edges.shape[1])}


def unit_layout(data, halo, budget=0.0):
    """Per unit: subgraph vertex list (members first, then halo) and its intra / cut edges.

    budget > 0 caps a unit's halo at floor(budget x members) vertices, keeping those with the most cut edges into the
    unit (ties: lower id); cut edges to dropped halo vertices are dropped too. This bounds the rows a unit reads.
    """
    cu, cv = data.comm[data.edges[0]], data.comm[data.edges[1]]
    intra = (cu == cv) & (cu != NO_COMM)
    e = data.edges
    by_unit = {}
    # intra edges grouped by unit
    idx = np.where(intra)[0]
    o = idx[np.argsort(cu[idx], kind="stable")]
    uq, first = np.unique(cu[o], return_index=True)
    for c, a, b in zip(uq, first, np.r_[first[1:], len(o)]):
        by_unit.setdefault(int(c), {})["intra"] = e[:, o[a:b]]
    if halo:
        # a cut edge (u, v) belongs to u's unit (v is halo there) and to v's unit (u is halo there)
        cut = np.where(~intra & (cu != NO_COMM) & (cv != NO_COMM))[0]
        side_unit = np.concatenate([cu[cut], cv[cut]])
        side_mem = np.concatenate([e[0, cut], e[1, cut]])
        side_halo = np.concatenate([e[1, cut], e[0, cut]])
        o = np.argsort(side_unit, kind="stable")
        uq, first = np.unique(side_unit[o], return_index=True)
        last = np.r_[first[1:], len(o)]
        for c, a, b in zip(uq, first, last):
            by_unit.setdefault(int(c), {})["cut"] = np.stack([side_mem[o[a:b]], side_halo[o[a:b]]])
    order = np.argsort(data.comm, kind="stable")
    cs = data.comm[order]
    out = {}
    for c in data.units:
        a, b = np.searchsorted(cs, c), np.searchsorted(cs, c, side="right")
        mem = order[a:b]
        d = by_unit.get(int(c), {})
        cut = d.get("cut", np.zeros((2, 0), dtype=np.int64))
        hv, cnt = np.unique(cut[1], return_counts=True) if cut.shape[1] else (np.zeros(0, dtype=np.int64), None)
        if budget > 0 and len(hv) > int(budget * len(mem)):
            hv = np.sort(hv[np.lexsort((hv, -cnt))[:int(budget * len(mem))]])
            cut = cut[:, np.isin(cut[1], hv)]
        out[int(c)] = {"members": mem, "halo": hv, "intra": d.get("intra", np.zeros((2, 0), dtype=np.int64)),
                       "cut": cut}
    return out


def run_units(data, pairs, cfg, device, gen, halo, cut_supervision, member_h=None):
    """Train one model per unit and score every pair side that falls in that unit. member_h ([n, hidden], optional)
    receives every member's embedding from its own unit's model."""
    t0 = time.time()
    lay = unit_layout(data, halo, cfg.get("halo_budget", 0.0))
    sides = {}   # pair set -> (side0 scores, side1 scores) filled per unit
    for k, p in pairs.items():
        sides[k] = [np.full(p.shape[1], np.nan, dtype=np.float32), np.full(p.shape[1], np.nan, dtype=np.float32)]
    stats = {"units": 0, "subgraph_rows": 0, "halo_rows": 0, "subgraph_edges": 0, "train_edges": 0}
    for c, u in lay.items():
        verts = np.concatenate([u["members"], u["halo"]]) if halo else u["members"]
        if len(u["members"]) < 2:
            continue
        sort_idx = np.argsort(verts)
        sorted_v = verts[sort_idx]

        def to_local(g):
            """Global ids -> (local ids, found mask) in this unit's subgraph."""
            i = np.clip(np.searchsorted(sorted_v, g), 0, len(verts) - 1)
            return sort_idx[i], sorted_v[i] == g
        ei = np.concatenate([u["intra"], u["cut"]], axis=1) if halo else u["intra"]
        li, ok = to_local(ei.reshape(-1))
        assert ok.all()
        li = li.reshape(2, -1)
        A = mean_adj(len(verts), li[0], li[1], device)
        if cut_supervision:
            pos = li
            dst_pool = np.arange(len(verts))
        else:
            pos = li[:, :u["intra"].shape[1]]
            dst_pool = np.arange(len(u["members"]))
        stats["units"] += 1; stats["subgraph_rows"] += len(verts)
        stats["halo_rows"] += len(u["halo"]) if halo else 0
        stats["subgraph_edges"] += ei.shape[1]
        stats["train_edges"] += pos.shape[1]
        if pos.shape[1] == 0:
            continue
        x = data.x[torch.from_numpy(verts)].to(device)
        enc, dec = train_link(x, A, pos, np.arange(len(u["members"])), dst_pool, cfg, device, gen)
        with torch.no_grad():
            h = enc(x, A)
            if member_h is not None:
                member_h[u["members"]] = h[:len(u["members"])].cpu().numpy()
            for k, p in pairs.items():
                for side in (0, 1):
                    a, b = p[side], p[1 - side]
                    sel = np.where(data.comm[a] == c)[0]
                    if len(sel) == 0:
                        continue
                    la, _ = to_local(a[sel])
                    lb, okb = to_local(b[sel])
                    ha = h[torch.from_numpy(la).to(device)]
                    hb = torch.empty_like(ha)
                    okb_t = torch.from_numpy(okb).to(device)
                    if okb.any():
                        hb[okb_t] = h[torch.from_numpy(lb[okb]).to(device)]
                    if (~okb).any():
                        hb[~okb_t] = enc.isolated(data.x[torch.from_numpy(b[sel][~okb])].to(device))
                    sides[k][side][sel] = dec.sym(ha, hb).cpu().numpy()
        del enc, dec, A, x, h
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores = {}
    for k, (s0, s1) in sides.items():
        both = np.vstack([s0, s1])
        none = np.isnan(both).all(axis=0)
        # vertices with no training edge are in no unit (Louvain never sees them): a pair with neither endpoint in
        # a unit has no model to score it and falls back to feature cosine; the count is reported per pair set
        sc = np.nanmean(np.where(none, 0.0, both), axis=0)       # none-columns are filled below
        if none.any():
            p = pairs[k][:, none]
            xa, xb = data.x[torch.from_numpy(p[0])], data.x[torch.from_numpy(p[1])]
            sc[none] = F.cosine_similarity(xa, xb, dim=1).numpy()
        stats[f"n_fallback_{k}"] = int(none.sum())
        scores[k] = sc
    stats["train_seconds"] = round(time.time() - t0, 1)
    stats["read_bytes_est"] = bytes_est(data, stats["subgraph_rows"], stats["subgraph_edges"])
    return scores, stats


class Fusion(nn.Module):
    """M5's cut-edge scorer: a shared projection of [h_local, h_macro, x] per endpoint, then the Siamese decoder."""
    def __init__(self, d, h, dropout):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, h))
        self.dec = Decoder(h, dropout)

    def sym(self, a, b):
        return self.dec.sym(self.phi(a), self.phi(b))


def run_m5(data, pairs, cfg, device, gen, local=True):
    """M5, late fusion for links: Stage 3a (M1a: members only, 1x rows) gives every member an embedding h_local; one
    shared decoder on [h_local, h_macro, x] of both endpoints, trained on all training edges, scores every pair. (Scoring
    intra pairs with the unit decoders and the rest with this one mixes two logit scales: a random cross negative of an
    intra positive would be compared across scorers.)
    h_local: the endpoint's embedding from its own unit's model. h_macro: its unit's row of a parameter-free 2-hop
    propagation (row-normalised, self-loops) of mean member features over the macro graph (units, cut-edge counts).
    Extra data read beyond Stage 3a: the cut-edge list and one macro row per unit.
    local=False is M3 (ablation): the same decoder on [h_macro, x] only, no per-unit training."""
    import scipy.sparse as sp
    t0 = time.time()
    H = np.zeros((data.n, cfg["hidden"] if local else 0), np.float32)
    if local:
        scores, stats = run_units(data, pairs, cfg, device, gen, halo=False, cut_supervision=False, member_h=H)
    else:
        scores = {key: np.zeros(p.shape[1], np.float32) for key, p in pairs.items()}
        stats = {"units": len(data.units), "subgraph_rows": int((data.comm != NO_COMM).sum()), "halo_rows": 0,
                 "subgraph_edges": 0, "train_edges": int(data.edges.shape[1])}
    k, valid = len(data.units), data.comm != NO_COMM
    ci = np.full(data.n, k, np.int64)               # k = "no unit": a zero macro row
    ci[valid] = np.searchsorted(data.units, data.comm[valid])
    cu, cv = ci[data.edges[0]], ci[data.edges[1]]
    cut = (cu != cv) & (cu < k) & (cv < k)
    W = sp.coo_matrix((np.ones(2 * cut.sum()), (np.r_[cu[cut], cv[cut]], np.r_[cv[cut], cu[cut]])), (k, k)).tocsr()
    W = W + sp.identity(k, format="csr")
    W = sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
    P = sp.coo_matrix((np.ones(valid.sum()), (ci[valid], np.where(valid)[0])), (k, data.n)).tocsr()
    P = sp.diags(1.0 / np.maximum(np.asarray(P.sum(1)).ravel(), 1)) @ P
    hm = W @ (W @ (P @ data.x.numpy()))
    hm = torch.from_numpy(np.vstack([hm, np.zeros((1, hm.shape[1]))]).astype(np.float32)).to(device)
    Ht, xg, cit = torch.from_numpy(H).to(device), data.x.to(device), torch.from_numpy(ci).to(device)
    feat = lambda idx: torch.cat([Ht[idx], hm[cit[idx]], xg[idx]], dim=1)
    # negatives: half uniform over all vertices, half from the unit of the positive's v (unit identity is not enough)
    order = np.argsort(ci, kind="stable")
    lo, hi = np.searchsorted(ci[order], np.arange(k + 1)), np.searchsorted(ci[order], np.arange(k + 1), side="right")
    pos = torch.from_numpy(data.edges).to(device)
    order_t, lo_t, n_t = (torch.from_numpy(a).to(device) for a in (order, lo, hi - lo))
    model = Fusion(H.shape[1] + 2 * data.x.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=5e-4)
    model.train()
    for _ in range(cfg["epochs"] if pos.shape[1] else 0):
        b = pos[:, torch.randint(0, pos.shape[1], (min(cfg["batch"], pos.shape[1]),), generator=gen).to(device)]
        half = b.shape[1] // 2
        w_glob = torch.randint(0, data.n, (half,), generator=gen).to(device)
        cvb = cit[b[1, half:]]
        r = (torch.rand(b.shape[1] - half, generator=gen).to(device) * n_t[cvb]).long()
        w = torch.cat([w_glob, order_t[lo_t[cvb] + r]])
        logit = torch.cat([model.sym(feat(b[0]), feat(b[1])), model.sym(feat(b[0]), feat(w))])
        y = torch.cat([torch.ones(b.shape[1], device=device), torch.zeros(b.shape[1], device=device)])
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        for key, p in pairs.items():
            for a in range(0, p.shape[1], 262144):
                j = np.arange(a, min(a + 262144, p.shape[1]))
                pa, pb = (torch.from_numpy(p[s, j]).to(device) for s in (0, 1))
                scores[key][j] = model.sym(feat(pa), feat(pb)).cpu().numpy()
    stats.update({f"n_fallback_{key}": 0 for key in pairs})       # the decoder scores every pair, no cosine fallback
    stats.update(cut_edges=int(cut.sum()), macro_units=k, train_seconds=round(time.time() - t0, 1))
    # members' rows and intra edges (Stage 3a; M3 reads the rows only) + the cut-edge list + one macro row per unit
    stats["read_bytes_est"] = bytes_est(data, stats["subgraph_rows"] + k, stats["subgraph_edges"] + int(cut.sum()))
    return scores, stats


METHODS = {
    "M0":  lambda d, p, c, dev, g: run_full_graph(d, p, c, dev, g),
    "M1a": lambda d, p, c, dev, g: run_units(d, p, c, dev, g, halo=False, cut_supervision=False),
    "M1b": lambda d, p, c, dev, g: run_units(d, p, c, dev, g, halo=True, cut_supervision=False),
    "M2":  lambda d, p, c, dev, g: run_units(d, p, c, dev, g, halo=True, cut_supervision=True),
    "M3":  lambda d, p, c, dev, g: run_m5(d, p, c, dev, g, local=False),
    "M5":  lambda d, p, c, dev, g: run_m5(d, p, c, dev, g),
}


# ----------------------------------------------------------------------------- evaluation

def evaluate(data, pos, negs, scores):
    """ROC-AUC per stratum (intra / cross / all) and negative scheme. scores[k] aligns with pairs[k]."""
    cu, cv = data.comm[pos[0]], data.comm[pos[1]]
    strata = {"intra": (cu == cv) & (cu != NO_COMM), "all": np.ones(pos.shape[1], bool)}
    strata["cross"] = ~strata["intra"]
    out = {}
    for scheme in negs:
        for name, m in strata.items():
            if m.sum() == 0:
                continue
            y = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
            s = np.r_[scores["pos"][m], scores[scheme][m]]
            out[f"auc_{scheme}_{name}"] = round(float(roc_auc_score(y, s)), 4)
    out["n_pos"] = int(pos.shape[1]); out["n_cross"] = int(strata["cross"].sum())
    return out


def halo_coverage(data, pos):
    """Share of cross positives (u, v) where v is in u's unit halo or u is in v's (the halo can score the pair)."""
    cu, cv = data.comm[pos[0]], data.comm[pos[1]]
    cross = np.where(~((cu == cv) & (cu != NO_COMM)))[0]
    if len(cross) == 0:
        return None
    hk = set(zip(data.halo_rows[0].tolist(), data.halo_rows[1].tolist()))
    ok = [(int(pos[1, i]), int(cu[i])) in hk or (int(pos[0, i]), int(cv[i])) in hk for i in cross]
    return round(float(np.mean(ok)), 4)


def git_hash():
    """HEAD, suffixed with -dirty when tracked files differ from it (the run's code is then not exactly HEAD)."""
    try:
        h = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short=10", "HEAD"], text=True).strip()
        # core.autocrlf=true as on the Windows side that owns the checkout: WSL's git would otherwise report every
        # CRLF working-tree file as modified
        dirty = subprocess.run(["git", "-c", "core.autocrlf=true", "-C", REPO, "diff", "--quiet", "HEAD", "--"],
                               capture_output=True).returncode
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alg", default="louvain")
    ap.add_argument("--methods", default="M0,M1a,M1b,M2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=65536, help="positive edges per step")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--halo-budget", type=float, default=0.0,
                    help="M1b / M2: cap each unit's halo at this many x its members (0 = full 1-hop halo)")
    ap.add_argument("--max-pos", type=int, default=0, help="evaluate on the first N positives only (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "trackA", "runs.jsonl"))
    args = ap.parse_args()

    store = os.environ.get("LAKEGRL_STORE", os.path.join(REPO, "data", ".local_pipeline"))
    data = Data(os.path.join(store, args.dataset, f"trackA_{args.alg}"))
    pos = data.val if args.split == "val" else data.test
    if args.max_pos:
        pos = pos[:, :args.max_pos]
    negs, neg_info = eval_negatives(data, pos, seed=1000 + (args.split == "test"))
    pairs = {"pos": pos, **negs}
    cfg = {k: getattr(args, k) for k in ("epochs", "batch", "hidden", "lr", "dropout")}
    if args.halo_budget:
        cfg["halo_budget"] = args.halo_budget       # only when set, so full-halo runs keep their report group
    device = torch.device(args.device)
    code = git_hash()       # at start: the code this process loaded, even if the checkout changes while it runs
    cov = halo_coverage(data, pos)
    print(f"{args.dataset}: {data.n:,} nodes, {data.edges.shape[1]:,} train edges, {len(data.units)} units, "
          f"{pos.shape[1]:,} {args.split} positives; halo coverage of cross positives {cov}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for m in args.methods.split(","):
            torch.manual_seed(seed); np.random.seed(seed)
            gen = torch.Generator().manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            scores, stats = METHODS[m](data, pairs, cfg, device, gen)
            res = evaluate(data, pos, negs, scores)
            rec = {"run_id": time.strftime("%Y%m%d-%H%M%S") + f"-{m}-s{seed}", "dataset": args.dataset,
                   "alg": args.alg, "method": m, "seed": seed, "split": args.split, "cfg": cfg,
                   "git": code, "device": str(device), "halo_coverage_cross": cov, "feat_dim": int(data.x.shape[1]),
                   **res, **neg_info, **stats,
                   "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}
            print(json.dumps(rec))
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
