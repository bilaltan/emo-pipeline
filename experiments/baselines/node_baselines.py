#!/usr/bin/env python3
"""
Single-GPU baselines for node classification, both on the same model and settings as Phase 3b's node task.

  full     full-graph GraphSAGE: one full-batch step per epoch on the whole training graph
  cluster  Cluster-GCN (Chiang et al., KDD 2019): METIS splits the graph into parts of about --part-nodes
           vertices; each step trains on the subgraph induced by --parts-per-batch random parts, and an epoch
           visits every part once (defaults 150 / 20: the paper's Reddit setting, 1,500 parts of 233K vertices)

Model, training and selection match Phase 3b: 2-layer mean SAGE (PyG SAGEConv parametrisation) + linear head,
L2-normalised features, Adam, the settings in experiment_config.TRAIN, and the best-validation checkpoint
(checked on the full graph every max(1, epochs // 40) epochs, no early stop). Reads the Track A cache
(experiments/trackA/export_cache.py), so graph, features and split are the ones the pipeline used.

    bash scripts/wsl_run.sh python experiments/baselines/node_baselines.py --dataset ogbn-arxiv-lp --method full
"""
import argparse
import copy
import json
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "trackA"))
from experiment_config import TRAIN  # noqa: E402
from harness import SAGE, git_hash, mean_adj  # noqa: E402


class Net(nn.Module):
    def __init__(self, f, h, c, dropout):
        super().__init__()
        self.enc, self.fc = SAGE(f, h, dropout), nn.Linear(h, c)

    def forward(self, x, A):
        return self.fc(self.enc(x, A))


def load(cache):
    L = lambda f: np.load(os.path.join(cache, f))
    x = L("x.npy")
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    x = torch.from_numpy((x / np.where(nrm > 0, nrm, 1.0)).astype(np.float32))
    return x, torch.from_numpy(L("y.npy")), L("split.npy"), L("edges.npy")


def cluster_batches(edges, n, n_parts, per_batch, device):
    """METIS parts of the training graph and a generator of one epoch of Cluster-GCN batches:
    (vertex ids, mean adjacency of the subgraph they induce). Returns (epoch_fn, partition seconds)."""
    import pymetis
    import scipy.sparse as sp
    t0 = time.time()
    G = sp.coo_matrix((np.ones(edges.shape[1], np.int8), (edges[0], edges[1])), shape=(n, n))
    G = (G + G.T).tocsr()
    _, parts = pymetis.part_graph(n_parts, xadj=G.indptr, adjncy=G.indices)
    parts = np.asarray(parts)
    order = np.argsort(parts, kind="stable")
    bounds = np.searchsorted(parts[order], np.arange(n_parts + 1))
    part_s = time.time() - t0

    def epoch(rng):
        perm = rng.permutation(n_parts)
        for i in range(0, n_parts, per_batch):
            idx = np.concatenate([order[bounds[p]:bounds[p + 1]] for p in perm[i:i + per_batch]])
            sub = sp.triu(G[idx][:, idx], 1).tocoo()
            yield torch.from_numpy(idx).to(device), mean_adj(len(idx), sub.row, sub.col, device)
    return epoch, part_s


def train_eval(x, y, split, A, seed, cfg, device, batches=None):
    """Returns (best val acc, test acc at that checkpoint, train seconds). batches: None = full-batch,
    else cluster_batches' epoch function."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(x.shape[1], cfg["hidden_dim"], int(y.max()) + 1, cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tr, va, te = (torch.from_numpy(split == k).to(device) for k in (0, 1, 2))

    def acc(m):
        model.eval()
        with torch.no_grad():
            pred = model(x, A).argmax(1)
        model.train()
        return float((pred[m] == y[m]).float().mean())

    def step(xb, Ab, yb, mb):
        loss = F.cross_entropy(model(xb, Ab)[mb], yb[mb])
        opt.zero_grad(); loss.backward(); opt.step()

    val_every = max(1, cfg["epochs"] // 40)
    best, state = -1.0, None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    model.train()
    for ep in range(cfg["epochs"]):
        if ep > 0 and ep % val_every == 0:
            v = acc(va)
            if v > best:
                best, state = v, copy.deepcopy(model.state_dict())
        if batches is None:
            step(x, A, y, tr)
        else:
            for idx, Ab in batches(rng):
                if tr[idx].any():
                    step(x[idx], Ab, y[idx], tr[idx])
    v = acc(va)
    if v > best:
        best, state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return best, acc(te), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="store name, e.g. ogbn-arxiv-lp")
    ap.add_argument("--method", default="full", choices=["full", "cluster"])
    ap.add_argument("--alg", default="louvain", help="only names the Track A cache directory")
    ap.add_argument("--cache", default=None, help="cache directory (default $LAKEGRL_STORE/<dataset>/trackA_<alg>)")
    ap.add_argument("--seeds", default=",".join(map(str, TRAIN["seeds"])))
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--part-nodes", type=int, default=150, help="cluster: vertices per METIS part")
    ap.add_argument("--parts-per-batch", type=int, default=20, help="cluster: parts per training step")
    ap.add_argument("--device", default="cpu")   # ICDE rule (Bilal, 2026-10-22): every model runs on CPU
    ap.add_argument("--out", default=os.path.join(REPO, "results", "icde", "baselines.jsonl"))
    args = ap.parse_args(argv)

    cache = args.cache or os.path.join(os.environ.get("LAKEGRL_STORE", "data/.local_pipeline"),
                                       args.dataset, f"trackA_{args.alg}")
    cfg = dict(TRAIN, epochs=args.epochs, lr=args.lr)
    device = torch.device(args.device)
    code = git_hash()
    t0 = time.time()
    x, y, split, edges = load(cache)
    with open(os.path.join(cache, "meta.json")) as fh:
        meta = json.load(fh)
    n = len(split)
    x, y = x.to(device), y.to(device)
    A = mean_adj(n, edges[0], edges[1], device)
    load_s = time.time() - t0
    extra, batches = {}, None
    if args.method == "cluster":
        n_parts = max(1, n // args.part_nodes)
        batches, part_s = cluster_batches(edges, n, n_parts, args.parts_per_batch, device)
        extra = {"n_parts": n_parts, "parts_per_batch": args.parts_per_batch, "partition_seconds": round(part_s, 1)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        val, test, train_s = train_eval(x, y, split, A, seed, cfg, device, batches)
        rec = {"dataset": args.dataset, "baseline": {"full": "full_graph_sage", "cluster": "cluster_gcn"}[args.method],
               "task": "node_classification", "seed": seed, "train": dict(cfg, seed=seed), **extra,
               "git": code, "cache": cache, "n_nodes": n, "n_edges_undirected": int(edges.shape[1]),
               "n_test": int((split == 2).sum()), "val_acc": val, "test_acc": test,
               "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1),
               "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
               # whole process so far (Linux reports KiB): the memory one machine needs for this baseline
               "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
               "threads": torch.get_num_threads(),
               # Spark only exports the Delta tables (None: cache written before export_cache saved it);
               # training is this one process, no executors
               "export_seconds": meta.get("export_seconds"), "export_spark": meta.get("spark"),
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec))
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
