#!/usr/bin/env python3
"""
Baseline: full-graph GraphSAGE on one GPU, node classification.

Phase 3b's node task with the whole training graph as one unit: the same 2-layer mean SAGE (PyG SAGEConv
parametrisation) + linear head, L2-normalised features, Adam, the settings in experiment_config.TRAIN, and the
same best-validation checkpoint (checked every max(1, epochs // 40) epochs, no early stop). Reads the Track A
cache (experiments/trackA/export_cache.py), so graph, features and split are the ones the pipeline used.

    bash scripts/wsl_run.sh python experiments/baselines/full_graph_sage.py --dataset ogbn-arxiv-lp --seeds 0,1,2
"""
import argparse
import copy
import json
import os
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


def train_eval(x, y, split, A, seed, cfg, device):
    """Returns (best val acc, test acc at that checkpoint, train seconds)."""
    torch.manual_seed(seed)
    model = Net(x.shape[1], cfg["hidden_dim"], int(y.max()) + 1, cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tr, va, te = (torch.from_numpy(split == k).to(device) for k in (0, 1, 2))

    def acc(m):
        model.eval()
        with torch.no_grad():
            pred = model(x, A).argmax(1)
        model.train()
        return float((pred[m] == y[m]).float().mean())

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
        loss = F.cross_entropy(model(x, A)[tr], y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
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
    ap.add_argument("--alg", default="louvain", help="only names the Track A cache directory")
    ap.add_argument("--cache", default=None, help="cache directory (default $LAKEGRL_STORE/<dataset>/trackA_<alg>)")
    ap.add_argument("--seeds", default=",".join(map(str, TRAIN["seeds"])))
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "icde", "baselines.jsonl"))
    args = ap.parse_args(argv)

    cache = args.cache or os.path.join(os.environ.get("LAKEGRL_STORE", "data/.local_pipeline"),
                                       args.dataset, f"trackA_{args.alg}")
    cfg = dict(TRAIN, epochs=args.epochs, lr=args.lr)
    device = torch.device(args.device)
    code = git_hash()
    t0 = time.time()
    x, y, split, edges = load(cache)
    x, y = x.to(device), y.to(device)
    A = mean_adj(len(split), edges[0], edges[1], device)
    load_s = time.time() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        val, test, train_s = train_eval(x, y, split, A, seed, cfg, device)
        rec = {"dataset": args.dataset, "baseline": "full_graph_sage", "task": "node_classification",
               "seed": seed, "train": dict(cfg, seed=seed), "git": code, "cache": cache,
               "n_nodes": len(split), "n_edges_undirected": int(edges.shape[1]),
               "n_test": int((split == 2).sum()), "val_acc": val, "test_acc": test,
               "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1),
               "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
               "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec))
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
