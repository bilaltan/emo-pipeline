#!/usr/bin/env python3
"""
Single-GPU baselines for node classification, both on the same model and settings as Phase 3b's node task.

  full     full-graph GraphSAGE: one full-batch step per epoch on the whole training graph
  cluster  Cluster-GCN (Chiang et al., KDD 2019): METIS splits the graph into parts of about --part-nodes
           vertices; each step trains on the subgraph induced by --parts-per-batch random parts, and an epoch
           visits every part once (defaults 150 / 20: the paper's Reddit setting, 1,500 parts of 233K vertices)

--communities (full only; advisor 2026-11-13, the community-ID ablation for reviewer 3): a learned --comm-dim
embedding of each vertex's Phase 1 community is concatenated to its features, x~ = [x || E[c(v)]]: major
communities (>= --min-size members) get rows 0..K-1, every other vertex (minor, isolated) row K.

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
    def __init__(self, f, h, c, dropout, comm=None, comm_dim=16):
        """comm: None, or (embedding row of every vertex, number of rows, comm_adj's M), tensors on the model's device."""
        super().__init__()
        self.comm, self.M = (comm[0], comm[2]) if comm else (None, None)
        self.emb = nn.Embedding(comm[1], comm_dim) if comm else None
        self.enc, self.fc = SAGE(f + (comm_dim if comm else 0), h, dropout), nn.Linear(h, c)

    def forward(self, x, A):
        if self.emb is None:
            return self.fc(self.enc(x, A))
        # = enc([x || E[c]], A), with layer 1's weights applied to the x and embedding columns apart, A @ E[c] as
        # M @ E, and the four products summed in one [N, h] buffer: backward builds no [N, f + d] gradient for the
        # constant features, never goes through A's transpose, and no extra [N, h] tensors are held (on
        # ogbn-products each of these filled the 24 GB card)
        s, f, e = self.enc, x.shape[1], self.emb(self.comm)
        h = F.linear(torch.sparse.mm(A, x), s.l1.weight[:, :f], s.l1.bias)
        h = h.addmm_(torch.sparse.mm(self.M, self.emb.weight), s.l1.weight[:, f:].t())
        h = h.addmm_(x, s.r1.weight[:, :f].t()).addmm_(e, s.r1.weight[:, f:].t())
        h = s.dr(F.relu(h))
        return self.fc(s.l2(torch.sparse.mm(A, h)) + s.r2(h))


def read_comms(url):
    """(vertex ids, community ids, Delta version) of a Phase 1 communities table, its current version."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.store_io import snapshot
    fs, files, ver = snapshot(url)
    t = pa.concat_tables([pq.read_table(f, columns=["id", "community_id"], filesystem=fs) for f in files])
    return t.column("id").to_numpy().astype(np.int64), t.column("community_id").to_numpy().astype(np.int64), ver


def comm_index(ids, cids, n, min_size):
    """Embedding row of every vertex: communities with >= min_size members get 0..K-1 (in community id order),
    every other vertex (minor community, isolated, in no community) gets K. Returns (rows [n], K)."""
    comms, size = np.unique(cids, return_counts=True)
    major = comms[size >= min_size]
    pos = np.searchsorted(major, cids)
    hit = major[np.minimum(pos, max(len(major) - 1, 0))] == cids if len(major) else np.zeros(len(cids), bool)
    rows = np.full(n, len(major), dtype=np.int64)
    rows[ids[hit]] = pos[hit]
    return rows, len(major)


def comm_adj(n, src, dst, rows, n_rows, device):
    """M = A @ onehot(rows) for mean_adj's A ([n, n_rows] CSR): row v holds the share of v's neighbours in each
    embedding row, so A @ E[rows] = M @ E."""
    import scipy.sparse as sp
    s, d = np.concatenate([src, dst]), np.concatenate([dst, src])
    deg = np.bincount(s, minlength=n).astype(np.float32)
    M = sp.csr_matrix(((1.0 / deg[s]).astype(np.float32), (s, rows[d])), shape=(n, n_rows))   # duplicates summed
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)), torch.from_numpy(M.indices.astype(np.int64)),
                                   torch.from_numpy(M.data), (n, n_rows)).to(device)


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


def train_eval(x, y, split, A, seed, cfg, device, batches=None, comm=None, comm_dim=16):
    """Returns (best val acc, test acc at that checkpoint, train seconds). batches: None = full-batch,
    else cluster_batches' epoch function. comm: Net's community embedding rows (full-batch only)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(x.shape[1], cfg["hidden_dim"], int(y.max()) + 1, cfg["dropout"], comm, comm_dim).to(device)
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
    ap.add_argument("--communities", default=None, metavar="URL",
                    help="full only: Phase 1 communities Delta table whose ids feed a learned embedding")
    ap.add_argument("--min-size", type=int, default=100, help="--communities: members of a major community")
    ap.add_argument("--comm-dim", type=int, default=16, help="--communities: embedding size")
    ap.add_argument("--device", default="cpu")   # ICDE rule (Bilal, 2026-10-22): every model runs on CPU
    ap.add_argument("--out", default=os.path.join(REPO, "results", "icde", "baselines.jsonl"))
    args = ap.parse_args(argv)
    if args.communities and args.method != "full":
        ap.error("--communities needs --method full")

    cache = args.cache or os.path.join(os.environ.get("LAKEGRL_STORE", "data/.local_pipeline"),
                                       args.dataset, f"trackA_{args.alg}")
    cfg = dict(TRAIN, epochs=args.epochs, lr=args.lr)
    device = torch.device(args.device)
    # Cluster-GCN takes thousands of Adam steps; the moments of zero-gradient weights then decay into denormals and
    # CPU math on them is several times slower (arxiv, 30 epochs: 128.5 s vs 71.2 s, identical accuracy). Every CPU
    # trainer flushes them: these baselines, Phase 3b and DistDGL.
    torch.set_flush_denormal(True)
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
    comm = None
    if args.communities:
        ids, cids, ver = read_comms(args.communities)
        rows, k = comm_index(ids, cids, n, args.min_size)
        comm = (torch.from_numpy(rows).to(device), k + 1, comm_adj(n, edges[0], edges[1], rows, k + 1, device))
        extra = {"communities": args.communities, "communities_version": ver, "min_size": args.min_size,
                 "comm_dim": args.comm_dim, "n_major": k, "n_other": int((rows == k).sum())}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        val, test, train_s = train_eval(x, y, split, A, seed, cfg, device, batches, comm, args.comm_dim)
        name = {"full": "full_graph_sage", "cluster": "cluster_gcn"}[args.method] + ("_commid" if comm else "")
        rec = {"dataset": args.dataset, "baseline": name,
               "task": "node_classification", "seed": seed, "train": dict(cfg, seed=seed), **extra,
               "git": code, "cache": cache, "n_nodes": n, "n_edges_undirected": int(edges.shape[1]),
               "n_test": int((split == 2).sum()), "val_acc": val, "test_acc": test,
               "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1),
               "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
               # whole process so far (Linux reports KiB): the memory one machine needs for this baseline
               "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
               "threads": torch.get_num_threads(), "flush_denormal": True,
               # Spark only exports the Delta tables (None: cache written before export_cache saved it);
               # training is this one process, no executors
               "export_seconds": meta.get("export_seconds"), "export_spark": meta.get("spark"),
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec))
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
