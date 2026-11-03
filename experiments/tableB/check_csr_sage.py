#!/usr/bin/env python3
"""
Check for commit 50fbdb2: PyG SAGEConv on the CSR adjacency of utils.common.undirected edges equals SAGEConv on
the edge list (forward and grads, GPU and CPU), and prints both peak GPU memories on a Reddit-sized dense unit.

    bash scripts/wsl_run.sh python experiments/tableB/check_csr_sage.py
"""
import sys, numpy as np, torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch_geometric
from torch_geometric.nn import SAGEConv
from utils.common import undirected

def run(n, m, f, dev, seed=0):
    rng = np.random.default_rng(seed)
    s, d = undirected(rng.integers(0, n, m), rng.integers(0, n, m), n)
    ei = torch.stack([torch.from_numpy(s), torch.from_numpy(d)])
    adj = torch_geometric.utils.to_torch_csr_tensor(ei, size=(n, n))
    x = torch.randn(n, f, device=dev)
    torch.manual_seed(seed); c1, c2 = SAGEConv(f, 256).to(dev), SAGEConv(256, 256).to(dev)
    out = {}
    for name, e in (("edge", ei.to(dev)), ("csr", adj.to(dev))):
        for p in list(c1.parameters()) + list(c2.parameters()): p.grad = None
        if dev == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            y = c2(torch.relu(c1(x, e)), e); y.square().mean().backward()
            out[name] = (y.detach().cpu(), c1.lin_l.weight.grad.cpu().clone(),
                         torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0)
        except torch.OutOfMemoryError:
            out[name] = (None, None, float("inf"))
        del e
    return len(s), out

E, o = run(3000, 40000, 602, "cuda")
assert torch.allclose(o["edge"][0], o["csr"][0], atol=1e-4), (o["edge"][0] - o["csr"][0]).abs().max()
assert torch.allclose(o["edge"][1], o["csr"][1], atol=1e-4), (o["edge"][1] - o["csr"][1]).abs().max()
E, oc = run(3000, 40000, 602, "cpu")
assert torch.allclose(oc["edge"][0], oc["csr"][0], atol=1e-4) and torch.allclose(oc["edge"][1], oc["csr"][1], atol=1e-4)
print("equal: forward and grads, cuda and cpu")
E, o = run(17500, 4_200_000, 602, "cuda")
print(f"dense unit: {E:,} directed edges, peak GPU MB edge list {o['edge'][2]:,.0f}, CSR {o['csr'][2]:,.0f}")
