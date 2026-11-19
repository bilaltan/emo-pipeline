"""GCN and GATv2 for the backbone runs (advisor 2026-11-19): one definition for the Stage 3b units, its global model
and the full-graph baseline (experiments/baselines/node_baselines.py --model), so all three train the same layers.
Same shape as the GraphSAGE of the units and the baseline: conv(f, h) -> activation -> dropout -> conv(h, h) -> linear.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, kind, f, h, c, dropout, heads=8):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, GCNConv
        if kind == "gcn":   # on backbone_adj's normalized matrix (PyG's default renormalizes A in every forward)
            self.c1, self.c2, self.act = GCNConv(f, h, normalize=False), GCNConv(h, h, normalize=False), F.relu
        elif kind == "gatv2":   # the units' GATv2 since the first code: 8 heads of h / 8, then one head of h
            self.c1 = GATv2Conv(f, h // heads, heads=heads, dropout=dropout)
            self.c2 = GATv2Conv(h, h, heads=1, concat=False, dropout=dropout)
            self.act = F.elu
        else:
            raise ValueError(f"backbone {kind!r}: gcn or gatv2")
        self.dr, self.fc = nn.Dropout(dropout), nn.Linear(h, c)

    def enc(self, x, adj):
        return self.c2(self.dr(self.act(self.c1(x, adj))), adj)

    def forward(self, x, adj):
        return self.fc(self.enc(x, adj))


def backbone_adj(kind, src, dst, n):
    """Backbone's graph input from an edge list over vertices < n: both directions, deduplicated, no self-loops. GATv2
    gets the edge list (its attention is per edge; the layer adds the self-loops). GCN gets GCN's propagation matrix
    D^-1/2 (A + I) D^-1/2 as a sparse CSR matrix, computed once here: sparse matmul as the SAGE units and baseline
    (no edges x channels message tensor); PyG's in-layer normalization of ogbn-products' 126M entries in every
    forward filled the 24 GB card on 2026-11-19, the same math."""
    from pipeline.utils.common import undirected
    s, d = undirected(src, dst, n)
    keep = s != d
    s, d = s[keep], d[keep]
    if kind != "gcn":
        return torch.from_numpy(np.stack([s, d]))
    s, d = np.concatenate([s, np.arange(n)]), np.concatenate([d, np.arange(n)])
    deg = np.bincount(s, minlength=n).astype(np.float32)   # symmetric: row = column degree, >= 1 with the self-loop
    w = (deg[s] ** -0.5 * deg[d] ** -0.5).astype(np.float32)
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack([s, d])), torch.from_numpy(w), (n, n)).coalesce().to_sparse_csr()
