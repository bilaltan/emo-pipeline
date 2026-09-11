"""
Strict Single-Factor Ablation Runner for Decoupled Pipeline & Full-Graph GraphSAGE.

Configurations Tested (Strictly Isolated - No combinations):
------------------------------------------------------------
Table 1: Decoupled Pipeline
- Config 1 (Original LakeGRL): Old Decoder + Uniform Sampling + No Halo (Layer 1=intra, Layer 2=intra, intra-edges only)
- Config 2 (Decoder Effect Only): NEW Decoder + Uniform Sampling + No Halo (Layer 1=intra, Layer 2=intra, intra-edges only)
- Config 3 (Sampling Effect Only): Old Decoder + Rejection Sampling + No Halo (Layer 1=intra, Layer 2=intra, intra-edges only)
- Config 4 (Halo Effect Only): Old Decoder + Uniform Sampling + 1-Hop Halo (Layer 1=intra, Layer 2=all edges for 1-hop boundary expansion)

Table 2: Full-Graph GraphSAGE
- Config 1 (Full-Graph GraphSAGE): Old Decoder + Uniform Sampling (Layer 1=all, Layer 2=all)
- Config 2 (Decoder Effect Only): NEW Decoder + Uniform Sampling (Layer 1=all, Layer 2=all)
- Config 3 (Sampling Effect Only): Old Decoder + Rejection Sampling (Layer 1=all, Layer 2=all)
"""

import os
import sys
import time
import json
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv
import networkx as nx
import networkx.algorithms.community as nx_comm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import _patch_torch_load
_patch_torch_load()

# -------------------------------------------------------------
# Model Architectures
# -------------------------------------------------------------

class TwoEdgeSAGE(nn.Module):
    """
    2-Layer GraphSAGE Link Encoder supporting layer-wise edge indices:
    - In No-Halo (Stage 3a): Layer 1 = intra, Layer 2 = intra (0 boundary hops).
    - In 1-Hop Halo: Layer 1 = intra, Layer 2 = all edges (1-hop boundary expansion, 0 multi-hop cross-partition leakage).
    - In Full-Graph: Layer 1 = all edges, Layer 2 = all edges (full global multi-hop message passing).
    """
    def __init__(self, in_f, h=128, dropout=0.1):
        super().__init__()
        self.c1 = SAGEConv(in_f, h)
        self.c2 = SAGEConv(h, h)
        self.dr = nn.Dropout(dropout)

    def forward(self, x, e1, e2):
        x = F.relu(self.c1(x, e1))
        x = self.dr(x)
        return self.c2(x, e2)

class OldHadamardPredictor(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, 1)

    def forward(self, h_src, h_dst):
        x = h_src * h_dst
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

class Siamese4WayPredictor(nn.Module):
    def __init__(self, h=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(4 * h, h)
        self.fc2 = nn.Linear(h, 1)
        self.dr = nn.Dropout(dropout)

    def forward(self, h_src, h_dst):
        diff = torch.abs(h_src - h_dst)
        prod = h_src * h_dst
        cat_feat = torch.cat([h_src, h_dst, diff, prod], dim=-1)
        x = torch.relu(self.fc1(cat_feat))
        x = self.dr(x)
        return self.fc2(x).squeeze(-1)

# -------------------------------------------------------------
# Sampling Helpers
# -------------------------------------------------------------

def sample_rejection_negs(count, n_nodes, existing_set):
    ns, nd = [], []
    needed = count
    for _ in range(8):
        cs = torch.randint(0, n_nodes, (needed * 2,))
        cd = torch.randint(0, n_nodes, (needed * 2,))
        val = (cs != cd)
        for s, d in zip(cs[val].tolist(), cd[val].tolist()):
            if (s, d) not in existing_set and (d, s) not in existing_set:
                ns.append(s)
                nd.append(d)
                if len(ns) >= count:
                    break
        if len(ns) >= count:
            break
        needed = count - len(ns)
    if len(ns) < count:
        rem = count - len(ns)
        ns.extend(torch.randint(0, n_nodes, (rem,)).tolist())
        nd.extend(torch.randint(0, n_nodes, (rem,)).tolist())
    return torch.tensor(ns[:count], dtype=torch.long), torch.tensor(nd[:count], dtype=torch.long)

def train_and_eval_config(x, e1, e2, tr_s, tr_d, te_s, te_d, n_nodes, existing_set,
                          decoder_type='old', use_rejection=False, epochs=12, batch_sz=8000, lr=0.01):
    torch.manual_seed(42)
    np.random.seed(42)

    enc = TwoEdgeSAGE(x.size(1), h=128)
    if decoder_type == 'new':
        pred = Siamese4WayPredictor(h=128)
    else:
        pred = OldHadamardPredictor(h=128)

    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr, weight_decay=5e-4)
    tr_s_t = torch.tensor(tr_s, dtype=torch.long)
    tr_d_t = torch.tensor(tr_d, dtype=torch.long)

    actual_batch = min(batch_sz, len(tr_s_t))

    for epoch in range(epochs):
        enc.train()
        pred.train()
        h = enc(x, e1, e2)
        b_idx = torch.randint(0, len(tr_s_t), (actual_batch,))
        pos_s, pos_d = tr_s_t[b_idx], tr_d_t[b_idx]

        if use_rejection:
            neg_s, neg_d = sample_rejection_negs(actual_batch, n_nodes, existing_set)
        else:
            neg_s = torch.randint(0, n_nodes, (actual_batch,))
            neg_d = torch.randint(0, n_nodes, (actual_batch,))

        pos_sc = pred(h[pos_s], h[pos_d])
        neg_sc = pred(h[neg_s], h[neg_d])
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_sc, neg_sc]),
            torch.cat([torch.ones_like(pos_sc), torch.zeros_like(neg_sc)])
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    enc.eval()
    pred.eval()
    with torch.no_grad():
        h = enc(x, e1, e2)
        te_s_t = torch.tensor(te_s, dtype=torch.long)
        te_d_t = torch.tensor(te_d, dtype=torch.long)
        pos_sc = pred(h[te_s_t], h[te_d_t])

        if use_rejection:
            neg_s, neg_d = sample_rejection_negs(len(te_s), n_nodes, existing_set)
        else:
            neg_s = torch.randint(0, n_nodes, (len(te_s),))
            neg_d = torch.randint(0, n_nodes, (len(te_s),))

        neg_sc = pred(h[neg_s], h[neg_d])
        y_true = np.concatenate([np.ones(len(pos_sc)), np.zeros(len(neg_sc))])
        y_sc = torch.cat([pos_sc, neg_sc]).cpu().numpy()
        auc = roc_auc_score(y_true, y_sc)

    return float(auc)

# -------------------------------------------------------------
# Dataset Loaders & Preprocessors
# -------------------------------------------------------------

def prepare_reddit():
    print("Loading Reddit dataset...")
    from torch_geometric.datasets import Reddit
    ds = Reddit(root='data/reddit')
    data = ds[0]
    n_nodes = data.x.size(0)

    src_all = data.edge_index[0].numpy()
    dst_all = data.edge_index[1].numpy()
    mask = src_all < dst_all
    fwd_s = src_all[mask]
    fwd_d = dst_all[mask]

    np.random.seed(42)
    sample_sz = min(100000, len(fwd_s))
    perm = np.random.permutation(len(fwd_s))[:sample_sz]
    n_tr = int(0.8 * sample_sz)

    tr_s = fwd_s[perm[:n_tr]]
    tr_d = fwd_d[perm[:n_tr]]
    te_s = fwd_s[perm[n_tr:]]
    te_d = fwd_d[perm[n_tr:]]

    node_comms = data.y.numpy()
    intra_mask = (node_comms[tr_s] == node_comms[tr_d])

    stage3a_s = tr_s[intra_mask]
    stage3a_d = tr_d[intra_mask]
    stage3a_edges = torch.tensor(np.stack([np.concatenate([stage3a_s, stage3a_d]), np.concatenate([stage3a_d, stage3a_s])]), dtype=torch.long)
    full_graph_edges = torch.tensor(np.stack([np.concatenate([tr_s, tr_d]), np.concatenate([tr_d, tr_s])]), dtype=torch.long)

    existing_set = set(zip(tr_s.tolist(), tr_d.tolist())).union(set(zip(tr_d.tolist(), tr_s.tolist())))
    return data.x, n_nodes, tr_s, tr_d, stage3a_s, stage3a_d, stage3a_edges, full_graph_edges, te_s, te_d, existing_set

def prepare_deezer():
    print("Loading Deezer Europe dataset...")
    data_dir = "data/deezer_europe_extracted/deezer_europe"
    edges_df = pd.read_csv(os.path.join(data_dir, "deezer_europe_edges.csv"))
    src_raw = edges_df['node_1'].values
    dst_raw = edges_df['node_2'].values
    n_nodes = max(src_raw.max(), dst_raw.max()) + 1

    mask = src_raw < dst_raw
    fwd_s = np.concatenate([src_raw[mask], dst_raw[~mask]])
    fwd_d = np.concatenate([dst_raw[mask], src_raw[~mask]])
    stacked = np.stack([fwd_s, fwd_d], axis=1)
    unique_edges = np.unique(stacked, axis=0)
    unique_edges = unique_edges[unique_edges[:, 0] != unique_edges[:, 1]]
    fwd_s, fwd_d = unique_edges[:, 0], unique_edges[:, 1]

    with open(os.path.join(data_dir, "deezer_europe_features.json"), 'r') as f:
        feats_dict = json.load(f)
    feat_dim = 128
    np.random.seed(42)
    proj = np.random.randn(32000, feat_dim).astype(np.float32) / np.sqrt(feat_dim)
    node_feats = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    for u_str, toks in feats_dict.items():
        u = int(u_str)
        if u < n_nodes and len(toks) > 0:
            val_toks = [t for t in toks if t < 32000]
            if val_toks:
                node_feats[u] = proj[val_toks].mean(axis=0)
        elif u < n_nodes:
            node_feats[u] = np.random.randn(feat_dim).astype(np.float32) * 0.01
    norm = np.linalg.norm(node_feats, axis=1, keepdims=True) + 1e-8
    x_t = torch.tensor(node_feats / norm, dtype=torch.float32)

    perm = np.random.permutation(len(fwd_s))
    n_tr = int(0.8 * len(fwd_s))
    tr_s = fwd_s[perm[:n_tr]]
    tr_d = fwd_d[perm[:n_tr]]
    te_s = fwd_s[perm[n_tr:]]
    te_d = fwd_d[perm[n_tr:]]

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for s, d in zip(tr_s, tr_d):
        G.add_edge(s, d)
    comms = nx_comm.louvain_communities(G, seed=42)
    node_to_comm = {}
    for cid, c in enumerate(comms):
        for u in c:
            node_to_comm[u] = cid

    c_s = np.array([node_to_comm.get(u, -1) for u in tr_s])
    c_d = np.array([node_to_comm.get(v, -2) for v in tr_d])
    intra_mask = (c_s == c_d)

    stage3a_s = tr_s[intra_mask]
    stage3a_d = tr_d[intra_mask]
    stage3a_edges = torch.tensor(np.stack([np.concatenate([stage3a_s, stage3a_d]), np.concatenate([stage3a_d, stage3a_s])]), dtype=torch.long)
    full_graph_edges = torch.tensor(np.stack([np.concatenate([tr_s, tr_d]), np.concatenate([tr_d, tr_s])]), dtype=torch.long)

    existing_set = set(zip(tr_s.tolist(), tr_d.tolist())).union(set(zip(tr_d.tolist(), tr_s.tolist())))
    return x_t, n_nodes, tr_s, tr_d, stage3a_s, stage3a_d, stage3a_edges, full_graph_edges, te_s, te_d, existing_set

def prepare_products():
    print("Loading ogbn-products dataset...")
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name='ogbn-products', root='data/ogb')
    data = ds[0]
    n_nodes = data.num_nodes
    node_comms = data.y.squeeze(-1).numpy()

    src_all = data.edge_index[0].numpy()
    dst_all = data.edge_index[1].numpy()
    mask = src_all < dst_all
    fwd_s = src_all[mask]
    fwd_d = dst_all[mask]

    np.random.seed(42)
    sample_sz = min(100000, len(fwd_s))
    perm = np.random.permutation(len(fwd_s))[:sample_sz]
    n_tr = int(0.8 * sample_sz)

    tr_s = fwd_s[perm[:n_tr]]
    tr_d = fwd_d[perm[:n_tr]]
    te_s = fwd_s[perm[n_tr:]]
    te_d = fwd_d[perm[n_tr:]]

    intra_mask = (node_comms[tr_s] == node_comms[tr_d])
    stage3a_s = tr_s[intra_mask]
    stage3a_d = tr_d[intra_mask]
    stage3a_edges = torch.tensor(np.stack([np.concatenate([stage3a_s, stage3a_d]), np.concatenate([stage3a_d, stage3a_s])]), dtype=torch.long)
    full_graph_edges = torch.tensor(np.stack([np.concatenate([tr_s, tr_d]), np.concatenate([tr_d, tr_s])]), dtype=torch.long)

    existing_set = set(zip(tr_s.tolist(), tr_d.tolist())).union(set(zip(tr_d.tolist(), tr_s.tolist())))
    return data.x, n_nodes, tr_s, tr_d, stage3a_s, stage3a_d, stage3a_edges, full_graph_edges, te_s, te_d, existing_set

def prepare_mag():
    print("Loading ogbn-mag dataset...")
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name='ogbn-mag', root='data/ogb')
    data = ds[0]
    n_nodes = data.num_nodes_dict['paper']
    feat_t = data.x_dict['paper']
    edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]
    node_comms = data.y_dict['paper'].squeeze(-1).numpy()

    src_all = edge_index[0].numpy()
    dst_all = edge_index[1].numpy()
    mask = src_all < dst_all
    fwd_s = src_all[mask]
    fwd_d = dst_all[mask]

    np.random.seed(42)
    sample_sz = min(100000, len(fwd_s))
    perm = np.random.permutation(len(fwd_s))[:sample_sz]
    n_tr = int(0.8 * sample_sz)

    tr_s = fwd_s[perm[:n_tr]]
    tr_d = fwd_d[perm[:n_tr]]
    te_s = fwd_s[perm[n_tr:]]
    te_d = fwd_d[perm[n_tr:]]

    intra_mask = (node_comms[tr_s] == node_comms[tr_d])
    stage3a_s = tr_s[intra_mask]
    stage3a_d = tr_d[intra_mask]
    stage3a_edges = torch.tensor(np.stack([np.concatenate([stage3a_s, stage3a_d]), np.concatenate([stage3a_d, stage3a_s])]), dtype=torch.long)
    full_graph_edges = torch.tensor(np.stack([np.concatenate([tr_s, tr_d]), np.concatenate([tr_d, tr_s])]), dtype=torch.long)

    existing_set = set(zip(tr_s.tolist(), tr_d.tolist())).union(set(zip(tr_d.tolist(), tr_s.tolist())))
    return feat_t, n_nodes, tr_s, tr_d, stage3a_s, stage3a_d, stage3a_edges, full_graph_edges, te_s, te_d, existing_set

def prepare_livejournal():
    print("Loading LiveJournal dataset...")
    gz_path = "data/livejournal/com-lj.ungraph.txt.gz"
    chunks = []
    chunk_sz = 200000
    total_read = 0
    with gzip.open(gz_path, 'rt') as f:
        reader = pd.read_csv(f, sep=r'\s+', comment='#', header=None, names=['src', 'dst'], chunksize=chunk_sz)
        for c in reader:
            chunks.append(c)
            total_read += len(c)
            if total_read >= 250000:
                break
    edges_df = pd.concat(chunks, ignore_index=True)
    raw_src = edges_df['src'].values.astype(np.int64)
    raw_dst = edges_df['dst'].values.astype(np.int64)

    unique_nodes = np.unique(np.concatenate([raw_src, raw_dst]))
    n_nodes = len(unique_nodes)
    node_map = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    src = np.array([node_map[u] for u in raw_src], dtype=np.int64)
    dst = np.array([node_map[v] for v in raw_dst], dtype=np.int64)

    np.random.seed(42)
    feat_dim = 128
    node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float32)
    deg = np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)
    node_feats[:, :16] += np.log1p(deg)[:, None] * 0.5
    node_feats = node_feats / (np.linalg.norm(node_feats, axis=1, keepdims=True) + 1e-8)
    x_t = torch.tensor(node_feats, dtype=torch.float32)

    mask = src < dst
    fwd_s, fwd_d = src[mask], dst[mask]
    sample_sz = min(60000, len(fwd_s))
    perm = np.random.permutation(len(fwd_s))[:sample_sz]
    n_tr = int(0.8 * sample_sz)

    tr_s = fwd_s[perm[:n_tr]]
    tr_d = fwd_d[perm[:n_tr]]
    te_s = fwd_s[perm[n_tr:]]
    te_d = fwd_d[perm[n_tr:]]

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for s, d in zip(tr_s[:30000], tr_d[:30000]):
        G.add_edge(s, d)
    comms = nx_comm.louvain_communities(G, seed=42)
    node_to_comm = {}
    for cid, c in enumerate(comms):
        for u in c:
            node_to_comm[u] = cid
    c_s = np.array([node_to_comm.get(u, -1) for u in tr_s])
    c_d = np.array([node_to_comm.get(v, -2) for v in tr_d])
    intra_mask = (c_s == c_d) & (c_s != -1)

    stage3a_s = tr_s[intra_mask]
    stage3a_d = tr_d[intra_mask]
    stage3a_edges = torch.tensor(np.stack([np.concatenate([stage3a_s, stage3a_d]), np.concatenate([stage3a_d, stage3a_s])]), dtype=torch.long)
    full_graph_edges = torch.tensor(np.stack([np.concatenate([tr_s, tr_d]), np.concatenate([tr_d, tr_s])]), dtype=torch.long)

    existing_set = set(zip(tr_s.tolist(), tr_d.tolist())).union(set(zip(tr_d.tolist(), tr_s.tolist())))
    return x_t, n_nodes, tr_s, tr_d, stage3a_s, stage3a_d, stage3a_edges, full_graph_edges, te_s, te_d, existing_set

def prepare_orkut():
    print("Loading Orkut dataset...")
    gz_path = "data/orkut/com-orkut.ungraph.txt.gz"
    chunks = []
    chunk_sz = 200000
    total_read = 0
    with gzip.open(gz_path, 'rt') as f:
        reader = pd.read_csv(f, sep=r'\s+', comment='#', header=None, names=['src', 'dst'], chunksize=chunk_sz)
        for c in reader:
            chunks.append(c)
            total_read += len(c)
            if total_read >= 250000:
                break
    edges_df = pd.concat(chunks, ignore_index=True)
    raw_src = edges_df['src'].values.astype(np.int64)
    raw_dst = edges_df['dst'].values.astype(np.int64)

    unique_nodes = np.unique(np.concatenate([raw_src, raw_dst]))
    n_nodes = len(unique_nodes)
    node_map = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    src = np.array([node_map[u] for u in raw_src], dtype=np.int64)
    dst = np.array([node_map[v] for v in raw_dst], dtype=np.int64)

    np.random.seed(42)
    feat_dim = 128
    node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float32)
    deg = np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)
    node_feats[:, :16] += np.log1p(deg)[:, None] * 0.5
    node_feats = node_feats / (np.linalg.norm(node_feats, axis=1, keepdims=True) + 1e-8)
    x_t = torch.tensor(node_feats, dtype=torch.float32)

    mask = src < dst
    fwd_s, fwd_d = src[mask], dst[mask]
    sample_sz = min(60000, len(fwd_s))
    perm = np.random.permutation(len(fwd_s))[:sample_sz]
    n_tr = int(0.8 * sample_sz)

    tr_s = fwd_s[perm[:n_tr]]
    tr_d = fwd_d[perm[:n_tr]]
    te_s = fwd_s[perm[n_tr:]]
    te_d = fwd_d[perm[n_tr:]]

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for s, d in zip(tr_s[:30000], tr_d[:30000]):
        G.add_edge(s, d)
    comms = nx_comm.louvain_communities(G, seed=42)
    node_to_comm = {}
    for cid, c in enumerate(comms):
        for u in c:
            node_to_comm[u] = cid
    c_s = np.array([node_to_comm.get(u, -1) for u in tr_s])
    c_d = np.array([node_to_comm.get(v, -2) for v in tr_d])
    intra_mask = (c_s == c_d) & (c_s != -1)

    stage3a_s = tr_s[intra_mask]
    stage3a_d = tr_d[intra_mask]
    stage3a_edges = torch.tensor(np.stack([np.concatenate([stage3a_s, stage3a_d]), np.concatenate([stage3a_d, stage3a_s])]), dtype=torch.long)
    full_graph_edges = torch.tensor(np.stack([np.concatenate([tr_s, tr_d]), np.concatenate([tr_d, tr_s])]), dtype=torch.long)

    existing_set = set(zip(tr_s.tolist(), tr_d.tolist())).union(set(zip(tr_d.tolist(), tr_s.tolist())))
    return x_t, n_nodes, tr_s, tr_d, stage3a_s, stage3a_d, stage3a_edges, full_graph_edges, te_s, te_d, existing_set

# -------------------------------------------------------------
# Main Runner
# -------------------------------------------------------------

def run_all_ablations():
    datasets = [
        ("Reddit", prepare_reddit),
        ("ogbn-products", prepare_products),
        ("ogbn-mag", prepare_mag),
        ("Deezer Europe", prepare_deezer),
        ("LiveJournal", prepare_livejournal),
        ("Orkut", prepare_orkut),
    ]

    decoupled_results = []
    fullgraph_results = []

    print("=" * 80)
    print("STARTING ISOLATED FACTOR ABLATION STUDY ACROSS ALL BENCHMARK DATASETS")
    print("=" * 80)

    for dname, prep_fn in datasets:
        print(f"\n>>> Running Ablations for Dataset: {dname}")
        t0_d = time.time()
        x, n_nodes, tr_s, tr_d, stage3a_s, stage3a_d, stage3a_edges, fg_edges, te_s, te_d, ex_set = prep_fn()

        # Decoupled Configurations:
        # Config 1: Old Decoder + Uniform Sampling + No Halo (Layer 1=intra, Layer 2=intra, intra-edges)
        auc_dec_c1 = train_and_eval_config(
            x, stage3a_edges, stage3a_edges, stage3a_s, stage3a_d, te_s, te_d, n_nodes, ex_set,
            decoder_type='old', use_rejection=False
        )
        # Config 2: NEW Decoder + Uniform Sampling + No Halo (Decoder Effect Only)
        auc_dec_c2 = train_and_eval_config(
            x, stage3a_edges, stage3a_edges, stage3a_s, stage3a_d, te_s, te_d, n_nodes, ex_set,
            decoder_type='new', use_rejection=False
        )
        # Config 3: Old Decoder + Rejection Sampling + No Halo (Sampling Effect Only)
        auc_dec_c3 = train_and_eval_config(
            x, stage3a_edges, stage3a_edges, stage3a_s, stage3a_d, te_s, te_d, n_nodes, ex_set,
            decoder_type='old', use_rejection=True
        )
        # Config 4: Old Decoder + Uniform Sampling + 1-Hop Halo (Halo Effect Only: Layer 1=intra, Layer 2=all 1-hop boundary edges)
        auc_dec_c4 = train_and_eval_config(
            x, stage3a_edges, fg_edges, tr_s, tr_d, te_s, te_d, n_nodes, ex_set,
            decoder_type='old', use_rejection=False
        )

        decoupled_results.append({
            'dataset': dname,
            'c1': auc_dec_c1,
            'c2': auc_dec_c2,
            'c3': auc_dec_c3,
            'c4': auc_dec_c4
        })

        # Full-Graph Configurations:
        # Config 1: Old Decoder + Uniform Sampling (Full-Graph GraphSAGE Baseline: Layer 1=all, Layer 2=all)
        auc_fg_c1 = train_and_eval_config(
            x, fg_edges, fg_edges, tr_s, tr_d, te_s, te_d, n_nodes, ex_set,
            decoder_type='old', use_rejection=False
        )
        # Config 2: NEW Decoder + Uniform Sampling (Decoder Effect Only)
        auc_fg_c2 = train_and_eval_config(
            x, fg_edges, fg_edges, tr_s, tr_d, te_s, te_d, n_nodes, ex_set,
            decoder_type='new', use_rejection=False
        )
        # Config 3: Old Decoder + Rejection Sampling (Sampling Effect Only)
        auc_fg_c3 = train_and_eval_config(
            x, fg_edges, fg_edges, tr_s, tr_d, te_s, te_d, n_nodes, ex_set,
            decoder_type='old', use_rejection=True
        )

        fullgraph_results.append({
            'dataset': dname,
            'c1': auc_fg_c1,
            'c2': auc_fg_c2,
            'c3': auc_fg_c3
        })

        print(f"  ✓ Finished {dname} in {time.time()-t0_d:.1f}s:")
        print(f"    Decoupled:  C1={auc_dec_c1:.4f}, C2={auc_dec_c2:.4f}, C3={auc_dec_c3:.4f}, C4={auc_dec_c4:.4f}")
        print(f"    Full-Graph: C1={auc_fg_c1:.4f}, C2={auc_fg_c2:.4f}, C3={auc_fg_c3:.4f}")

    # -------------------------------------------------------------
    # Render Tables
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TABLE 1: DECOUPLED PIPELINE ISOLATED FACTOR ABLATION (ROC-AUC)")
    print("=" * 90)
    print(f"{'Dataset':<16} | {'Config 1 (Original)':<20} | {'Config 2 (Decoder Only)':<24} | {'Config 3 (Sampling Only)':<24} | {'Config 4 (Halo Only)':<20}")
    print(f"{'':<16} | {'Old Dec + Uniform':<20} | {'NEW Dec + Uniform':<24} | {'Old Dec + Rejection':<24} | {'Old Dec + 1-Hop Halo':<20}")
    print("-" * 114)
    for r in decoupled_results:
        print(f"{r['dataset']:<16} | {r['c1']:<20.4f} | {r['c2']:<24.4f} | {r['c3']:<24.4f} | {r['c4']:<20.4f}")
    print("=" * 114)

    print("\n" + "=" * 90)
    print("TABLE 2: FULL-GRAPH GRAPHSAGE ISOLATED FACTOR ABLATION (ROC-AUC)")
    print("=" * 90)
    print(f"{'Dataset':<16} | {'Config 1 (Full-Graph SAGE)':<26} | {'Config 2 (Decoder Only)':<24} | {'Config 3 (Sampling Only)':<24}")
    print(f"{'':<16} | {'Old Dec + Uniform':<26} | {'NEW Dec + Uniform':<24} | {'Old Dec + Rejection':<24}")
    print("-" * 96)
    for r in fullgraph_results:
        print(f"{r['dataset']:<16} | {r['c1']:<26.4f} | {r['c2']:<24.4f} | {r['c3']:<24.4f}")
    print("=" * 96 + "\n")

    out_path = "data/isolated_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump({'decoupled': decoupled_results, 'full_graph': fullgraph_results}, f, indent=2)
    print(f"Results successfully saved to {out_path}")

if __name__ == '__main__':
    run_all_ablations()
