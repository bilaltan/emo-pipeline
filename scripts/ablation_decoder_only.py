"""
Rigorous Decoder-Only Ablation Study & Fair Comparison on Upper Bound (Full Graph).

Controlled Variables:
- Negative Sampling: Naive uniform torch.randint for ALL configurations (NO rejection sampling).
- Halo Expansion: Completely disabled for Decoupled configurations (severed cut edges ONLY).

2x2 Comparison Matrix:
1. Full Graph Baseline   + Old Hadamard Decoder
2. Full Graph Baseline   + New 4-Way Siamese Decoder
3. Decoupled (Stage 3a)  + Old Hadamard Decoder
4. Decoupled (Stage 3a)  + New 4-Way Siamese Decoder ONLY
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Reddit
from torch_geometric.nn import SAGEConv

class PyGLinkEncoder(nn.Module):
    def __init__(self, in_f, h=128, dropout=0.1):
        super().__init__()
        self.c1 = SAGEConv(in_f, h)
        self.c2 = SAGEConv(h, h)
        self.dr = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = self.dr(x)
        return self.c2(x, edge_index)

# 1. Old Hadamard Predictor
class OldHadamardPredictor(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, 1)
    def forward(self, h_src, h_dst):
        x = h_src * h_dst
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# 2. New 4-Way Siamese Predictor
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

def train_and_eval(name, enc, pred, x, graph_edges, train_src, train_dst, test_src, test_dst, n_nodes, epochs=15, batch_sz=10000, lr=0.01):
    torch.manual_seed(42)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr, weight_decay=5e-4)
    tr_s_t = torch.tensor(train_src, dtype=torch.long)
    tr_d_t = torch.tensor(train_dst, dtype=torch.long)
    
    t0 = time.time()
    for epoch in range(epochs):
        enc.train(); pred.train()
        h = enc(x, graph_edges)
        b_idx = torch.randint(0, len(tr_s_t), (batch_sz,))
        pos_s, pos_d = tr_s_t[b_idx], tr_d_t[b_idx]
        # STRICT CONTROL: Naive random negative sampling for all
        neg_s = torch.randint(0, n_nodes, (batch_sz,))
        neg_d = torch.randint(0, n_nodes, (batch_sz,))
        
        pos_sc = pred(h[pos_s], h[pos_d])
        neg_sc = pred(h[neg_s], h[neg_d])
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_sc, neg_sc]),
            torch.cat([torch.ones_like(pos_sc), torch.zeros_like(neg_sc)])
        )
        opt.zero_grad(); loss.backward(); opt.step()
        
    enc.eval(); pred.eval()
    with torch.no_grad():
        h = enc(x, graph_edges)
        te_s = torch.tensor(test_src, dtype=torch.long)
        te_d = torch.tensor(test_dst, dtype=torch.long)
        pos_sc = pred(h[te_s], h[te_d])
        # Naive random negative evaluation
        neg_s = torch.randint(0, n_nodes, (len(test_src),))
        neg_d = torch.randint(0, n_nodes, (len(test_src),))
        neg_sc = pred(neg_s, neg_d) if not callable(pred) else pred(h[neg_s], h[neg_d])
        y_true = np.concatenate([np.ones(len(pos_sc)), np.zeros(len(neg_sc))])
        y_sc = torch.cat([pos_sc, neg_sc]).cpu().numpy()
        auc = roc_auc_score(y_true, y_sc)
    t_elapsed = time.time() - t0
    print(f"  ✓ {name:<42}: ROC-AUC = {auc:.4f} (took {t_elapsed:.1f}s)")
    return auc

def run_reddit_ablation():
    print("=" * 80)
    print("   STRICT DECODER-ONLY ABLATION & FAIR FULL-GRAPH BASELINE ON REDDIT")
    print("=" * 80)
    print("Loading Reddit dataset from data/reddit...")
    t0 = time.time()
    ds = Reddit(root='data/reddit')
    data = ds[0]
    n_nodes = data.x.size(0)
    print(f"  ✓ Loaded Reddit: {n_nodes:,} nodes in {time.time()-t0:.1f}s")
    
    src_all = data.edge_index[0].numpy()
    dst_all = data.edge_index[1].numpy()
    mask = src_all < dst_all
    fwd_src = src_all[mask]
    fwd_dst = dst_all[mask]
    
    np.random.seed(42)
    torch.manual_seed(42)
    sample_size = min(100000, len(fwd_src))
    perm = np.random.permutation(len(fwd_src))[:sample_size]
    
    n_tr = int(0.8 * sample_size)
    train_src = fwd_src[perm[:n_tr]]
    train_dst = fwd_dst[perm[:n_tr]]
    test_src = fwd_src[perm[n_tr:]]
    test_dst = fwd_dst[perm[n_tr:]]
    
    # Community boundaries: 41 subreddits
    node_comms = data.y.numpy()
    intra_mask = (node_comms[train_src] == node_comms[train_dst])
    cut_mask = ~intra_mask
    print(f"  Sample: {len(train_src):,} train edges ({intra_mask.sum():,} intra, {cut_mask.sum():,} boundary cut).")
    print(f"  Test:   {len(test_src):,} test edges.")
    print("-" * 80)
    
    # 1. Full Graph graph edges (Intra + Boundary edges preserved)
    full_graph_edges = torch.tensor(
        np.stack([np.concatenate([train_src, train_dst]), np.concatenate([train_dst, train_src])]),
        dtype=torch.long
    )
    
    # 2. Decoupled graph edges (Intra-community ONLY, boundary cut edges SEVERED)
    stage3a_src = train_src[intra_mask]
    stage3a_dst = train_dst[intra_mask]
    stage3a_edges = torch.tensor(
        np.stack([np.concatenate([stage3a_src, stage3a_dst]), np.concatenate([stage3a_dst, stage3a_src])]),
        dtype=torch.long
    )
    
    print("\nRunning 2x2 Controlled Experiments (All using identical naive negative sampling):\n")
    
    # Experiment 1: Full Graph + Old Hadamard
    enc_fg_old = PyGLinkEncoder(data.x.size(1), h=128)
    pred_fg_old = OldHadamardPredictor(h=128)
    auc_fg_old = train_and_eval(
        "1. Full Graph Upper Bound + Old Hadamard",
        enc_fg_old, pred_fg_old, data.x, full_graph_edges,
        train_src, train_dst, test_src, test_dst, n_nodes
    )
    
    # Experiment 2: Full Graph + New 4-Way Siamese (Fair baseline comparison)
    enc_fg_new = PyGLinkEncoder(data.x.size(1), h=128)
    pred_fg_new = Siamese4WayPredictor(h=128)
    auc_fg_new = train_and_eval(
        "2. Full Graph Upper Bound + New 4-Way Siamese",
        enc_fg_new, pred_fg_new, data.x, full_graph_edges,
        train_src, train_dst, test_src, test_dst, n_nodes
    )
    
    # Experiment 3: Decoupled Stage 3a + Old Hadamard (Original Decoupled Baseline)
    enc_dec_old = PyGLinkEncoder(data.x.size(1), h=128)
    pred_dec_old = OldHadamardPredictor(h=128)
    auc_dec_old = train_and_eval(
        "3. Decoupled Stage 3a   + Old Hadamard",
        enc_dec_old, pred_dec_old, data.x, stage3a_edges,
        stage3a_src, stage3a_dst, test_src, test_dst, n_nodes
    )
    
    # Experiment 4: Decoupled Stage 3a + New 4-Way Siamese ONLY (Isolated Decoder Effect)
    # NOTE: NO halo expansion (severed stage3a_edges), NO rejection sampling!
    enc_dec_new = PyGLinkEncoder(data.x.size(1), h=128)
    pred_dec_new = Siamese4WayPredictor(h=128)
    auc_dec_new = train_and_eval(
        "4. Decoupled Stage 3a   + New 4-Way Siamese ONLY",
        enc_dec_new, pred_dec_new, data.x, stage3a_edges,
        stage3a_src, stage3a_dst, test_src, test_dst, n_nodes
    )
    
    print("\n" + "=" * 80)
    print("                    ABLATION & FAIR COMPARISON MATRIX (REDDIT)")
    print("=" * 80)
    print(f"{'Configuration':<45} | {'ROC-AUC':<9} | {'Decoder Gain':<13} | {'Retention':<10}")
    print("-" * 80)
    print(f"{'Full Graph (Upper Bound) - Old Hadamard':<45} | {auc_fg_old:<9.4f} | {'Baseline':<13} | {'100.0%':<10}")
    print(f"{'Full Graph (Upper Bound) - New 4-Way Siamese':<45} | {auc_fg_new:<9.4f} | {f'{auc_fg_new - auc_fg_old:+.4f}':<13} | {'100.0%':<10}")
    print("-" * 80)
    print(f"{'Decoupled Stage 3a - Old Hadamard':<45} | {auc_dec_old:<9.4f} | {'Baseline':<13} | {f'{auc_dec_old / auc_fg_old * 100:.1f}%':<10}")
    print(f"{'Decoupled Stage 3a - New 4-Way Siamese ONLY':<45} | {auc_dec_new:<9.4f} | {f'{auc_dec_new - auc_dec_old:+.4f}':<13} | {f'{auc_dec_new / auc_fg_new * 100:.1f}%':<10}")
    print("=" * 80)
    
    dec_gain_fg = (auc_fg_new - auc_fg_old) * 100
    dec_gain_decoupled = (auc_dec_new - auc_dec_old) * 100
    ret_old = (auc_dec_old / auc_fg_old) * 100
    ret_new = (auc_dec_new / auc_fg_new) * 100
    
    print(f"\nKey Takeaways for Your Professor:")
    print(f"1. Isolated Decoder Gain on Decoupled Stage 3a:  {dec_gain_decoupled:+.2f} percentage points (without any halo or rejection sampling).")
    print(f"2. Fair Comparison on Full Graph Upper Bound:    {dec_gain_fg:+.2f} percentage points.")
    print(f"3. Apples-to-Apples Retention Comparison:")
    print(f"   - Under Old Hadamard: Decoupled retains {ret_old:.1f}% of Full Graph.")
    print(f"   - Under New Siamese:  Decoupled retains {ret_new:.1f}% of Full Graph.")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    run_reddit_ablation()
