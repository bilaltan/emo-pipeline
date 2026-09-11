"""
Unit test for Link Prediction components:
1. 4-way Siamese Link Predictor vs. Hadamard baseline
2. Rejection negative sampling (verifying zero false negatives & no self loops)
3. Simulated community link prediction training & ROC-AUC evaluation
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def test_rejection_sampling():
    print("=== Testing Rejection Negative Sampling ===")
    n_nodes = 100
    # Create a dense cluster with 500 edges
    src = np.random.randint(0, n_nodes, 500)
    dst = np.random.randint(0, n_nodes, 500)
    # remove self loops from positive edges
    valid = src != dst
    src, dst = src[valid], dst[valid]
    existing_edge_set = set(zip(src.tolist(), dst.tolist()))
    
    def sample_negative_edges_local(n_nodes_cnt, n_samples):
        neg_s_list = []
        neg_d_list = []
        needed = n_samples
        for _ in range(8):
            c_s = torch.randint(0, n_nodes_cnt, (needed * 2,))
            c_d = torch.randint(0, n_nodes_cnt, (needed * 2,))
            valid_mask = (c_s != c_d)
            c_s_np = c_s[valid_mask].tolist()
            c_d_np = c_d[valid_mask].tolist()
            for s_cand, d_cand in zip(c_s_np, c_d_np):
                if (s_cand, d_cand) not in existing_edge_set and (d_cand, s_cand) not in existing_edge_set:
                    neg_s_list.append(s_cand)
                    neg_d_list.append(d_cand)
                    if len(neg_s_list) >= n_samples:
                        break
            if len(neg_s_list) >= n_samples:
                break
            needed = n_samples - len(neg_s_list)
        if len(neg_s_list) < n_samples:
            rem = n_samples - len(neg_s_list)
            f_s = torch.randint(0, n_nodes_cnt, (rem,))
            f_d = torch.randint(0, n_nodes_cnt, (rem,))
            neg_s_list.extend(f_s.tolist())
            neg_d_list.extend(f_d.tolist())
        return torch.tensor(neg_s_list[:n_samples], dtype=torch.int64), torch.tensor(neg_d_list[:n_samples], dtype=torch.int64)

    neg_s, neg_d = sample_negative_edges_local(n_nodes, 200)
    assert len(neg_s) == 200, f"Expected 200, got {len(neg_s)}"
    assert len(neg_d) == 200, f"Expected 200, got {len(neg_d)}"
    
    # Check no false negatives
    false_negs = 0
    self_loops = 0
    for s, d in zip(neg_s.tolist(), neg_d.tolist()):
        if (s, d) in existing_edge_set or (d, s) in existing_edge_set:
            false_negs += 1
        if s == d:
            self_loops += 1
            
    print(f"  Sampled 200 negatives: False Negatives = {false_negs}, Self Loops = {self_loops}")
    assert false_negs == 0, f"Rejection sampling failed: found {false_negs} false negatives!"
    assert self_loops == 0, f"Rejection sampling failed: found {self_loops} self loops!"
    print("  ✓ Rejection sampling passed with 100% purity!")


def test_decoder_architectures():
    print("\n=== Testing Decoder Architectures & Expressiveness ===")
    h_dim = 128
    n_pairs = 1000
    
    # Old Hadamard Predictor
    class OldHadamardPredictor(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.fc1 = nn.Linear(h, h)
            self.fc2 = nn.Linear(h, 1)
        def forward(self, h_src, h_dst):
            x = h_src * h_dst
            x = torch.relu(self.fc1(x))
            return self.fc2(x).squeeze(-1)

    # New 4-way Siamese Predictor
    class Siamese4WayPredictor(nn.Module):
        def __init__(self, h, dropout=0.1):
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

    old_pred = OldHadamardPredictor(h_dim)
    new_pred = Siamese4WayPredictor(h_dim)

    # Synthetic node embeddings
    torch.manual_seed(42)
    h_src = torch.randn(n_pairs, h_dim)
    h_dst = torch.randn(n_pairs, h_dim)

    out_old = old_pred(h_src, h_dst)
    out_new = new_pred(h_src, h_dst)

    assert out_old.shape == (n_pairs,), f"Old output shape mismatch: {out_old.shape}"
    assert out_new.shape == (n_pairs,), f"New output shape mismatch: {out_new.shape}"
    print(f"  Old Hadamard output shape: {out_old.shape} | New Siamese output shape: {out_new.shape}")
    print("  ✓ Decoder shapes and forward passes verified!")

    # Check gradient flow
    loss_new = out_new.sum()
    loss_new.backward()
    assert new_pred.fc1.weight.grad is not None
    assert new_pred.fc2.weight.grad is not None
    print("  ✓ Gradient backward pass verified for 4-way Siamese decoder!")


def test_link_prediction_synthetic_simulation():
    print("\n=== Simulating Community Link Prediction Training & ROC-AUC ===")
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_nodes = 250
    feat_dim = 64
    hidden_dim = 64
    
    # Generate structured graph with 2 sub-communities
    # Nodes 0-124 in cluster A, 125-249 in cluster B
    feats = torch.randn(n_nodes, feat_dim)
    feats[:125] += 1.5  # cluster A signal
    feats[125:] -= 1.5  # cluster B signal
    
    # Intra-cluster edges have high probability, inter-cluster edges have low probability
    edge_src = []
    edge_dst = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            prob = 0.12 if (i < 125 and j < 125) or (i >= 125 and j >= 125) else 0.01
            if np.random.rand() < prob:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                
    edge_src = np.array(edge_src, dtype=np.int64)
    edge_dst = np.array(edge_dst, dtype=np.int64)
    n_edges = len(edge_src)
    existing_edge_set = set(zip(edge_src.tolist(), edge_dst.tolist()))
    print(f"  Generated graph: {n_nodes} nodes, {n_edges} directed edges.")

    # Split train / test
    perm = torch.randperm(n_edges)
    n_tr = int(0.8 * n_edges)
    tr_idx = perm[:n_tr]
    te_idx = perm[n_tr:]
    
    src_t = torch.tensor(edge_src, dtype=torch.int64)
    dst_t = torch.tensor(edge_dst, dtype=torch.int64)

    # Simple GCN-like encoder
    class SimpleEncoder(nn.Module):
        def __init__(self, in_f, h):
            super().__init__()
            self.l1 = nn.Linear(in_f, h)
            self.l2 = nn.Linear(h, h)
        def forward(self, x, adj_norm):
            h = F.relu(self.l1(adj_norm @ x))
            return self.l2(adj_norm @ h)

    # Build adjacency matrix with self loop
    adj = torch.zeros(n_nodes, n_nodes)
    adj[src_t[tr_idx], dst_t[tr_idx]] = 1.0
    adj += torch.eye(n_nodes)
    deg = adj.sum(dim=1, keepdim=True)
    adj_norm = adj / deg

    # Test 1: Old (Hadamard + Naive Random Negatives)
    torch.manual_seed(42)
    enc1 = SimpleEncoder(feat_dim, hidden_dim)
    pred1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    opt1 = torch.optim.Adam(list(enc1.parameters()) + list(pred1.parameters()), lr=0.01, weight_decay=5e-4)

    for epoch in range(25):
        enc1.train(); pred1.train()
        h = enc1(feats, adj_norm)
        pos_s, pos_d = src_t[tr_idx], dst_t[tr_idx]
        neg_s = torch.randint(0, n_nodes, (len(tr_idx),))
        neg_d = torch.randint(0, n_nodes, (len(tr_idx),))
        
        pos_scores = pred1(h[pos_s] * h[pos_d]).squeeze(-1)
        neg_scores = pred1(h[neg_s] * h[neg_d]).squeeze(-1)
        
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        )
        opt1.zero_grad(); loss.backward(); opt1.step()

    enc1.eval(); pred1.eval()
    with torch.no_grad():
        h = enc1(feats, adj_norm)
        pos_scores = pred1(h[src_t[te_idx]] * h[dst_t[te_idx]]).squeeze(-1)
        neg_s = torch.randint(0, n_nodes, (len(te_idx),))
        neg_d = torch.randint(0, n_nodes, (len(te_idx),))
        neg_scores = pred1(h[neg_s] * h[neg_d]).squeeze(-1)
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        auc_old = roc_auc_score(y_true, y_score)

    # Test 2: New (4-Way Siamese Decoder + Rejection Sampling)
    def sample_clean_negs(n_samples):
        neg_s_list, neg_d_list = [], []
        needed = n_samples
        for _ in range(8):
            cs = torch.randint(0, n_nodes, (needed * 2,))
            cd = torch.randint(0, n_nodes, (needed * 2,))
            valid = (cs != cd)
            for s, d in zip(cs[valid].tolist(), cd[valid].tolist()):
                if (s, d) not in existing_edge_set and (d, s) not in existing_edge_set:
                    neg_s_list.append(s)
                    neg_d_list.append(d)
                    if len(neg_s_list) >= n_samples:
                        break
            if len(neg_s_list) >= n_samples:
                break
            needed = n_samples - len(neg_s_list)
        return torch.tensor(neg_s_list[:n_samples]), torch.tensor(neg_d_list[:n_samples])

    torch.manual_seed(42)
    enc2 = SimpleEncoder(feat_dim, hidden_dim)
    pred2 = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    opt2 = torch.optim.Adam(list(enc2.parameters()) + list(pred2.parameters()), lr=0.01, weight_decay=5e-4)

    def score_siamese(pred, h, src_idx, dst_idx):
        h_s = h[src_idx]
        h_d = h[dst_idx]
        cat = torch.cat([h_s, h_d, torch.abs(h_s - h_d), h_s * h_d], dim=-1)
        return pred(cat).squeeze(-1)

    for epoch in range(25):
        enc2.train(); pred2.train()
        h = enc2(feats, adj_norm)
        pos_s, pos_d = src_t[tr_idx], dst_t[tr_idx]
        neg_s, neg_d = sample_clean_negs(len(tr_idx))
        
        pos_scores = score_siamese(pred2, h, pos_s, pos_d)
        neg_scores = score_siamese(pred2, h, neg_s, neg_d)
        
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        )
        opt2.zero_grad(); loss.backward(); opt2.step()

    enc2.eval(); pred2.eval()
    with torch.no_grad():
        h = enc2(feats, adj_norm)
        pos_scores = score_siamese(pred2, h, src_t[te_idx], dst_t[te_idx])
        neg_s, neg_d = sample_clean_negs(len(te_idx))
        neg_scores = score_siamese(pred2, h, neg_s, neg_d)
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        auc_new = roc_auc_score(y_true, y_score)

    print(f"\n  Results Comparison:")
    print(f"    - Old (Hadamard + Naive Random Negatives): ROC-AUC = {auc_old:.4f}")
    print(f"    - New (4-Way Siamese + Rejection Negatives): ROC-AUC = {auc_new:.4f}")
    print(f"    - Improvement: {(auc_new - auc_old)*100:+.2f} percentage points!")
    assert auc_new > auc_old, "Expected upgraded architecture to beat naive baseline!"


if __name__ == '__main__':
    test_rejection_sampling()
    test_decoder_architectures()
    test_link_prediction_synthetic_simulation()
    print("\nAll unit tests passed successfully!")
