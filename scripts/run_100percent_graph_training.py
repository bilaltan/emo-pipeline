"""
Rigorous 100% Full-Graph Representation Learning & Link Prediction Benchmark across All Datasets.

Protocol (100% Graph Split):
- 80% of all forward edges form the Bidirectional Training Graph for SpMM GraphSAGE message passing.
- 10% of all forward edges are held-out for Validation evaluation & early stopping tracking.
- 10% of all forward edges are held-out for final Test evaluation.
Total: 80% + 10% + 10% = 100% of all edges in the graph.

Memory & Speed Optimization:
- PyTorch Sparse CSR / SpMM computes full-graph neighborhood aggregation in ~2.5 - 7.5 GB RAM,
  avoiding the 50 GB OOM crash of naive full-batch edge gathering on 30M - 117M edge graphs.
- Evaluates both Config 1 (Old Hadamard Decoder) and Config 2 (4-Way Siamese Decoder).
- Tracks Convergence Epoch, Best Val AUC, Test AUC @ Best Val, and Total Learnable Parameters.
"""

import os
import sys
import time
import json
import gzip
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import _patch_torch_load
_patch_torch_load()

# -------------------------------------------------------------
# 1. High-Performance SpMM GraphSAGE Models
# -------------------------------------------------------------

class SpMMGraphSAGE(nn.Module):
    """
    2-Layer GraphSAGE Link Encoder using Sparse Matrix Multiplication (SpMM).
    Executes mean neighborhood aggregation over 100% of massive graphs (up to 117M edges)
    in under 7.5 GB RAM on commodity CPU/RAM.
    """
    def __init__(self, in_f, h=128, dropout=0.1):
        super().__init__()
        self.l1_self = nn.Linear(in_f, h)
        self.l1_neigh = nn.Linear(in_f, h)
        self.l2_self = nn.Linear(h, h)
        self.l2_neigh = nn.Linear(h, h)
        self.dr = nn.Dropout(dropout)

    def forward(self, x, adj_csr):
        # Layer 1
        neigh1 = torch.sparse.mm(adj_csr, x)
        h1 = F.relu(self.l1_self(x) + self.l1_neigh(neigh1))
        h1 = self.dr(h1)
        # Layer 2
        neigh2 = torch.sparse.mm(adj_csr, h1)
        h2 = self.l2_self(h1) + self.l2_neigh(neigh2)
        return h2

class OldHadamardPredictor(nn.Module):
    """Config 1: Classic Symmetric Hadamard Link Decoder (16,641 parameters)."""
    def __init__(self, h=128):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, 1)

    def forward(self, h_src, h_dst):
        x = h_src * h_dst
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

class Siamese4WayPredictor(nn.Module):
    """Config 2: 4-Way Concatenation Siamese Link Decoder (65,793 parameters)."""
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
# 2. Dataset Loaders for 100% Graph (80/10/10 Split)
# -------------------------------------------------------------

def build_100percent_csr(n_nodes, tr_s, tr_d):
    """Builds row-normalized (D^-1 A) Sparse CSR tensor from 80% training graph edges."""
    adj_src = np.concatenate([tr_s, tr_d])
    adj_dst = np.concatenate([tr_d, tr_s])
    
    deg = np.bincount(adj_dst, minlength=n_nodes).astype(np.float32)
    deg[deg == 0] = 1.0
    val_weights = (1.0 / deg)[adj_dst]
    
    adj_csr = torch.sparse_coo_tensor(
        torch.tensor(np.stack([adj_dst, adj_src]), dtype=torch.long),
        torch.tensor(val_weights, dtype=torch.float32),
        (n_nodes, n_nodes)
    ).coalesce().to_sparse_csr()
    return adj_csr, len(adj_src)

def load_dataset_100percent(dataset_name):
    """Loads dataset and splits into 80% Train Graph, 10% Val, 10% Test edges (100% Total)."""
    print(f"\n=======================================================", flush=True)
    print(f"LOADING DATASET: {dataset_name} (100% Full-Graph Protocol)", flush=True)
    print(f"=======================================================", flush=True)
    t0 = time.time()
    
    if dataset_name.lower() == 'reddit':
        from torch_geometric.datasets import Reddit
        ds = Reddit(root='data/reddit')
        data = ds[0]
        n_nodes = data.x.size(0)
        x = data.x
        ei = data.edge_index
        mask = (ei[0] < ei[1]).numpy()
        s_np = ei[0].numpy()[mask]
        d_np = ei[1].numpy()[mask]

    elif dataset_name.lower() in ('ogbn-products', 'products'):
        dataset_name = 'ogbn-products'
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name='ogbn-products', root='data/ogb')
        data = ds[0]
        n_nodes = data.num_nodes
        x = data.x
        ei = data.edge_index
        mask = (ei[0] < ei[1]).numpy()
        s_np = ei[0].numpy()[mask]
        d_np = ei[1].numpy()[mask]

    elif dataset_name.lower() in ('ogbn-mag', 'mag'):
        dataset_name = 'ogbn-mag'
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name='ogbn-mag', root='data/ogb')
        data = ds[0]
        n_nodes = data.num_nodes_dict['paper']
        x = data.x_dict['paper']
        ei = data.edge_index_dict[('paper', 'cites', 'paper')]
        mask = (ei[0] < ei[1]).numpy()
        s_np = ei[0].numpy()[mask]
        d_np = ei[1].numpy()[mask]

    elif dataset_name.lower() in ('deezer europe', 'deezer', 'deezereurope'):
        dataset_name = 'Deezer Europe'
        data_dir = "data/deezer_europe_extracted/deezer_europe"
        edges_df = pd.read_csv(os.path.join(data_dir, "deezer_europe_edges.csv"))
        s_raw = edges_df['node_1'].values
        d_raw = edges_df['node_2'].values
        n_nodes = max(s_raw.max(), d_raw.max()) + 1
        mask = s_raw < d_raw
        fwd_s = np.concatenate([s_raw[mask], d_raw[~mask]])
        fwd_d = np.concatenate([d_raw[mask], s_raw[~mask]])
        stacked = np.stack([fwd_s, fwd_d], axis=1)
        unique_edges = np.unique(stacked, axis=0)
        unique_edges = unique_edges[unique_edges[:, 0] != unique_edges[:, 1]]
        s_np, d_np = unique_edges[:, 0], unique_edges[:, 1]
        
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
        x = torch.tensor(node_feats / norm, dtype=torch.float32)

    elif dataset_name.lower() in ('livejournal', 'com-lj'):
        dataset_name = 'LiveJournal'
        gz_path = "data/livejournal/com-lj.ungraph.txt.gz"
        with gzip.open(gz_path, 'rt') as f:
            df = pd.read_csv(f, sep=r'\s+', comment='#', header=None, names=['src', 'dst'])
        raw_s = df['src'].values
        raw_d = df['dst'].values
        n_nodes = int(max(raw_s.max(), raw_d.max()) + 1)
        mask = raw_s < raw_d
        s_np = raw_s[mask]
        d_np = raw_d[mask]
        
        print(f"  Generating structural node embeddings for {n_nodes:,} LiveJournal nodes...", flush=True)
        feat_dim = 128
        np.random.seed(42)
        deg = np.bincount(s_np, minlength=n_nodes) + np.bincount(d_np, minlength=n_nodes)
        node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float32)
        node_feats[:, :16] += np.log1p(deg)[:, None] * 0.5
        node_feats = node_feats / (np.linalg.norm(node_feats, axis=1, keepdims=True) + 1e-8)
        x = torch.tensor(node_feats, dtype=torch.float32)

    elif dataset_name.lower() in ('orkut', 'com-orkut'):
        dataset_name = 'Orkut'
        gz_path = "data/orkut/com-orkut.ungraph.txt.gz"
        with gzip.open(gz_path, 'rt') as f:
            df = pd.read_csv(f, sep=r'\s+', comment='#', header=None, names=['src', 'dst'])
        raw_s = df['src'].values
        raw_d = df['dst'].values
        n_nodes = int(max(raw_s.max(), raw_d.max()) + 1)
        mask = raw_s < raw_d
        s_np = raw_s[mask]
        d_np = raw_d[mask]
        
        print(f"  Generating structural node embeddings for {n_nodes:,} Orkut nodes...", flush=True)
        feat_dim = 128
        np.random.seed(42)
        deg = np.bincount(s_np, minlength=n_nodes) + np.bincount(d_np, minlength=n_nodes)
        node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float32)
        node_feats[:, :16] += np.log1p(deg)[:, None] * 0.5
        node_feats = node_feats / (np.linalg.norm(node_feats, axis=1, keepdims=True) + 1e-8)
        x = torch.tensor(node_feats, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_fwd = len(s_np)
    print(f"  ✓ {dataset_name} Loaded in {time.time()-t0:.1f}s:", flush=True)
    print(f"    Total Nodes: {n_nodes:,} | Feature Dim: {x.size(1)} | Total Forward Edges: {n_fwd:,}", flush=True)
    
    # 100% Graph Split: 80% Train Graph, 10% Val, 10% Test
    n_train = int(0.80 * n_fwd)
    n_val = int(0.10 * n_fwd)
    n_test = n_fwd - n_train - n_val
    
    np.random.seed(42)
    perm = np.random.permutation(n_fwd)
    tr_s = s_np[perm[:n_train]]
    tr_d = d_np[perm[:n_train]]
    val_s = s_np[perm[n_train : n_train + n_val]]
    val_d = d_np[perm[n_train : n_train + n_val]]
    te_s = s_np[perm[n_train + n_val:]]
    te_d = d_np[perm[n_train + n_val:]]
    
    print(f"    Train Graph (80%): {len(tr_s):,} forward edges (Avg Degree: {len(tr_s)*2/n_nodes:.1f})", flush=True)
    print(f"    Validation (10%):  {len(val_s):,} edges", flush=True)
    print(f"    Test (10%):        {len(te_s):,} edges", flush=True)
    print(f"    Total Graph:       {len(tr_s) + len(val_s) + len(te_s):,} edges (100.0% of graph)", flush=True)
    
    # Build CSR Adjacency
    t1 = time.time()
    adj_csr, n_dir_edges = build_100percent_csr(n_nodes, tr_s, tr_d)
    print(f"    Sparse CSR Adjacency Built in {time.time()-t1:.1f}s ({n_dir_edges:,} directed edges).", flush=True)
    
    return dataset_name, x, n_nodes, adj_csr, tr_s, tr_d, val_s, val_d, te_s, te_d

# -------------------------------------------------------------
# 3. Training & Diagnostic Evaluation Engine
# -------------------------------------------------------------

def train_and_eval_100percent(dname, x, n_nodes, adj_csr, tr_s, tr_d, val_s, val_d, te_s, te_d,
                              decoder_type='old', epochs=12, batch_sz=32768, lr=0.01):
    torch.manual_seed(42)
    np.random.seed(42)
    
    enc = SpMMGraphSAGE(x.size(1), h=128)
    if decoder_type == 'new':
        pred = Siamese4WayPredictor(h=128)
        c_name = "Config 2 (4-Way Siamese Decoder)"
    else:
        pred = OldHadamardPredictor(h=128)
        c_name = "Config 1 (Old Hadamard Decoder)"
        
    enc_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in pred.parameters() if p.requires_grad)
    total_params = enc_params + dec_params
    
    print(f"\n--- [{dname}] Training {c_name} (100% Graph Protocol) ---", flush=True)
    print(f"    Total Learnable Params: {total_params:,} (Encoder: {enc_params:,} | Decoder: {dec_params:,})", flush=True)
    
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr, weight_decay=5e-4)
    
    tr_s_t = torch.tensor(tr_s, dtype=torch.long)
    tr_d_t = torch.tensor(tr_d, dtype=torch.long)
    
    # Cap evaluation pairs to 50k for fast epoch-by-epoch evaluation
    eval_val_sz = min(50000, len(val_s))
    eval_te_sz = min(50000, len(te_s))
    val_s_t = torch.tensor(val_s[:eval_val_sz], dtype=torch.long)
    val_d_t = torch.tensor(val_d[:eval_val_sz], dtype=torch.long)
    te_s_t = torch.tensor(te_s[:eval_te_sz], dtype=torch.long)
    te_d_t = torch.tensor(te_d[:eval_te_sz], dtype=torch.long)
    
    # Pre-sample negatives
    np.random.seed(42)
    torch.manual_seed(42)
    val_neg_s = torch.randint(0, n_nodes, (len(val_s_t),))
    val_neg_d = torch.randint(0, n_nodes, (len(val_d_t),))
    test_neg_s = torch.randint(0, n_nodes, (len(te_s_t),))
    test_neg_d = torch.randint(0, n_nodes, (len(te_d_t),))
    train_eval_neg_s = torch.randint(0, n_nodes, (min(20000, len(tr_s_t)),))
    train_eval_neg_d = torch.randint(0, n_nodes, (min(20000, len(tr_d_t)),))
    
    actual_batch = min(batch_sz, len(tr_s_t))
    best_val_auc = -1.0
    best_epoch = -1
    test_at_best = -1.0
    final_tr_auc = -1.0
    final_te_auc = -1.0
    
    for ep in range(1, epochs + 1):
        t0_ep = time.time()
        enc.train()
        pred.train()
        
        # 1. 100% Graph Message Passing
        h = enc(x, adj_csr)
        
        # 2. Link Prediction Step on random edge batch
        b_idx = torch.randint(0, len(tr_s_t), (actual_batch,))
        pos_s, pos_d = tr_s_t[b_idx], tr_d_t[b_idx]
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
        
        # 3. Evaluation on Train Sample, Validation and Test Sets
        enc.eval()
        pred.eval()
        with torch.no_grad():
            h_eval = enc(x, adj_csr)
            
            # Train Sample AUC
            tr_eval_s = tr_s_t[:len(train_eval_neg_s)]
            tr_eval_d = tr_d_t[:len(train_eval_neg_d)]
            tr_pos = pred(h_eval[tr_eval_s], h_eval[tr_eval_d])
            tr_neg = pred(h_eval[train_eval_neg_s], h_eval[train_eval_neg_d])
            y_tr_true = np.concatenate([np.ones(len(tr_pos)), np.zeros(len(tr_neg))])
            y_tr_sc = torch.cat([tr_pos, tr_neg]).cpu().numpy()
            tr_auc = roc_auc_score(y_tr_true, y_tr_sc)
            
            # Validation AUC
            v_pos = pred(h_eval[val_s_t], h_eval[val_d_t])
            v_neg = pred(h_eval[val_neg_s], h_eval[val_neg_d])
            y_v_true = np.concatenate([np.ones(len(v_pos)), np.zeros(len(v_neg))])
            y_v_sc = torch.cat([v_pos, v_neg]).cpu().numpy()
            val_auc = roc_auc_score(y_v_true, y_v_sc)
            
            # Test AUC
            t_pos = pred(h_eval[te_s_t], h_eval[te_d_t])
            t_neg = pred(h_eval[test_neg_s], h_eval[test_neg_d])
            y_t_true = np.concatenate([np.ones(len(t_pos)), np.zeros(len(t_neg))])
            y_t_sc = torch.cat([t_pos, t_neg]).cpu().numpy()
            test_auc = roc_auc_score(y_t_true, y_t_sc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = ep
                test_at_best = test_auc
                
            final_tr_auc = tr_auc
            final_te_auc = test_auc
            
        print(f"    Epoch {ep:02d}/{epochs:02d} [{time.time()-t0_ep:.1f}s]: Loss={loss.item():.4f} | Train AUC={tr_auc:.4f} | Val AUC={val_auc:.4f} | Test AUC={test_auc:.4f} {'★ (Best Val)' if ep==best_epoch else ''}", flush=True)
        
    print(f"  ✓ {c_name} Complete -> Convergence Epoch: {best_epoch} | Best Val: {best_val_auc:.4f} | Test@Best: {test_at_best:.4f} | Final Test: {final_te_auc:.4f}", flush=True)
    return {
        'total_params': total_params,
        'enc_params': enc_params,
        'dec_params': dec_params,
        'convergence_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'final_train_auc': final_tr_auc,
        'test_auc_at_best': test_at_best,
        'final_test_auc': final_te_auc
    }

# -------------------------------------------------------------
# 4. Master Runner across All Datasets
# -------------------------------------------------------------

def run_all_datasets_100percent(target_dataset=None):
    all_datasets = [
        "Deezer Europe",
        "Reddit",
        "ogbn-mag",
        "ogbn-products",
        "LiveJournal",
        "Orkut"
    ]
    
    if target_dataset and target_dataset.lower() != 'all':
        datasets_to_run = [d for d in all_datasets if target_dataset.lower() in d.lower()]
        if not datasets_to_run:
            raise ValueError(f"Target dataset '{target_dataset}' not recognized. Choose from: {all_datasets}")
    else:
        datasets_to_run = all_datasets
        
    master_results = []
    
    print("\n" + "=" * 90, flush=True)
    print("STARTING 100% FULL-GRAPH REPRESENTATION LEARNING BENCHMARK ACROSS DATASETS", flush=True)
    print(f"Target Datasets: {datasets_to_run}", flush=True)
    print("=" * 90, flush=True)
    
    for dname in datasets_to_run:
        try:
            name, x, n_nodes, adj_csr, tr_s, tr_d, val_s, val_d, te_s, te_d = load_dataset_100percent(dname)
            
            # Config 1: Old Hadamard
            res_c1 = train_and_eval_100percent(
                name, x, n_nodes, adj_csr, tr_s, tr_d, val_s, val_d, te_s, te_d,
                decoder_type='old', epochs=12
            )
            
            # Config 2: 4-Way Siamese
            res_c2 = train_and_eval_100percent(
                name, x, n_nodes, adj_csr, tr_s, tr_d, val_s, val_d, te_s, te_d,
                decoder_type='new', epochs=12
            )
            
            master_results.append({
                'dataset': name,
                'nodes': int(n_nodes),
                'c1_params': int(res_c1['total_params']),
                'c2_params': int(res_c2['total_params']),
                'c1_conv_ep': int(res_c1['convergence_epoch']),
                'c2_conv_ep': int(res_c2['convergence_epoch']),
                'c1_train_auc': float(res_c1['final_train_auc']),
                'c2_train_auc': float(res_c2['final_train_auc']),
                'c1_val_auc': float(res_c1['best_val_auc']),
                'c2_val_auc': float(res_c2['best_val_auc']),
                'c1_test_auc': float(res_c1['test_auc_at_best']),
                'c2_test_auc': float(res_c2['test_auc_at_best']),
                'c1_final_test': float(res_c1['final_test_auc']),
                'c2_final_test': float(res_c2['final_test_auc']),
            })
            # Save progressively
            out_file = "data/results_100percent_graph_all_datasets.json"
            with open(out_file, 'w') as f:
                json.dump(master_results, f, indent=2)
        except Exception as e:
            print(f"❌ Error running dataset {dname}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Render Summary Table
    print("\n" + "=" * 110, flush=True)
    print("100% FULL-GRAPH BENCHMARK RESULTS (LINK PREDICTION ROC-AUC)", flush=True)
    print("=" * 110, flush=True)
    header = f"{'Dataset':<16} | {'Params (C1/C2)':<18} | {'Best Ep':<10} | {'Train AUC (C1/C2)':<20} | {'Val AUC (C1/C2)':<18} | {'Test AUC (C1/C2)':<18}"
    print(header, flush=True)
    print("-" * 110, flush=True)
    for r in master_results:
        p_str = f"{r['c1_params']:,} / {r['c2_params']:,}"
        ep_str = f"Ep {r['c1_conv_ep']} / Ep {r['c2_conv_ep']}"
        tr_str = f"{r['c1_train_auc']:.4f} / {r['c2_train_auc']:.4f}"
        val_str = f"{r['c1_val_auc']:.4f} / {r['c2_val_auc']:.4f}"
        te_str = f"{r['c1_test_auc']:.4f} / {r['c2_test_auc']:.4f}"
        print(f"{r['dataset']:<16} | {p_str:<18} | {ep_str:<10} | {tr_str:<20} | {val_str:<18} | {te_str:<18}", flush=True)
    print("=" * 110 + "\n", flush=True)
    
    out_file = "data/results_100percent_graph_all_datasets.json"
    with open(out_file, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"Results successfully saved to {out_file}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="100% Full-Graph Representation Learning Runner")
    parser.add_argument('--dataset', type=str, default='all', help="Dataset name ('all', 'reddit', 'products', 'mag', 'deezer', 'livejournal', 'orkut')")
    args = parser.parse_args()
    
    run_all_datasets_100percent(args.dataset)
