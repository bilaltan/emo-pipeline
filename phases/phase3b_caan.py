import os
import time
import pandas as pd
import numpy as np

_DS_CACHE = {}

def _get_dataset(url):
    if url not in _DS_CACHE:
        import pyarrow.dataset as ds
        import subprocess, hashlib

        if url.startswith("s3://"):
            candidates = ["/mnt1/tmp", "/mnt/tmp", "/mnt2/tmp", "/mnt/spark", "/mnt1/spark", "/mnt2/spark", "/tmp"]
            worker_tmp = "/tmp"
            for c in candidates:
                if os.path.exists(c) and os.access(c, os.W_OK):
                    worker_tmp = c
                    break
            
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
            local_dir = f"{worker_tmp}/s3_cache_{url_hash}"
            flag_file = f"{local_dir}/_SYNC_COMPLETE"

            if not os.path.exists(flag_file):
                os.makedirs(local_dir, exist_ok=True)
                cmd = ["aws", "s3", "sync", url, local_dir, "--quiet", "--exclude", "_delta_log/*"]
                try:
                    subprocess.run(cmd, check=True)
                    with open(flag_file, "w") as f:
                        f.write("OK")
                except Exception:
                    pass

            if os.path.exists(flag_file):
                local_path = local_dir
            else:
                import pyarrow.fs as fs
                s3, path = fs.S3FileSystem.from_uri(url)
                _DS_CACHE[url] = ds.dataset(path, filesystem=s3, format="parquet", ignore_prefixes=['_delta_log', '.'])
                return _DS_CACHE[url]
        else:
            local_path = url.replace("file://", "")

        _DS_CACHE[url] = ds.dataset(local_path, format="parquet", ignore_prefixes=['_delta_log', '.'])

    return _DS_CACHE[url]

def _load_communities_data_batch(nodes_url, edges_url, comm_ids):
    import pyarrow.dataset as ds
    import pandas as pd

    try:
        nodes_ds = _get_dataset(nodes_url)
        nodes_pdf = nodes_ds.to_table(filter=(ds.field("community_id").isin(comm_ids)), use_threads=True).to_pandas()
    except Exception:
        nodes_pdf = pd.DataFrame()

    try:
        edges_ds = _get_dataset(edges_url)
        edges_pdf = edges_ds.to_table(filter=(ds.field("community_id").isin(comm_ids)), use_threads=True).to_pandas()
    except Exception:
        edges_pdf = pd.DataFrame()

    return nodes_pdf, edges_pdf


def _train_minor_global_caan(dataset, gcn_cfg, dataset_cfg, caan_components, model_type, task_type='node_classification'):
    import os, time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    super_nodes_dict = caan_components['super_nodes_dict']
    minor_nodes_dict = caan_components['minor_nodes_dict']
    caan_edges = caan_components['caan_edges']
    
    super_ids = [-1000 - cid for cid in super_nodes_dict.keys()]
    minor_ids = list(minor_nodes_dict.keys())
    
    if len(minor_ids) == 0:
        return None
        
    feat_dim = dataset_cfg[dataset]['in_feats']
    num_classes = dataset_cfg[dataset]['num_classes']
    hidden_dim = gcn_cfg['hidden_dim']
    num_epochs = gcn_cfg['num_epochs']
    lr = gcn_cfg['lr']
    dropout = gcn_cfg['dropout']
    
    super_feats = [super_nodes_dict[cid] for cid in super_nodes_dict.keys()]
    minor_feats = [minor_nodes_dict[nid]['features'] for nid in minor_ids]
    
    if len(super_feats) > 0:
        super_feats = np.stack(super_feats).astype(np.float32)
    else:
        super_feats = np.empty((0, feat_dim), dtype=np.float32)
        
    if len(minor_feats) > 0:
        minor_feats = np.stack(minor_feats).astype(np.float32)
    else:
        minor_feats = np.empty((0, feat_dim), dtype=np.float32)
        
    feat_arr = np.concatenate([super_feats, minor_feats], axis=0)
    
    # L2 normalize
    feats_norms = np.linalg.norm(feat_arr, axis=1, keepdims=True)
    feat_arr = feat_arr / np.where(feats_norms > 0, feats_norms, 1.0)
    
    # Labels
    super_labels = np.full(len(super_ids), -1, dtype=np.int64)
    minor_labels = np.array([minor_nodes_dict[nid]['label'] for nid in minor_ids], dtype=np.int64)
    label_arr = np.concatenate([super_labels, minor_labels], axis=0)
    
    # Splits
    super_splits = ['none'] * len(super_ids)
    minor_splits = [minor_nodes_dict[nid]['split'] for nid in minor_ids]
    split_arr = super_splits + minor_splits
    
    # Mapping to indices
    node_map = {}
    idx = 0
    for nid in super_ids:
        node_map[nid] = idx
        idx += 1
    for nid in minor_ids:
        node_map[nid] = idx
        idx += 1
        
    # Map caan_edges
    src_l = []
    dst_l = []
    for u, v in caan_edges:
        u_idx = node_map.get(u)
        v_idx = node_map.get(v)
        if u_idx is not None and v_idx is not None:
            src_l.append(u_idx)
            dst_l.append(v_idx)
            
    src_l = np.array(src_l, dtype=np.int64)
    dst_l = np.array(dst_l, dtype=np.int64)
    
    if len(src_l) > 0:
        edges_stacked = np.stack([src_l, dst_l], axis=1)
        unique_edges = np.unique(edges_stacked, axis=0)
        src_l_g = unique_edges[:, 0]
        dst_l_g = unique_edges[:, 1]
    else:
        src_l_g = src_l
        dst_l_g = dst_l
        
    n_nodes = len(node_map)
    if len(src_l_g) > 0:
        mask = (src_l_g >= 0) & (src_l_g < n_nodes) & (dst_l_g >= 0) & (dst_l_g < n_nodes)
        src_l_g = src_l_g[mask]
        dst_l_g = dst_l_g[mask]
    n_edges = len(src_l_g)
    
    has_lbl = torch.tensor(label_arr >= 0, dtype=torch.bool)
    train_m = torch.tensor([s == 'train' and i >= len(super_ids) for i, s in enumerate(split_arr)], dtype=torch.bool) & has_lbl
    val_m   = torch.tensor([s == 'valid' and i >= len(super_ids) for i, s in enumerate(split_arr)], dtype=torch.bool) & has_lbl
    test_m  = torch.tensor([s == 'test'  and i >= len(super_ids) for i, s in enumerate(split_arr)], dtype=torch.bool) & has_lbl
    
    is_pyg = (model_type in ('gat', 'transformer', 'clusterscl'))
    if is_pyg:
        import torch_geometric
        pyg_edge_index = torch.stack([
            torch.tensor(src_l_g, dtype=torch.long),
            torch.tensor(dst_l_g, dtype=torch.long)
        ], dim=0)
        feat_t = torch.tensor(feat_arr, dtype=torch.float32)
        lbl_t = torch.tensor(label_arr, dtype=torch.long)
    else:
        import dgl
        import dgl.nn as dglnn
        g = dgl.graph((src_l_g, dst_l_g), num_nodes=n_nodes)
        g = dgl.to_simple(g)
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = torch.tensor(feat_arr, dtype=torch.float32)
        g.ndata['label'] = torch.tensor(label_arr, dtype=torch.int64)
        feat_t, lbl_t = g.ndata['feat'], g.ndata['label']
        
    t_train_start = time.time()
    if model_type == 'gat':
        from torch_geometric.nn import GATConv
        class GATNet(nn.Module):
            def __init__(self, in_f, h, nc, num_heads=8):
                super().__init__()
                self.c1 = GATConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                self.c2 = GATConv(h, nc, heads=1, concat=False, dropout=dropout)
                self.dr = nn.Dropout(dropout)
            def forward(self, x, edge_index):
                x = F.elu(self.c1(x, edge_index))
                x = self.dr(x)
                x = self.c2(x, edge_index)
                return x
        model = GATNet(feat_arr.shape[1], hidden_dim, num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        crit = nn.CrossEntropyLoss()
        
    elif model_type == 'transformer':
        from torch_geometric.nn import TransformerConv
        class GraphTransformerNet(nn.Module):
            def __init__(self, in_f, h, nc, num_heads=8):
                super().__init__()
                self.c1 = TransformerConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                self.c2 = TransformerConv(h, nc, heads=1, concat=False, dropout=dropout)
                self.dr = nn.Dropout(dropout)
            def forward(self, x, edge_index):
                x = F.relu(self.c1(x, edge_index))
                x = self.dr(x)
                x = self.c2(x, edge_index)
                return x
        model = GraphTransformerNet(feat_arr.shape[1], hidden_dim, num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        crit = nn.CrossEntropyLoss()
        
    elif model_type == 'clusterscl':
        from torch_geometric.nn import GATConv
        class ELBO(nn.Module):
            def __init__(self, num_class, num_cluster, feat_dim, tau, kappa, eta, device):
                super().__init__()
                self.num_class = num_class
                self.num_cluster = num_cluster
                self.feat_dim = feat_dim
                self.tau = tau
                self.kappa = kappa
                self.eta = eta
                self.device = device
                self.prototype = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(self.feat_dim, self.num_cluster), a=0, b=1))
                self.logSoftmax = torch.nn.LogSoftmax(dim=1)
            def forward(self, emb, emb2, y):
                features = torch.cat((emb, emb2), dim=0)
                batchSize = features.shape[0]
                y = y.contiguous().view(-1, 1)
                mask = torch.eq(y, y.T).float().to(self.device)
                mask = mask.repeat(2, 2)
                anchor_dot_cluster = torch.matmul(features, self.prototype)
                anchor_dot_contrast = torch.matmul(features, features.T)
                pi_logit = torch.div(anchor_dot_cluster, self.kappa)
                log_pi = self.logSoftmax(pi_logit + 1e-18)
                pi = torch.exp(log_pi)
                loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))
                align_cluster = anchor_dot_cluster.T.view(self.num_cluster, batchSize, 1).repeat(1, 1, batchSize)
                align_contrast = anchor_dot_contrast.repeat(self.num_cluster, 1).view(self.num_cluster, batchSize, batchSize)
                denom = torch.exp(align_cluster) + torch.exp(align_contrast) + 1e-18
                weight1 = torch.div(torch.exp(align_cluster), denom)
                weight2 = torch.div(torch.exp(align_contrast), denom)
                anchor_dot_augmentation = (weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-18
                logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
                logits = anchor_dot_augmentation - logits_max.detach()
                logits_mask = torch.scatter(
                    torch.ones_like(mask),
                    1,
                    torch.arange(batchSize).view(-1, 1).to(self.device),
                    0
                )
                mask = mask * logits_mask
                exp_logits = torch.exp(logits) * logits_mask
                log_logits = logits - torch.log(exp_logits.sum(2, keepdim=True) + 1e-18)
                normalized_logits = torch.exp(log_logits)
                log_logits_pos = torch.mul(log_logits, mask)
                normalized_logits_pos = torch.mul(normalized_logits, mask)
                pi_normalized_logits_pos = pi.T.view(self.num_cluster, batchSize, 1) * normalized_logits_pos
                posterior = torch.div(pi_normalized_logits_pos, torch.add(torch.sum(pi_normalized_logits_pos, 0), 1 - mask) + 1e-18)
                posterior = torch.mul(posterior, mask)
                pos_sum = torch.clamp(torch.sum(mask, 1), min=1.0)
                loss = -torch.mean(torch.div(torch.sum(torch.sum(posterior * (log_pi.T.view(self.num_cluster, batchSize, 1) + log_logits_pos - torch.log(posterior + 1e-18)), 0), 1), pos_sum))
                return loss + self.eta * loss_0

        class ClusterSCLModel(nn.Module):
            def __init__(self, in_f, h, nc, num_heads=8):
                super().__init__()
                self.encoder = GATNet(in_f, h, h, num_heads)
                self.proj_head = nn.Sequential(
                    nn.Linear(h, h),
                    nn.ReLU(),
                    nn.Linear(h, 128)
                )
                self.fc = nn.Linear(h, nc)
                self.dropout_layer = nn.Dropout(dropout)
            def forward(self, x, edge_index):
                z = self.encoder(x, edge_index)
                return self.fc(z)
            def get_embeddings_and_logits(self, x, edge_index):
                z = self.encoder(x, edge_index)
                proj = F.normalize(self.proj_head(z), p=2, dim=1)
                logits = self.fc(z)
                return z, proj, logits

        class GATNet(nn.Module):
            def __init__(self, in_f, h, nc, num_heads=8):
                super().__init__()
                self.c1 = GATConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                self.c2 = GATConv(h, nc, heads=1, concat=False, dropout=dropout)
                self.dr = nn.Dropout(dropout)
            def forward(self, x, edge_index):
                x = F.elu(self.c1(x, edge_index))
                x = self.dr(x)
                x = self.c2(x, edge_index)
                return x

        model = ClusterSCLModel(feat_arr.shape[1], hidden_dim, num_classes)
        elbo_loss_fn = ELBO(
            num_class=num_classes,
            num_cluster=max(2, num_classes),
            feat_dim=128,
            tau=0.07,
            kappa=0.1,
            eta=0.1,
            device=torch.device('cpu')
        )
        opt = torch.optim.Adam(list(model.parameters()) + list(elbo_loss_fn.parameters()), lr=lr, weight_decay=5e-4)
        crit = nn.CrossEntropyLoss()
        
    else:
        import dgl.nn as dglnn
        class GraphSAGECommunity(nn.Module):
            def __init__(self, in_f, h, nc):
                super().__init__()
                self.c1 = dglnn.SAGEConv(in_f, h, 'mean')
                self.c2 = dglnn.SAGEConv(h,    h, 'mean')
                self.fc = nn.Linear(h, nc)
                self.dr = nn.Dropout(dropout)
            def forward(self, g, x):
                x = torch.relu(self.c1(g, x)); x = self.dr(x)
                x = self.c2(g, x)
                return self.fc(x)
            def encode(self, g, x):
                x = torch.relu(self.c1(g, x)); x = self.dr(x)
                return self.c2(g, x)
        model = GraphSAGECommunity(feat_arr.shape[1], hidden_dim, num_classes)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        crit  = nn.CrossEntropyLoss()

    model.train()
    for _ in range(num_epochs):
        if model_type == 'clusterscl':
            opt.zero_grad()
            z1, proj1, logits1 = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
            z2, proj2, logits2 = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
            loss_ce = crit(logits1[train_m], lbl_t[train_m])
            if train_m.sum() >= 2:
                loss_ce.backward(retain_graph=True)
                n_tr = train_m.sum().item()
                tr_indices = torch.where(train_m)[0]
                shuf_indices = tr_indices[torch.randperm(n_tr)]
                batch_size = 512
                for i in range(0, n_tr, batch_size):
                    b_idx = shuf_indices[i : i + batch_size]
                    p1_b = proj1[b_idx]
                    p2_b = proj2[b_idx]
                    y_b = lbl_t[b_idx]
                    loss_elbo_b = elbo_loss_fn(p1_b, p2_b, y_b)
                    scaled_elbo = loss_elbo_b * (len(b_idx) / n_tr)
                    is_last = (i + batch_size >= n_tr)
                    scaled_elbo.backward(retain_graph=not is_last)
            else:
                loss_ce.backward()
            opt.step()
        elif is_pyg:
            opt.zero_grad()
            logits = model(feat_t, pyg_edge_index)
            loss   = crit(logits[train_m], lbl_t[train_m])
            loss.backward()
            opt.step()
        else:
            logits = model(g, feat_t)
            loss   = crit(logits[train_m], lbl_t[train_m])
            opt.zero_grad(); loss.backward(); opt.step()
            
    train_time = time.time() - t_train_start
    
    model.eval()
    with torch.no_grad():
        if is_pyg:
            logits = model(feat_t, pyg_edge_index)
        else:
            logits = model(g, feat_t)
        preds = logits.argmax(dim=1)
        
    def safe_acc(mask):
        n = int(mask.sum())
        if n == 0:
            return 0.0, 0
        return float((preds[mask] == lbl_t[mask]).float().mean()), n
        
    comm_acc, n_test_node = safe_acc(test_m)
    bnd_acc, _ = safe_acc(test_m)
    int_acc, _ = safe_acc(test_m)
    
    return pd.DataFrame([{
        'community_id': -1,
        'n_nodes': len(minor_ids),
        'n_edges': n_edges,
        'n_train': int(train_m.sum()),
        'n_val': int(val_m.sum()),
        'n_test': n_test_node,
        'n_boundary': 0,
        'n_internal': len(minor_ids),
        'comm_test_acc': comm_acc,
        'boundary_acc': bnd_acc,
        'internal_acc': int_acc,
        'comm_link_auc': 0.5,
        'size_bucket': 'small',
        'load_time_s': 0.0,
        'node_train_time_s': train_time,
        'link_train_time_s': 0.0,
        'peak_mem_mb': 0.0,
    }])

def make_caan_udf(super_nodes_dict_bc, minor_node_to_idx_bc, minor_feats_arr_bc,
                  minor_labels_arr_bc, minor_splits_arr_bc, minor_ids_arr_bc,
                  caan_adj_bc, node_to_comm_bc, major_comms_bc,
                  base_weights_bc=None, base_embeddings_bc=None, base_node_map_bc=None,
                  p2_nodes_url=None, p2_edges_url=None):
    def _train_gnn_community_caan_single(pdf, comm_edges_pdf):
        import os, time, subprocess, sys, resource
        import numpy as np
        import pandas as pd
        import inspect

        # 1. Dynamically resolve node site-packages to sys.path
        candidates = ["/mnt/tmp", "/mnt1/tmp", "/mnt2/tmp", "/mnt/spark", "/mnt1/spark", "/mnt2/spark", "/tmp"]
        worker_tmp = "/tmp"
        for c in candidates:
            if os.path.exists(c) and os.access(c, os.W_OK):
                worker_tmp = c
                break
        site_packages = f"{worker_tmp}/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
        os.environ["PYTHONUSERBASE"] = f"{worker_tmp}/.local"

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

        try:
            if not hasattr(torch, '_orig_load_patched'):
                _orig = torch.load
                torch._orig_load_patched = _orig
                def _patched(*args, **kwargs):
                    sig = inspect.signature(_orig)
                    if 'weights_only' in sig.parameters:
                        kwargs['weights_only'] = False
                    return _orig(*args, **kwargs)
                torch.load = _patched
        except Exception:
            pass

        os.environ.setdefault('HOME', '/tmp')
        os.environ.setdefault('DGLBACKEND', 'pytorch')
        os.makedirs('/tmp/.dgl', exist_ok=True)

        try:
            import dgl
            import dgl.nn as dglnn
        except Exception:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir',
                            'dgl==1.1.3', '-f', 'https://data.dgl.ai/wheels/repo.html'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            import dgl
            import dgl.nn as dglnn

        try:
            omp_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
            torch.set_num_threads(omp_threads)
        except Exception:
            pass
        
        t_start = time.time()
        comm_id = int(pdf['community_id'].iloc[0])
        num_classes = int(pdf['_num_classes'].iloc[0])
        hidden_dim = int(pdf['_hidden'].iloc[0])
        num_epochs = int(pdf['_epochs'].iloc[0])
        lr = float(pdf['_lr'].iloc[0])
        dropout = float(pdf['_dropout'].iloc[0])
        task_type = str(pdf['_task_type'].iloc[0]) if '_task_type' in pdf.columns else 'node_classification'
        model_type = str(pdf['_model_type'].iloc[0]) if '_model_type' in pdf.columns else 'sage'
        
        super_nodes_dict = super_nodes_dict_bc.value
        minor_node_to_idx = minor_node_to_idx_bc.value
        minor_feats_arr = minor_feats_arr_bc.value
        minor_labels_arr = minor_labels_arr_bc.value
        minor_splits_arr = minor_splits_arr_bc.value
        minor_ids_arr = minor_ids_arr_bc.value
        caan_adj = caan_adj_bc.value
        node_to_comm = node_to_comm_bc.value
        major_comms = major_comms_bc.value
        
        local_ids = pdf['id'].values
        n_local = len(local_ids)

        # Dynamic dataset/community size-aware epochs scale down to speed up training
        # Small communities (nodes < 1K) adapt in 2 epochs, medium (nodes < 10K) in 4 epochs, large/massive in 10 epochs.
        if n_local < 1000:
            num_epochs = min(2, num_epochs)
        elif n_local < 10000:
            num_epochs = min(4, num_epochs)
        else:
            num_epochs = num_epochs
            
        # Warm-start logic safety floor
        if base_weights_bc is not None and model_type == 'sage':
            num_epochs = max(2, num_epochs)
        local_feats = np.stack(pdf['features'].values).astype(np.float32)
        local_labels = np.array([int(v) if not pd.isna(v) else -1 for v in pdf['label'].values], dtype=np.int64)
        local_splits = list(pdf['split'].values)
        local_bnd = np.array([bool(v) if not (pd.isna(v) or v is None) else False for v in pdf['is_boundary'].values], dtype=bool)
        
        super_feats = []
        super_ids = []
        for cid, feat in super_nodes_dict.items():
            if cid != comm_id:
                super_feats.append(feat)
                super_ids.append(-1000 - cid)
                
        feat_dim = local_feats.shape[1]
        if len(super_feats) > 0:
            super_feats = np.stack(super_feats).astype(np.float32)
        else:
            super_feats = np.empty((0, feat_dim), dtype=np.float32)
            
        # 1. Map neighbors first to identify connected minor nodes
        if comm_edges_pdf is not None and len(comm_edges_pdf) > 0:
            exploded = pd.DataFrame({
                'id': comm_edges_pdf['src'].values,
                'neighbors': comm_edges_pdf['dst'].values
            })
        else:
            exploded = pd.DataFrame(columns=['id', 'neighbors'])
        connected_minor_ids = set()
        
        if len(exploded) > 0:
            exploded['neighbors'] = exploded['neighbors'].astype(np.int64)
            
            def map_dst(w):
                w_comm = node_to_comm.get(w, -1)
                if w_comm == comm_id:
                    return w
                elif w_comm in major_comms:
                    return -1000 - w_comm
                else:
                    return w
                    
            exploded['dst_mapped'] = exploded['neighbors'].map(map_dst)
            
            local_ids_set = set(local_ids)
            for w in exploded['dst_mapped'].unique():
                w_int = int(w)
                if w_int >= 0 and w_int not in local_ids_set:
                    if w_int in minor_node_to_idx:
                        connected_minor_ids.add(w_int)
                        
        # 2. Slice minor node arrays using indexed positions (O(1) vector slice)
        minor_ids = list(connected_minor_ids)
        minor_indices = [minor_node_to_idx[nid] for nid in minor_ids]
        
        if len(minor_indices) > 0:
            minor_feats = minor_feats_arr[minor_indices]
            minor_labels_arr_sliced = minor_labels_arr[minor_indices]
            minor_splits = [minor_splits_arr[i] for i in minor_indices]
        else:
            minor_feats = np.empty((0, feat_dim), dtype=np.float32)
            minor_labels_arr_sliced = np.empty((0,), dtype=np.int64)
            minor_splits = []
            
        feat_arr = np.concatenate([local_feats, super_feats, minor_feats], axis=0)
        
        feat_norms = np.linalg.norm(feat_arr, axis=1, keepdims=True)
        feat_arr = feat_arr / np.where(feat_norms > 0, feat_norms, 1.0)
        
        super_labels = np.full(len(super_ids), -1, dtype=np.int64)
        label_arr = np.concatenate([local_labels, super_labels, minor_labels_arr_sliced], axis=0)
        
        super_splits = ['none'] * len(super_ids)
        split_arr = local_splits + super_splits + minor_splits
        
        super_bnd = np.full(len(super_ids), False, dtype=bool)
        minor_bnd = np.full(len(minor_ids), False, dtype=bool)
        bnd_arr = np.concatenate([local_bnd, super_bnd, minor_bnd], axis=0)
        
        node_map = {}
        idx = 0
        for nid in local_ids:
            node_map[int(nid)] = idx
            idx += 1
        for nid in super_ids:
            node_map[nid] = idx
            idx += 1
        for nid in minor_ids:
            node_map[int(nid)] = idx
            idx += 1
            
        src_l = []
        dst_l = []
        
        if len(exploded) > 0:
            exploded['src_idx'] = exploded['id'].map(node_map)
            exploded['dst_idx'] = exploded['dst_mapped'].map(node_map)
            
            valid = exploded.dropna(subset=['src_idx', 'dst_idx'])
            if len(valid) > 0:
                src_l.extend(valid['src_idx'].values.astype(np.int64))
                dst_l.extend(valid['dst_idx'].values.astype(np.int64))
                
        exclude_id = -1000 - comm_id
        valid_nodes_set = set(super_ids).union(set(minor_ids))
        
        # O(V_subgraph + E_subgraph) edge mapping using adjacency lookup
        for u in valid_nodes_set:
            u_neighbors = caan_adj.get(u, [])
            for v in u_neighbors:
                if v in valid_nodes_set and v != exclude_id:
                    u_idx = node_map.get(u)
                    v_idx = node_map.get(v)
                    if u_idx is not None and v_idx is not None:
                        src_l.append(u_idx)
                        dst_l.append(v_idx)
                    
        src_l = np.array(src_l, dtype=np.int64)
        dst_l = np.array(dst_l, dtype=np.int64)
        
        if len(src_l) > 0:
            edges_stacked = np.stack([src_l, dst_l], axis=1)
            unique_edges = np.unique(edges_stacked, axis=0)
            src_l_g = unique_edges[:, 0]
            dst_l_g = unique_edges[:, 1]
        else:
            src_l_g = src_l
            dst_l_g = dst_l
            
        n_nodes = len(node_map)
        if len(src_l_g) > 0:
            mask = (src_l_g >= 0) & (src_l_g < n_nodes) & (dst_l_g >= 0) & (dst_l_g < n_nodes)
            src_l_g = src_l_g[mask]
            dst_l_g = dst_l_g[mask]
        n_edges = len(src_l_g)
        
        t_load = time.time() - t_start
        t_dgl_conv_start = time.time()
        
        train_m = torch.tensor([s == 'train' and i < n_local for i, s in enumerate(split_arr)], dtype=torch.bool)
        val_m   = torch.tensor([s == 'valid' and i < n_local for i, s in enumerate(split_arr)], dtype=torch.bool)
        test_m  = torch.tensor([s == 'test'  and i < n_local for i, s in enumerate(split_arr)], dtype=torch.bool)
        bnd_t   = torch.tensor(bnd_arr, dtype=torch.bool)
        
        is_pyg = (model_type in ('gat', 'transformer', 'clusterscl'))
        if is_pyg:
            import torch_geometric
            pyg_edge_index = torch.stack([
                torch.tensor(src_l_g, dtype=torch.long),
                torch.tensor(dst_l_g, dtype=torch.long)
            ], dim=0)
            feat_t = torch.tensor(feat_arr, dtype=torch.float32)
            lbl_t = torch.tensor(label_arr, dtype=torch.long)
        else:
            g = dgl.graph((src_l_g, dst_l_g), num_nodes=n_nodes)
            g = dgl.to_simple(g)
            g = dgl.add_self_loop(g)
            g.ndata['feat']  = torch.tensor(feat_arr,  dtype=torch.float32)
            g.ndata['label'] = torch.tensor(label_arr, dtype=torch.int64)
            feat_t, lbl_t = g.ndata['feat'], g.ndata['label']

        t_dgl_conv = time.time() - t_dgl_conv_start
        node_train_time = 0.0
        comm_acc = 0.0
        bnd_acc = 0.0
        int_acc = 0.0
        n_test_node = int(test_m.sum())
        
        run_node = (task_type in ('node_classification', 'both'))
        run_link = (task_type in ('link_prediction', 'both'))
        
        if run_node and train_m.sum() > 0:
            t_node_start = time.time()
            if model_type == 'gat':
                from torch_geometric.nn import GATConv
                class GATEncoder(nn.Module):
                    def __init__(self, in_f, h, num_heads=8):
                        super().__init__()
                        self.c1 = GATConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                        self.c2 = GATConv(h, h, heads=1, concat=False, dropout=dropout)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.elu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
                class GATNet(nn.Module):
                    def __init__(self, in_f, h, nc):
                        super().__init__()
                        self.enc = GATEncoder(in_f, h)
                        self.fc = nn.Linear(h, nc)
                    def forward(self, x, edge_index):
                        z = self.enc(x, edge_index)
                        return self.fc(z)
                model = GATNet(feat_arr.shape[1], hidden_dim, num_classes)
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
                crit = nn.CrossEntropyLoss()
                
            elif model_type == 'transformer':
                from torch_geometric.nn import TransformerConv
                class GraphTransformerEncoder(nn.Module):
                    def __init__(self, in_f, h, num_heads=8):
                        super().__init__()
                        self.c1 = TransformerConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                        self.c2 = TransformerConv(h, h, heads=1, concat=False, dropout=dropout)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.relu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
                class GraphTransformerNet(nn.Module):
                    def __init__(self, in_f, h, nc):
                        super().__init__()
                        self.enc = GraphTransformerEncoder(in_f, h)
                        self.fc = nn.Linear(h, nc)
                    def forward(self, x, edge_index):
                        z = self.enc(x, edge_index)
                        return self.fc(z)
                model = GraphTransformerNet(feat_arr.shape[1], hidden_dim, num_classes)
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
                crit = nn.CrossEntropyLoss()
                
            elif model_type == 'clusterscl':
                from torch_geometric.nn import GATConv
                class ELBO(nn.Module):
                    def __init__(self, num_class, num_cluster, feat_dim, tau, kappa, eta, device):
                        super().__init__()
                        self.num_class = num_class
                        self.num_cluster = num_cluster
                        self.feat_dim = feat_dim
                        self.tau = tau
                        self.kappa = kappa
                        self.eta = eta
                        self.device = device
                        self.prototype = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(self.feat_dim, self.num_cluster), a=0, b=1))
                        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
                    def forward(self, emb, emb2, y):
                        features = torch.cat((emb, emb2), dim=0)
                        batchSize = features.shape[0]
                        y = y.contiguous().view(-1, 1)
                        mask = torch.eq(y, y.T).float().to(self.device)
                        mask = mask.repeat(2, 2)
                        anchor_dot_cluster = torch.matmul(features, self.prototype)
                        anchor_dot_contrast = torch.matmul(features, features.T)
                        pi_logit = torch.div(anchor_dot_cluster, self.kappa)
                        log_pi = self.logSoftmax(pi_logit + 1e-18)
                        pi = torch.exp(log_pi)
                        loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))
                        align_cluster = anchor_dot_cluster.T.view(self.num_cluster, batchSize, 1).repeat(1, 1, batchSize)
                        align_contrast = anchor_dot_contrast.repeat(self.num_cluster, 1).view(self.num_cluster, batchSize, batchSize)
                        denom = torch.exp(align_cluster) + torch.exp(align_contrast) + 1e-18
                        weight1 = torch.div(torch.exp(align_cluster), denom)
                        weight2 = torch.div(torch.exp(align_contrast), denom)
                        anchor_dot_augmentation = (weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-18
                        logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
                        logits = anchor_dot_augmentation - logits_max.detach()
                        logits_mask = torch.scatter(
                            torch.ones_like(mask),
                            1,
                            torch.arange(batchSize).view(-1, 1).to(self.device),
                            0
                        )
                        mask = mask * logits_mask
                        exp_logits = torch.exp(logits) * logits_mask
                        log_logits = logits - torch.log(exp_logits.sum(2, keepdim=True) + 1e-18)
                        normalized_logits = torch.exp(log_logits)
                        log_logits_pos = torch.mul(log_logits, mask)
                        normalized_logits_pos = torch.mul(normalized_logits, mask)
                        pi_normalized_logits_pos = pi.T.view(self.num_cluster, batchSize, 1) * normalized_logits_pos
                        posterior = torch.div(pi_normalized_logits_pos, torch.add(torch.sum(pi_normalized_logits_pos, 0), 1 - mask) + 1e-18)
                        posterior = torch.mul(posterior, mask)
                        pos_sum = torch.clamp(torch.sum(mask, 1), min=1.0)
                        loss = -torch.mean(torch.div(torch.sum(torch.sum(posterior * (log_pi.T.view(self.num_cluster, batchSize, 1) + log_logits_pos - torch.log(posterior + 1e-18)), 0), 1), pos_sum))
                        return loss + self.eta * loss_0

                class ClusterSCLModel(nn.Module):
                    def __init__(self, in_f, h, nc, num_heads=8):
                        super().__init__()
                        self.encoder = GATNet(in_f, h, h, num_heads)
                        self.proj_head = nn.Sequential(
                            nn.Linear(h, h),
                            nn.ReLU(),
                            nn.Linear(h, 128)
                        )
                        self.fc = nn.Linear(h, nc)
                        self.dropout_layer = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        z = self.encoder(x, edge_index)
                        return self.fc(z)
                    def get_embeddings_and_logits(self, x, edge_index):
                        z = self.encoder(x, edge_index)
                        proj = F.normalize(self.proj_head(z), p=2, dim=1)
                        logits = self.fc(z)
                        return z, proj, logits

                class GATNet(nn.Module):
                    def __init__(self, in_f, h, nc, num_heads=8):
                        super().__init__()
                        self.c1 = GATConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                        self.c2 = GATConv(h, nc, heads=1, concat=False, dropout=dropout)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.elu(self.c1(x, edge_index))
                        x = self.dr(x)
                        x = self.c2(x, edge_index)
                        return x

                model = ClusterSCLModel(feat_arr.shape[1], hidden_dim, num_classes)
                elbo_loss_fn = ELBO(
                    num_class=num_classes,
                    num_cluster=max(2, num_classes),
                    feat_dim=128,
                    tau=0.07,
                    kappa=0.1,
                    eta=0.1,
                    device=torch.device('cpu')
                )
                opt = torch.optim.Adam(list(model.parameters()) + list(elbo_loss_fn.parameters()), lr=lr, weight_decay=5e-4)
                crit = nn.CrossEntropyLoss()
                
            else:
                class GraphSAGECommunity(nn.Module):
                    def __init__(self, in_f, h, nc):
                        super().__init__()
                        self.c1 = dglnn.SAGEConv(in_f, h, 'mean')
                        self.c2 = dglnn.SAGEConv(h,    h, 'mean')
                        self.fc = nn.Linear(h, nc)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, g, x):
                        x = torch.relu(self.c1(g, x)); x = self.dr(x)
                        x = self.c2(g, x)
                        return self.fc(x)
                    def encode(self, g, x):
                        x = torch.relu(self.c1(g, x)); x = self.dr(x)
                        return self.c2(g, x)
                model = GraphSAGECommunity(feat_arr.shape[1], hidden_dim, num_classes)
                opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
                crit  = nn.CrossEntropyLoss()

            # Warm-start weights load
            if base_weights_bc is not None and model_type == 'sage':
                try:
                    model.load_state_dict(base_weights_bc.value)
                except Exception:
                    pass

            # Embedding Regularization setup
            valid_emb_mask = None
            if base_embeddings_bc is not None and base_node_map_bc is not None:
                try:
                    global_node_map = base_node_map_bc.value
                    emb_idx = [global_node_map.get(int(nid), -1) for nid in local_ids]
                    emb_idx_clean = [idx if idx != -1 else 0 for idx in emb_idx]
                    global_emb_t = torch.tensor(base_embeddings_bc.value[emb_idx_clean], dtype=torch.float32)
                    valid_emb_mask = torch.tensor([idx != -1 for idx in emb_idx], dtype=torch.bool)
                except Exception:
                    pass

            model.train()
            for _ in range(num_epochs):
                if model_type == 'clusterscl':
                    opt.zero_grad()
                    z1, proj1, logits1 = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
                    z2, proj2, logits2 = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
                    loss_ce = crit(logits1[train_m], lbl_t[train_m])
                    
                    # Apply regularization if applicable
                    if valid_emb_mask is not None and valid_emb_mask.sum() > 0:
                        try:
                            local_emb, _, _ = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
                            local_emb_slice = local_emb[:n_local]
                            loss_reg = F.mse_loss(local_emb_slice[valid_emb_mask], global_emb_t[valid_emb_mask])
                            loss_ce = loss_ce + 0.01 * loss_reg
                        except Exception:
                            pass

                    if train_m.sum() >= 2:
                        loss_ce.backward(retain_graph=True)
                        n_tr = train_m.sum().item()
                        tr_indices = torch.where(train_m)[0]
                        shuf_indices = tr_indices[torch.randperm(n_tr)]
                        batch_size = 128
                        for i in range(0, n_tr, batch_size):
                            b_idx = shuf_indices[i : i + batch_size]
                            p1_b = proj1[b_idx]
                            p2_b = proj2[b_idx]
                            y_b = lbl_t[b_idx]
                            loss_elbo_b = elbo_loss_fn(p1_b, p2_b, y_b)
                            scaled_elbo = loss_elbo_b * (len(b_idx) / n_tr)
                            is_last = (i + batch_size >= n_tr)
                            scaled_elbo.backward(retain_graph=not is_last)
                    else:
                        loss_ce.backward()
                    opt.step()
                elif is_pyg:
                    opt.zero_grad()
                    logits = model(feat_t, pyg_edge_index)
                    loss   = crit(logits[train_m], lbl_t[train_m])
                    
                    # Apply regularization if applicable
                    if valid_emb_mask is not None and valid_emb_mask.sum() > 0:
                        try:
                            local_emb = model.enc(feat_t, pyg_edge_index)
                            local_emb_slice = local_emb[:n_local]
                            loss_reg = F.mse_loss(local_emb_slice[valid_emb_mask], global_emb_t[valid_emb_mask])
                            loss = loss + 0.01 * loss_reg
                        except Exception:
                            pass

                    loss.backward()
                    opt.step()
                else:
                    logits = model(g, feat_t)
                    loss   = crit(logits[train_m], lbl_t[train_m])
                    
                    # Apply regularization if applicable
                    if valid_emb_mask is not None and valid_emb_mask.sum() > 0:
                        try:
                            local_emb = model.encode(g, feat_t)
                            local_emb_slice = local_emb[:n_local]
                            loss_reg = F.mse_loss(local_emb_slice[valid_emb_mask], global_emb_t[valid_emb_mask])
                            loss = loss + 0.01 * loss_reg
                        except Exception:
                            pass

                    opt.zero_grad(); loss.backward(); opt.step()
            node_train_time = time.time() - t_node_start
            
            model.eval()
            with torch.no_grad():
                if is_pyg:
                    if model_type == 'clusterscl':
                        embed = model.encoder(feat_t, pyg_edge_index)
                    else:
                        embed = model.enc(feat_t, pyg_edge_index)
                else:
                    embed = model.encode(g, feat_t)
                embed_np = embed.cpu().numpy()

            import copy
            from torch.utils.data import TensorDataset, DataLoader

            class DownstreamNodeClassifierUDF(nn.Module):
                def __init__(self, input_dim, classes):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, classes)
                    )
                def forward(self, x):
                    return self.layers(x)

            input_dim = embed_np.shape[1]
            mlp_model = DownstreamNodeClassifierUDF(input_dim, num_classes)

            train_idx = np.where(train_m.numpy())[0]
            val_idx = np.where(val_m.numpy())[0]
            test_idx = np.where(test_m.numpy())[0]

            if len(train_idx) > 0:
                train_embed = embed[train_idx]
                train_labels = lbl_t[train_idx]
                mlp_dataset = TensorDataset(train_embed, train_labels)
                mlp_loader = DataLoader(mlp_dataset, batch_size=128, shuffle=True)
                mlp_loss_fn = nn.CrossEntropyLoss()
                mlp_opt = torch.optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=5e-4)

                best_acc = -1.0
                best_weights = copy.deepcopy(mlp_model.state_dict())

                for mlp_epoch in range(num_epochs):
                    mlp_model.train()
                    for x_b, y_b in mlp_loader:
                        mlp_opt.zero_grad()
                        y_pred = mlp_model(x_b)
                        loss_mlp = mlp_loss_fn(y_pred, y_b)
                        loss_mlp.backward()
                        mlp_opt.step()

                    if len(val_idx) > 0:
                        mlp_model.eval()
                        with torch.no_grad():
                            val_embed = embed[val_idx]
                            val_labels = lbl_t[val_idx]
                            y_pred_val = mlp_model(val_embed)
                            acc = (y_pred_val.argmax(dim=1) == val_labels).float().mean().item()
                            if acc > best_acc:
                                best_acc = acc
                                best_weights = copy.deepcopy(mlp_model.state_dict())
                    else:
                        best_weights = copy.deepcopy(mlp_model.state_dict())

                mlp_model.load_state_dict(best_weights)

            mlp_model.eval()
            with torch.no_grad():
                y_pred_all = mlp_model(embed)
                preds = y_pred_all.argmax(dim=1)

            def safe_acc(mask):
                n = int(mask.sum())
                if n == 0:
                    return 0.0, 0
                return float((preds[mask] == lbl_t[mask]).float().mean()), n

            comm_acc, n_test_node = safe_acc(test_m)
            bnd_acc,  _     = safe_acc(test_m & bnd_t)
            int_acc,  _     = safe_acc(test_m & ~bnd_t)

        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0
        bucket = 'large' if n_local > 200 else ('medium' if n_local >= 50 else 'small')

        return pd.DataFrame([{
            'community_id':  comm_id,
            'n_nodes':       n_local,
            'n_edges':       n_edges,
            'n_train':       int(train_m.sum()) if run_node else 0,
            'n_val':         int(val_m.sum()) if run_node else 0,
            'n_test':        n_test_node if run_node else 0,
            'n_boundary':    int(bnd_t[:n_local].sum()),
            'n_internal':    n_local - int(bnd_t[:n_local].sum()),
            'comm_test_acc': comm_acc,
            'boundary_acc':  bnd_acc,
            'internal_acc':  int_acc,
            'comm_link_auc': 0.5,
            'size_bucket':   bucket,
            'load_time_s':   t_load,
            'node_train_time_s': node_train_time,
            'link_train_time_s': 0.0,
            'peak_mem_mb':   peak_mem / 1e6,
        }])
        
    def _train_gnn_bin_caan(key, manifest_pdf):
        # Set inter-op and intra-op thread counts strictly on the worker
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        import pandas as pd
        comm_ids = manifest_pdf['community_id'].unique().tolist()
        nodes_pdf_all, edges_pdf_all = _load_communities_data_batch(p2_nodes_url, p2_edges_url, comm_ids)

        results = []
        for _, row in manifest_pdf.iterrows():
            comm_id = int(row['community_id'])
            
            if len(nodes_pdf_all) > 0:
                pdf = nodes_pdf_all[nodes_pdf_all['community_id'] == comm_id].copy()
            else:
                pdf = pd.DataFrame()

            if len(edges_pdf_all) > 0:
                comm_edges_pdf = edges_pdf_all[edges_pdf_all['community_id'] == comm_id].copy()
            else:
                comm_edges_pdf = pd.DataFrame()
            
            if pdf is None or len(pdf) == 0:
                results.append(pd.DataFrame([{
                    'community_id':   comm_id,
                    'n_nodes':        0, 'n_edges': 0, 'n_train': 0, 'n_val': 0, 'n_test': 0,
                    'n_boundary':     0, 'n_internal': 0, 'comm_test_acc': 0.0, 'boundary_acc': 0.0,
                    'internal_acc':   0.0, 'comm_link_auc': 0.5, 'size_bucket': 'empty',
                    'load_time_s':    0.0, 'node_train_time_s': 0.0, 'link_train_time_s': 0.0, 'peak_mem_mb': 0.0
                }]))
                continue

            pdf['_num_classes'] = int(row['_num_classes'])
            pdf['_hidden']      = int(row['_hidden'])
            pdf['_epochs']      = int(row['_epochs'])
            pdf['_lr']          = float(row['_lr'])
            pdf['_dropout']     = float(row['_dropout'])
            pdf['_task_type']   = str(row['_task_type'])
            pdf['_model_type']  = str(row['_model_type'])

            res_row = _train_gnn_community_caan_single(pdf, comm_edges_pdf)
            results.append(res_row)

        if len(results) == 0:
            return pd.DataFrame(columns=[
                'community_id', 'n_nodes', 'n_edges', 'n_train', 'n_val', 'n_test',
                'n_boundary', 'n_internal', 'comm_test_acc', 'boundary_acc',
                'internal_acc', 'comm_link_auc', 'size_bucket', 'load_time_s',
                'node_train_time_s', 'link_train_time_s', 'peak_mem_mb'
            ])
        return pd.concat(results, ignore_index=True)
        
    return _train_gnn_bin_caan

def run_phase3b(spark, sc, datasets, algorithms, use_global_mapping,
                dataset_cfg, gcn_cfg, get_paths_fn, timing, results, **kwargs):
    """
    Train GNN models per community using Spark groupBy().applyInPandas() with CaaN Global Graph.
    Optimized for massive EMR clusters and shuffless execution.
    """
    task_type = kwargs.get('task_type', 'node_classification')
    gnn_models = kwargs.get('models', ['sage'])
    min_size = kwargs.get('min_size', 100)
    
    from pyspark.sql import functions as F
    from pipeline.phases.phase3_training import _make_result_schema
    result_schema = _make_result_schema()
    
    for dataset in datasets:
        cfg = dataset_cfg[dataset]
        p = get_paths_fn(dataset)
        
        # Load raw nodes and edges
        raw_nodes_df = spark.read.format('delta').load(p['nodes'])
        raw_edges_df = spark.read.format('delta').load(p['edges'])
        
        for alg in algorithms:
            p_alg = get_paths_fn(dataset, alg)
            
            # Load community assignments and p2_nodes (already contains split, community_id, and is_boundary)
            comms_df = spark.read.format('delta').load(p_alg['communities'])
            nodes_w_comm = spark.read.format('delta').load(p_alg['p2_nodes'])
            
            # Compute community sizes
            comm_sizes = comms_df.groupBy('community_id').count().collect()
            
            major_comms = set()
            minor_comms = set()
            for row in comm_sizes:
                cid = row['community_id']
                cnt = row['count']
                if cnt >= min_size:
                    major_comms.add(cid)
                else:
                    minor_comms.add(cid)
            
            print(f"  Dataset: {dataset} | Algorithm: {alg}")
            print(f"    - Major communities (>= {min_size}): {len(major_comms)}")
            print(f"    - Minor communities (< {min_size}): {len(minor_comms)}")
            
            if len(major_comms) == 0:
                print("    - No major communities found, skipping Phase 3b.")
                continue
                
            # Compute super-node features for major communities
            major_nodes_df = nodes_w_comm.filter(F.col('community_id').isin(list(major_comms)))
            
            def compute_means_udf(pdf):
                import numpy as np
                import pandas as pd
                cid = pdf['community_id'].iloc[0]
                feats = np.stack(pdf['features'].values)
                mean_feat = feats.mean(axis=0).tolist()
                return pd.DataFrame([{'community_id': cid, 'mean_features': mean_feat}])
                
            mean_df = major_nodes_df.groupBy('community_id').applyInPandas(
                compute_means_udf,
                schema='community_id long, mean_features array<float>'
            )
            
            super_nodes_dict = {
                row['community_id']: np.array(row['mean_features'], dtype=np.float32)
                for row in mean_df.collect()
            }
            
            # Gather minor nodes features, labels, and splits
            minor_nodes_pd = nodes_w_comm.filter(F.col('community_id').isin(list(minor_comms))).toPandas()
            minor_nodes_dict = {}
            for _, row in minor_nodes_pd.iterrows():
                minor_nodes_dict[int(row['id'])] = {
                    'features': np.array(row['features'], dtype=np.float32),
                    'label': int(row['label']) if not pd.isna(row['label']) else -1,
                    'split': str(row['split'])
                }
                
            # Map-side broadcast edge mapping (completely shuffless!)
            node_to_comm_pd = comms_df.toPandas()
            node_to_comm = dict(zip(node_to_comm_pd['id'].astype(int), node_to_comm_pd['community_id'].astype(int)))

            node_to_comm_bc = sc.broadcast(node_to_comm)
            major_comms_bc = sc.broadcast(set(major_comms))

            def map_edges(iterator):
                n2c = node_to_comm_bc.value
                maj = major_comms_bc.value
                for row in iterator:
                    u, v = row.src, row.dst
                    u_comm = n2c.get(u, -1)
                    v_comm = n2c.get(v, -1)
                    u_mapped = -1000 - u_comm if u_comm in maj else u
                    v_mapped = -1000 - v_comm if v_comm in maj else v
                    if u_mapped != v_mapped:
                        yield (u_mapped, v_mapped)

            print("    - Computing global CAAN edges map-side (shuffless RDD partitioning)...")
            caan_edges = (raw_edges_df.rdd
                          .mapPartitions(map_edges)
                          .distinct()
                          .collect())
            
            # Broadcast caan components - Driver-side pre-stacking for optimization
            minor_ids = list(minor_nodes_dict.keys())
            minor_node_to_idx = {int(nid): idx for idx, nid in enumerate(minor_ids)}
            
            feat_dim = cfg['in_feats']
            if len(minor_ids) > 0:
                minor_feats_arr = np.stack([minor_nodes_dict[nid]['features'] for nid in minor_ids]).astype(np.float32)
                minor_labels_arr = np.array([minor_nodes_dict[nid]['label'] for nid in minor_ids], dtype=np.int64)
                minor_splits_arr = np.array([minor_nodes_dict[nid]['split'] for nid in minor_ids], dtype=object)
                minor_ids_arr = np.array(minor_ids, dtype=np.int64)
            else:
                minor_feats_arr = np.empty((0, feat_dim), dtype=np.float32)
                minor_labels_arr = np.empty((0,), dtype=np.int64)
                minor_splits_arr = np.empty((0,), dtype=object)
                minor_ids_arr = np.empty((0,), dtype=np.int64)
                
            from collections import defaultdict
            caan_adj = defaultdict(list)
            for u, v in caan_edges:
                caan_adj[int(u)].append(int(v))
                
            super_nodes_dict_bc = sc.broadcast(super_nodes_dict)
            minor_node_to_idx_bc = sc.broadcast(minor_node_to_idx)
            minor_feats_arr_bc = sc.broadcast(minor_feats_arr)
            minor_labels_arr_bc = sc.broadcast(minor_labels_arr)
            minor_splits_arr_bc = sc.broadcast(minor_splits_arr)
            minor_ids_arr_bc = sc.broadcast(minor_ids_arr)
            caan_adj_bc = sc.broadcast(dict(caan_adj))
            node_to_comm_bc = sc.broadcast(node_to_comm)
            major_comms_bc = sc.broadcast(major_comms)
            
            caan_components = {
                'super_nodes_dict': super_nodes_dict,
                'minor_nodes_dict': minor_nodes_dict,
                'caan_edges': caan_edges
            }
            
            for model_type in gnn_models:
                key = (dataset, alg, model_type)
                
                # Checkpoint Path on S3 or local depending on local_data_dir
                s3_bucket = kwargs.get('s3_bucket', 'us-east-1-s3-gnn')
                experiment_name = kwargs.get('experiment_name', 'run-all')
                local_data_dir = kwargs.get('local_data_dir', None)
                
                if local_data_dir:
                    ckpt_dir = os.path.join(local_data_dir, "gnn-bench-checkpoint", "phase3b", experiment_name)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"{dataset}_{alg}_{model_type}.parquet")
                else:
                    ckpt_path = f"s3://{s3_bucket}/gnn-bench-checkpoint/phase3b/{experiment_name}/{dataset}_{alg}_{model_type}.parquet"

                # Checkpoint loading
                force_rerun = kwargs.get('force_rerun', False)
                loaded_from_ckpt = False
                
                if not force_rerun:
                    try:
                        if local_data_dir:
                            if os.path.exists(ckpt_path):
                                comm_pd = pd.read_parquet(ckpt_path)
                                loaded_from_ckpt = True
                        else:
                            # S3 check and download using boto3
                            import boto3
                            from botocore.exceptions import ClientError
                            import tempfile
                            s3_client = boto3.client('s3')
                            tmp_file = tempfile.mktemp(suffix=".parquet")
                            s3_key = f"gnn-bench-checkpoint/phase3b/{experiment_name}/{dataset}_{alg}_{model_type}.parquet"
                            try:
                                s3_client.download_file(s3_bucket, s3_key, tmp_file)
                                comm_pd = pd.read_parquet(tmp_file)
                                if os.path.exists(tmp_file):
                                    os.remove(tmp_file)
                                loaded_from_ckpt = True
                            except ClientError as e:
                                if e.response['Error']['Code'] == "404":
                                    pass
                                else:
                                    raise e
                    except Exception as ex:
                        print(f"    - Failed to load checkpoint: {ex}")
                
                if loaded_from_ckpt:
                    print(f"\n{'='*60}")
                    print(f"  PHASE 3b — GNN Training (CaaN): {dataset} / {alg} / {model_type} (Loaded from Checkpoint)")
                    print(f"{'='*60}")
                    total_test_nodes = comm_pd['n_test'].sum()
                    weighted_comm_acc = (comm_pd['comm_test_acc'] * comm_pd['n_test']).sum() / total_test_nodes if total_test_nodes > 0 else 0.0
                    total_edges = comm_pd['n_edges'].sum()
                    weighted_comm_link_auc = (comm_pd['comm_link_auc'] * comm_pd['n_edges']).sum() / total_edges if total_edges > 0 else 0.5
                    
                    # Store with attrs attached
                    results[key] = comm_pd.copy()
                    results[key].attrs['weighted_comm_acc']  = weighted_comm_acc
                    results[key].attrs['weighted_comm_link_auc'] = weighted_comm_link_auc
                    results[key].attrs['wall_time_s'] = 0.0
                    results[key].attrs['dataset']     = dataset
                    results[key].attrs['alg']         = alg
                    results[key].attrs['model_type']  = model_type
                    
                    timing[('phase3b', dataset, alg, model_type)] = 0.0
                    print(f"  ✓ Loaded model accuracy: {weighted_comm_acc:.4f}, skipping training.")
                    continue

                t0 = time.time()
                print(f"\n{'='*60}")
                print(f"  PHASE 3b — GNN Training (CaaN): {dataset} / {alg} / {model_type}")
                print(f"  tag={p_alg['tag']}")
                print(f"{'='*60}")
                
                # 1. Driver-side warmup training (GraphSAGE base weights and embeddings)
                base_weights_bc = None
                base_embeddings_bc = None
                base_node_map_bc = None
                
                try:
                    import torch
                    import torch.nn as nn
                    import dgl
                    
                    print("  [Driver Warmstart] Extracting representative community for driver-side pre-training...")
                    comms_node_counts_df = nodes_w_comm.select('community_id').groupBy('community_id').count().toPandas()
                    comms_sorted = comms_node_counts_df.sort_values(by='count', ascending=False)
                    
                    valid_comms = comms_sorted[comms_sorted['count'] <= 100_000]
                    if len(valid_comms) == 0:
                        target_comm_row = comms_sorted.iloc[-1]
                    else:
                        target_comm_row = valid_comms.iloc[0]
                        
                    largest_comm_id = int(target_comm_row['community_id'])
                    comm_count = int(target_comm_row['count'])
                    
                    if comm_count > 100_000:
                        print(f"  [Driver Warmstart] Community size ({comm_count:,} nodes) exceeds driver RAM safety limit — sub-sampling 50,000 nodes...")
                        large_comm_pdf = nodes_w_comm.filter(F.col('community_id') == largest_comm_id).limit(50000).toPandas()
                    else:
                        large_comm_pdf = nodes_w_comm.filter(F.col('community_id') == largest_comm_id).toPandas()
                    
                    in_feats = len(large_comm_pdf['features'].iloc[0])
                    num_classes = int(cfg['num_classes'])
                    hidden_dim = int(gcn_cfg['hidden_dim'])
                    
                    # Map nodes
                    all_nodes = large_comm_pdf['id'].values
                    n_nodes = len(all_nodes)
                    node_map = {int(n): i for i, n in enumerate(all_nodes)}
                    
                    large_edges_pdf = spark.read.format('delta').load(p_alg['p2_edges']).filter(F.col('community_id') == largest_comm_id).toPandas()
                    if len(large_edges_pdf) > 0:
                        src_arr = large_edges_pdf['src'].values.astype(np.int64)
                        dst_arr = large_edges_pdf['dst'].values.astype(np.int64)
                        valid = np.isin(src_arr, all_nodes) & np.isin(dst_arr, all_nodes)
                        src_l = np.array([node_map[s] for s in src_arr[valid]], dtype=np.int64)
                        dst_l = np.array([node_map[d] for d in dst_arr[valid]], dtype=np.int64)
                    else:
                        src_l = np.array([], dtype=np.int64)
                        dst_l = np.array([], dtype=np.int64)
                        
                    g_large = dgl.graph((src_l, dst_l), num_nodes=n_nodes)
                    g_large = dgl.add_self_loop(g_large)
                    
                    feat_arr = np.stack(large_comm_pdf['features'].values).astype(np.float32)
                    feat_norms = np.linalg.norm(feat_arr, axis=1, keepdims=True)
                    feat_arr = feat_arr / np.where(feat_norms > 0, feat_norms, 1.0)
                    feat_t = torch.tensor(feat_arr, dtype=torch.float32)
                    
                    lbl_arr = np.array([int(v) if not pd.isna(v) else -1 for v in large_comm_pdf['label'].values], dtype=np.int64)
                    lbl_t = torch.tensor(lbl_arr, dtype=torch.long)
                    has_lbl = torch.tensor(lbl_arr >= 0, dtype=torch.bool)
                    
                    splits = list(large_comm_pdf['split'].values)
                    train_m = torch.tensor([s == 'train' for s in splits], dtype=torch.bool) & has_lbl
                    
                    class DriverGraphSAGE(nn.Module):
                        def __init__(self, in_f, h, nc):
                            super().__init__()
                            import dgl.nn as dglnn
                            self.c1 = dglnn.SAGEConv(in_f, h, 'mean')
                            self.c2 = dglnn.SAGEConv(h,    h, 'mean')
                            self.fc = nn.Linear(h, nc)
                            self.dr = nn.Dropout(float(gcn_cfg['dropout']))
                        def forward(self, g, x):
                            x = torch.relu(self.c1(g, x)); x = self.dr(x)
                            x = self.c2(g, x)
                            return self.fc(x)
                        def encode(self, g, x):
                            x = torch.relu(self.c1(g, x)); x = self.dr(x)
                            return self.c2(g, x)
                            
                    base_model = DriverGraphSAGE(in_feats, hidden_dim, num_classes)
                    opt = torch.optim.Adam(base_model.parameters(), lr=float(gcn_cfg['lr']))
                    crit = nn.CrossEntropyLoss()
                    
                    base_model.train()
                    for _ in range(5):
                        opt.zero_grad()
                        logits = base_model(g_large, feat_t)
                        if train_m.sum() > 0:
                            loss = crit(logits[train_m], lbl_t[train_m])
                            loss.backward()
                            opt.step()
                            
                    base_model.eval()
                    with torch.no_grad():
                        global_embeddings = base_model.encode(g_large, feat_t).numpy()
                        
                    # Broadcast state dict and embeddings safely
                    base_weights_bc = sc.broadcast(base_model.state_dict())
                    if len(node_map) <= 5_000_000:
                        base_embeddings_bc = sc.broadcast(global_embeddings)
                        base_node_map_bc = sc.broadcast(node_map)
                    else:
                        base_embeddings_bc = None
                        base_node_map_bc = None
                    print(f"  ✓ Driver-side base model trained successfully on Comm {largest_comm_id} ({n_nodes:,} nodes).")
                except Exception as base_err:
                    print(f"  Warning: Skipped warm-start driver pre-training: {base_err}")

                minor_df = _train_minor_global_caan(
                    dataset=dataset,
                    gcn_cfg=gcn_cfg,
                    dataset_cfg=dataset_cfg,
                    caan_components=caan_components,
                    model_type=model_type,
                    task_type=task_type
                )
                
                # 2. Driver-side Community Binning
                comms_node_counts_df = nodes_w_comm.select('community_id').groupBy('community_id').count().toPandas()
                comms_node_counts_df = comms_node_counts_df.sort_values(by='count', ascending=False).reset_index(drop=True)
                
                num_comms = len(comms_node_counts_df)
                if num_comms <= 2000:
                    num_bins = num_comms
                else:
                    default_para = sc.defaultParallelism
                    num_bins = min(max(default_para * 4, 1000), num_comms)
                comms_node_counts_df['bin_id'] = [i % num_bins for i in range(len(comms_node_counts_df))]
                
                bin_mapping_df = spark.createDataFrame(comms_node_counts_df[['community_id', 'bin_id']])
                
                manifest_df = (bin_mapping_df
                    .withColumn('_num_classes', F.lit(int(cfg['num_classes'])))
                    .withColumn('_hidden',      F.lit(int(gcn_cfg['hidden_dim'])))
                    .withColumn('_epochs',      F.lit(int(gcn_cfg['num_epochs'])))
                    .withColumn('_lr',          F.lit(float(gcn_cfg['lr'])))
                    .withColumn('_dropout',     F.lit(float(gcn_cfg['dropout'])))
                    .withColumn('_task_type',   F.lit(str(task_type)))
                    .withColumn('_model_type',  F.lit(str(model_type))))
                
                caan_udf = make_caan_udf(
                    super_nodes_dict_bc=super_nodes_dict_bc,
                    minor_node_to_idx_bc=minor_node_to_idx_bc,
                    minor_feats_arr_bc=minor_feats_arr_bc,
                    minor_labels_arr_bc=minor_labels_arr_bc,
                    minor_splits_arr_bc=minor_splits_arr_bc,
                    minor_ids_arr_bc=minor_ids_arr_bc,
                    caan_adj_bc=caan_adj_bc,
                    node_to_comm_bc=node_to_comm_bc,
                    major_comms_bc=major_comms_bc,
                    base_weights_bc=base_weights_bc,
                    base_embeddings_bc=base_embeddings_bc,
                    base_node_map_bc=base_node_map_bc,
                    p2_nodes_url=p_alg['p2_nodes'],
                    p2_edges_url=p_alg['p2_edges']
                )
                
                sc.setJobDescription(f'phase3b_{dataset}_{alg}_{model_type}')
                major_results = (manifest_df
                                 .repartition(num_bins, 'bin_id')
                                 .groupBy('bin_id')
                                 .applyInPandas(caan_udf, schema=result_schema))
                major_pd = major_results.toPandas()
                sc.setJobDescription('')
                
                if minor_df is not None:
                    comm_pd = pd.concat([minor_df, major_pd], ignore_index=True)
                else:
                    comm_pd = major_pd
                    
                total_test_nodes = comm_pd['n_test'].sum()
                weighted_comm_acc = (comm_pd['comm_test_acc'] * comm_pd['n_test']).sum() / total_test_nodes if total_test_nodes > 0 else 0.0
                total_edges = comm_pd['n_edges'].sum()
                weighted_comm_link_auc = (comm_pd['comm_link_auc'] * comm_pd['n_edges']).sum() / total_edges if total_edges > 0 else 0.5
                
                elapsed = time.time() - t0
                timing[('phase3b', dataset, alg, model_type)] = elapsed
                
                results[key] = comm_pd.copy()
                results[key].attrs['weighted_comm_acc']  = weighted_comm_acc
                results[key].attrs['weighted_comm_link_auc'] = weighted_comm_link_auc
                results[key].attrs['wall_time_s'] = elapsed
                results[key].attrs['dataset']     = dataset
                results[key].attrs['alg']         = alg
                results[key].attrs['model_type']  = model_type
                
                print(f"  ✓ Mean CaaN comm acc = {comm_pd['comm_test_acc'].mean():.4f}")
                print(f"  ✓ Weighted CaaN comm acc = {weighted_comm_acc:.4f}")
                print(f"  ✓ Wall time: {elapsed:.1f}s")

                # Save checkpoint
                try:
                    if local_data_dir:
                        comm_pd.to_parquet(ckpt_path, index=False)
                        print(f"    ✓ Saved checkpoint locally: {ckpt_path}")
                    else:
                        import tempfile
                        import boto3
                        tmp_file = tempfile.mktemp(suffix=".parquet")
                        comm_pd.to_parquet(tmp_file, index=False)
                        s3_client = boto3.client('s3')
                        s3_key = f"gnn-bench-checkpoint/phase3b/{experiment_name}/{dataset}_{alg}_{model_type}.parquet"
                        s3_client.upload_file(tmp_file, s3_bucket, s3_key)
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        print(f"    ✓ Saved S3 checkpoint: s3://{s3_bucket}/{s3_key}")
                except Exception as ex:
                    print(f"    ⚠️ Failed to save checkpoint: {ex}")
