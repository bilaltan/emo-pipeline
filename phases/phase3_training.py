import os
import sys
import time
import pandas as pd
import numpy as np

# Forward-compatibility shim for NumPy 1.x / 2.x cross-node pickle compatibility
try:
    import numpy.core as _np_core
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = _np_core
        sys.modules['numpy._core.numeric'] = _np_core.numeric
        sys.modules['numpy._core.multiarray'] = _np_core.multiarray
        sys.modules['numpy._core._multiarray_umath'] = getattr(_np_core, '_multiarray_umath', _np_core)
        sys.modules['numpy._core._exceptions'] = getattr(_np_core, '_exceptions', _np_core)
        sys.modules['numpy._core.umath'] = getattr(_np_core, 'umath', _np_core)
except Exception:
    pass

from pipeline.utils.common import _patch_torch_load

def _make_result_schema():
    from pyspark.sql.types import (StructType, StructField,
                                   LongType, DoubleType, StringType)
    return StructType([
        StructField('community_id',   LongType()),
        StructField('n_nodes',        LongType()),
        StructField('n_edges',        LongType()),
        StructField('n_train',        LongType()),
        StructField('n_val',          LongType()),
        StructField('n_test',         LongType()),
        StructField('n_boundary',     LongType()),
        StructField('n_internal',     LongType()),
        StructField('comm_test_acc',  DoubleType()),
        # Both classifier heads are recorded so the choice between them is a
        # measurement rather than an assumption, and so a dataset where they
        # disagree is visible instead of silently reported as one or the other.
        StructField('acc_gnn_head',   DoubleType()),
        StructField('acc_mlp_head',   DoubleType()),
        StructField('head_used',      StringType()),
        StructField('boundary_acc',   DoubleType()),
        StructField('internal_acc',   DoubleType()),
        StructField('comm_link_auc',  DoubleType()),
        StructField('size_bucket',    StringType()),
        StructField('load_time_s',    DoubleType()),
        StructField('node_train_time_s', DoubleType()),
        StructField('link_train_time_s', DoubleType()),
        StructField('peak_mem_mb',    DoubleType()),
    ])

_DS_CACHE = {}

def _get_dataset(url):
    if url not in _DS_CACHE:
        import pyarrow.dataset as ds
        if url.startswith("s3://"):
            import pyarrow.fs as fs
            clean_url = url.replace("s3://", "").rstrip('/')
            s3 = fs.S3FileSystem(region="us-east-1")
            _DS_CACHE[url] = ds.dataset(clean_url, filesystem=s3, format="parquet", ignore_prefixes=['_delta_log', '.'])
        else:
            local_path = url.replace("file://", "")
            _DS_CACHE[url] = ds.dataset(local_path, format="parquet", ignore_prefixes=['_delta_log', '.'])
    return _DS_CACHE[url]

def _load_communities_data_batch(nodes_url, edges_url, comm_ids):
    """
    Direct C++ PyArrow Dataset reader for worker tasks.
    Reads multiple community partition nodes and edges directly from S3/disk Delta Parquet files in a single batch.
    """
    import pyarrow.dataset as ds
    import pandas as pd

    try:
        nodes_ds = _get_dataset(nodes_url)
        nodes_pdf = nodes_ds.to_table(filter=(ds.field("community_id").isin(comm_ids)), use_threads=True).to_pandas()
    except Exception as e:
        print(f"  ⚠ _load_communities_data_batch NODES ERROR: {e}")
        nodes_pdf = pd.DataFrame()

    try:
        edges_ds = _get_dataset(edges_url)
        edges_pdf = edges_ds.to_table(filter=(ds.field("community_id").isin(comm_ids)), use_threads=True).to_pandas()
    except Exception as e:
        print(f"  ⚠ _load_communities_data_batch EDGES ERROR: {e}")
        edges_pdf = pd.DataFrame()

    return nodes_pdf, edges_pdf


def _train_gnn_community_single(pdf, comm_edges_pdf=None, base_weights_bc=None, base_embeddings_bc=None, base_node_map_bc=None):
    """
    Spark Pandas UDF — runs on executor, one call per community.
    Hyperparams are read from constant DataFrame columns to avoid closure issues.
    Returns one-row DataFrame per community with all metrics.
    """
    if pdf is None or len(pdf) == 0:
        return pd.DataFrame([{
            'community_id':   -1,
            'n_nodes':        0,
            'n_edges':        0,
            'n_train':        0,
            'n_val':          0,
            'n_test':         0,
            'n_boundary':     0,
            'n_internal':     0,
            'comm_test_acc':  0.0,
            'acc_gnn_head':   0.0,
            'acc_mlp_head':   0.0,
            'head_used':      'none',
            'boundary_acc':   0.0,
            'internal_acc':   0.0,
            'comm_link_auc':  0.5,
            'size_bucket':    'empty',
            'load_time_s':    0.0,
            'node_train_time_s': 0.0,
            'link_train_time_s': 0.0,
            'peak_mem_mb':    0.0,
        }])

    import os, time, subprocess, sys, resource
    import numpy as np
    import pandas as pd
    import inspect
    worker_start = time.time()

    # Patch torch.load inline on workers to prevent weights_only=True unpickling error
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
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir',
                            'dgl==1.1.3', '-f', 'https://data.dgl.ai/wheels/repo.html'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            import dgl
            import dgl.nn as dglnn
        except Exception:
            # DGL is unavailable (no wheel for this platform). PyG covers every
            # backbone, so route SAGE through PyG rather than failing the run.
            dgl = None
            dglnn = None

    try:
        omp_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
        torch.set_num_threads(omp_threads)
    except Exception:
        pass

    t_start     = time.time()
    comm_id     = int(pdf['community_id'].iloc[0])
    diagnostics = bool(pdf['_phase3_diagnostics'].iloc[0]) if '_phase3_diagnostics' in pdf.columns else False
    worker_name = os.environ.get('HOSTNAME', 'unknown-worker')

    def diagnostic(message):
        if diagnostics:
            print(f"[PHASE3-WORKER] host={worker_name} community={comm_id} {message}", flush=True)

    diagnostic(f"started with {len(pdf):,} node rows")
    num_classes = int(pdf['_num_classes'].iloc[0])
    hidden_dim  = int(pdf['_hidden'].iloc[0])
    num_epochs  = int(pdf['_epochs'].iloc[0])
    lr          = float(pdf['_lr'].iloc[0])
    dropout     = float(pdf['_dropout'].iloc[0])
    task_type   = str(pdf['_task_type'].iloc[0]) if '_task_type' in pdf.columns else 'node_classification'
    model_type  = str(pdf['_model_type'].iloc[0]) if '_model_type' in pdf.columns else 'sage'
    max_nodes_per_community = int(pdf['_max_nodes'].iloc[0]) if '_max_nodes' in pdf.columns else 10000
    max_edges_per_community = int(pdf['_max_edges'].iloc[0]) if '_max_edges' in pdf.columns else 200000
    mlp_epochs = int(pdf['_mlp_epochs'].iloc[0]) if '_mlp_epochs' in pdf.columns else 5
    mlp_patience = int(pdf['_mlp_patience'].iloc[0]) if '_mlp_patience' in pdf.columns else 15
    node_patience = int(pdf['_node_patience'].iloc[0]) if '_node_patience' in pdf.columns else 10
    diagnostic(f"dependencies ready after {time.time() - worker_start:.1f}s; model={model_type}; task={task_type}")

    # Deterministic weight init per unit. Without this the same configuration moves
    # several accuracy points between runs purely from random initialisation, which
    # makes any reported mean±std measure noise rather than data variance. Seeding
    # by community keeps units independent while staying reproducible.
    torch.manual_seed(42 + int(comm_id))
    np.random.seed((42 + int(comm_id)) % (2**32))

    # Support both bundled single-row community representations and multi-row frames
    if '_id_list' in pdf.columns and len(pdf) == 1:
        raw_ids = pdf['_id_list'].iloc[0]
        if raw_ids is None or len(raw_ids) == 0:
            return pd.DataFrame([{
                'community_id':   int(comm_id),
                'n_nodes':        0,
                'n_edges':        0,
                'n_train':        0,
                'n_val':          0,
                'n_test':         0,
                'n_boundary':     0,
                'n_internal':     0,
                'comm_test_acc':  0.0,
                'boundary_acc':   0.0,
                'internal_acc':   0.0,
                'comm_link_auc':  0.5,
                'size_bucket':    'empty',
                'load_time_s':    0.0,
                'node_train_time_s': 0.0,
                'link_train_time_s': 0.0,
                'peak_mem_mb':    0.0,
            }])
        all_nodes = np.array(raw_ids, dtype=np.int64)
        raw_labels = pdf['_label_list'].iloc[0] if '_label_list' in pdf.columns else []
        raw_feats  = pdf['_features_list'].iloc[0] if '_features_list' in pdf.columns else []
        split_arr  = list(pdf['_split_list'].iloc[0]) if '_split_list' in pdf.columns else []
        bnd_arr    = np.array([bool(v) if not (pd.isna(v) or v is None) else False for v in pdf['_is_boundary_list'].iloc[0]], dtype=bool) if '_is_boundary_list' in pdf.columns else np.zeros(len(all_nodes), dtype=bool)
        label_arr  = np.array([int(v) if not pd.isna(v) else -1 for v in raw_labels], dtype=np.int64)

        n_nodes = len(all_nodes)
        if len(label_arr) != n_nodes:
            label_arr = np.resize(label_arr, (n_nodes,))
        if len(split_arr) != n_nodes:
            split_arr = split_arr[:n_nodes] if len(split_arr) > n_nodes else split_arr + ['none'] * (n_nodes - len(split_arr))
        if len(bnd_arr) != n_nodes:
            bnd_arr = np.resize(bnd_arr, (n_nodes,))
    else:
        all_nodes = pdf['id'].values.astype(np.int64)
        raw_labels = pdf['label'].values
        raw_feats  = pdf['features'].values
        split_arr  = list(pdf['split'].values)
        bnd_arr    = np.array([bool(v) if not (pd.isna(v) or v is None) else False for v in pdf['is_boundary'].values], dtype=bool)
        label_arr  = np.array([int(v) if not pd.isna(v) else -1 for v in raw_labels], dtype=np.int64)
        n_nodes    = len(all_nodes)

    # Fast-Path for Micro-Communities (< 5 nodes) — instant evaluation without PyTorch setup
    if n_nodes < 5:
        has_lbl = label_arr >= 0
        train_m = np.array([s == 'train' for s in split_arr]) & has_lbl
        test_m  = np.array([s == 'test'  for s in split_arr]) & has_lbl

        if train_m.sum() > 0:
            maj_class = int(pd.Series(label_arr[train_m]).mode().iloc[0])
        elif has_lbl.sum() > 0:
            maj_class = int(pd.Series(label_arr[has_lbl]).mode().iloc[0])
        else:
            maj_class = 0

        n_test = int(test_m.sum())
        if n_test > 0:
            test_acc = float((label_arr[test_m] == maj_class).sum() / n_test)
            bnd_test_m = test_m & bnd_arr
            int_test_m = test_m & (~bnd_arr)
            bnd_acc = float((label_arr[bnd_test_m] == maj_class).sum() / bnd_test_m.sum()) if bnd_test_m.sum() > 0 else 0.0
            int_acc = float((label_arr[int_test_m] == maj_class).sum() / int_test_m.sum()) if int_test_m.sum() > 0 else 0.0
        else:
            test_acc = 0.0
            bnd_acc  = 0.0
            int_acc  = 0.0

        return pd.DataFrame([{
            'community_id':   int(comm_id),
            'n_nodes':        int(n_nodes),
            'n_edges':        0,
            'n_train':        int(train_m.sum()),
            'n_val':          int((np.array([s == 'valid' for s in split_arr]) & has_lbl).sum()),
            'n_test':         int(n_test),
            'n_boundary':     int(bnd_arr.sum()),
            'n_internal':     int((~bnd_arr).sum()),
            'comm_test_acc':  test_acc,
            'acc_gnn_head':   test_acc,
            'acc_mlp_head':   test_acc,
            'head_used':      'majority',
            'boundary_acc':   bnd_acc,
            'internal_acc':   int_acc,
            'comm_link_auc':  0.5,
            'size_bucket':    'small',
            'load_time_s':    0.0,
            'node_train_time_s': 0.0,
            'link_train_time_s': 0.0,
            'peak_mem_mb':    0.0,
        }])

    # Final RAM safety floor for the small amount of variation left by Spark-side sampling.
    if n_nodes > max_nodes_per_community:
        all_nodes = all_nodes[:max_nodes_per_community]
        label_arr = label_arr[:max_nodes_per_community]
        split_arr = split_arr[:max_nodes_per_community]
        bnd_arr   = bnd_arr[:max_nodes_per_community]
        n_nodes   = len(all_nodes)
        if len(raw_feats) > max_nodes_per_community:
            raw_feats = raw_feats[:max_nodes_per_community]

    sorted_ids = np.sort(all_nodes)
    sort_idx   = np.argsort(all_nodes)
    
    # Fast vectorized edge mapping
    if '_src_list' in pdf.columns and '_dst_list' in pdf.columns:
        raw_src = pdf['_src_list'].iloc[0]
        raw_dst = pdf['_dst_list'].iloc[0]
        if raw_src is not None and not (isinstance(raw_src, float) and np.isnan(raw_src)) and len(raw_src) > 0:
            src_arr = np.array(raw_src, dtype=np.int64)
            dst_arr = np.array(raw_dst, dtype=np.int64)
            idx_src = np.searchsorted(sorted_ids, src_arr)
            idx_dst = np.searchsorted(sorted_ids, dst_arr)

            valid = (idx_src < n_nodes) & (sorted_ids[np.minimum(idx_src, n_nodes - 1)] == src_arr) & \
                    (idx_dst < n_nodes) & (sorted_ids[np.minimum(idx_dst, n_nodes - 1)] == dst_arr)

            src_l = sort_idx[idx_src[valid]].astype(np.int64)
            dst_l = sort_idx[idx_dst[valid]].astype(np.int64)
        else:
            src_l = np.array([], dtype=np.int64)
            dst_l = np.array([], dtype=np.int64)
    elif comm_edges_pdf is not None and len(comm_edges_pdf) > 0:
        src_arr = comm_edges_pdf['src'].values.astype(np.int64)
        dst_arr = comm_edges_pdf['dst'].values.astype(np.int64)
        idx_src = np.searchsorted(sorted_ids, src_arr)
        idx_dst = np.searchsorted(sorted_ids, dst_arr)

        valid = (idx_src < n_nodes) & (sorted_ids[np.minimum(idx_src, n_nodes - 1)] == src_arr) & \
                (idx_dst < n_nodes) & (sorted_ids[np.minimum(idx_dst, n_nodes - 1)] == dst_arr)

        src_l = sort_idx[idx_src[valid]].astype(np.int64)
        dst_l = sort_idx[idx_dst[valid]].astype(np.int64)
    elif 'neighbors' in pdf.columns:
        exploded = pdf[['id', 'neighbors']].explode('neighbors').dropna()
        if len(exploded) > 0:
            src_arr = exploded['id'].values.astype(np.int64)
            dst_arr = exploded['neighbors'].values.astype(np.int64)

            idx_src = np.searchsorted(sorted_ids, src_arr)
            idx_dst = np.searchsorted(sorted_ids, dst_arr)

            valid = (idx_src < n_nodes) & (sorted_ids[np.minimum(idx_src, n_nodes - 1)] == src_arr) & \
                    (idx_dst < n_nodes) & (sorted_ids[np.minimum(idx_dst, n_nodes - 1)] == dst_arr)

            src_l = sort_idx[idx_src[valid]].astype(np.int64)
            dst_l = sort_idx[idx_dst[valid]].astype(np.int64)
        else:
            src_l = np.array([], dtype=np.int64)
            dst_l = np.array([], dtype=np.int64)
    else:
        src_l = np.array([], dtype=np.int64)
        dst_l = np.array([], dtype=np.int64)
        
    n_edges = len(src_l)

    t_load = time.time() - t_start
    t_dgl_conv_start = time.time()

    if len(raw_feats) > 0 and isinstance(raw_feats[0], (np.ndarray, list, tuple)):
        feat_arr = np.ascontiguousarray(np.vstack(raw_feats), dtype=np.float32)
    elif len(raw_feats) > 0:
        feat_arr = np.stack(raw_feats).astype(np.float32)
    else:
        feat_arr = np.zeros((len(all_nodes), 128), dtype=np.float32)

    feat_norms = np.linalg.norm(feat_arr, axis=1, keepdims=True)
    feat_arr  = feat_arr / np.where(feat_norms > 0, feat_norms, 1.0)

    if n_edges > max_edges_per_community:
        np.random.seed(42)
        keep_idx = np.random.choice(n_edges, max_edges_per_community, replace=False)
        src_l_g = src_l[keep_idx]
        dst_l_g = dst_l[keep_idx]
    else:
        src_l_g = src_l
        dst_l_g = dst_l

    # Which rows are this community's own vertices, as opposed to 1-hop halo
    # vertices borrowed from neighbours. Phase 2 marks these with is_member; the
    # local harness signals the same split positionally via _n_local. Halo
    # vertices are message-passing context only — counting them would score the
    # same vertex once per community that borrows it.
    from pipeline.utils.common import resolve_member_mask
    member_arr = resolve_member_mask(pdf, n_nodes)
    member_t = torch.tensor(member_arr, dtype=torch.bool)

    has_label = torch.tensor(label_arr >= 0, dtype=torch.bool) & member_t
    train_m = torch.tensor([s == 'train' for s in split_arr], dtype=torch.bool) & has_label
    val_m   = torch.tensor([s == 'valid' for s in split_arr], dtype=torch.bool) & has_label
    test_m  = torch.tensor([s == 'test'  for s in split_arr], dtype=torch.bool) & has_label
    bnd_t = torch.tensor(bnd_arr, dtype=torch.bool)

    is_pyg = (model_type in ('gat', 'gatv2', 'transformer', 'clusterscl', 'arma', 'asap')) or dgl is None
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
        g = dgl.add_self_loop(g)
        g.ndata['feat']  = torch.tensor(feat_arr,  dtype=torch.float32)
        g.ndata['label'] = torch.tensor(label_arr, dtype=torch.int64)
        feat_t, lbl_t = g.ndata['feat'], g.ndata['label']

    t_dgl_conv = time.time() - t_dgl_conv_start
    diagnostic(f"graph ready: nodes={n_nodes:,}, edges={n_edges:,}, load={t_load:.1f}s, graph_build={t_dgl_conv:.1f}s")

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
            
        elif model_type == 'gatv2':
            from torch_geometric.nn import GATv2Conv
            class GATv2Encoder(nn.Module):
                def __init__(self, in_f, h, num_heads=8):
                    super().__init__()
                    self.c1 = GATv2Conv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                    self.c2 = GATv2Conv(h, h, heads=1, concat=False, dropout=dropout)
                    self.dr = nn.Dropout(dropout)
                def forward(self, x, edge_index):
                    x = F.elu(self.c1(x, edge_index))
                    x = self.dr(x)
                    return self.c2(x, edge_index)
            class GATv2Net(nn.Module):
                def __init__(self, in_f, h, nc):
                    super().__init__()
                    self.enc = GATv2Encoder(in_f, h)
                    self.fc = nn.Linear(h, nc)
                def forward(self, x, edge_index):
                    z = self.enc(x, edge_index)
                    return self.fc(z)
            model = GATv2Net(feat_arr.shape[1], hidden_dim, num_classes)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            crit = nn.CrossEntropyLoss()
            
        elif model_type == 'arma':
            from torch_geometric.nn import ARMAConv
            class ARMAEncoder(nn.Module):
                def __init__(self, in_f, h):
                    super().__init__()
                    self.c1 = ARMAConv(in_f, h, dropout=dropout)
                    self.c2 = ARMAConv(h, h, dropout=dropout)
                    self.dr = nn.Dropout(dropout)
                def forward(self, x, edge_index):
                    x = F.relu(self.c1(x, edge_index))
                    x = self.dr(x)
                    return self.c2(x, edge_index)
            class ARMANet(nn.Module):
                def __init__(self, in_f, h, nc):
                    super().__init__()
                    self.enc = ARMAEncoder(in_f, h)
                    self.fc = nn.Linear(h, nc)
                def forward(self, x, edge_index):
                    z = self.enc(x, edge_index)
                    return self.fc(z)
            model = ARMANet(feat_arr.shape[1], hidden_dim, num_classes)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            crit = nn.CrossEntropyLoss()
            
        elif model_type == 'asap':
            from torch_geometric.nn import LEConv
            class ASAPEncoder(nn.Module):
                def __init__(self, in_f, h):
                    super().__init__()
                    self.c1 = LEConv(in_f, h)
                    self.c2 = LEConv(h, h)
                    self.dr = nn.Dropout(dropout)
                def forward(self, x, edge_index):
                    x = F.relu(self.c1(x, edge_index))
                    x = self.dr(x)
                    return self.c2(x, edge_index)
            class ASAPNet(nn.Module):
                def __init__(self, in_f, h, nc):
                    super().__init__()
                    self.enc = ASAPEncoder(in_f, h)
                    self.fc = nn.Linear(h, nc)
                def forward(self, x, edge_index):
                    z = self.enc(x, edge_index)
                    return self.fc(z)
            model = ASAPNet(feat_arr.shape[1], hidden_dim, num_classes)
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
            
        elif dgl is None:
            # PyG realization of the default GraphSAGE backbone, used wherever DGL
            # has no wheel for the platform. Same architecture as the DGL branch.
            from torch_geometric.nn import SAGEConv
            class PyGSAGEEncoder(nn.Module):
                def __init__(self, in_f, h):
                    super().__init__()
                    self.c1 = SAGEConv(in_f, h)
                    self.c2 = SAGEConv(h, h)
                    self.dr = nn.Dropout(dropout)
                def forward(self, x, edge_index):
                    x = torch.relu(self.c1(x, edge_index))
                    x = self.dr(x)
                    return self.c2(x, edge_index)
            class PyGSAGECommunity(nn.Module):
                def __init__(self, in_f, h, nc):
                    super().__init__()
                    self.enc = PyGSAGEEncoder(in_f, h)
                    self.fc = nn.Linear(h, nc)
                def forward(self, x, edge_index):
                    return self.fc(self.enc(x, edge_index))
            model = PyGSAGECommunity(feat_arr.shape[1], hidden_dim, num_classes)
            opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            crit  = nn.CrossEntropyLoss()

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
                raw_emb = base_embeddings_bc.value
                emb_arr = np.array(raw_emb, dtype=np.float32) if not isinstance(raw_emb, np.ndarray) else raw_emb
                emb_idx = [global_node_map.get(int(nid), -1) for nid in all_nodes]
                emb_idx_clean = [idx if idx != -1 else 0 for idx in emb_idx]
                global_emb_t = torch.tensor(emb_arr[emb_idx_clean], dtype=torch.float32)
                valid_emb_mask = torch.tensor([idx != -1 for idx in emb_idx], dtype=torch.bool)
            except Exception:
                pass

        model.train()
        if train_m.sum() > 0:
            # Train to convergence on validation accuracy, not to a fixed budget.
            # This loop takes one full-batch step per epoch, so a 10-epoch budget is
            # 10 gradient updates for the whole community — the same defect that left
            # the Phase 4 baseline below chance. Stopping on training loss (the old
            # rule) cannot detect overfitting and fires early on any flat step.
            import copy as _copy
            best_loss = float('inf')
            patience_counter = 0
            best_val_acc, best_state = -1.0, None
            val_every = max(1, int(num_epochs // 40) or 1)
            has_val = bool(val_m.sum() > 0)

            def _val_logits():
                if model_type == 'clusterscl':
                    return model.get_embeddings_and_logits(feat_t, pyg_edge_index)[2]
                if is_pyg:
                    return model(feat_t, pyg_edge_index)
                return model(g, feat_t)

            for _ep in range(num_epochs):
                if model_type == 'clusterscl':
                    opt.zero_grad()
                    z1, proj1, logits1 = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
                    z2, proj2, logits2 = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
                    loss_ce = crit(logits1[train_m], lbl_t[train_m])
                    
                    if valid_emb_mask is not None and valid_emb_mask.sum() > 0:
                        try:
                            local_emb, _, _ = model.get_embeddings_and_logits(feat_t, pyg_edge_index)
                            loss_reg = F.mse_loss(local_emb[valid_emb_mask], global_emb_t[valid_emb_mask])
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
                    curr_loss = loss_ce.item()
                elif is_pyg:
                    opt.zero_grad()
                    logits = model(feat_t, pyg_edge_index)
                    loss   = crit(logits[train_m], lbl_t[train_m])
                    
                    if valid_emb_mask is not None and valid_emb_mask.sum() > 0:
                        try:
                            local_emb = model.enc(feat_t, pyg_edge_index)
                            loss_reg = F.mse_loss(local_emb[valid_emb_mask], global_emb_t[valid_emb_mask])
                            loss = loss + 0.01 * loss_reg
                        except Exception:
                            pass

                    loss.backward()
                    opt.step()
                    curr_loss = loss.item()
                else:
                    opt.zero_grad()
                    logits = model(g, feat_t)
                    loss   = crit(logits[train_m], lbl_t[train_m])
                    
                    if valid_emb_mask is not None and valid_emb_mask.sum() > 0:
                        try:
                            local_emb = model.encode(g, feat_t)
                            loss_reg = F.mse_loss(local_emb[valid_emb_mask], global_emb_t[valid_emb_mask])
                            loss = loss + 0.01 * loss_reg
                        except Exception:
                            pass

                    loss.backward()
                    opt.step()
                    curr_loss = loss.item()

                if has_val and ((_ep + 1) % val_every == 0 or _ep == num_epochs - 1):
                    model.eval()
                    with torch.no_grad():
                        v_acc = float((_val_logits()[val_m].argmax(dim=1)
                                       == lbl_t[val_m]).float().mean())
                    model.train()
                    if v_acc > best_val_acc:
                        best_val_acc, patience_counter = v_acc, 0
                        best_state = _copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += val_every
                        if patience_counter >= node_patience:
                            break
                elif not has_val:
                    # No validation split in this unit: fall back to training loss.
                    if curr_loss < best_loss - 1e-4:
                        best_loss, patience_counter = curr_loss, 0
                    else:
                        patience_counter += 1
                        if patience_counter >= node_patience:
                            break

            if best_state is not None:
                model.load_state_dict(best_state)
        node_train_time = time.time() - t_node_start
        diagnostic(f"node training/evaluation finished in {node_train_time:.1f}s")
        
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
            """Probe head over frozen embeddings, sized from the data.

            The previous fixed 64->32 funnel was narrower than the label space on
            any dataset with more than 32 classes — on ogbn-arxiv's 40 classes the
            head, not the encoder, was the accuracy ceiling. Widths now scale with
            both the embedding dimension and the class count so the same code is
            reasonable from 2 classes to hundreds.
            """
            def __init__(self, input_dim, classes):
                super().__init__()
                h1 = max(64, min(input_dim, 4 * classes))
                h2 = max(32, min(h1, 2 * classes))
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, h1),
                    nn.ReLU(),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Linear(h2, classes)
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
            mlp_loader = DataLoader(mlp_dataset, batch_size=min(4096, max(128, len(train_idx))), shuffle=True)
            mlp_loss_fn = nn.CrossEntropyLoss()
            mlp_opt = torch.optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=5e-4)

            best_acc = -1.0
            best_weights = copy.deepcopy(mlp_model.state_dict())

            for mlp_epoch in range(mlp_epochs):
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
                            _mlp_wait = 0
                        else:
                            _mlp_wait = locals().get('_mlp_wait', 0) + 1
                            if _mlp_wait >= mlp_patience:
                                break
                else:
                    best_weights = copy.deepcopy(mlp_model.state_dict())

            mlp_model.load_state_dict(best_weights)

        # The encoder already carries a classifier trained end-to-end for this task;
        # the probe head is a second, weaker model over frozen features. Score both
        # on validation and keep the better one, so no dataset is stuck with a head
        # that happens to suit a different one.
        def _acc_of(pred_vec, mask):
            n = int(mask.sum())
            return float((pred_vec[mask] == lbl_t[mask]).float().mean()) if n else 0.0

        mlp_model.eval()
        model.eval()
        with torch.no_grad():
            preds_mlp = mlp_model(embed).argmax(dim=1)
            try:
                if model_type == 'clusterscl':
                    gnn_logits = model.get_embeddings_and_logits(feat_t, pyg_edge_index)[2]
                elif is_pyg:
                    gnn_logits = model(feat_t, pyg_edge_index)
                else:
                    gnn_logits = model(g, feat_t)
                preds_gnn = gnn_logits.argmax(dim=1)
            except Exception:
                preds_gnn = preds_mlp

            if int(val_m.sum()) > 0:
                use_gnn = _acc_of(preds_gnn, val_m) > _acc_of(preds_mlp, val_m)
            else:
                use_gnn = False
            preds = preds_gnn if use_gnn else preds_mlp
            head_used = 'gnn' if use_gnn else 'mlp'
            acc_gnn_head = _acc_of(preds_gnn, test_m)
            acc_mlp_head = _acc_of(preds_mlp, test_m)

        def safe_acc(mask):
            n = int(mask.sum())
            if n == 0:
                return 0.0, 0
            return float((preds[mask] == lbl_t[mask]).float().mean()), n

        comm_acc, n_test_node = safe_acc(test_m)
        bnd_acc,  _     = safe_acc(test_m & bnd_t)
        int_acc,  _     = safe_acc(test_m & ~bnd_t)
    
    link_train_time = 0.0
    comm_link_auc = 0.5
    test_edges_idx = []
    
    # Link prediction scores member-to-member pairs only; halo vertices stay as
    # message-passing context. Negatives are drawn from members for the same reason.
    _member_idx = np.where(member_arr)[0].astype(np.int64)
    if n_edges > 0 and len(_member_idx) < n_nodes:
        _real_edge_pos = np.where(member_arr[src_l] & member_arr[dst_l])[0]
    else:
        _real_edge_pos = np.arange(n_edges, dtype=np.int64)
    n_real_edges = len(_real_edge_pos)

    if run_link and n_real_edges >= 5:
        t_link_start = time.time()
        torch.manual_seed(42)
        _real_pos_t = torch.tensor(_real_edge_pos, dtype=torch.int64)
        shuffled_edge_ids = _real_pos_t[torch.randperm(n_real_edges)]
        n_tr_edges = int(0.8 * n_real_edges)
        n_val_edges = int(0.1 * n_real_edges)

        max_local_train = min(10000, n_tr_edges)
        max_local_test = min(2000, n_real_edges - n_tr_edges - n_val_edges)

        train_edges_idx = shuffled_edge_ids[:max_local_train]
        test_edges_idx = shuffled_edge_ids[n_tr_edges + n_val_edges : n_tr_edges + n_val_edges + max_local_test]

        src_l_t = torch.tensor(src_l, dtype=torch.int64)
        dst_l_t = torch.tensor(dst_l, dtype=torch.int64)
        # Sorted int64 edge keys for vectorized rejection sampling. A Python set of
        # edge tuples costs several GB on a multi-million-edge community.
        _edge_keys = np.unique(
            (src_l.astype(np.int64) * np.int64(n_nodes) + dst_l.astype(np.int64))
        ) if n_edges > 0 else np.empty(0, dtype=np.int64)

        def sample_negative_edges_local(pool, n_samples):
            """Uniform negatives over `pool`, with true edges rejected both ways."""
            if n_samples <= 0 or len(pool) < 2:
                return (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
            n_pool = len(pool)
            kept_s, kept_d, collected = [], [], 0
            for _ in range(8):
                need = n_samples - collected
                c_s = torch.from_numpy(pool[torch.randint(0, n_pool, (max(need * 2, 1024),)).numpy()])
                c_d = torch.from_numpy(pool[torch.randint(0, n_pool, (max(need * 2, 1024),)).numpy()])
                keep = (c_s != c_d)
                c_s, c_d = c_s[keep], c_d[keep]
                if len(c_s) == 0:
                    continue
                s_np = c_s.numpy().astype(np.int64)
                d_np = c_d.numpy().astype(np.int64)
                fwd = s_np * np.int64(n_nodes) + d_np
                rev = d_np * np.int64(n_nodes) + s_np
                ok = ~(np.isin(fwd, _edge_keys) | np.isin(rev, _edge_keys))
                if ok.any():
                    kept_s.append(s_np[ok][:need])
                    kept_d.append(d_np[ok][:need])
                    collected += len(kept_s[-1])
                if collected >= n_samples:
                    break
            if collected < n_samples:
                # Corner case: a near-complete subgraph leaves too few true non-edges.
                rem = n_samples - collected
                kept_s.append(pool[torch.randint(0, n_pool, (rem,)).numpy()].astype(np.int64))
                kept_d.append(pool[torch.randint(0, n_pool, (rem,)).numpy()].astype(np.int64))
            neg_s = np.concatenate(kept_s)[:n_samples]
            neg_d = np.concatenate(kept_d)[:n_samples]
            return (torch.from_numpy(neg_s.copy()), torch.from_numpy(neg_d.copy()))
        
        if is_pyg:
            if model_type == 'gat' or model_type == 'clusterscl':
                from torch_geometric.nn import GATConv
                class PyGEncoder(nn.Module):
                    def __init__(self, in_f, h, num_heads=8):
                        super().__init__()
                        self.c1 = GATConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                        self.c2 = GATConv(h, h, heads=1, concat=False, dropout=dropout)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.elu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
            elif model_type == 'gatv2':
                from torch_geometric.nn import GATv2Conv
                class PyGEncoder(nn.Module):
                    def __init__(self, in_f, h, num_heads=8):
                        super().__init__()
                        self.c1 = GATv2Conv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                        self.c2 = GATv2Conv(h, h, heads=1, concat=False, dropout=dropout)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.elu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
            elif model_type == 'arma':
                from torch_geometric.nn import ARMAConv
                class PyGEncoder(nn.Module):
                    def __init__(self, in_f, h):
                        super().__init__()
                        self.c1 = ARMAConv(in_f, h, dropout=dropout)
                        self.c2 = ARMAConv(h, h, dropout=dropout)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.relu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
            elif model_type == 'asap':
                from torch_geometric.nn import LEConv
                class PyGEncoder(nn.Module):
                    def __init__(self, in_f, h):
                        super().__init__()
                        self.c1 = LEConv(in_f, h)
                        self.c2 = LEConv(h, h)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.relu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
            elif model_type == 'transformer':
                from torch_geometric.nn import TransformerConv
                class PyGEncoder(nn.Module):
                    def __init__(self, in_f, h, num_heads=8):
                        super().__init__()
                        self.c1 = TransformerConv(in_f, h // num_heads, heads=num_heads, dropout=dropout)
                        self.c2 = TransformerConv(h, h, heads=1, concat=False, dropout=dropout)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.relu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
            else:
                from torch_geometric.nn import SAGEConv
                class PyGEncoder(nn.Module):
                    def __init__(self, in_f, h):
                        super().__init__()
                        self.c1 = SAGEConv(in_f, h)
                        self.c2 = SAGEConv(h, h)
                        self.dr = nn.Dropout(dropout)
                    def forward(self, x, edge_index):
                        x = F.relu(self.c1(x, edge_index))
                        x = self.dr(x)
                        return self.c2(x, edge_index)
            
            class PyGLinkPredictor(nn.Module):
                def __init__(self, h, dropout=0.1):
                    super().__init__()
                    # 4-way Siamese edge representation: [h_src, h_dst, |h_src - h_dst|, h_src * h_dst]
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
            
            encoder = PyGEncoder(feat_arr.shape[1], hidden_dim)
            predictor = PyGLinkPredictor(hidden_dim, dropout=dropout)
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(predictor.parameters()),
                lr=lr, weight_decay=5e-4
            )
            
            feat_t = torch.tensor(feat_arr, dtype=torch.float32)
            pyg_train_edge_index = torch.stack([src_l_t[train_edges_idx], dst_l_t[train_edges_idx]], dim=0)
            
            encoder.train()
            predictor.train()
            for _ in range(num_epochs):
                h = encoder(feat_t, pyg_train_edge_index)
                pos_src = src_l_t[train_edges_idx]
                pos_dst = dst_l_t[train_edges_idx]
                neg_src, neg_dst = sample_negative_edges_local(_member_idx, len(train_edges_idx))
                
                pos_scores = predictor(h[pos_src], h[pos_dst])
                neg_scores = predictor(h[neg_src], h[neg_dst])
                
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            encoder.eval()
            predictor.eval()
            with torch.no_grad():
                h = encoder(feat_t, pyg_train_edge_index)
                test_src = src_l_t[test_edges_idx]
                test_dst = dst_l_t[test_edges_idx]
                if len(test_src) > 0:
                    pos_scores = predictor(h[test_src], h[test_dst])
                    neg_src, neg_dst = sample_negative_edges_local(_member_idx, len(test_src))
                    neg_scores = predictor(h[neg_src], h[neg_dst])
                    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
                    y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                    from sklearn.metrics import roc_auc_score
                    comm_link_auc = float(roc_auc_score(y_true, y_scores))
                else:
                    comm_link_auc = 0.5
            link_train_time = time.time() - t_link_start
            
        else:
            train_g = dgl.graph((src_l_t[train_edges_idx], dst_l_t[train_edges_idx]), num_nodes=n_nodes)
            train_g = dgl.to_simple(train_g)
            train_g = dgl.add_self_loop(train_g)
            
            class LinkPredictor(nn.Module):
                def __init__(self, h, dropout=0.1):
                    super().__init__()
                    # 4-way Siamese edge representation: [h_src, h_dst, |h_src - h_dst|, h_src * h_dst]
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
            
            class GCNEncoder(nn.Module):
                def __init__(self, in_f, h):
                    super().__init__()
                    self.c1 = dglnn.SAGEConv(in_f, h, 'mean')
                    self.c2 = dglnn.SAGEConv(h,    h, 'mean')
                    self.dr = nn.Dropout(dropout)
                def forward(self, g, x):
                    x = torch.relu(self.c1(g, x)); x = self.dr(x)
                    x = self.c2(g, x)
                    return x
            
            encoder = GCNEncoder(feat_arr.shape[1], hidden_dim)
            predictor = LinkPredictor(hidden_dim, dropout=dropout)
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(predictor.parameters()),
                lr=lr, weight_decay=5e-4
            )
            
            feat_t = torch.tensor(feat_arr, dtype=torch.float32)
            
            encoder.train()
            predictor.train()
            for _ in range(num_epochs):
                pos_src = src_l_t[train_edges_idx]
                pos_dst = dst_l_t[train_edges_idx]
                neg_src, neg_dst = sample_negative_edges_local(_member_idx, len(train_edges_idx))
                
                h = encoder(train_g, feat_t)
                pos_scores = predictor(h[pos_src], h[pos_dst])
                neg_scores = predictor(h[neg_src], h[neg_dst])
                
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            encoder.eval()
            predictor.eval()
            with torch.no_grad():
                h = encoder(train_g, feat_t)
                test_src = src_l_t[test_edges_idx]
                test_dst = dst_l_t[test_edges_idx]
                
                if len(test_src) > 0:
                    pos_scores = predictor(h[test_src], h[test_dst])
                    neg_src, neg_dst = sample_negative_edges_local(_member_idx, len(test_src))
                    neg_scores = predictor(h[neg_src], h[neg_dst])
                    
                    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
                    y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                    from sklearn.metrics import roc_auc_score
                    comm_link_auc = float(roc_auc_score(y_true, y_scores))
                else:
                    comm_link_auc = 0.5
            link_train_time = time.time() - t_link_start

        diagnostic(f"link training/evaluation finished in {link_train_time:.1f}s")

    peak_mem = resource.getrusage(resource.RESOURCE_SELF if hasattr(resource, 'RESOURCE_SELF') else resource.RUSAGE_SELF).ru_maxrss * 1024.0

    n_nodes_bnd = int(bnd_t.sum())
    bucket = 'large' if n_nodes > 200 else ('medium' if n_nodes >= 50 else 'small')
    diagnostic(f"completed in {time.time() - t_start:.1f}s; peak_rss={peak_mem / 1e6:.1f} MB")

    return pd.DataFrame([{
        'community_id':  comm_id,
        'n_nodes':       n_nodes,
        'n_edges':       n_edges,
        'n_train':       int(train_m.sum()) if run_node else 0,
        'n_val':         int(val_m.sum()) if run_node else 0,
        'n_test':        n_test_node if run_node else len(test_edges_idx) if run_link else 0,
        'n_boundary':    n_nodes_bnd,
        'n_internal':    n_nodes - n_nodes_bnd,
        'comm_test_acc': comm_acc,
        'acc_gnn_head':  float(locals().get('acc_gnn_head', comm_acc)),
        'acc_mlp_head':  float(locals().get('acc_mlp_head', comm_acc)),
        'head_used':     str(locals().get('head_used', 'mlp')),
        'boundary_acc':  bnd_acc,
        'internal_acc':  int_acc,
        'comm_link_auc': comm_link_auc,
        'size_bucket':   bucket,
        'load_time_s':   t_load,
        'node_train_time_s': node_train_time,
        'link_train_time_s': link_train_time,
        'peak_mem_mb':   peak_mem / 1e6,
    }])

def run_phase3(spark, sc, datasets, algorithms, use_global_mapping,
               dataset_cfg, gcn_cfg, get_paths_fn, timing, results, **kwargs):
    """
    Train GNN models per community using Spark groupBy().applyInPandas().
    Hyperparams injected as constant DataFrame columns (avoids UDF closure issues).
    Results stored in results[(dataset, alg, model_type)] — isolated per model.
    """
    task_type = kwargs.get('task_type', 'node_classification')
    gnn_models = kwargs.get('models', ['sage'])
    diagnostics = kwargs.get('diagnostics', False)
    max_nodes_per_community = int(kwargs.get('max_nodes_per_community', 10000))
    max_edges_per_community = int(kwargs.get('max_edges_per_community', 50000))
    edge_sample_modulus = max(1, int(kwargs.get('edge_sample_modulus', 1)))
    mlp_epochs = max(1, int(kwargs.get('mlp_epochs', 5)))
    # Split communities larger than the per-unit cap into bounded blocks and stream
    # them via cogroup, instead of hash-sampling them down and packing each into a
    # single Spark row. Required for billion-edge graphs; harmless below the cap,
    # where every community yields exactly one block.
    block_oversized = bool(kwargs.get('block_oversized', True))
    node_patience = max(1, int(kwargs.get('node_patience', 10)))
    mlp_patience = max(1, int(kwargs.get('mlp_patience', 15)))
    
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    result_schema = _make_result_schema()

    for dataset in datasets:
        cfg = dataset_cfg.get(dataset)
        if cfg is None:
            for k, v in dataset_cfg.items():
                if str(k).lower() == str(dataset).lower() or str(k).lower().replace('_', '-') == str(dataset).lower().replace('_', '-'):
                    cfg = v
                    break
        if cfg is None:
            cfg = {'in_feats': 128, 'num_classes': 10}
        for alg in algorithms:
            p_alg = get_paths_fn(dataset, alg)

            print(f"  [Phase 3 setup] Loading Phase 2 Delta tables for {dataset}/{alg}...")
            cache_start = time.time()
            nodes_df = (spark.read.format('delta').load(p_alg['p2_nodes'])
                        .persist(StorageLevel.MEMORY_AND_DISK))
            edges_df = (spark.read.format('delta').load(p_alg['p2_edges'])
                        .persist(StorageLevel.MEMORY_AND_DISK))
            n_phase2_nodes = nodes_df.count()
            n_phase2_edges = edges_df.count()
            print(f"  [Phase 3 setup] Cached {n_phase2_nodes:,} node rows and {n_phase2_edges:,} edge rows in {time.time() - cache_start:.1f}s "
                  f"(nodes partitions={nodes_df.rdd.getNumPartitions()}, edges partitions={edges_df.rdd.getNumPartitions()}).")

            # Standard metadata + 128-float features array in DataFrame
            # Fixed 128-float arrays with maxRecordsPerBatch=1000 produce tiny 512KB Arrow batches
            # while avoiding all C++ PyArrow dataset disk scanning in Python workers.
            # is_member is absent from Phase 2 tables written before halo tracking
            # existed; default those to member so older checkpoints still run.
            if 'is_member' not in nodes_df.columns:
                nodes_df = nodes_df.withColumn('is_member', F.lit(True))
            training_df_base = nodes_df.select('id', 'label', 'features', 'split', 'community_id', 'is_boundary', 'is_member')

            # Compute model-independent community layout once per (dataset, alg).
            layout_start = time.time()
            comms_node_counts_pd = nodes_df.select('community_id').groupBy('community_id').count().toPandas()
            comms_node_counts_pd = comms_node_counts_pd.sort_values(by='count', ascending=False).reset_index(drop=True)
            num_comms = len(comms_node_counts_pd)
            largest_communities = ', '.join(
                f"{int(row.community_id)}:{int(row.count):,}"
                for row in comms_node_counts_pd.head(5).itertuples(index=False)
            )
            print(f"  [Phase 3 setup] Community layout computed in {time.time() - layout_start:.1f}s: "
                  f"{num_comms:,} communities; largest community_id:nodes = {largest_communities}.")

            # Stride must exceed any per-community block count so unit ids stay unique.
            BLOCK_STRIDE = int(max(1024, comms_node_counts_pd['count'].max() // max(1, max_nodes_per_community) + 2)) \
                if len(comms_node_counts_pd) else 1024
            default_para = sc.defaultParallelism
            if num_comms <= 2000:
                num_bins = num_comms
            else:
                num_bins = min(max(default_para * 4, 1000), num_comms)

            # Group communities into bins once, then reuse for each model type.
            comms_node_counts_pd['bin_id'] = [i % num_bins for i in range(num_comms)]
            # Keep a safety margin below the hard UDF limit. Hash sampling avoids
            # a per-community window sort over every Papers100M node row.
            node_sampling_target = max(1, int(max_nodes_per_community * 0.8))
            comms_node_counts_pd['_phase3_node_mod'] = np.maximum(
                1,
                np.ceil(comms_node_counts_pd['count'] / node_sampling_target).astype(np.int64)
            )
            # Number of bounded training units a community is split into. 1 leaves the
            # community intact; larger values split it rather than sampling it away.
            comms_node_counts_pd['_phase3_n_blocks'] = np.maximum(
                1,
                np.ceil(comms_node_counts_pd['count'] / max_nodes_per_community).astype(np.int64)
            )
            comms_node_counts = spark.createDataFrame(
                comms_node_counts_pd[['community_id', 'bin_id', '_phase3_node_mod',
                                      '_phase3_n_blocks']]
            )

            base_manifest_df = (comms_node_counts
                .withColumn('_num_classes', F.lit(int(cfg['num_classes'])))
                .withColumn('_hidden',      F.lit(int(gcn_cfg['hidden_dim'])))
                .withColumn('_epochs',      F.lit(int(gcn_cfg['num_epochs'])))
                .withColumn('_lr',          F.lit(float(gcn_cfg['lr'])))
                .withColumn('_dropout',     F.lit(float(gcn_cfg['dropout'])))
                .withColumn('_task_type',   F.lit(str(task_type))))

            for model_type in gnn_models:
                key   = (dataset, alg, model_type)
                
                # Checkpoint Path on S3 or local depending on local_data_dir
                s3_bucket = kwargs.get('s3_bucket', 'us-east-1-s3-gnn')
                experiment_name = kwargs.get('experiment_name', 'run-all')
                local_data_dir = kwargs.get('local_data_dir', None)
                
                if local_data_dir:
                    ckpt_dir = os.path.join(local_data_dir, "gnn-bench-checkpoint", "phase3", experiment_name)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"{dataset}_{alg}_{model_type}.parquet")
                else:
                    ckpt_path = f"s3://{s3_bucket}/gnn-bench-checkpoint/phase3/{experiment_name}/{dataset}_{alg}_{model_type}.parquet"

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
                            s3_key = f"gnn-bench-checkpoint/phase3/{experiment_name}/{dataset}_{alg}_{model_type}.parquet"
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
                    print(f"  PHASE 3 — GNN Training: {dataset} / {alg} / {model_type} (Loaded from Checkpoint)")
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
                    
                    timing[('phase3', dataset, alg, model_type)] = 0.0
                    print(f"  ✓ Loaded model accuracy: {weighted_comm_acc:.4f}, skipping training.")
                    continue

                t0    = time.time()
                print(f"\n{'='*60}")
                print(f"  PHASE 3 — GNN Training: {dataset} / {alg} / {model_type}")
                print(f"  tag={p_alg['tag']}")
                print(f"  task={task_type}; epochs={gcn_cfg['num_epochs']}; diagnostics={'on' if diagnostics else 'off'}")
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
                    comms_sorted = comms_node_counts_pd

                    pos_comms = comms_sorted[comms_sorted['community_id'] >= 0]
                    valid_comms = pos_comms[pos_comms['count'] <= 10_000] if len(pos_comms) > 0 else comms_sorted[comms_sorted['count'] <= 10_000]
                    if len(valid_comms) == 0:
                        target_comm_row = comms_sorted.iloc[-1]
                    else:
                        target_comm_row = valid_comms.iloc[0]

                    largest_comm_id = int(target_comm_row['community_id'])
                    comm_count = int(target_comm_row['count'])

                    if comm_count > 10_000:
                        print(f"  [Driver Warmstart] Community size ({comm_count:,} nodes) exceeds driver RAM safety limit — sub-sampling 5,000 nodes...")
                        large_comm_pdf = nodes_df.filter(F.col('community_id') == largest_comm_id).limit(5000).toPandas()
                    else:
                        large_comm_pdf = nodes_df.filter(F.col('community_id') == largest_comm_id).toPandas()

                    in_feats = len(large_comm_pdf['features'].iloc[0]) if len(large_comm_pdf) > 0 else int(cfg.get('in_feats', 128))
                    num_classes = int(cfg.get('num_classes', 10))
                    hidden_dim = int(gcn_cfg['hidden_dim'])

                    # Map nodes
                    all_nodes = large_comm_pdf['id'].values
                    n_nodes = len(all_nodes)
                    node_map = {int(n): i for i, n in enumerate(all_nodes)}
                    large_edges_pdf = edges_df.filter(F.col('community_id') == largest_comm_id).toPandas()
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
                        base_embeddings_bc = sc.broadcast(global_embeddings.tolist() if isinstance(global_embeddings, np.ndarray) else global_embeddings)
                        base_node_map_bc = sc.broadcast(node_map)
                    else:
                        base_embeddings_bc = None
                        base_node_map_bc = None
                    print(f"  ✓ Driver-side base model trained successfully on Comm {largest_comm_id} ({n_nodes:,} nodes).")
                except Exception as base_err:
                    print(f"  Warning: Skipped warm-start driver pre-training: {base_err}")

                # 2. Parallel grouped Pandas execution with community binning.
                spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
                spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

                p2_nodes_url = p_alg['p2_nodes']
                p2_edges_url = p_alg['p2_edges']

                print(f"  [Manifest Architecture] Reusing cached community manifest...")

                manifest_df = (base_manifest_df
                    .withColumn('_num_classes', F.lit(int(cfg['num_classes'])))
                    .withColumn('_hidden',      F.lit(int(gcn_cfg['hidden_dim'])))
                    .withColumn('_epochs',      F.lit(int(gcn_cfg['num_epochs'])))
                    .withColumn('_lr',          F.lit(float(gcn_cfg['lr'])))
                    .withColumn('_dropout',     F.lit(float(gcn_cfg['dropout'])))
                    .withColumn('_task_type',   F.lit(str(task_type)))
                    .withColumn('_model_type',  F.lit(str(model_type))))

                def _train_gnn_community_single_wrapper(pdf):
                    try:
                        import torch
                        torch.set_num_threads(1)
                        torch.set_num_interop_threads(1)
                    except Exception:
                        pass

                    return _train_gnn_community_single(
                        pdf,
                        comm_edges_pdf=None,
                        base_weights_bc=base_weights_bc,
                        base_embeddings_bc=base_embeddings_bc,
                        base_node_map_bc=base_node_map_bc
                    )

                if block_oversized:
                    # ── Recursive block partitioning (billion-edge path) ──────────
                    #
                    # The sampling path below bounds a community by keeping an induced
                    # subgraph on ~8k hash-sampled nodes. Because an edge survives only
                    # if BOTH endpoints do, edge retention falls as 1/mod^2: a 10M-node
                    # community keeps ~8k nodes and a millionth of its edges, so most
                    # test vertices are never evaluated and the model trains on a
                    # fragment. That is why Papers100M could not be processed here.
                    #
                    # Instead, split an oversized community into ceil(N/cap) blocks and
                    # train each independently. Every vertex is kept, every intra-block
                    # edge is kept, and only edges crossing a block boundary are lost —
                    # the same trade the method already makes at community boundaries,
                    # applied recursively. Each unit is bounded by construction, so no
                    # group can exceed a worker's budget.
                    block_plan = F.broadcast(
                        comms_node_counts.select('community_id', '_phase3_n_blocks'))

                    nodes_blocked = (nodes_df
                        .join(block_plan, on='community_id', how='inner')
                        .withColumn('_block',
                                    F.pmod(F.xxhash64('id'), F.col('_phase3_n_blocks')))
                        .withColumn('_unit',
                                    F.col('community_id') * F.lit(BLOCK_STRIDE) + F.col('_block')))

                    # Join on (id, community_id), not id alone. Under 1-hop expansion a
                    # vertex belongs to every community that borrows it as halo — on
                    # WikiCS the average vertex appears in 3.8 communities — so a map
                    # keyed only on id is not a function, and joining both endpoints
                    # through it replicates each edge across every community where both
                    # endpoints happen to appear. That inflates per-unit edge payloads
                    # far beyond the community's own edge count.
                    unit_map = nodes_blocked.select('id', 'community_id', '_unit')
                    edges_blocked = (edges_df
                        .join(unit_map.withColumnRenamed('id', 'src')
                                      .withColumnRenamed('_unit', '_src_unit'),
                              on=['src', 'community_id'], how='inner')
                        .join(unit_map.withColumnRenamed('id', 'dst')
                                      .withColumnRenamed('_unit', '_dst_unit'),
                              on=['dst', 'community_id'], how='inner')
                        .filter(F.col('_src_unit') == F.col('_dst_unit'))
                        .withColumn('_unit', F.col('_src_unit')))

                    # A dense block can still exceed the per-unit edge budget; sample
                    # edges directly rather than via endpoint survival, so the sample
                    # stays uniform over the block's edges.
                    if edge_sample_modulus > 1:
                        edges_blocked = edges_blocked.filter(
                            F.pmod(F.xxhash64('src', 'dst'), F.lit(edge_sample_modulus)) == F.lit(0))

                    retained_nodes = (nodes_blocked
                        .withColumn('label', F.coalesce(F.col('label'), F.lit(-1)))
                        .withColumn('split', F.coalesce(F.col('split'), F.lit('none')))
                        .withColumn('is_boundary', F.coalesce(F.col('is_boundary'), F.lit(False)))
                        .withColumn('is_member', F.coalesce(F.col('is_member'), F.lit(True)))
                        .drop('community_id').withColumnRenamed('_unit', 'community_id')
                        .select('id', 'label', 'features', 'split', 'community_id',
                                'is_boundary', 'is_member'))
                    bounded_edges = (edges_blocked
                        .drop('community_id')
                        .withColumnRenamed('_unit', 'community_id')
                        .select('src', 'dst', 'community_id'))

                    # Cap edges per unit in Spark, upstream of the shuffle.
                    #
                    # The UDF already bounds each unit to max_edges_per_community, but it
                    # does so *after* Arrow has shipped the rows: on Reddit the largest
                    # unit received 1,692,338 edges and discarded 98% of them to train on
                    # 30,000. The cost was data movement, not compute, which is why unit
                    # skew reached 10.13x while every unit trained on the same bounded
                    # subgraph. Expressing the bound as a relational filter moves that
                    # work into Spark and cuts shuffle volume without changing what the
                    # UDF would have kept. Same deterministic-hash pattern as the node
                    # sampler: a pure filter, no window sort.
                    _unit_counts = bounded_edges.groupBy('community_id').count()
                    _edge_plan = F.broadcast(
                        _unit_counts
                        .withColumn('_edge_mod',
                                    F.greatest(
                                        F.lit(1),
                                        F.ceil(F.col('count') /
                                               F.lit(float(max_edges_per_community)))
                                    ).cast('long'))
                        .select('community_id', '_edge_mod'))
                    bounded_edges = (bounded_edges
                        .join(_edge_plan, on='community_id', how='inner')
                        .filter(F.pmod(F.xxhash64('src', 'dst'),
                                       F.col('_edge_mod')) == F.lit(0))
                        .select('src', 'dst', 'community_id'))

                    unit_edges = (bounded_edges.groupBy('community_id').count()
                                  .agg(F.max('count').alias('mx'),
                                       F.avg('count').alias('av'),
                                       F.min('count').alias('mn')).collect()[0])
                    _mx, _av = int(unit_edges['mx'] or 0), float(unit_edges['av'] or 1.0)
                    print(f"  [Phase 3 balance] edges/unit min={int(unit_edges['mn'] or 0):,} "
                          f"avg={_av:,.0f} max={_mx:,} — skew {_mx / max(_av, 1.0):.2f}x "
                                                    f"wall-clock follows the slowest unit.")

                    n_units = int(comms_node_counts_pd['_phase3_n_blocks'].sum())
                    print(f"  [Phase 3 blocks] {num_comms:,} communities -> {n_units:,} bounded "
                          f"training units (cap {max_nodes_per_community:,} nodes/unit); "
                          f"all vertices retained, only cross-block edges dropped.")
                    num_bins = min(max(sc.defaultParallelism * 4, 1000), max(n_units, 1))

                # Legacy bounded-sampling path, retained for graphs that fit it.
                # Apply the same deterministic hash predicate to node IDs and both
                # endpoints. This creates an induced sampled subgraph without window
                # sorts or two large edge/node joins. The 80% node target leaves room
                # for normal sampling variation; the UDF retains the final hard cap.
                if not block_oversized:
                    sampling_plan = F.broadcast(comms_node_counts.select('community_id', '_phase3_node_mod'))
                    retained_nodes = (nodes_df
                        .join(sampling_plan, on='community_id', how='inner')
                        .filter(F.pmod(F.xxhash64('id'), F.col('_phase3_node_mod')) == F.lit(0)))

                    bounded_edges = (edges_df
                        .join(sampling_plan, on='community_id', how='inner')
                        .filter(F.pmod(F.xxhash64('src'), F.col('_phase3_node_mod')) == F.lit(0))
                        .filter(F.pmod(F.xxhash64('dst'), F.col('_phase3_node_mod')) == F.lit(0))
                        .filter(F.pmod(F.xxhash64('src', 'dst'), F.lit(edge_sample_modulus)) == F.lit(0)))

                    print(f"  [Phase 3 safety] Hash-sampling up to ~{node_sampling_target:,} nodes/community "
                          f"(hard cap {max_nodes_per_community:,}); retaining ~1/{edge_sample_modulus} eligible edges "
                          f"before aggregation (UDF hard edge cap {max_edges_per_community:,}).")

                community_bundles = None
                if not block_oversized:
                  edges_agg = (bounded_edges
                    .groupBy('community_id')
                    .agg(
                        F.collect_list('src').alias('_src_list'),
                        F.collect_list('dst').alias('_dst_list')
                    ))

                  nodes_prepared = (retained_nodes
                    .withColumn('label', F.coalesce(F.col('label'), F.lit(-1)))
                    .withColumn('split', F.coalesce(F.col('split'), F.lit('none')))
                    .withColumn('is_boundary', F.coalesce(F.col('is_boundary'), F.lit(False))))

                  nodes_agg = (nodes_prepared
                    .groupBy('community_id')
                    .agg(
                        F.collect_list('id').alias('_id_list'),
                        F.collect_list('label').alias('_label_list'),
                        F.collect_list('features').alias('_features_list'),
                        F.collect_list('split').alias('_split_list'),
                        F.collect_list('is_boundary').alias('_is_boundary_list'),
                        F.collect_list('is_member').alias('_is_member_list')
                    ))

                  community_bundles = (nodes_agg
                    .join(edges_agg, on='community_id', how='left')
                    .withColumn('_num_classes', F.lit(int(cfg['num_classes'])))
                    .withColumn('_hidden',      F.lit(int(gcn_cfg['hidden_dim'])))
                    .withColumn('_epochs',      F.lit(int(gcn_cfg['num_epochs'])))
                    .withColumn('_lr',          F.lit(float(gcn_cfg['lr'])))
                    .withColumn('_dropout',     F.lit(float(gcn_cfg['dropout'])))
                    .withColumn('_task_type',   F.lit(str(task_type)))
                    .withColumn('_model_type',  F.lit(str(model_type)))
                    .withColumn('_max_nodes',   F.lit(int(max_nodes_per_community)))
                    .withColumn('_max_edges',   F.lit(int(max_edges_per_community)))
                    .withColumn('_mlp_epochs',  F.lit(int(mlp_epochs)))
                    .withColumn('_mlp_patience', F.lit(int(mlp_patience)))
                    .withColumn('_node_patience', F.lit(int(node_patience)))
                    .withColumn('_phase3_diagnostics', F.lit(bool(diagnostics))))

                sc.setJobDescription(f'phase3_{dataset}_{alg}_{model_type}')
                if block_oversized:
                    # cogroup streams two Arrow record batches per unit instead of
                    # packing a community into one Spark row. collect_list caps a
                    # community at the JVM's 2 GB array limit — about 4M nodes of
                    # 128-dim float32 — which a Papers100M community exceeds outright.
                    def _unit_wrapper(nodes_pdf, edges_pdf):
                        try:
                            import torch
                            torch.set_num_threads(1)
                            torch.set_num_interop_threads(1)
                        except Exception:
                            pass
                        return _train_gnn_community_single(
                            nodes_pdf,
                            comm_edges_pdf=edges_pdf,
                            base_weights_bc=base_weights_bc,
                            base_embeddings_bc=base_embeddings_bc,
                            base_node_map_bc=base_node_map_bc)

                    nodes_for_units = (retained_nodes
                        .withColumn('_num_classes', F.lit(int(cfg['num_classes'])))
                        .withColumn('_hidden',      F.lit(int(gcn_cfg['hidden_dim'])))
                        .withColumn('_epochs',      F.lit(int(gcn_cfg['num_epochs'])))
                        .withColumn('_lr',          F.lit(float(gcn_cfg['lr'])))
                        .withColumn('_dropout',     F.lit(float(gcn_cfg['dropout'])))
                        .withColumn('_task_type',   F.lit(str(task_type)))
                        .withColumn('_model_type',  F.lit(str(model_type)))
                        .withColumn('_max_nodes',   F.lit(int(max_nodes_per_community)))
                        .withColumn('_max_edges',   F.lit(int(max_edges_per_community)))
                        .withColumn('_mlp_epochs',  F.lit(int(mlp_epochs)))
                        .withColumn('_mlp_patience', F.lit(int(mlp_patience)))
                    .withColumn('_mlp_patience', F.lit(int(mlp_patience)))
                        .withColumn('_node_patience', F.lit(int(node_patience)))
                    .withColumn('_node_patience', F.lit(int(node_patience)))
                        .withColumn('_phase3_diagnostics', F.lit(bool(diagnostics)))
                        .repartition(num_bins, 'community_id'))

                    comm_results = (nodes_for_units.groupBy('community_id')
                                    .cogroup(bounded_edges.repartition(num_bins, 'community_id')
                                                          .groupBy('community_id'))
                                    .applyInPandas(_unit_wrapper, schema=result_schema)
                                    .persist(StorageLevel.MEMORY_AND_DISK))
                else:
                    comm_results = (community_bundles
                                    .repartition(num_bins, 'community_id')
                                    .groupBy('community_id')
                                    .applyInPandas(_train_gnn_community_single_wrapper, schema=result_schema)
                                    .persist(StorageLevel.MEMORY_AND_DISK))

                # Keep the full per-community frame in Spark; only collect the compact summaries
                # needed for reporting and checkpoints.
                summary_row = comm_results.agg(
                    F.count('*').alias('n_communities'),
                    F.sum('n_nodes').alias('total_nodes'),
                    F.sum('n_edges').alias('total_edges'),
                    F.sum('n_train').alias('total_train_nodes'),
                    F.sum('n_val').alias('total_val_nodes'),
                    F.sum('n_test').alias('total_test_nodes'),
                    F.sum('n_boundary').alias('total_boundary_nodes'),
                    F.sum('n_internal').alias('total_internal_nodes'),
                    F.sum(F.col('comm_test_acc') * F.col('n_test')).alias('weighted_comm_acc_num'),
                    F.sum(F.col('comm_link_auc') * F.col('n_edges')).alias('weighted_comm_link_auc_num'),
                    F.avg('comm_test_acc').alias('mean_comm_acc'),
                    F.avg('acc_gnn_head').alias('mean_acc_gnn_head'),
                    F.avg('acc_mlp_head').alias('mean_acc_mlp_head'),
                    F.sum(F.when(F.col('head_used') == 'gnn', 1).otherwise(0)).alias('n_head_gnn'),
                    F.sum(F.when(F.col('head_used') == 'mlp', 1).otherwise(0)).alias('n_head_mlp'),
                    F.avg('boundary_acc').alias('mean_boundary_acc'),
                    F.avg('internal_acc').alias('mean_internal_acc'),
                    F.avg('comm_link_auc').alias('mean_comm_link_auc'),
                    F.avg('load_time_s').alias('avg_load_time_s'),
                    F.avg('node_train_time_s').alias('avg_node_train_time_s'),
                    F.avg('link_train_time_s').alias('avg_link_train_time_s'),
                    F.avg('peak_mem_mb').alias('avg_peak_mem_mb'),
                    F.max('peak_mem_mb').alias('max_peak_mem_mb'),
                ).toPandas()

                bucket_stats_pd = (comm_results
                    .groupBy('size_bucket')
                    .agg(
                        F.count('*').alias('n_communities'),
                        F.sum('n_test').alias('total_test_nodes'),
                        F.sum('n_edges').alias('total_edges'),
                        F.avg('comm_test_acc').alias('mean_comm_acc'),
                    F.avg('acc_gnn_head').alias('mean_acc_gnn_head'),
                    F.avg('acc_mlp_head').alias('mean_acc_mlp_head'),
                    F.sum(F.when(F.col('head_used') == 'gnn', 1).otherwise(0)).alias('n_head_gnn'),
                    F.sum(F.when(F.col('head_used') == 'mlp', 1).otherwise(0)).alias('n_head_mlp'),
                        F.avg('boundary_acc').alias('mean_boundary_acc'),
                        F.avg('internal_acc').alias('mean_internal_acc'),
                        F.avg('comm_link_auc').alias('mean_comm_link_auc'),
                        F.avg('load_time_s').alias('avg_load_time_s'),
                        F.avg('node_train_time_s').alias('avg_node_train_time_s'),
                        F.avg('link_train_time_s').alias('avg_link_train_time_s'),
                        F.avg('peak_mem_mb').alias('avg_peak_mem_mb'),
                    )
                    .orderBy('size_bucket')
                    .toPandas())

                weighted_comm_acc = (
                    float(summary_row['weighted_comm_acc_num'].iloc[0]) / float(summary_row['total_test_nodes'].iloc[0])
                    if float(summary_row['total_test_nodes'].iloc[0]) > 0 else 0.0
                )
                weighted_comm_link_auc = (
                    float(summary_row['weighted_comm_link_auc_num'].iloc[0]) / float(summary_row['total_edges'].iloc[0])
                    if float(summary_row['total_edges'].iloc[0]) > 0 else 0.5
                )

                summary_pdf = pd.DataFrame([{
                    'community_id': -1,
                    'n_nodes': float(summary_row['total_nodes'].iloc[0]),
                    'n_edges': float(summary_row['total_edges'].iloc[0]),
                    'n_train': float(summary_row['total_train_nodes'].iloc[0]),
                    'n_val': float(summary_row['total_val_nodes'].iloc[0]),
                    'n_test': float(summary_row['total_test_nodes'].iloc[0]),
                    'n_boundary': float(summary_row['total_boundary_nodes'].iloc[0]),
                    'n_internal': float(summary_row['total_internal_nodes'].iloc[0]),
                    'comm_test_acc': float(summary_row['mean_comm_acc'].iloc[0]),
                    'boundary_acc': float(summary_row['mean_boundary_acc'].iloc[0]),
                    'internal_acc': float(summary_row['mean_internal_acc'].iloc[0]),
                    'comm_link_auc': float(summary_row['mean_comm_link_auc'].iloc[0]),
                    'size_bucket': 'summary',
                    'load_time_s': float(summary_row['avg_load_time_s'].iloc[0]),
                    'node_train_time_s': float(summary_row['avg_node_train_time_s'].iloc[0]),
                    'link_train_time_s': float(summary_row['avg_link_train_time_s'].iloc[0]),
                    'peak_mem_mb': float(summary_row['avg_peak_mem_mb'].iloc[0]),
                }])
                sc.setJobDescription('')

                elapsed = time.time() - t0
                timing[('phase3', dataset, alg, model_type)] = elapsed

                # Store with attrs attached
                results[key] = summary_pdf.copy()
                results[key].attrs['weighted_comm_acc']  = weighted_comm_acc
                results[key].attrs['weighted_comm_link_auc'] = weighted_comm_link_auc
                results[key].attrs['mean_comm_acc'] = float(summary_row['mean_comm_acc'].iloc[0])
                results[key].attrs['mean_boundary_acc'] = float(summary_row['mean_boundary_acc'].iloc[0])
                results[key].attrs['mean_internal_acc'] = float(summary_row['mean_internal_acc'].iloc[0])
                results[key].attrs['mean_comm_link_auc'] = float(summary_row['mean_comm_link_auc'].iloc[0])
                results[key].attrs['bucket_stats'] = bucket_stats_pd.to_dict('records')
                results[key].attrs['summary_only'] = True
                results[key].attrs['wall_time_s'] = elapsed
                results[key].attrs['dataset']     = dataset
                results[key].attrs['alg']         = alg
                results[key].attrs['model_type']  = model_type
                results[key].attrs['n_communities'] = int(summary_row['n_communities'].iloc[0])
                # Edges actually trained on, so a run can prove nothing was silently
                # capped rather than assuming it.
                _ng = int(summary_row['n_head_gnn'].iloc[0] or 0)
                _nm = int(summary_row['n_head_mlp'].iloc[0] or 0)
                results[key].attrs['n_head_gnn'] = _ng
                results[key].attrs['n_head_mlp'] = _nm
                results[key].attrs['head_used'] = (
                    f"gnn {_ng}/{_ng + _nm}" if (_ng + _nm) else 'n/a')
                results[key].attrs['acc_gnn_head'] = float(summary_row['mean_acc_gnn_head'].iloc[0])
                results[key].attrs['acc_mlp_head'] = float(summary_row['mean_acc_mlp_head'].iloc[0])
                results[key].attrs['total_edges'] = int(summary_row['total_edges'].iloc[0])
                results[key].attrs['total_nodes'] = int(summary_row['total_nodes'].iloc[0])

                print(f"  ✓ Mean comm acc = {results[key].attrs['mean_comm_acc']:.4f}")
                print(f"  ✓ Weighted comm acc = {weighted_comm_acc:.4f}")
                print(f"  ✓ Mean comm link AUC = {results[key].attrs['mean_comm_link_auc']:.4f}")
                print(f"  ✓ Weighted comm link AUC = {weighted_comm_link_auc:.4f}")
                print(f"  ✓ Wall time: {elapsed:.1f}s")
                print(f"    - Avg Load time: {summary_pdf['load_time_s'].iloc[0]:.2f}s")
                print(f"    - Avg Node Train: {summary_pdf['node_train_time_s'].iloc[0]:.2f}s")
                print(f"    - Avg Link Train: {summary_pdf['link_train_time_s'].iloc[0]:.2f}s")

                # Save checkpoint
                try:
                    s3_result_path = f"s3://{s3_bucket}/gnn-bench-results/phase3/{experiment_name}/{dataset}_{alg}_{model_type}"
                    comm_results.write.mode('overwrite').parquet(s3_result_path)
                    results[key].attrs['spark_result_path'] = s3_result_path
                    print(f"    ✓ Saved Spark result frame: {s3_result_path}")

                    if local_data_dir:
                        import json
                        ckpt_dir = os.path.join(local_data_dir, "gnn-bench-checkpoint", "phase3", experiment_name)
                        os.makedirs(ckpt_dir, exist_ok=True)
                        ckpt_path = os.path.join(ckpt_dir, f"{dataset}_{alg}_{model_type}.json")
                        payload = {
                            'summary': results[key].to_dict(orient='records'),
                            'bucket_stats': results[key].attrs.get('bucket_stats', []),
                            'weighted_comm_acc': weighted_comm_acc,
                            'weighted_comm_link_auc': weighted_comm_link_auc,
                            'spark_result_path': s3_result_path,
                        }
                        with open(ckpt_path, 'w') as f:
                            json.dump(payload, f, indent=2)
                        print(f"    ✓ Saved summary checkpoint locally: {ckpt_path}")
                except Exception as ex:
                    print(f"    ⚠️ Failed to save checkpoint: {ex}")

            nodes_df.unpersist()
            edges_df.unpersist()
            try:
                bounded_edges.unpersist()
            except NameError:
                pass
