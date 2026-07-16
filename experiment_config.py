# ══════════════════════════════════════════════════════════════════════════════
#  experiment_config.py
#  ► Edit ONLY this file to configure your experiment.
#  ► Upload to S3 before running the notebook:
#      python upload_to_s3.py
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME   = 'ogbn-arxiv'          # labels ALL S3 outputs; change per experiment run

# ── Datasets ───────────────────────────────────────────────────────────────────
# To add ogbn-arxiv: set RUN_PHASE0=True first to ingest it.
DATASETS_TO_RUN = ['ogbn-arxiv']

# ── GNN Models to Run ─────────────────────────────────────────────────────────
# Supported choices: 'sage', 'gat', 'gatv2', 'transformer', 'clusterscl'
GNN_MODELS = ['sage', 'gatv2']

# ── Phase 0: Delta Lake Ingestion ─────────────────────────────────────────────
# True  = re-download OGB dataset and overwrite Delta tables.
#         REQUIRED when using a dataset for the first time.
# False = skip (Delta tables already exist).
RUN_PHASE0        = True
FORCE_REINGEST    = True   # Set to True to force overwrite even if tables already exist
FORCE_RERUN       = True   # Set to True to ignore all S3 checkpoints and rerun the pipeline
USE_OGB_SPLITS    = True    # True = OGB official splits | False = stratified 60/20/20
RANDOM_SEED       = 42
N_BASELINE_RUNS   = 1          # number of runs per baseline for mean ± std

# ── Phase 1: Community Detection ──────────────────────────────────────────────
# All listed algorithms run independently. Results are NEVER mixed.
#   'lpa'     = distributed Spark (fast, lower community quality)
#   'louvain' = driver/igraph   (moderate quality, pulls graph to driver RAM)
#   'igraph_lpa' = driver/igraph   (LPA using igraph)
RUN_PHASE1         = True             # Set to False to skip community detection phase
ALGORITHMS_TO_RUN  = ['lpa', 'louvain']
LPA_MAX_ITER       = 5
RESOLUTION         = 1.0              # louvain / leiden resolution parameter
MIN_COMMUNITY_SIZE = 100              # communities smaller than this are excluded

# ── Phase 2 / 3: Partitioning & GNN Training ──────────────────────────────────
# USE_GLOBAL_MAPPING = True  (RECOMMENDED):
#   Global OGB masks used for Phase 3 AND Phase 4 → directly comparable.
#   Required for valid global accuracy comparison (Pipelines.txt §5).
# USE_GLOBAL_MAPPING = False (ablation only):
#   Per-community 70/15/15 random split inside UDF → NOT globally comparable.
RUN_PHASE2         = True             # Set to False to skip subgraph generation phase
RUN_PHASE3         = True             # Set to False to skip parallel GNN UDF training phase
USE_GLOBAL_MAPPING = True

GCN_HIDDEN_DIM    = 256
GCN_NUM_EPOCHS    = 10
GCN_LR            = 0.01
GCN_DROPOUT       = 0.5
RUN_PHASE3B       = True              # Phase 3b: CaaN Global Graph GNN Training

# ── New Advanced Features ──────────────────────────────────────────────────────
# Tiny community handling: 'drop' (ignore them), 'misc' (group them all into community_id = -1)
TINY_COMM_HANDLING  = 'misc'

# 1-hop boundary expansion: If True, include 1-hop external neighbors for boundary nodes.
# NOTE: increases data size but improves boundary accuracy significantly.
EXPAND_BOUNDARY_NODES = True

# Task Type: 'node_classification' or 'link_prediction'
TASK_TYPE = 'both'

# ── Phase 4: Full-Graph Baseline ──────────────────────────────────────────────
# Runs ONCE per dataset (not per algorithm). Uses SAME masks as Phase 3.
RUN_PHASE4        = True       # Set to False to skip OOM-prone driver-bound baselines
BASELINE_EPOCHS   = 15          # reduced from 50 epochs to speed up CPU full-graph training
BASELINE_BATCH    = 1024
BASELINE_FANOUT   = [15, 10]
BASELINE_LR       = GCN_LR
RUN_PHASE4B       = True       # DistDGL Baseline Simulation
RUN_PHASE4C       = True       # ARMA Baseline
RUN_PHASE4D       = True       # ASAP Baseline
RUN_PHASE4E       = True       # GAT Baseline
RUN_PHASE4F       = True       # Graph Transformer Baseline
RUN_PHASE4G       = True       # ClusterSCL Baseline
RUN_PHASE4H       = True       # GATv2 Baseline

# ── Infrastructure ─────────────────────────────────────────────────────────────
S3_BUCKET         = 'us-east-1-s3-gnn'
S3_CODE_PREFIX    = 'pipeline'   # where upload_to_s3.py puts .py files

# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED CONFIG — do not edit below this line
# ══════════════════════════════════════════════════════════════════════════════

# Dataset-specific architecture configs (auto-applied per dataset)
DATASET_CFG = {
    'ogbn-products':   {'in_feats': 100, 'num_classes': 47},
    'ogbn-arxiv':      {'in_feats': 128, 'num_classes': 40},
    'ogbn-mag':        {'in_feats': 128, 'num_classes': 349},
    'ogbn-papers100M': {'in_feats': 128, 'num_classes': 172},
    'ogbn-proteins':   {'in_feats': 8, 'num_classes': 112},
    'reddit':          {'in_feats': 602, 'num_classes': 41},
    'flickr':          {'in_feats': 500, 'num_classes': 7},
    'wikics':          {'in_feats': 300, 'num_classes': 10},
    'coauthor-cs':     {'in_feats': 6805, 'num_classes': 15},
    'coauthor-physics':{'in_feats': 8415, 'num_classes': 5},
    'deezereurope':    {'in_feats': 128, 'num_classes': 2},
}

# Bundled GCN config dict (passed to pipeline functions)
GCN_CFG = {
    'hidden_dim': GCN_HIDDEN_DIM,
    'num_epochs': GCN_NUM_EPOCHS,
    'lr':         GCN_LR,
    'dropout':    GCN_DROPOUT,
}

# Bundled baseline config dict
BASELINE_CFG = {
    'epochs': BASELINE_EPOCHS,
    'batch':  BASELINE_BATCH,
    'fanout': BASELINE_FANOUT,
    'lr':     BASELINE_LR,
    'hidden_dim': GCN_HIDDEN_DIM,
    'dropout':    GCN_DROPOUT,
}


def get_paths(dataset, alg=None):
    """
    Returns all S3/Delta paths for a (dataset, algorithm) pair.

    Isolation contract:
      delta-data/{dataset}/nodes|edges|masks/
        → shared, written once by Phase 0
      delta-data/{dataset}/communities/{alg}/
        → algorithm-specific; shared across experiment runs with same algorithm
      delta-data/{dataset}/phase2_nodes/{EXPERIMENT_NAME}_{dataset}_{alg}/
      delta-data/{dataset}/phase2_edges/{EXPERIMENT_NAME}_{dataset}_{alg}/
        → fully isolated per EXPERIMENT_NAME + dataset + algorithm
      gnn-bench-out/{EXPERIMENT_NAME}_{dataset}_{alg}_phase3.xlsx
      gnn-bench-out/{EXPERIMENT_NAME}_{dataset}_phase4.xlsx
        → isolated per tag; never overlap between algorithms or datasets
    """
    root = f's3://{S3_BUCKET}/delta-data/{dataset}'
    p = {
        'root':            root,
        'nodes':           f'{root}/nodes/',
        'edges':           f'{root}/edges/',
        'masks':           f'{root}/masks/',
        'original_nodes':  f'{root}/original_nodes/',
        'original_edges':  f'{root}/original_edges/',
        'checkpoints':     f's3://{S3_BUCKET}/checkpoints/{dataset}/',
        'phase4_xlsx':     (f's3://{S3_BUCKET}/gnn-bench-out/'
                            f'{EXPERIMENT_NAME}_{dataset}_phase4.xlsx'),
    }
    if alg:
        tag = f'{EXPERIMENT_NAME}_{dataset}_{alg}'
        p.update({
            'communities': f'{root}/communities/{alg}/',
            'p2_nodes':    f'{root}/phase2_nodes/{tag}/',
            'p2_edges':    f'{root}/phase2_edges/{tag}/',
            'phase3_xlsx': (f's3://{S3_BUCKET}/gnn-bench-out/'
                            f'{tag}_phase3.xlsx'),
            'models':      f's3://{S3_BUCKET}/gnn-bench-out/models/{tag}/',
            'tag':         tag,
        })
    return p


# ── Results isolation ──────────────────────────────────────────────────────────
# Keyed by (dataset, algorithm) or dataset alone.
# DO NOT share or compare values across different keys without explicit intent.
phase1_results = {}   # (dataset, alg)  → {n_comms, runtime_s, nmi}
phase2_results = {}   # (dataset, alg)  → {n_valid_comms, n_nodes_kept, n_boundary, ...}
phase3_results = {}   # (dataset, alg)  → pd.DataFrame of per-community rows
phase4_results = {}   # dataset         → {test_acc, train_time_s, peak_mem_gb}
phase4b_results = {}  # dataset         → {test_acc, train_time_s, peak_mem_gb}
phase4c_results = {}  # dataset         → {test_acc, train_time_s, peak_mem_gb}
phase4d_results = {}  # dataset         → {test_acc, train_time_s, peak_mem_gb}
phase4e_results = {}  # dataset         → {test_acc, train_time_s, peak_mem_gb}
phase4f_results = {}  # dataset         → {test_acc, train_time_s, peak_mem_gb}
phase4g_results = {}  # dataset         → {test_acc, train_time_s, peak_mem_gb}
phase4h_results = {}  # dataset         → {test_acc, train_time_s, peak_mem_gb}

# Timing registry — every wall-clock duration stored here
# Keys: ('phase0', dataset)  |  ('phase1', dataset, alg)  |  ('phase2', dataset, alg)
#       ('phase3', dataset, alg)  |  ('phase4', dataset)  |  ('phase4b', dataset)
#       ('phase4c', dataset)      |  ('phase4d', dataset) |  ('phase4e', dataset)
#       ('phase4f', dataset)      |  ('phase4g', dataset)
#       ('phase4h', dataset)
timing = {}
