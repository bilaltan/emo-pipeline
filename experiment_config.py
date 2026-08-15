# ══════════════════════════════════════════════════════════════════════════════
#  experiment_config.py
#  ► Edit ONLY this file to configure your experiment.
#  ► Upload to S3 before running the notebook:
#      python upload_to_s3.py
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME   = 'ogbn-papers100M'          # labels ALL S3 outputs; change per experiment run

# ── Datasets ───────────────────────────────────────────────────────────────────
# Available options:
# Standard: 'WikiCS', 'Coauthor-Physics', 'Coauthor-CS', 'DeezerEurope', 'Foursquare'
# 100M-scale: 'reddit', 'ogbn-products'
# 1B-scale:   'ogbn-papers100M'
DATASETS_TO_RUN = ['WikiCS', 'ogbn-products', 'reddit', 'ogbn-papers100M']

# ── GNN Models to Run ─────────────────────────────────────────────────────────
# Supported choices: 'sage', 'gat', 'gatv2', 'transformer', 'clusterscl', 'arma', 'asap'
GNN_MODELS = ['sage']

# ── Phase 0: Delta Lake Ingestion ─────────────────────────────────────────────
# True  = re-download OGB dataset and overwrite Delta tables.
#         REQUIRED when using a dataset for the first time.
# False = skip (Delta tables already exist).
RUN_PHASE0        = True
FORCE_REINGEST    = False   # Set to False to use existing S3 Delta tables
FORCE_RERUN       = True    # Force clean, fresh training and evaluation for all phases
USE_OGB_SPLITS    = True    # True = OGB official splits | False = stratified 60/20/20
RANDOM_SEED       = 42
N_BASELINE_RUNS   = 1          # number of runs per baseline for mean ± std


# ── Phase 1: Community Detection ──────────────────────────────────────────────
# All listed algorithms run independently. Results are NEVER mixed.
#   'lpa'     = distributed Spark (fast, lower community quality)
#   'louvain' = driver/igraph   (moderate quality, pulls graph to driver RAM)
#   'igraph_lpa' = driver/igraph   (LPA using igraph)
RUN_PHASE1         = True            # Reuse the completed LPA checkpoint for the Phase 3 retry
ALGORITHMS_TO_RUN  = ['lpa', 'louvain']  # subset of ['lpa', 'louvain', 'igraph_lpa']
LPA_MAX_ITER       = 6
RESOLUTION         = 1.0              # louvain / leiden resolution parameter
MIN_COMMUNITY_SIZE = 1000             # communities smaller than this are excluded

# ── Phase 2 / 3: Partitioning & GNN Training ──────────────────────────────────
# USE_GLOBAL_MAPPING = True  (RECOMMENDED):
#   Global OGB masks used for Phase 3 AND Phase 4 → directly comparable.
#   Required for valid global accuracy comparison (Pipelines.txt §5).
# USE_GLOBAL_MAPPING = False (ablation only):
#   Per-community 70/15/15 random split inside UDF → NOT globally comparable.
RUN_PHASE2         = True            # Reuse the completed bounded Phase 2 subgraphs
# Phase 2.5 writes a lossless, shard-addressable node/adjacency graph store.
# Enable this once to prepare the direct-parquet Phase 3 redesign; it does not
# change the current sampled-community Phase 3 path yet.
RUN_PHASE25        = False            # Reuse the completed lossless graph-store checkpoint
PHASE25_NUM_SHARDS = 512
RUN_PHASE26        = False            # Correct independent source-seed layout is now materialized
PHASE26_SEED_BLOCKS = 16
PHASE26_NEIGHBOR_BLOCKS = 4
# Reunites the four edge partitions logically per source seed unit and records
# exact halo/workload statistics. It does not train or materialize adjacency.
RUN_PHASE27        = False            # Correct full seed-unit audit is now materialized
PHASE27_WORKING_SET_HEADROOM = 4.0
# Initial direct-Delta validation: trains bounded, complete source-seed units
# without collect_list. This is intentionally a local-block validation, not a
# synchronized full-graph model. Start with a small deterministic subset.
RUN_PHASE35        = False            # Completed direct-I/O validation; preserve output while shared training is implemented
PHASE35_MAX_UNITS  = 8
# Synchronous FedAvg proof on bounded direct blocks. The driver averages only
# this validation subset's compact model vectors; it is not the final 8,192-
# unit distributed parameter-service implementation.
RUN_PHASE36        = False            # Timing A/B complete; do not rerun before selecting a faster training architecture
PHASE36_TRAIN_UNITS = 1024            # Scale validated FedAdam to the 118M-edge direct-training workload
PHASE36_HOLDOUT_UNITS = 64            # Fixed deterministic evaluation population for fair scale comparisons
PHASE36_ROUNDS     = 30               # Match the fixed-holdout 256-unit baseline before comparing scale
PHASE36_LOCAL_EPOCHS = 2
PHASE36_AGGREGATION_PARTITIONS = 64   # One weighted model vector returned per partition, not per unit
PHASE36_SERVER_OPTIMIZER = 'fedadam'  # Compare against the 256-unit FedAvg fixed-holdout score of 0.2748
PHASE36_SERVER_LR = 0.003
PHASE36_SERVER_BETA1 = 0.9
PHASE36_SERVER_BETA2 = 0.99
PHASE36_SERVER_EPSILON = 1e-8
PHASE36_USE_WORKSET_CHECKPOINT = True   # Delta checkpoint selected train/holdout unit records for fast reruns
PHASE36_REPARTITION_BY_UNIT = True      # Pre-cluster by (src_shard, seed_block) before grouped UDF rounds
# Phase 3.7 is a CPU-oriented SIGN-style preprocessing path. It computes each
# graph hop once with Spark vector aggregation and caches reusable features in
# Delta. The Phase 0 source runs the original 111M-node graph and its
# symmetrized ~3.23B propagation edges. Its cache is separate from the prior
# validated Phase 2 cache, so this is a new full-graph materialization.
RUN_PHASE37        = False
PHASE37_GRAPH_SOURCE = 'phase0'
PHASE37_NUM_HOPS   = 2
PHASE37_NUM_PARTITIONS = 512
# Phase 3.8 evaluates one globally optimized Spark ML classifier on Phase 3.7
# features. It is an edge-free, distributed linear probe; it does not create
# independent partition models and therefore reports a valid global metric.
RUN_PHASE38        = False
PHASE38_MAX_ITER   = 30
PHASE38_REG_PARAM  = 0.0001
PHASE38_ELASTIC_NET_PARAM = 0.0
RUN_PHASE3         = True
USE_GLOBAL_MAPPING = True

# Emits Phase 3 driver/executor timing markers to diagnose slow or stalled runs.
# Executor markers are written to the relevant YARN container logs.
PHASE3_DIAGNOSTICS = True

# Phase 3 graph limits. Spark hash-samples toward the node limit and filters
# edges before aggregation; the UDF then enforces these final hard limits.
PHASE3_MAX_NODES_PER_COMMUNITY = 10000
PHASE3_MAX_EDGES_PER_COMMUNITY = 30000
# Samples one of every N eligible edges before aggregation. Combined with the
# node cap this keeps the grouped edge payloads bounded without sorting all
# Papers100M edges by community.
# Accuracy experiment: retain twice as many eligible edges as the successful
# 1/8 run while remaining below the unstable full-edge (1/1) configuration.
PHASE3_EDGE_SAMPLE_MODULUS = 4
PHASE3_MLP_EPOCHS = 10

GCN_HIDDEN_DIM    = 256
GCN_NUM_EPOCHS    = 10
GCN_LR            = 0.001
GCN_DROPOUT       = 0.5
RUN_PHASE3B       = True              # Phase 3b: CaaN Global Graph GNN Training

# ── New Advanced Features ──────────────────────────────────────────────────────
# Tiny community handling: 'drop' (ignore them), 'misc' (group them all into community_id = -1)
TINY_COMM_HANDLING  = 'drop'

# 1-hop boundary expansion: If True, include 1-hop external neighbors for boundary nodes.
# NOTE: increases data size but improves boundary accuracy significantly.
EXPAND_BOUNDARY_NODES = False

# Task Type: 'node_classification' or 'link_prediction'
TASK_TYPE = 'both'

# ── Phase 4: Full-Graph Baseline ──────────────────────────────────────────────
# Runs ONCE per dataset (not per algorithm). Uses SAME masks as Phase 3.
RUN_PHASE4        = False       # Set to False to skip OOM-prone driver-bound baselines
BASELINE_EPOCHS   = 10          # reduced from 50 epochs to speed up CPU full-graph training
BASELINE_BATCH    = 1024
BASELINE_FANOUT   = [15, 10]
BASELINE_LR       = GCN_LR
RUN_PHASE4B       = False      # DistDGL Baseline Simulation
RUN_PHASE4C       = False       # ARMA Baseline
RUN_PHASE4D       = False       # ASAP Baseline
RUN_PHASE4E       = False       # GAT Baseline
RUN_PHASE4F       = False       # Graph Transformer Baseline
RUN_PHASE4G       = False       # ClusterSCL Baseline
RUN_PHASE4H       = False       # GATv2 Baseline

# ── Infrastructure ─────────────────────────────────────────────────────────────
S3_BUCKET         = 'us-east-1-s3-gnn'
S3_CODE_PREFIX    = 'pipeline'   # where upload_to_s3.py puts .py files
SKIP_PKG_SYNC     = False        # Required after worker replacement/restart; verifies NumPy and GNN dependencies on every executor node

# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED CONFIG — do not edit below this line
# ══════════════════════════════════════════════════════════════════════════════

# Dataset-specific architecture configs (auto-applied per dataset, case-insensitive)
class _CaseInsensitiveDict(dict):
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        for k in self:
            if str(k).lower() == str(key).lower() or str(k).lower().replace('_', '-') == str(key).lower().replace('_', '-'):
                return super().__getitem__(k)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

_RAW_DATASET_CFG = {
    'ogbn-products':   {'in_feats': 100, 'num_classes': 47},
    'ogbn-arxiv':      {'in_feats': 128, 'num_classes': 40},
    'ogbn-mag':        {'in_feats': 128, 'num_classes': 349},
    'ogbn-papers100M': {'in_feats': 128, 'num_classes': 172},
    'ogbn-proteins':   {'in_feats': 8, 'num_classes': 112},
    'reddit':          {'in_feats': 602, 'num_classes': 41},
    'Reddit':          {'in_feats': 602, 'num_classes': 41},
    'flickr':          {'in_feats': 500, 'num_classes': 7},
    'Flickr':          {'in_feats': 500, 'num_classes': 7},
    'wikics':          {'in_feats': 300, 'num_classes': 10},
    'WikiCS':          {'in_feats': 300, 'num_classes': 10},
    'coauthor-cs':     {'in_feats': 6805, 'num_classes': 15},
    'Coauthor-CS':     {'in_feats': 6805, 'num_classes': 15},
    'coauthor-physics':{'in_feats': 8415, 'num_classes': 5},
    'Coauthor-Physics':{'in_feats': 8415, 'num_classes': 5},
    'deezereurope':    {'in_feats': 128, 'num_classes': 2},
    'DeezerEurope':    {'in_feats': 128, 'num_classes': 2},
    'livejournal':     {'in_feats': 128, 'num_classes': 100},
    'LiveJournal':     {'in_feats': 128, 'num_classes': 100},
    'orkut':           {'in_feats': 128, 'num_classes': 100},
    'Orkut':           {'in_feats': 128, 'num_classes': 100},
}

DATASET_CFG = _CaseInsensitiveDict(_RAW_DATASET_CFG)

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
            'p25_nodes':   f'{root}/phase25_nodes/{tag}/',
            'p25_edges':   f'{root}/phase25_edges/{tag}/',
            'p25_manifest': f'{root}/phase25_manifest/{tag}/',
            'p26_nodes':   f'{root}/phase26_nodes/{tag}/',
            'p26_edges':   f'{root}/phase26_edges/{tag}/',
            'p26_manifest': f'{root}/phase26_manifest/{tag}/',
            'p27_manifest': f'{root}/phase27_manifest/{tag}/',
            'p37_base':    f'{root}/phase37_propagation/{tag}/',
            'p38_base':    f'{root}/phase38_classifier/{tag}/',
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
