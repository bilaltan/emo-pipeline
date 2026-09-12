# ══════════════════════════════════════════════════════════════════════════════
#  experiment_config.py
#  ► Edit ONLY this file to configure your experiment.
#  ► Upload to S3 before running the notebook:
#      python upload_to_s3.py
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME   = 'gatv2_full_sweep'          # labels ALL S3 outputs; change per experiment run

# ── Datasets ───────────────────────────────────────────────────────────────────
# Available options:
# Standard: 'WikiCS', 'Coauthor-Physics', 'Coauthor-CS', 'DeezerEurope', 'Foursquare'
# 100M-scale: 'reddit', 'ogbn-products'
# 1B-scale:   'ogbn-papers100M'
DATASETS_TO_RUN = ['WikiCS', 'ogbn-products', 'reddit', 'ogbn-mag', 'LiveJournal', 'Orkut',]

# ── GNN Models to Run ─────────────────────────────────────────────────────────
# Supported choices: 'sage', 'gat', 'gatv2', 'transformer', 'clusterscl', 'arma', 'asap'
GNN_MODELS = ['gatv2']

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
ALGORITHMS_TO_RUN  = ['louvain']  # subset of ['lpa', 'louvain', 'igraph_lpa']
# LPA propagates one hop per iteration, so a small budget leaves a large-diameter
# graph fragmented. Runs until labels settle or the budget is exhausted.
LPA_MAX_ITER       = 20
LPA_TOL            = 0.001   # stop when <0.1% of labels move in an iteration
# Fold communities smaller than MIN_COMMUNITY_SIZE into the major community they
# have the most edges into, instead of leaving a long tail for Phase 2 to drop
# and Phase 3b to carry as explicit minor nodes.
MERGE_MINOR_COMMUNITIES = True
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
# ── Scalable path for billion-edge graphs (2.5 -> 2.6 -> 2.7 -> 3) ────────────
# Phase 3's bundled path collapses each community into one Spark row via
# collect_list, and bounds it by hash-sampling large communities down to ~8k
# nodes. On Papers100M that discards almost every edge of a large community
# (retention falls as 1/mod^2), so the run either fails or returns a model
# trained on a fragment. These phases materialize the graph as shard-addressable
# Delta blocks so workers read bounded units directly instead.
RUN_PHASE25        = True             # Shard-addressable graph store (required for 100M+)
PHASE25_NUM_SHARDS = 512
RUN_PHASE26        = True             # Bounded source-seed training blocks
PHASE26_SEED_BLOCKS = 16
PHASE26_NEIGHBOR_BLOCKS = 4
# Reunites the four edge partitions logically per source seed unit and records
# exact halo/workload statistics. It does not train or materialize adjacency.
RUN_PHASE27        = True             # Audits block halo/workload before training
PHASE27_WORKING_SET_HEADROOM = 4.0
# Initial direct-Delta validation: trains bounded, complete source-seed units
# without collect_list. This is intentionally a local-block validation, not a
# synchronized full-graph model. Start with a small deterministic subset.
RUN_PHASE35        = False            # Direct-Delta block training; enable for Papers100M-scale runs
PHASE35_MAX_UNITS  = 8
# Synchronous FedAvg proof on bounded direct blocks. The driver averages only
# this validation subset's compact model vectors; it is not the final 8,192-
# unit distributed parameter-service implementation.
# DISABLED — not thesis-consistent. Phase 3.6 performs synchronized FedAvg/FedAdam
# across PHASE36_ROUNDS rounds, which reintroduces exactly the cross-worker
# parameter communication LakeGRL claims to eliminate. Kept for reference only;
# do not report its results as LakeGRL.
RUN_PHASE36        = False
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
# DISABLED — not the method under test. Phases 3.7/3.8 are SIGN-style multi-hop
# feature propagation followed by a linear probe. They use no communities, no
# super-node abstraction and no decoupled per-partition training, so they
# demonstrate that Spark can scale a different (2020) technique rather than that
# LakeGRL scales. Scale claims must come from the 2.5/2.6/2.7/3 path instead.
# Retained because the submitted paper reports their results — do not delete.
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
# Units are also split so no unit carries more than this many TRAIN nodes.
# Cost tracks n_train (corr +0.52), which varied 9.1x across units and made
# Phase 3 straggler-bound; row count alone does not balance it.
PHASE3_MAX_TRAIN_PER_UNIT = 1500
PHASE3_MAX_EDGES_PER_COMMUNITY = 30000
# Samples one of every N eligible edges before aggregation. Combined with the
# node cap this keeps the grouped edge payloads bounded without sorting all
# Papers100M edges by community.
# Accuracy experiment: retain twice as many eligible edges as the successful
# 1/8 run while remaining below the unstable full-edge (1/1) configuration.
# Set modulus to 1 to retain 100% of eligible edges for link prediction accuracy
PHASE3_EDGE_SAMPLE_MODULUS = 1
# Probe-head budget. It early-stops on validation, so this is a ceiling. A flat 10
# was the accuracy ceiling on datasets with many classes.
PHASE3_MLP_EPOCHS = 50
PHASE3_MLP_PATIENCE = 15
# Communities larger than PHASE3_MAX_NODES_PER_COMMUNITY are split into bounded
# blocks and streamed via cogroup rather than hash-sampled down and packed into a
# single Spark row. The old path lost edges as 1/mod^2 on large communities and
# hit the JVM 2GB array limit, which is why Papers100M could not be processed.
PHASE3_BLOCK_OVERSIZED = True

GCN_HIDDEN_DIM    = 256
# Phase 3 takes one full-batch step per epoch, so this is a gradient-update budget,
# not a pass count. Training now early-stops on each unit's own validation split, so
# this is a ceiling rather than a target.
GCN_NUM_EPOCHS = 50
PHASE3_NODE_PATIENCE = 20
GCN_LR            = 0.001
GCN_DROPOUT       = 0.5
RUN_PHASE3B       = True              # Phase 3b: CaaN Global Graph GNN Training

# ── New Advanced Features ──────────────────────────────────────────────────────
# Tiny community handling: 'drop' (ignore them), 'misc' (group them all into community_id = -1)
TINY_COMM_HANDLING  = 'drop'

# 1-hop boundary expansion: If True, include 1-hop external neighbors for boundary nodes.
# Retains incident cut edges in local partitions for boundary link prediction recovery.
EXPAND_BOUNDARY_NODES = True

# Task Type: 'node_classification' or 'link_prediction'
TASK_TYPE = 'both'

# ── Phase 4: Full-Graph Baseline ──────────────────────────────────────────────
# Runs ONCE per dataset (not per algorithm). Uses SAME masks as Phase 3.
RUN_PHASE4        = True        # Re-enabled: the baseline now trains to convergence (was 10 steps)
# The baseline defines the upper bound every retention claim is measured against,
# so it trains to convergence with early stopping rather than to a fixed budget.
BASELINE_EPOCHS   = 100         # max node-classification epochs (early-stopped on val)
BASELINE_NODE_PATIENCE = 10     # epochs without val improvement before stopping
BASELINE_LINK_EPOCHS   = 300    # max link-prediction epochs (full-batch, early-stopped)
BASELINE_LINK_PATIENCE = 30

# Link-split ratios must match Phase 3's per-community 80/10/10 split so both
# sides measure the same task.
BASELINE_LINK_VAL_FRAC  = 0.10
BASELINE_LINK_TEST_FRAC = 0.10

# Substrate parity. Phase 3 trains under per-community node/edge caps; an
# unconstrained Phase 4 sees the whole graph, so the gap between them mixes
# partitioning loss with sampling loss. Set these to the totals Phase 3 reports
# on its summary row (n_nodes / n_edges) to run the matched-budget condition,
# which separates the two. Leave at 0 for the true upper bound.
#   Run both: 0 -> upper bound, matched -> partitioning loss in isolation.
BASELINE_NODE_BUDGET = 0
BASELINE_EDGE_BUDGET = 0

# Phase 4 is deliberately single-machine: it collects the graph to one driver.
# That is the constraint LakeGRL exists to remove, so the baseline is expected to
# be the component that cannot scale — run it on a single large EC2 instance.
BASELINE_SINGLE_MACHINE = True
BASELINE_BATCH    = 1024
BASELINE_FANOUT   = [15, 10]
BASELINE_LR       = GCN_LR
RUN_PHASE4B       = False      # DistDGL Baseline Simulation
RUN_PHASE4C       = False       # ARMA Baseline
RUN_PHASE4D       = False       # ASAP Baseline
RUN_PHASE4E       = False       # GAT Baseline
RUN_PHASE4F       = False       # Graph Transformer Baseline
RUN_PHASE4G       = False       # ClusterSCL Baseline
RUN_PHASE4H       = True        # GATv2 Baseline

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
    'node_epochs': BASELINE_EPOCHS,
    'node_patience': BASELINE_NODE_PATIENCE,
    'link_epochs': BASELINE_LINK_EPOCHS,
    'link_patience': BASELINE_LINK_PATIENCE,
    'link_val_frac': BASELINE_LINK_VAL_FRAC,
    'link_test_frac': BASELINE_LINK_TEST_FRAC,
    'node_budget': BASELINE_NODE_BUDGET,
    'edge_budget': BASELINE_EDGE_BUDGET,
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
