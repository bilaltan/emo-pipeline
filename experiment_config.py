# Settings shared by every experiment in the paper. The drivers (scripts/run_local_pipeline.py,
# experiments/baselines/node_baselines.py, experiments/trackA/harness.py) import them from here.

# Maximum unit capacity C_max: rows per training unit, core members + 1-hop halo. Phase 1 splits communities to
# <= C_max / 2 core members and Phase 2 caps the halo at the core size, so a unit stores <= C_max rows; Stage 3 packs
# coalesced work items up to C_max rows as well. 10,000 by default (experiments/tableA/probe_units.py varies it).
C_MAX = 10_000

# 'leiden': Phase 1's recursive Leiden (phase1_community.split_units) keeps every unit within C_MAX and Stage 3 never
# cuts a unit. 'idmod': no Phase 1 split; Stage 3 cuts units above C_MAX into id & (n - 1) blocks (Table B row 4).
UNIT_SPLIT = 'leiden'

# One training setting for every model: LakeGRL's stages, the full-graph baseline, Cluster-GCN and the link-prediction
# methods all read this block, and every result record stores a copy. An epoch is one pass over the training nodes;
# for full-batch training that is one gradient step. No early stopping: each model trains for `epochs`, and node
# classification keeps its best-validation checkpoint.
TRAIN = {
    'epochs':       100,
    'lr':           0.01,
    'weight_decay': 5e-4,
    'hidden_dim':   256,
    'num_layers':   2,          # every encoder in the code is 2-layer
    'dropout':      0.5,
    'seeds':        [0, 1, 2],
    'probe_head':   False,      # True: also fit an MLP on frozen embeddings and keep the better head on validation
    'link_batch':   65536,      # positive edges per step in link training
}
