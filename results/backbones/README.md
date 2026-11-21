# Backbones: GCN and GATv2 on the stored GraphSAGE view (advisor 2026-11-19)

LakeGRL trains GCN or GATv2 by Stage 3b alone (`--stage3-only`) on the Phase 2 view `leiden_c10000` that Table B row 5
wrote for GraphSAGE: no Phase 1-2 rerun. The full-graph baseline (`experiments/baselines/node_baselines.py --model`)
trains the same layers (`utils/backbones.py`: conv(f, 256) -> act -> dropout -> conv(256, 256) -> linear; GATv2 = 8
heads x 32, then 1 head x 256). Official split, 100 epochs, lr 0.01, seeds 0-2, one RTX A5000 (24 GB), LakeGRL with 8
workers (local[8], 1 executor x 8 cores). Driver: `experiments/tableB/backbones.sh <dataset> <gcn|gatv2>`; table:
`python experiments/tableB/summary_backbones_cmax.py` -> `SUMMARY.md`.

| Folder | Contents |
|---|---|
| `gcn/`, `gatv2/` | `baselines.jsonl` (full graph), `pipeline.jsonl` (LakeGRL), `peak.jsonl` + `peak_trace/` (peak.sh: GPU = nvidia-smi minus idle, the Table 4 measure), `work/` (bins) |
| `gatv2_1worker/` | Reddit GATv2 LakeGRL with ONE worker, seed 0: the second unit alone fills the card |
| `gcn_pre/` | 11-20: every GCN cell again (full graph + LakeGRL) on the precomputed matrix, d658119: the table's GCN rows |
| `sage/` | 11-20: LakeGRL GraphSAGE by Stage 3b alone on the same stored view (no full graph), d658119 |
| `gcn/` | 11-19 GCN runs, the first code (superseded by `gcn_pre/`, kept) |
| `rerun_1120.sh` | Driver of `gcn_pre/` and `sage/`: backbones.sh with another output folder (untracked, so no repo change) |

OOM: the card filled and `peak.sh` stopped the run (`gpu_full: 1`, WSL would otherwise spill into Windows memory), or
CUDA refused one allocation (`oom` in the baseline record; Reddit full graph GATv2 asked for 109.5 GiB). Reddit's
densest unit holds ~4.2M edges; GATv2 keeps ~8 KB per edge, so it does not fit even alone.

**Advisor 2026-11-20** (decisions on the 11-19 tables): one GCN code version, so every GCN cell was rerun on the
precomputed matrix (`gcn_pre/`); LakeGRL GraphSAGE rerun by Stage 3b alone like GCN and GATv2 (`sage/`), its full graph
stays Table 4 (results/tableB: node_baselines' GraphSAGE path is the same code); GATv2 OOM cells stay OOM; C_max stays
10,000. Jobs grl-bb2-{arxiv,reddit,products}-{gcn,sage}, `bash results/backbones/rerun_1120.sh <dataset> <gcn|sage>`.
GATv2's code is the same at 2cbc57e and d658119 (only the GCN branch changed), so `gatv2/` was not rerun.

**GCN code versions:**
- 2cbc57e / d0cff08 (`gcn/`): PyG `GCNConv` with its default in-layer normalization, recomputed in every forward.
  ogbn-products' full graph filled the card with it (11-19 17:15 record, `gpu_full: 1`).
- 8a6e1e7 (records say d658119 = 8a6e1e7 + a summary-script change only; `gcn_pre/`, and the 11-19 ogbn-products
  full graph in `gcn/`): the same D^-1/2 (A + I) D^-1/2 computed once in `backbone_adj`, `GCNConv(normalize=False)`;
  self-check 6e-8 from PyG's default on the edge list. Same accuracy; Stage 3 37 / 90 / 328 s against 58 / 131 / 553 s.
