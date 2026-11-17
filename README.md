# LakeGRL: Topology-Aware Lakehouse Storage and Disaggregated Execution for Scalable GNN Training

LakeGRL trains graph neural networks on graphs stored as Delta Lake tables on a cloud object store. Storage and
compute are separate: the graph and everything derived from it are transactional tables, and training runs on
stateless workers that read these tables and do not exchange data with each other.

## Pipeline

| Stage | What it does | Code |
|---|---|---|
| Phase 0 | Writes the raw tables: vertices with features and labels (T_V), edges (T_E) and the split (T_S). For link prediction, the validation and test edges are held out here, before Phase 1. | `phases/phase0_ingestion.py`, `scripts/run_local_pipeline.py` |
| Phase 1 | Bounded recursive Leiden. Leiden runs once on the whole graph; every community with more than C_max/2 core vertices (5,000 for C_max = 10,000) is split again by Leiden on its own subgraph, and the resolution is doubled when a piece does not split. | `phases/phase1_community.py` |
| Phase 2 | Writes the unit view MV_unit (each unit's core vertices, its degree-capped 1-hop halo and its edges) in the topology-aware layout: one partition per major unit, minor units packed into shared buckets, rows sorted by unit, square-root row groups with a 2 MB floor. The same pass writes the macro-graph view MV_macro (super-nodes, super-edges, minor vertices) and the unit plan. | `phases/phase2_subgraph.py`, `utils/partition_layout.py`, `utils/macro_graph.py` |
| Stage 3 | Packs consecutive minor units into coalesced work items, reads each one with contiguous range reads, and trains the two-level CaaN model: Global GRL on the stored macro-graph and Local GRL on every major unit, with the other communities as super-nodes. | `phases/phase3b_caan.py` |

Every model uses the training setting `TRAIN` in `experiment_config.py` (100 epochs, lr 0.01, weight decay 5e-4,
2 layers, hidden size 256, dropout 0.5).

## Setup

The local runs use Python 3.11, OpenJDK 17, Spark 3.5 and Delta Lake 3.2, as on Amazon EMR 7.5.0. The exact
package versions are in `requirements-local.txt`; GPU training uses the CUDA 12.8 build of PyTorch.

The code imports the repository as the package `pipeline`, a symlink to the repository root. `scripts/wsl_run.sh`
runs a command in the micromamba environment `lakegrl` (`~/mamba/envs/lakegrl`) with this symlink, Java and Spark set
up, and writes the Delta tables under `$LAKEGRL_STORE` (default `~/lakegrl/store`). On Windows, run it in WSL.

LakeGRL on one dataset (`ogbn-arxiv`, `reddit` or `ogbn-products`), as in Table B row 5:

```bash
bash scripts/wsl_run.sh python scripts/run_local_pipeline.py --dataset ogbn-arxiv --alg leiden \
  --unit-split leiden --tiny-comm drop --c-max 10000 \
  --p2-layout hive_merged --p2-rg-floor-mb 2 --p2-one-file --device cuda --seed 0
```

## Experiments in the paper

| Section | Driver | Notes |
|---|---|---|
| 4.1 Storage layouts (Table A) | `RS=rleiden_c10000 bash experiments/tableA/run.sh <dataset>` | `write_layouts.py` writes the 8 layouts, `read_layouts.py` reads them cold through an S3 cost model (20 ms + 80 MB/s per GET), `summary.py` builds the table. Row-group floor sweep: `bash experiments/tableA/rg_floor.sh ogbn-products 0 2 4 8 16`. Units vs C_max: `probe_units.py`. |
| 4.2 Node classification (Table B) | `bash experiments/tableB/run.sh <dataset>` | Rows 1-2 (full-graph GraphSAGE, Cluster-GCN), the 500-epoch check and the community-ID ablation: `experiments/baselines/node_baselines.py`. Rows 3-5: `scripts/run_local_pipeline.py`. Cold reads: `read_units.py`, `read_full.py`. GPU peak: `peak.sh`. |
| 4.3 Halo and one-time cost | `experiments/tableA/halo_table.py` | Halo storage and read; link prediction AUC with `experiments/trackA/harness.py`. Phase 1-2 timing: `run_local_pipeline.py --p2-time-split`. |
| 4.4 Amazon EMR | `experiments/aws/README.md` | `scripts/emr_cluster.sh` creates, resizes and terminates the cluster; `experiments/aws/run.sh smoke\|expA\|expB-read\|expB-train\|nodes N` runs the steps on it. |

Earlier versions of the project left more code in this repository (for example `runners/` and most of `scripts/`);
the files above are the ones the paper uses.
