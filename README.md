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

In the code, the storage layout ablation is "Table A" (`experiments/tableA/`, Table 2 of the paper) and node
classification is "Table B" (`experiments/tableB/`, Table 3). Every timed read goes through an S3 cost model
(20 ms + 80 MB/s per GET) with the page cache dropped first. `<ds>` is `ogbn-arxiv`, `reddit` or `ogbn-products`.

**Section 4.1, storage layouts.** The row-group floor sweep comes first; it picked the 2 MB floor.

```bash
bash experiments/tableA/rg_floor.sh ogbn-products 0 2 4 8 16
RS=rleiden_c10000_v2 bash experiments/tableA/run.sh <ds> --cap 10000 --unit-split leiden --rg-floor-mb 2
#   (ogbn-products: ZL_SAMPLE=10, so Z-order and Liquid clustering read every tenth work item)
bash scripts/wsl_run.sh python experiments/tableA/read_layouts.py --dataset <ds> --runs 3 --coalesce-minor \
  --out-dir ~/lakegrl/tableA/rleiden_c10000_v2 --out results/tableA/read_coalesce.jsonl   # coalesced schedule
#   (Z-order and Liquid clustering: --layouts zorder liquid --procs 3, plus --bin-sample 10 on ogbn-products)
bash scripts/wsl_run.sh python experiments/tableA/probe_units.py <ds> --c-max 5000 40000 --seed 42 \
  --out results/tableA/unit_probe.jsonl                                                       # units vs C_max
```

`write_layouts.py` writes the eight layouts, `read_layouts.py` reads them, and `summary.py` builds the table.

**Section 4.2, node classification.** One command per dataset runs rows 1-5 for seeds 0-2, with their cold reads
and GPU peaks:

```bash
bash experiments/tableB/run.sh <ds>
TB_CORES=1 TB_OUT=results/tableB/mem1 TB_SEEDS=0 TB_REUSE_P1=1 TB_READS=0 TB_PIPELINE_ONLY=1 \
  bash experiments/tableB/run.sh <ds>                                       # one worker: GPU peak
```

Rows 1-2 are `experiments/baselines/node_baselines.py` (full-graph GraphSAGE, Cluster-GCN), and rows 3-5 are
`scripts/run_local_pipeline.py`. The footnote and the community-ID paragraph use the same script:

```bash
S=~/lakegrl/store
bash scripts/wsl_run.sh python experiments/baselines/node_baselines.py --dataset <ds> --method full \
  --cache $S/<ds>/tableB_full --device cuda --epochs 500                    # 500 epochs
bash scripts/wsl_run.sh python experiments/baselines/node_baselines.py --dataset <ds> --method full \
  --cache $S/<ds>/tableB_full --communities $S/<ds>/communities/leiden_c10000 --device cuda   # community IDs
```

**Section 4.3, one view for two tasks.** `<lp>` is `reddit-lp` or `ogbn-products-lp`, the graphs with 50,000
validation and 50,000 test edges held out before Phase 1 (`run_local_pipeline.py --link-split 50000`).

```bash
bash scripts/wsl_run.sh python experiments/tableA/halo_table.py --dataset <lp> --alg leiden_c10000 --rg-floor-mb 2
bash scripts/wsl_run.sh python experiments/trackA/harness.py --dataset <lp> --alg leiden_c10000 --methods M2 \
  --seeds 0,1,2 --device cpu [--halo-budget 1]                              # uncapped / capped halo
TB_OUT=results/tableB/breakdown2 TB_ROWS=5 TB_SEEDS=0 TB_READS=0 TB_ARGS=--p2-time-split TB_PIPELINE_ONLY=1 \
  bash experiments/tableB/run.sh <ds>                                       # one-time cost of Phases 1-2
```

**Section 4.4, Amazon EMR.** `experiments/aws/README.md` lists the steps: upload the tables, create the cluster with
`scripts/emr_cluster.sh`, then run `experiments/aws/run.sh smoke`, `expA`, `expB-read`, `expB-train` and `nodes N`.

## Results

`results/` holds the records behind every number of Section 4, one folder per subsection. `results/README.md` says
which file feeds which table, and `python results/verify_numbers.py` recomputes every number and compares it with
the paper's text.
