# EMO: Executor-level Multi-community Orchestrator
### Scalable Graph Representation Learning over Transactional Cloud Data Lakes

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%2B-orange.svg)](https://spark.apache.org/)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-3.0%2B-00ADD8.svg)](https://delta.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyG--Distributed-3C2179.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**EMO** is an end-to-end distributed system for partition-centric Graph Representation Learning (GRL) executed natively over transactional cloud data lakes. By modeling graph ingestion, community detection, boundary extraction, and auxiliary network contraction as declarative queries in **Apache Spark SQL** over **Delta Lake** on Amazon S3, EMO enables high-throughput GNN training on commodity cloud clusters without expensive cross-worker parameter synchronization.

---

## 🌟 Key Features

- **Decoupled Partition-Centric GRL**: Eliminates the high network communication bottleneck of distributed GNN frameworks (e.g., DistDGL) by training locally on modular subgraphs.
- **Community-Aware Auxiliary Network (CAAN)**: Compresses major communities into centroid-feature super-nodes and broadcasts them to workers, restoring global topological context with **0.0 GB** inter-worker network overhead during training.
- **Transactional Data Lake Architecture**: Uses Delta Lake ACID transaction logs on Amazon S3 for seamless phase-level checkpointing, automated recovery, and zero-RAM disk-safe ingestion.
- **Scalability from Small Graphs to Billion-Edge Scale**: Supports single-machine debugging and large-scale AWS EMR multi-worker clusters across datasets ranging from WikiCS to `ogbn-papers100M`.
- **Distributed Multi-Hop Feature Propagation (SIGN-Style)**: Computes high-hop diffused features natively via Spark SQL relational joins for ultra-fast, memory-bounded graph learning.
- **Automated Reporting & Artifacts**: Generates multi-tab Excel performance reports and publication-ready LaTeX tables.

---

## 🏗️ System Architecture & Pipeline Phases

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EMO Pipeline Lifecycle                                   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
  Phase 0: Delta Ingestion  ──►  Phase 1: Community Detection  ──►  Phase 2: Subgraph Extract
  [Transactional S3 Tables]       [Louvain / LPA / METIS]           [Boundary & Adjacency]
              │
              ▼
  Phase 3: Decoupled GNNs   ──►  Phase 3b: CAAN Context        ──►  Phase 3.7: SIGN Propagation
  [Local PyG Training]            [Super-node Compression]          [Multi-hop Spark SQL]
              │
              ▼
  Phase 4: Baselines        ──►  Phase 5: Reporting
  [DistDGL / Full-Graph PyG]     [Excel Sheets & LaTeX Tables]
```

| Phase | Module | Description |
| :--- | :--- | :--- |
| **Phase 0** | `phases.phase0_ingestion` | Ingests graphs and feature matrices into transactional Delta Lake tables on S3. |
| **Phase 1** | `phases.phase1_community` | Partitions vertices using Louvain modularity or distributed Label Propagation (LPA). |
| **Phase 2** | `phases.phase2_subgraph` | Extracts local partition subgraphs and identifies cross-community boundary vertices. |
| **Phase 2.5–2.7** | `phases.phase25_graph_store` | Materializes sharded Delta tables, training blocks, and verifies halo overlaps. |
| **Phase 3** | `phases.phase3_training` | Distributed PyTorch GNN training per executor (GraphSAGE, GAT, GATv2, Transformer). |
| **Phase 3b** | `phases.phase3b_caan` | Compiles CAAN super-node auxiliary graph to recover global context. |
| **Phase 3.7** | `phases.phase37_feature_propagation` | Distributed multi-hop feature propagation on Spark SQL for massive graphs. |
| **Phase 3.8** | `phases.phase38_edge_free_classifier` | Edge-free Spark ML classification probe on propagated features. |
| **Phase 4** | `phases.phase4_baselines` | Standalone full-graph PyG and DistDGL distributed baseline evaluations. |
| **Phase 5** | `phases.phase5_reporting` | Automated LaTeX table generation and multi-sheet master Excel reports. |

---

## 📁 Repository Structure

```text
emo-pipeline/
├── experiment_config.py     # Central experiment configuration (phases, datasets, models)
├── upload_to_s3.py          # Synchronizes code and configs to Amazon S3
├── diagnose_workers.py      # EMR worker and executor diagnostics
│
├── phases/                  # Core pipeline phase implementations
│   ├── phase0_ingestion.py
│   ├── phase1_community.py
│   ├── phase2_subgraph.py
│   ├── phase25_graph_store.py
│   ├── phase26_training_blocks.py
│   ├── phase27_block_audit.py
│   ├── phase3_training.py
│   ├── phase3b_caan.py
│   ├── phase35_direct_block_training.py
│   ├── phase36_sync_training.py
│   ├── phase37_feature_propagation.py
│   ├── phase38_edge_free_classifier.py
│   ├── phase4_baselines.py
│   ├── phase5_reporting.py
│   └── metis_runner.py
│
├── runners/                 # Execution drivers and figure visualizers
│   ├── run_emr.py           # AWS EMR cluster driver
│   ├── run_local.py         # Local / single-machine driver
│   ├── run_emr_step.py      # EMR step execution runner
│   ├── create_emo_architecture_figure.py
│   ├── create_infrastructure_figure.py
│   ├── create_cpu_scaling_figure.py
│   ├── collect_propagation_benchmark_results.py
│   └── generate_emr_excel_report.py
│
├── scripts/                 # Shell execution & orchestrator scripts
│   ├── run_all.sh                        # Master unified benchmark execution
│   ├── run_single_fast_run.sh            # 1-minute fast sanity execution on WikiCS
│   ├── run_emr_experiments.sh            # Main runner for AWS EMR clusters
│   ├── run_all_experiments_master.sh     # Comprehensive multi-dataset benchmark
│   ├── run_all_experiments_2worker.sh    # 2-worker scaling sweep script
│   ├── run_all_experiments.sh            # Step-by-step suite execution
│   ├── run_full_paper_benchmark.sh       # Full academic benchmark suite
│   ├── run_missing_results.sh            # Ablation & missing experiment runner
│   ├── emr_setup.sh                      # EMR bootstrap & environment installer
│   └── fetch_s3_results_last_24h.py      # Pulls latest results from S3
│
├── utils/                   # Shared utility modules
│   ├── common.py            # Spark session builder & logging helpers
│   ├── models.py            # PyTorch Geometric model definitions
│   ├── paths.py             # Delta table path resolvers
│   ├── graph_validator.py   # Dataset & partition validation
│   └── emr_bootstrap.py     # Dynamic storage & disk redirection
│
└── results/                 # Output logs, Excel summaries, and benchmarks
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ (tested on Python 3.9, 3.11, 3.14)
- Java 8 or Java 11 / 17 (for Apache Spark)
- Apache Spark 3.4+ / 3.5+ with Delta Lake support
- PyTorch 2.0+ & PyTorch Geometric

Install the required Python packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric ogb delta-spark pyspark pandas numpy openpyxl matplotlib igraph networkx boto3
```

---

### 2. Fast Sanity Run (Local Execution)

Run a fast, 1-minute end-to-end execution across all phases on the **WikiCS** dataset:

```bash
./scripts/run_single_fast_run.sh
```

Or run directly with the Python local runner:

```bash
python3 -m runners.run_local --dataset WikiCS --model sage --fast
```

---

### 3. Configuring Experiments

Modify `experiment_config.py` to select target datasets, models, and phases:

```python
# experiment_config.py
EXPERIMENT_NAME = 'scaling_sweep'

# Supported: 'WikiCS', 'Coauthor-Physics', 'Coauthor-CS', 'DeezerEurope', 
#            'reddit', 'ogbn-products', 'ogbn-mag', 'LiveJournal', 'Orkut', 'ogbn-papers100M'
DATASETS_TO_RUN = ['WikiCS', 'reddit', 'ogbn-products']

# Supported: 'sage', 'gat', 'gatv2', 'transformer', 'clusterscl', 'arma', 'asap'
GNN_MODELS = ['sage', 'gatv2']

# Enable / Disable Pipeline Phases
RUN_PHASE0  = True    # Ingestion into Delta Lake
RUN_PHASE1  = True    # Community Partitioning (Louvain / LPA)
RUN_PHASE2  = True    # Subgraph & Boundary Extraction
RUN_PHASE3  = True    # Decoupled PyG Training
RUN_PHASE4  = True    # Baseline Evaluations
```

---

### 4. Running on AWS EMR

1. **Bootstrap / Setup Driver**:
   Connect to your EMR Driver node and run:
   ```bash
   bash scripts/emr_setup.sh
   ```

2. **Upload Code to S3**:
   From your local environment or driver:
   ```bash
   python3 upload_to_s3.py --code-only
   ```

3. **Execute Cluster Suite**:
   ```bash
   # Run standard 2-worker scaling sweep (4 -> 8 -> 16 Executors)
   ./scripts/run_emr_experiments.sh 2worker

   # Run on a single dataset (e.g. reddit or ogbn-products)
   ./scripts/run_emr_experiments.sh reddit
   ./scripts/run_emr_experiments.sh products

   # Run full 9-dataset master benchmark
   ./scripts/run_all_experiments_master.sh
   ```

---

## 📊 Supported Datasets & Benchmarks

| Dataset | Nodes | Edges | Classes | Task Type |
| :--- | :--- | :--- | :--- | :--- |
| **WikiCS** | 11,701 | 431,726 | 10 | Node Classification / Link Prediction |
| **Coauthor-CS** | 18,333 | 163,788 | 15 | Node Classification / Link Prediction |
| **Coauthor-Physics** | 34,493 | 495,924 | 5 | Node Classification / Link Prediction |
| **DeezerEurope** | 28,281 | 185,504 | 2 | Node Classification / Link Prediction |
| **Reddit** | 232,965 | 11,606,919 | 41 | Node Classification / Link Prediction |
| **ogbn-products** | 2,449,029 | 61,859,140 | 47 | Node Classification / Link Prediction |
| **ogbn-mag** | 1,939,743 | 5,422,810 | 349 | Node Classification |
| **LiveJournal** | 4,847,571 | 34,681,189 | - | Link Prediction / Scaling |
| **Orkut** | 3,072,441 | 117,185,083 | - | Link Prediction / Scaling |
| **ogbn-papers100M** | 111,059,956 | 1,615,685,872 | 172 | Distributed SIGN Feature Propagation |

---

## 📈 Performance & Results

- **Zero-Network Training**: Achieves **0.0 GB** inter-worker network communication during PyTorch training epochs via the CAAN super-node compression mechanism.
- **High Accuracy Retention**: Recovers up to **97.7%** of global graph accuracy while delivering up to **+22.6% higher link prediction AUC** compared to unaugmented community partitioning.
- **Linear Scaling on Cloud Storage**: Delta Lake columnar Parquet layouts with dynamic EBS redirection ensure resilient, crash-free execution on large-scale commodity instances.

Results and logs are automatically synced to `results/` and Amazon S3 in Excel format (`.xlsx`) and LaTeX format (`.tex`).

---

## 📜 Citation & License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
