# Implementation Plan - LaTeX Paper Skeleton Draft

This plan outlines the design and sections to be written into [bare_conf_compsoc.tex](file:///Users/bilaltan/Desktop/emo-pipeline/bare_conf_compsoc.tex) for our submission to the **IEEE Big Data 2026** conference.

---

## Proposed Changes

We will replace the placeholder text in the LaTeX file with a fully-fleshed academic skeleton for the paper **"Towards Scalable and Efficient Graph Representation Learning with Modern Data Lakes"**, detailing the design, implementation, and empirical results of the **EMO** framework.

### Document Configuration

We will load necessary mathematical, algorithmic, and table packages (e.g., `amsmath`, `amssymb`, `booktabs`, `graphicx`, `url`) at the top of the file to support the systems figures and tables.

### Paper Title & Author Placeholders
* **Title**: `Towards Scalable and Efficient Graph Representation Learning with Modern Data Lakes`
* **Authors**: Leave default templates or set to placeholder authors representing your group.

### Section-by-Section Content

1. **Abstract**:
   * Present the scalability challenges of distributed GRL (neighborhood explosion, network bottlenecks in DistDGL).
   * Introduce EMO (Executor-level Multi-community Orchestrator) as a systems-level architecture bridging transactional data lakes (Delta Lake) and decentralized GNN training on AWS EMR.
   * State that EMO supports complex partitioned workloads like CAAN (Community-aware Auxiliary Networks, from past literature) with zero cross-worker training communication.
   * Report key results: up to a 2× accuracy increase over isolated models, matching or beating DistDGL in throughput and cost efficiency.

2. **Section I: Introduction**:
   * Motivate GRL and explain why distributed scaling is challenging.
   * Define the trade-offs of decentralized community-based GRL: communication-free but suffers from boundary accuracy degradation and minor community sample scarcity.
   * Cite your professor's past paper (CAAN) as the theoretical foundation for resolving these accuracy trade-offs.
   * Introduce EMO as the systems framework that scales CAAN and other GRL workloads to massive datasets directly within Delta Lake.
   * List the paper's three core systems contributions:
     1. Unification of GRL with transactional data lakes (Delta Lake) using relational queries.
     2. A scalable, serverless orchestration layer (EMO) using PySpark and YARN on AWS EMR.
     3. An empirical evaluation proving EMO scales efficiently with commodity hardware.

3. **Section II: Related Work**:
   * *Distributed GNN Engines*: Contrast EMO's communication-free training with high-overhead messaging engines like DistDGL.
   * *Graph Partitioning*: Survey METIS, Louvain, and LPA as partitioning methods.
   * *Transactional Data Lakes*: Introduce Delta Lake, Hudi, and Iceberg, highlighting their utility in machine learning pipelines.

4. **Section III: EMO System Architecture**:
   * Walk through the pipeline phases (Phase 0: Ingestion, Phase 1: Community Detection, Phase 2: Partitioning, Phase 3: GNN training, Phase 5: Reporting).
   * Explain Delta Lake's role in enforcing transaction isolation between phases, preventing concurrent experiment conflicts, and implementing lightweight metadata checkpoints.

5. **Section IV: Relational Graph Operations & CAAN Workload**:
   * Detail how EMO expresses graph boundary analysis and CAAN auxiliary graph construction as relational algebra (joins, group-bys) in PySpark.
   * Explain the compression of major communities into super-nodes (averaging node features) and the integration of minor communities into a global topology using YARN broadcasts.
   * Show how parallel GNN models are trained on executors via Spark Pandas UDFs (`applyInPandas`) with Arrow-based serialization.

6. **Section V: Infrastructure Orchestration & EMR Optimizations**:
   * Discuss off-heap memory configuration (`spark.executor.memoryOverhead = 12g`) to support PyTorch/DGL tensor allocations outside JVM heap.
   * Detail the dynamic host scanning algorithm to locate high-capacity local drives and prevent root-disk EBS OOM during CUDA compile/PyPI download.
   * Explain executor sync patterns using `sc.install_pypi_package` over YARN.

7. **Section VI: Experimental Evaluation**:
   * Compare performance on WikiCS and DeezerEurope.
   * Include a structured LaTeX Table summarizing: Node Accuracy, Internal Accuracy, Link AUC, and Training Time for:
     * LPA + SAGE / LPA + SAGE-CAAN
     * Louvain + SAGE / Louvain + SAGE-CAAN
     * Baselines: DistDGL, ARMA, ASAP, and SAGE-BL.
   * Outline the key takeaways: Louvain + SAGE-CAAN achieves 60.31% node accuracy, recovering boundary accuracy, and executing in 41.5 seconds.

8. **Section VII: Discussion & Future Database Extensions**:
   * Outline future extensions:
     * *Incremental Time-Travel GRL*: Querying Delta logs to retrain only dynamic communities.
     * *Z-Ordering*: Physically clustering tables by `community_id` to speed up S3 reads.

9. **Section VIII: Conclusion**:
   * Reiterate that EMO bridges database query engines and GRL, presenting a scalable cloud pipeline for graph mining.

---

## Verification Plan

### Compilation Check
- Run a LaTeX syntax checker or compilation check on the generated file to ensure all commands, mathematical formulations, and tables are structurally valid LaTeX.
- Command: `pdflatex -interaction=nonstopmode bare_conf_compsoc.tex` (if pdflatex is present on user's system, otherwise check brackets and tags manually).
- We will double check all LaTeX markup to ensure correct escaping of characters (like `%`, `_`, `&`, `#`).
