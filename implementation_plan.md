# Implementation Plan - EMO Paper Framing, Figures, Tables & Ablation Study

This plan details the technical structure, manuscript refactoring, and experimental table/figure generation to align the **EMO** manuscript ([bare_conf_compsoc.tex](file:///Users/bilaltan/Desktop/emo-pipeline/bare_conf_compsoc.tex)) with top-tier CS systems research standards.

---

## Key Framing & Strategic Decisions

1. **Primary Paper Framing (Approach 1)**:
   - **Main System Focus**: Phase 0 (Ingestion) $\rightarrow$ Phase 1 (Community Detection) $\rightarrow$ Phase 2 (Partitioning & Subgraphs) $\rightarrow$ Phase 3 (Decoupled Community GNN Training).
   - **Supplementary Component**: Phase 3b (CaaN - Community-aware Auxiliary Networks) is framed as a supplementary relational context recovery mechanism to fix boundary degradation.
   - **Approach 2 (Distributed Scale)**: Phase 3.7 & 3.8 (SIGN-style 2-hop neighborhood feature propagation + global Spark ML classifier) is framed specifically as the big-dataset scaling solution, demonstrating the capabilities of AWS EMR + Apache Spark + Delta Lake on massive graphs like `ogbn-papers100M`.

2. **Strict Research Paper Tone (No Early Implementation Dumping)**:
   - Sections 1–4 are kept strictly academic, formal, and problem/systems-first (focusing on formal graph definitions, boundary equations, relational algebra operators, complexity, and design principles).
   - All environment details (e.g. PyPI YARN sync, `memoryOverhead`, local disk remapping, specific PyTorch flags) are isolated to **Section 5 (Runtime System Design on EMR)** and **Section 6 (Experimental Setup)**.

3. **Separate DistDGL Reporting**:
   - DistDGL is evaluated in a dedicated subsection (Section 6.D) and stand-alone comparison table (**Table 12**), contrasting EMO's communication-free training against DistDGL's RPC network traffic and memory overhead.

4. **ML-GRL Style Figures & Tables**:
   - **Figure 8 (ML-GRL Style)**: CPU Cores / Executor Scalability plot showing execution time and speedup as cluster parallelism increases across datasets.
   - **Table 12 (ML-GRL Style)**: Fair performance & efficiency comparison with DistDGL (Node Acc, Link AUC, Runtime, Network Volume GB).
   - **Table 9 (ML-GRL Style)**: Complete component-wise ablation study demonstrating the impact of each subsystem (Global Graph / CaaN, S3/Delta Lake optimizations, Partitioning algorithms).

---

## User Review Required

> [!IMPORTANT]
> **Manuscript Restructuring**: Sections 3 & 4 will emphasize Approach 1 (Phases 0-3) as the core systems paper narrative, while positioning Phases 3.7-3.8 as the big-data Spark/EMR processing path for 100M+ scale graphs.
>
> **Ablation Data Verification**: Table 9 will include specific measurements for:
> 1. Full EMO (Louvain-SAGE-CAAN)
> 2. Full EMO (LPA-SAGE-CAAN)
> 3. w/o Global Graph / CAAN (Decoupled Louvain-SAGE & LPA-SAGE)
> 4. w/o S3/Delta Lake Level Optimizations (Uncached S3 reads vs. Delta transaction log caching)

---

## Proposed Changes

---

### Component 1: Manuscript Structural Refactoring ([bare_conf_compsoc.tex](file:///Users/bilaltan/Desktop/emo-pipeline/bare_conf_compsoc.tex))

#### [MODIFY] [bare_conf_compsoc.tex](file:///Users/bilaltan/Desktop/emo-pipeline/bare_conf_compsoc.tex)

- **Section 3 (EMO Model and System Architecture)**:
  - Formally introduce the two-approach paradigm:
    - **Approach 1 (Core)**: Partition-centric, communication-free multi-community GNN execution (Phases 0–3).
    - **Approach 2 (Distributed Scale)**: Full-graph relational feature propagation (Phases 3.7–3.8) for billion-scale graphs on EMR/Spark.
  - Highlight Phase 3b (CaaN) as a supplementary relational context recovery layer.
  - Clean out premature deployment notes, ensuring Sections 3 & 4 contain zero low-level execution noise.

- **Section 4 (Relational Formulation for Community-Aware GRL)**:
  - Detail relational graph encoding, boundary detection equations, super-node centroid abstraction, and distributed execution complexity ($T_{\text{total}} = T_{\text{io}} + T_{\text{shuffle}} + T_{\text{compute}} + T_{\text{sync}}$).
  - Include Algorithm 1 (Boundary Detection and Super-node Context Construction).

- **Section 5 (Runtime System Design on AWS EMR)**:
  - Consolidate all infrastructure, YARN memory overhead (`12g`), local volume redirection (`TMPDIR`), and PyPI executor synchronization logic here.

---

### Component 2: Dedicated DistDGL Evaluation & Table 12

#### [MODIFY] [results/table12_distdgl_comparison.tex](file:///Users/bilaltan/Desktop/emo-pipeline/results/table12_distdgl_comparison.tex)
#### [MODIFY] [bare_conf_compsoc.tex](file:///Users/bilaltan/Desktop/emo-pipeline/bare_conf_compsoc.tex) (Section 6.D)

- Format Table 12 to match ML-GRL Table 12 standards:
  - Datasets: `WikiCS`, `ogbn-products`, `reddit`, `ogbn-arxiv`.
  - Metrics: Node Accuracy (%), Link AUC (%), Execution Time (s), Inter-Node Communication Volume (GB), Memory Footprint (GB).
  - Explicitly emphasize EMO's **0.0 GB inter-node training communication** vs. DistDGL's heavy RPC network traffic (e.g., 28.66 GB RAM / 46 GB RAM).

---

### Component 3: CPU Cores / Executor Scalability Figure (ML-GRL Fig. 8 Style)

#### [NEW] [runners/create_cpu_scaling_figure.py](file:///Users/bilaltan/Desktop/emo-pipeline/runners/create_cpu_scaling_figure.py)
#### [MODIFY] [bare_conf_compsoc.tex](file:///Users/bilaltan/Desktop/emo-pipeline/bare_conf_compsoc.tex)

- Write a standalone script to generate a publication-quality figure (`fig_cpu_scaling.pdf` / `fig8_cpu_cores_scaling.pdf`):
  - **x-axis**: Number of CPU Cores / Spark Executors (e.g., 16, 32, 64 vCPUs / executors).
  - **y-axis**: Execution Time (seconds) & Speedup multiplier across datasets (`WikiCS`, `ogbn-products`, `ogbn-papers100M`).
  - Embed into Section 6 to visually illustrate horizontal cluster scaling.

---

### Component 4: Complete Ablation Study (ML-GRL Table 9 Style)

#### [MODIFY] [results/table9_ablation_study.tex](file:///Users/bilaltan/Desktop/emo-pipeline/results/table9_ablation_study.tex)

- Complete the multi-column ablation matrix to report:
  1. **Full EMO Pipeline (Louvain-SAGE-CAAN + Delta Lake Opt)**
  2. **EMO Pipeline (LPA-SAGE-CAAN + Delta Lake Opt)**
  3. **w/o Global Graph / CAAN (Decoupled Louvain-SAGE)**
  4. **w/o Global Graph / CAAN (Decoupled LPA-SAGE)**
  5. **w/o S3/Delta Lake-Level Optimizations (Uncached S3 reads)**
  6. **Full-Graph Baseline (GraphSAGE)**
  7. **Distributed GNN Engine (DistDGL)**
- Columns: `Configuration / Variant`, `Node Acc (%)`, `Comm Acc (%)`, `Ingestion Time (s)`, `Partition Time (s)`, `GNN Train Time (s)`, `Total Time (s)`.

---

## Detailed Step-by-Step Methodology for Table 9 (Ablation Study)

### How We Complete Table 9:
1. **Data Source Extraction**:
   - Extract raw baseline timings and accuracy values from [`results/run-all_results-7.xlsx`](file:///Users/bilaltan/Desktop/emo-pipeline/results/run-all_results-7.xlsx) and [`paper_results_reference.json`](file:///Users/bilaltan/Desktop/emo-pipeline/paper_results_reference.json).
2. **Delta Lake Optimization Delta**:
   - Compare Phase 0 ingestion & Phase 3 data loading times with Delta Lake ACID transaction log caching enabled vs. cold uncached S3 Parquet re-reads.
3. **Global Graph (CAAN) Delta**:
   - Quantify the exact accuracy jump on `reddit` (52.96% $\rightarrow$ 92.73% for Louvain; 72.74% $\rightarrow$ 89.11% for LPA) to prove boundary recovery effect.
4. **LaTeX Integration**:
   - Format [`results/table9_ablation_study.tex`](file:///Users/bilaltan/Desktop/emo-pipeline/results/table9_ablation_study.tex) cleanly and link it directly into Section 6.C of [`bare_conf_compsoc.tex`](file:///Users/bilaltan/Desktop/emo-pipeline/bare_conf_compsoc.tex).

---

## Verification Plan

### Automated Checks
- Compile LaTeX manuscript to ensure clean build without syntax errors or broken figure/table references:
  `pdflatex -interaction=nonstopmode bare_conf_compsoc.tex`
- Run figure generation script:
  `python3 runners/create_cpu_scaling_figure.py`

### Visual & Structural Inspection
- Inspect generated `fig8_cpu_cores_scaling.pdf` and ensure print-ready GFM formatting.
- Verify LaTeX Table 9, Table 12, and Figure 8 alignment and caption accuracy.
