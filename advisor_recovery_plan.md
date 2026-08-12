# Advisor Recovery Plan (Sections 3-5, Figure 1, 2-hop visibility)

Date: 2026-08-11

## 1) Diagnosis of current draft

### What triggered the advisor's reaction
- Section 3, 4, and 5 are currently implementation-first and read like engineering notes.
- The narrative lacks the canonical research flow: problem formalization -> method design -> analysis -> implications.
- Figure 1 is descriptive but not analytical: it shows pipeline blocks, but does not communicate hypotheses, design principles, or where contributions map.

### Concrete symptoms in current manuscript
- Section 3 (EMO System Architecture) mostly lists phases and storage details.
- Section 4 (Relational Graph Operations & CAAN Workload) gives operator-level notes, but limited formalism and no complexity/guarantee discussion.
- Section 5 (AWS EMR Cluster Optimizations) reads as deployment tips rather than research-level systems methodology.
- The 2-hop setting used in Phase 3.7/3.8 scaling exists in run artifacts but is not visibly foregrounded in the main Overleaf text/tables.

## 2) Target structure expected in top-tier CS systems papers

Use this exact logic chain for Sections 3-5:

1. Problem and design goals
2. Formal model and assumptions
3. Method/architecture with design rationale
4. Algorithm or execution semantics
5. Complexity/cost/scalability analysis
6. Failure modes and mitigation
7. What this enables experimentally

## 3) Rewrite blueprint (drop-in replacement plan)

## Section 3 -> "EMO Model and Architecture"

### 3.1 Problem Setup and Notation
- Define graph and partition setup:
  - Graph $G=(V,E,X)$
  - Partition mapping $\pi: V \rightarrow \{1,\dots,m\}$
  - Boundary node definition
- State objective explicitly:
  - maximize predictive quality while minimizing cross-worker communication.

### 3.2 Design Goals and Constraints
- G1: communication-free local training path
- G2: recover global context for boundary and minor communities
- G3: transactional reproducibility for large-scale runs

### 3.3 Two-Branch Architecture with Execution Contracts
- Branch A: full-graph propagation + global classifier (Phase 3.7/3.8)
- Branch B: community branch for ablation/variants (Phase 1/2/3b)
- Define branch-level contracts (inputs, outputs, checkpoint boundaries).

### 3.4 Storage Semantics as a First-Class System Component
- Formalize experiment isolation with experiment-id namespacing and Delta transaction logs.
- Explain resumability semantics and correctness of checkpoint reuse.

## Section 4 -> "Relational Formulation of Graph Operations"

### 4.1 Relational Encoding
- Specify tables and keys (nodes, edges, partitions, boundary labels).
- Describe required joins/group-bys as algebraic operators.

### 4.2 Boundary and Super-Node Construction
- Provide equations for:
  - boundary detection
  - super-node feature aggregation
- Add an algorithm box (Algorithm 1): boundary/super-node construction workflow.

### 4.3 Distributed Execution and Data Movement Analysis
- Where shuffles happen, where broadcasts happen.
- Complexity terms with practical interpretation:
  - e.g., join/group-by cost under partitioning assumptions.

### 4.4 Correctness/Consistency Notes
- Argue consistency between relational outputs and graph-level intent.
- Clarify what is exact vs approximate.

## Section 5 -> "Runtime System Design on EMR"

### 5.1 Resource Model
- Separate JVM heap, off-heap tensor memory, local disk pressure, and network.

### 5.2 Runtime Failure Modes and Mitigations
- F1: container kill from off-heap under-provisioning
- F2: root volume exhaustion from temp/cache activity
- F3: package drift across executors
- Map each to mitigation and observed effect.

### 5.3 Operational Cost Model
- Simple throughput/cost framing:
  - runtime decomposition $T = T_{io}+T_{shuffle}+T_{compute}+T_{sync}$
  - explain why communication-free branch reduces $T_{sync}$.

### 5.4 Reproducibility and Determinism Checklist
- Pin environment versions, table versions, and run metadata.

## 4) Figure 1 redesign plan (academic style)

Current Figure 1 is pipeline-heavy and insight-light. Replace with a multi-panel overview figure:

- Panel (a): Problem tension diagram
  - x-axis: communication overhead
  - y-axis: global-context fidelity
  - place full-graph distributed baseline vs isolated community baseline vs EMO target zone
- Panel (b): EMO architecture
  - two branches, with explicit transactional plane and checkpoint boundaries
  - contribution callouts C1/C2/C3 linked to sections
- Panel (c): execution timeline
  - where shuffles/broadcast/checkpoints occur
  - where 2-hop propagation lives in branch A

Design requirements:
- Minimal text per box, larger semantic labels.
- Consistent visual grammar: data artifact, compute stage, synchronization boundary.
- Black-and-white print-friendly with one accent color.
- Caption should state claim, not only describe layout.

## 5) Where the 2-hop results are right now

### Present in Overleaf
- Scaling table inclusion: overleaf/bare_conf_compsoc.tex via results/table13_scalability_multi_instance.tex
- Scaling figure inclusion: overleaf/bare_conf_compsoc.tex via results/phase37_scaling/phase37_paper_run1/phase37_paper_figure_main.pdf

### Not explicit enough
- The table caption/text does not explicitly state "2-hop".
- The figure caption/text does not explicitly state feature vector includes $x^{(0)}, x^{(1)}, x^{(2)}$.

### Evidence in run artifacts (outside Overleaf main tex)
- Manifest confirms num_hops=2 in results/phase37_scaling/phase37_paper_run1/phase37-scaling-20260810T210437Z-manifest.csv

## 6) Fast patch list (high-impact edits in order)

1. Rewrite Section 3 opening into formal problem + design goals.
2. Add notation table (small) before architecture subsection.
3. Replace Section 4 with relational formalization + Algorithm 1 + complexity discussion.
4. Reframe Section 5 as runtime model and failure analysis, not operational notes.
5. Replace Figure 1 with 3-panel overview + contribution mapping.
6. Add one explicit sentence in Experimental Setup: "All Phase 3.7/3.8 scaling experiments use 2-hop propagation (num_hops=2)."
7. Update Table 13 caption to include 2-hop.
8. Update scaling figure caption to mention 2-hop feature construction.

## 7) Suggested writing workflow for the next 24-48h

1. Lock technical claims first (no prose polishing yet).
2. Draft Section 3-5 with equation/algorithm placeholders before beautifying language.
3. Redraw Figure 1 after section skeleton is stable.
4. Only then tighten Related Work and Introduction alignment.

## 8) Ready-to-use subsection templates (short)

Section opener pattern:
- "We address [problem] under [constraints]. Existing systems optimize [A] or [B], but not both. EMO is designed to satisfy [G1-G3] by [core mechanism]."

Method subsection pattern:
- "Given inputs [..], stage S computes [..] using [operator]. The output contract is [..], persisted as [artifact]."

Analysis subsection pattern:
- "The dominant cost is [..]. Under [assumption], expected overhead scales as [..]. This explains the empirical trend in Section 6."