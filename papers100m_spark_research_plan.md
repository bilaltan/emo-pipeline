# Papers100M Spark Training Research Plan

## Research objective

Demonstrate what Spark contributes to CPU-only training and evaluation of a graph at Papers100M scale:

1. lossless handling of Phase 2 community subgraphs at edge-sampling modulus `1/1`;
2. improved accuracy through a controlled, reproducible partition-training protocol; and
3. eventual **full-data-coverage** training/evaluation, where every train/test node and every graph edge is available to the computation.

This plan distinguishes full-data coverage from full-batch GNN training. A model can see every node and edge across distributed mini-batches without placing the entire graph in one Python process.

## Current baseline and conclusions

| Run | Eligible-edge retention | Weighted node accuracy | Phase 3 time | Outcome |
|---|---:|---:|---:|---|
| Sparse baseline | `1/64` | 0.2344 | 145.3s | Stable, overly sparse |
| Balanced baseline | `1/16` | 0.3818 | 351.8s | Stable |
| Accuracy baseline | `1/8` | **0.3996** | 687.2s | Stable, best observed accuracy |
| Dense ablation | `1/4` | 0.3981 | 1667.7s | Stable, worse efficiency |
| Dense attempt | `1/1` | — | — | Executor loss |

The current best operational point is `1/8`. It is a comparison baseline, not the final objective.

## What the present pipeline does and does not prove

### It does prove

- Spark/EMR can ingest, partition, sample, schedule, and execute many bounded CPU GNN jobs on a 111M-node graph.
- Community-local GNN training can run without collecting the whole graph into one driver process.
- More retained local structure improves accuracy until the current pipeline reaches an efficiency plateau.

### It does not yet prove

- Full-graph training.
- That every node or every edge contributes to the current objective.
- A single global GraphSAGE model: Phase 3 currently trains independent local models per community.
- Full message passing across community cuts.

The final paper must label the current results as **bounded sampled-community training**.

## Non-negotiable design rules

1. Never use `collect_list` for an unbounded community edge list.
2. Never collect large edge/node tables to the driver.
3. Apply every node/edge bound before Arrow/Pandas conversion.
4. Keep data access distributed; return only metrics, checkpoints, or bounded parameter tensors to the driver.
5. Separate correctness checks from throughput tests.
6. Use the official OGB masks for all comparable accuracy measurements.
7. Record node/edge coverage, cut-edge coverage, memory, shuffle bytes, S3 bytes, runtime, and cost for every experiment.

## Milestone 0 — Stabilize the experiment harness

### Goal

Make every run reproducible and make environment failures distinct from graph-training failures.

### Required work

- Preserve the per-worker NumPy/Pandas repair and health verification.
- Write a machine-readable run manifest containing config, Git revision, Spark resource settings, worker count, input row counts, and timings.
- Use a unique `EXPERIMENT_NAME` for each ablation, for example:
  - `papers100m-lpa-sage-e8`
  - `papers100m-lpa-sage-e1-lossless`
- Save per-stage Spark metrics: input bytes, shuffle read/write, spill bytes, completed/failed tasks, executor loss, and peak memory.
- Add assertions that Phase 3 receives non-empty train and test masks and that each output row has bounded node/edge counts.

### Acceptance gate

A repeated `1/8` run completes twice with no environment errors and weighted accuracy within a predeclared tolerance, such as ±0.5 percentage points.

## Milestone 1 — Lossless community edge processing at `1/1`

### Goal

Set `PHASE3_EDGE_SAMPLE_MODULUS = 1` without executor loss **while retaining all edges induced by the selected node set**.

This is not yet full-graph training because large communities are still node-bounded. It is the necessary first step: remove artificial edge thinning safely.

### Why the current `1/1` path fails

Even with node hashing, all retained edges are grouped into Python/Arrow payloads. Dense communities create large edge arrays, high shuffle pressure, and concurrent memory use. The current per-community cap in Python runs too late to protect Spark aggregation.

### Implementation design

Replace unbounded `groupBy(...).agg(collect_list(...))` edge construction with a storage-first bounded graph format:

1. In Phase 2 or a new Phase 2.5, write an explicit `community_id`, `chunk_id` edge table.
2. Split large communities into deterministic chunks before training. A chunk has a target number of seed nodes and a hard edge budget.
3. Write each chunk as Parquet/Delta files physically partitioned by `community_id` and `chunk_id`.
4. Include a bounded halo/neighbor list for each seed block, not an unbounded all-community edge list.
5. Create a compact manifest table containing only chunk metadata: `community_id`, `chunk_id`, node path, edge path, counts, and split counts.
6. Send manifests to Pandas UDFs. Each worker loads only its own bounded chunk directly from S3/local Parquet.
7. Train/evaluate each chunk; aggregate metrics by global test-node count, ensuring no duplicate test seeds.

### Correctness controls

For a selected community, compare Phase 2 edge count with the sum of its chunk edge counts. Report:

$$
\text{edge coverage} = \frac{\sum \text{chunk edges}}{\text{Phase 2 eligible edges}}
$$

Report duplicate seed-node rate, unassigned-node rate, and chunk-size percentiles.

### Acceptance gate

- `PHASE3_EDGE_SAMPLE_MODULUS = 1`
- no executor loss;
- no chunk exceeds its configured node or edge budget;
- 100% edge coverage of the chosen Phase 3 node subset;
- all selected test nodes are evaluated exactly once;
- observed runtime and resource counters are saved.

## Milestone 2 — Increase data coverage and accuracy fairly

### Goal

Move from a fixed 8K-node sample per large community toward full community-node coverage through bounded chunks.

### Experiment sequence

Run the same model/seed/masks with increasing chunk budgets:

| Experiment | Seed-node chunk budget | Edge budget | Edge modulus | Purpose |
|---|---:|---:|---:|---|
| A | 8K | 30K | 1 | Lossless edges for current node subset |
| B | 16K | 60K | 1 | More node and edge coverage |
| C | 32K | 120K | 1 | Scaling test |
| D | adaptive | adaptive | 1 | Full community-node coverage through multiple chunks |

Use multiple chunks for a large community instead of increasing a single Pandas group indefinitely.

### Accuracy mechanisms to evaluate separately

1. **Chunk ownership:** each target/seed node belongs to exactly one training/evaluation chunk.
2. **Halo nodes:** include sampled boundary neighbors as message-passing context but do not double-count them as seeds.
3. **Training-label sufficiency:** merge only training-starved chunks using deterministic rules; report label counts.
4. **Model synchronization options:**
   - independent local models: baseline only;
   - epoch-level FedAvg/weighted model averaging: a single global model approximation;
   - periodic community-representative/CaaN refinement: recovery of cut-edge information.
5. **Validation early stopping:** choose checkpoints using global validation metrics, not per-community averages.

### Valid comparisons

Compare global OGB test accuracy, weighted by unique test nodes:

$$
\mathrm{Accuracy} = \frac{\sum_{v \in V_{test}} \mathbb{1}[\hat y_v=y_v]}{|V_{test}|}
$$

Do not compare unweighted community averages with a full-graph baseline.

### Acceptance gate

- increasing node coverage produces a coverage/accuracy/runtime curve;
- no test-node duplication in the global metric;
- the selected protocol improves over the `1/8` sampled-community baseline or documents why it does not;
- every result includes coverage values, not only accuracy.

## Milestone 3 — Full-data-coverage distributed GNN training

### Goal

Train and evaluate a model whose mini-batches cover every training node and whose evaluation covers every OGB test node, using all graph edges as a distributed adjacency store.

### Recommended architecture: distributed neighbor-sampled GraphSAGE

Full-batch CPU GraphSAGE for Papers100M is not a realistic target. Use exact seed coverage with bounded neighbor-sampled computation blocks instead.

1. **Graph store tables**
   - `nodes_by_shard`: ID, features, label, official split, shard ID.
   - `adjacency_by_src_shard`: source ID, neighbor IDs or edge rows, shard ID.
   - `seed_manifest`: every train/validation/test node assigned exactly once to an epoch/batch shard.
2. **Batch construction**
   - Spark partitions the seed manifest into bounded batches.
   - Worker reads seed features and samples a fixed fanout, e.g. `[15, 10]`, from adjacency shards.
   - Worker fetches only sampled neighbor features.
3. **Model consistency**
   - Start with synchronous epoch-level parameter averaging. Workers train from the same epoch weights; aggregate weighted model deltas/gradients once per epoch.
   - The driver transfers only model tensors, never graph data.
   - If synchronization is too expensive, use a distributed parameter server or a GNN framework designed for distributed training; document that Spark remains responsible for data preparation and orchestration.
4. **Evaluation**
   - deterministically partition every OGB test seed once;
   - sample/evaluate all test seeds;
   - aggregate correct predictions and count, not per-community means.

### Full-coverage definitions to report

- node coverage: fraction of official train/validation/test seeds processed;
- edge availability: fraction of original edges present in adjacency storage;
- sampling coverage: fraction of available neighbors sampled per layer, expected to be less than 100% by design;
- cut-edge availability: fraction of cross-community edges still queryable by the graph store.

The final system can truthfully claim full-data coverage if all original node/edge records are available to distributed batch construction and every official seed is processed. It must not call neighbor sampling “full-batch training.”

### Acceptance gate

- 100% official test-seed coverage exactly once;
- 100% original edge records represented in adjacency storage;
- no driver collection of graph tables;
- bounded executor memory and no executor loss;
- global OGB accuracy calculated from total correct/total test seeds;
- end-to-end resource profile and cost report.

## Milestone 4 — Establish Spark’s contribution with baselines

### Research question

Does Spark provide a useful systems advantage for graph-scale data preparation and CPU GNN orchestration compared with a non-Spark implementation using the same model and machine budget?

### Baselines

1. Current bounded sampled-community training, `1/8`.
2. Lossless chunked community training, `1/1`.
3. Distributed neighbor-sampled global GraphSAGE with exact seed coverage.
4. Single-driver baseline on a dataset that fits memory, such as ogbn-arxiv or ogbn-products.
5. If available, a DGL/PyG distributed baseline using equivalent fanout, epochs, and splits.

### Measure

- preprocessing runtime: ingestion, partitioning, graph-store/chunk construction;
- training runtime and epoch time;
- input/shuffle/S3 traffic;
- executor peak memory and failure rate;
- accuracy, coverage, and variance;
- cluster size and estimated cost.

### Claims that are supportable

- Spark scales relational graph preparation and chunk scheduling without a driver graph collection.
- Spark can execute bounded CPU GNN workloads across distributed graph storage.
- The accuracy/runtime/coverage trade-off is quantified.

Avoid claiming that Spark is inherently a faster GNN kernel than specialized DGL/PyG systems unless measured directly.

## Immediate next implementation task

Do not run `1/2` or retry the current unbounded `1/1` aggregation.

Implement Phase 2.5 chunk materialization plus a manifest-driven direct-parquet loader, then run the Milestone 1 `1/1` experiment. The current `_get_dataset()` and `_load_communities_data_batch()` helpers in Phase 3 are a starting point, but they need a chunk manifest and bounded Parquet layout before they can replace `collect_list` safely.
