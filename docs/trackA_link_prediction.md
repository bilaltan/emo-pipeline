# Track A: boundary-aware link prediction in LakeGRL

Status: design, 2026-10-12. Local only (WSL Spark + this machine); datasets: Reddit and ogbn-products.

## 1. Problem

LakeGRL trains one model per community on intra-community edges only. The BigData paper names link prediction as
the main limitation: once cross-community edges are cut, the macro-context recovers little of the link AUC.

Two facts about the current code shape the plan:

1. **Phase 3 never scores a cross-community pair.** Positives and negatives are both drawn from members of one
   community (`phase3_training.py`, `sample_negative_edges_local`). Halo vertices are message-passing context only.
   So the Stage 3a link AUC answers "which pairs inside this community are linked", not the full-graph question.
2. **Phase 4 uses a different split.** It runs `RandomLinkSplit` on the full graph, so the full-graph and LakeGRL
   AUCs are computed on different test edges. Since d61f10d, Phase 4 also reports a partition-matched variant, but
   the positives still differ.

Without one shared test set, no method comparison on links is valid. The first deliverable is therefore an
evaluation protocol, and the methods come second.

## 2. Evaluation protocol (shared by every method)

- **Held-out edges before Phase 1.** From the undirected edge set, sample val and test positives (default
  50,000 each, seeded) and remove them from the `edges` table *before* community detection. Phases 1–3 and the
  full-graph baseline all see only the training graph. Store the split in a `link_split` table
  (`src, dst, split, label, neg_scheme`).
- **Stratify by community after Phase 1:** each positive is `intra` (π(u) = π(v)) or `cross`. Report both strata
  plus their union (weighted by the true stratum sizes).
- **Two negative schemes** per positive (u, v), with true edges rejected:
  - `global`: (u, w) with w uniform over all vertices, the standard full-graph protocol.
  - `matched`: w drawn from v's community, so a model cannot win just by learning "same community ⇒ linked".
- **Metrics:** ROC-AUC per stratum and scheme; MRR with 100 negatives per positive as a secondary metric.
- **Cost:** for every method, the extra bytes read from Delta, extra rows shuffled, and extra training time
  relative to Stage 3a.
- **Coverage:** the fraction of cross test pairs a method can score at all. A method that cannot score a pair
  falls back to the feature-cosine score, and that is reported.
- 3 seeds (split seed fixed, model seeds 0–2), mean ± std.

## 3. Methods

| Id | Method | Shared space for cross pairs? | Extra cost |
|---|---|---|---|
| M0 | Full-graph GraphSAGE, link objective (upper bound) | yes | needs the whole graph (not LakeGRL) |
| M1 | Stage 3a as today: per-community models, members only; cross pairs fall back to feature cosine | no | none |
| M2 | **Halo scoring:** score (u, v) with u's community model when v is in that community's 1-hop halo (both directions, averaged); train with cut edges whose halo endpoint is present as positives | partial (within the halo) | halo rows, already materialized when `EXPAND_BOUNDARY_NODES = True` |
| M3 | **Macro-space decoder:** decoder on Phase 3b's shared macro embeddings + raw features, trained on training cut edges | yes | the macro graph (already built) |
| M4 | **Aligned local models:** every community model is initialized from one shared base model (Phase 3 has a driver warm start) and its encoder is regularized toward it, so local spaces stay comparable | approximately | one base model broadcast |
| M5 | **Cut-edge decoder (late fusion for links):** after Stage 3a, an MLP on `[h_local ‖ h_macro ‖ x]` of both endpoints, trained only on training cut edges | yes (via h_macro and x) | one pass over E_cross |

M2 and M5 are the main candidates: M2 uses data LakeGRL already stores, and M5 mirrors the paper's late fusion for
nodes. M3 and M4 are ablations that explain *why* one of them works.

## 4. Hypothesis and gate

- **H-A:** most of the M1 → M0 AUC gap sits in the `cross` stratum. Restoring boundary context (M2 or M5) closes
  ≥ 50% of the union-AUC gap on both datasets under both negative schemes, at ≤ 2× Stage 3a's bytes read.
- **Go** (write it up as a contribution): H-A holds on Reddit and ogbn-products.
- **Otherwise:** report the measured gap decomposition (intra vs cross, coverage) as a characterization result.
  That is still useful for the journal version and for Track B.

## 5. Implementation steps (local, kept small)

1. **Environment:** micromamba env `lakegrl` in WSL (Python 3.11, OpenJDK 17, PySpark 3.5.3, delta-spark 3.2.0,
   torch, PyG). The Delta store lives on the WSL file system (`LAKEGRL_STORE`), not on `/mnt/c`.
2. **Ingestion:** add Reddit (`C:\data\raw\reddit`) and ogbn-products (`C:\data\raw\ogb`) to
   `scripts/run_local_pipeline.py`, plus a `--link-split` option that writes the training graph and `link_split`.
3. **Phases 1–2 once per dataset** (Louvain, K = 100, halo on), so every method reads the same partition tables.
4. **Evaluation harness** `experiments/trackA/`: loads `link_split` and the partition tables and scores any method
   with the protocol above; one JSON result per run, appended to `results/REGISTRY.md`.
5. **Methods in order:** M1 and M0 (to measure the gap and its intra/cross split), then M2, then M5, then M3 and M4.
   Each new method first runs on a small community subset (smoke), then on the full dataset.

Run sizes: Reddit (233K nodes, 57M undirected edges) and ogbn-products (2.45M nodes, 62M edges). Phase 1 Louvain on
the driver is the biggest local job. If it does not fit in WSL's memory, use the Leiden/igraph path or a coarser
resolution, and record that in the doc, rather than running anything larger.

## 6. Run log

- 2026-10-12, set before any test result was seen: every method trains 100 full-batch epochs, lr 0.005, 65,536
  positives per step, hidden 256, dropout 0.5, seeds 0-2. One shared budget for all methods; no per-method tuning in v1.
- Reddit (`reddit-lp`): 25 Louvain units; the 1-hop halo adds 1.0M rows (4.3× the members). 18.6% of the test
  positives cross units, and every cross positive is coverable by the halo (coverage 1.0).
