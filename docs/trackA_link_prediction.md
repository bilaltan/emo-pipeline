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
| M5 | **Late fusion for links:** Stage 3a (members only) gives h_local; h_macro = parameter-free 2-hop propagation of mean member features over the macro graph; one shared decoder on `[h_local ‖ h_macro ‖ x]` of both endpoints, trained on all training edges, scores every pair | yes (via h_macro and x) | the cut-edge list and one macro row per unit; 1× Stage 3a rows |

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
- ogbn-products (`ogbn-products-lp`): 4,389 Louvain communities → 1,063 units (tiny ones in shared misc bins); the
  halo adds 1.32× the members (2.32× the rows of Stage 3a). 9.5% of the test positives cross units. Phases 0–2 took
  28 min locally.
- 100-epoch main runs, 3 seeds (T_A.md): cross-unit AUC (global / matched negatives):
  - Reddit: M1a 0.704 / 0.684, M1b 0.868 / 0.842, M2 0.912 / 0.879, M0 0.853 / 0.784.
  - Products: M1a 0.532 / 0.546, M1b 0.955 / 0.921, M2 0.958 / 0.923, M0 0.940 / 0.854.
  - M0 scoring below M1b raised the question of whether 100 epochs under-trains the full graph.
- Reddit validation sweep, seed 0 (cross AUC, global negatives):
  - M0: 0.838 (100 ep), 0.868 (300), 0.873 (1000).
  - M1b: 0.861 (100), 0.898 (300). M2: 0.907 (100), 0.935 (300).
  - Chosen on validation: M0 1000 epochs, M1b / M2 300 epochs. Tested on 3 seeds (tuned test).
- Tuned Reddit test, M0 at 1000 epochs: seeds 0 and 2 reach cross AUC 0.873. Seed 1 collapses (intra 0.857, cross 0.682;
  20261012-223714-M0-s1).
  - That record is flagged `-dirty`: the harness took the git hash when it wrote each record, and a commit landed
    during the run. Since 535c575 the hash is taken at process start.
  - The seed is rerun clean, and M0 at 300 epochs (3 seeds) is added as a stability check. The validation-chosen
    1000-epoch budget stays the primary row.
- I/O side of H-A: M1b / M2 read 5.30× Stage 3a's rows on Reddit and 2.32× on products, so the full halo fails the
  "≤ 2× bytes" clause on both.
  - `--halo-budget R` caps each unit's halo at R × members, keeping the halo vertices with the most edges into the
    unit; R = 1 is the gate's 2×.
  - The Reddit validation sweep runs R = 1, 0.5 and 0.25 for M1b and M2.
- M5 (late fusion for links) implemented. Stage 3a (members only, 1× rows) gives h_local. h_macro is a parameter-free
  2-hop propagation of mean member features over the macro graph (units, cut-edge counts). One shared decoder on
  `[h_local ‖ h_macro ‖ x]` of both endpoints, trained on all training edges, scores **every** pair.
  - An earlier version scored intra pairs with the unit decoders and the rest with the shared one. On an arxiv smoke run
    that dropped intra AUC from 0.88 to 0.58 with identical intra scores, because global negatives of intra positives are
    mostly cross pairs scored on another logit scale.
- ogbn-arxiv (`ogbn-arxiv-lp`) added as the third dataset: 476 Louvain communities → 392 units; halo 0.98× the members.
  Phases 0–2 took 2 min.
- Tuned Reddit test, complete (test split, 3 seeds, budgets chosen on validation; cross-unit AUC, global / matched):
  - M2 (300 epochs): 0.935 / 0.905. M1b (300 epochs): 0.903 / 0.873.
  - M0 (1000 epochs): 0.871 / 0.810. Its seed 1 is the clean rerun 20261013-002456-M0-s1 (0.866 / 0.804).
  - M0 at 300 epochs: 0.870 / 0.807, so the full graph plateaus at ≈ 0.87.
  - So with tuned budgets the halo still beats the full graph on cross-unit links, by +0.064 / +0.095 for M2.
  - The excluded seed-1 record (0.682) used the same seed and code but did not reproduce. Full-graph training at 1000
    epochs is nondeterministic on the GPU and collapsed once in five runs. It is not counted (its record is `-dirty`)
    and is mentioned here so the collapse is on record.
  - The M1b / M2 records of this run show the commit that was HEAD when each record was written (535c575 … 40a3ac5).
    All of them ran 3f0c583's harness, whose M1b / M2 code is unchanged since.
- ogbn-arxiv, 100-epoch protocol (test, 3 seeds; cross AUC global / matched): M0 0.849 / 0.723, M1a 0.590 / 0.556,
  M1b 0.825 / 0.767, M2 0.826 / 0.753.
  - Gate table: M1b and M2 close 111% / 142% and 110% / 109% of the Stage 3a → M0 gap on the "all" AUC, at 1.92× Stage
    3a's estimated bytes. **H-A holds on arxiv at the pre-registered protocol.**
- Arxiv validation sweep (seed 0): M0 keeps improving with epochs (cross 0.860 → 0.873 → 0.891 at 100 / 300 / 1000), so
  100 epochs under-trains it here. M1b / M2 are best at 300.
- Arxiv tuned test (3 seeds; cross / all AUC, global / matched):
  - M0 (1000 epochs): 0.874 / 0.756, 0.955 / 0.833.
  - M1b (300): 0.828 / 0.781, 0.949 / 0.858. M2 (300): 0.825 / 0.763, 0.946 / 0.834.
  - With tuned budgets the full graph wins on global negatives (+0.046 cross), and the halo wins on matched negatives
    (+0.025 cross and all). Cut-edge supervision does not help on arxiv.
  - M0 seed 2 at 1000 epochs is again low (0.853 / 0.716 vs 0.883–0.888 / 0.773–0.779 for seeds 0–1): full-graph training
    at 1000 epochs is unstable on arxiv as well as on Reddit.
  - So the arxiv gate result depends on the budget: at the protocol it holds, and tuned it is a draw that depends on
    the negative scheme. Both go into the write-up.
- Products validation, M0 at 300 epochs (seed 0): cross 0.940 / 0.862, about the 100-epoch test level (0.940 / 0.854).
  Products' full halo reads an estimated 1.79× Stage 3a's bytes (2.32× its rows), so **H-A holds on products** for M1b
  and M2 at the protocol (gap closed +105% / +235% and +106% / +231%).
- Reddit halo-budget curve (validation, 300 epochs, seed 0; M2 cross AUC global / matched, estimated bytes):
  - full halo 0.935 / 0.908 (3.1× Stage 3a);
  - R = 1: 0.892 / 0.867 (2.11 GB);
  - R = 0.5: 0.858 / 0.831 (1.81 GB);
  - R = 0.25: 0.816 / 0.792 (1.62 GB).
  - For reference: Stage 3a ≈ 1.3 GB, full graph 0.873 / 0.810.
  - Keeping the halo vertices with the most edges into the unit loses AUC smoothly; at the gate's 2× limit M2 still
    beats the full graph on cross-unit links.
- Reddit at the protocol (test, 100 epochs, 3 seeds):
  - M2@R=1 closes 57% of the "all" gap on global negatives at 1.61× bytes and stays above Stage 3a on matched
    negatives (+0.026), so it **meets H-A**. M1b@R=1 closes 49% (no).
  - M5 (late fusion) closes 78% at 1.13× bytes (+0.016 vs Stage 3a on matched), so it **meets H-A**.
  - M3 (M5 without h_local) falls below Stage 3a (−132%, −0.121): the local embeddings carry M5. The same holds on
    arxiv, where M5 closes 94% / 137% at 1.04× bytes and M3 −302% / −284%.
- Gate bookkeeping: where M0 does not beat Stage 3a (Reddit, matched negatives: 0.773 vs 0.815) there is no gap to
  close; since 25ba8da a candidate must then at least not fall below Stage 3a.
- Open for the H-A decision: products M2@R=1 and M5 (queued). Reddit and arxiv already have a variant that meets.
- Products at the protocol (test, 100 epochs, 3 seeds):
  - The halo capped at R = 1 (1.53× bytes) meets H-A: M1b@R=1 +103% / +229%, M2@R=1 +104% / +225%; cross 0.950 / 0.912
    for M2@R=1 vs 0.958 / 0.923 with the full halo.
  - M5 meets H-A at 1.05× bytes (+69% / +132%) but is well below the halo on cross-unit links (0.815 / 0.700). M3 does
    not meet it (−58% / −792%).
- Products validation, complete (seed 0): M0 best at 300 epochs (0.940 / 0.862; 100: 0.935 / 0.852; 1000: 0.897 / 0.771).
  M1b best at 100 (0.9524 vs 0.9522 at 300). M2 best at 300 (0.957 / 0.928). Tuned test queued: M0 300 and M2 300 ×
  3 seeds; M1b's 100-epoch test already exists.
- **Gate decision: GO** (docs/trackA_gate_report.md). M2@R=1 meets H-A on Reddit and products (and arxiv) at the
  protocol, and M5 meets it on all three at about 1.1× Stage 3a's bytes.
