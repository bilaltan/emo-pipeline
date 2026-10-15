# Track B: graph-aware storage layout for LakeGRL

## 1. Question
Which physical layout of LakeGRL's Delta tables minimizes what Phases 2–3 read (bytes, files, row groups, requests),
and what does that layout cost to write and keep up to date? The output is a **layout rule**: from cheap graph and
partition statistics, pick a layout. The BigData paper lists this as future work.

## 2. What already exists (merged branch, all local benches)
| Bench | Measures |
|---|---|
| `experiments/zorder_bench.py` | Delta OPTIMIZE vs OPTIMIZE ZORDER(src, dst) on raw edges (arxiv, mag), with compaction as the control |
| `experiments/community_layout_bench.py` | whether Phase 2's layout lets Phase 3's per-unit reads skip data (hash vs community-clustered) |
| `experiments/partition_layout_bench.py` | p2 tables as range / range + small row groups / hive per community / hive merged (+ row groups) / naive hive: write cost, metadata, per-unit direct reads, full scan, point reads, checksums across layouts |
| `experiments/join_skipping_bench.py` | how much of the raw nodes table a unit reads through a normalized partition table, vs file size and layout |
| `experiments/global_join_vs_mv_bench.py` | materialized partition view vs one global join (broadcast and sort-merge) |
| `experiments/block_split_study.py` | splitting oversized units by id bits vs graph-aware (Louvain / Leiden / LPA) blocks |

Those runs used arxiv, mag and products from an older store. Reddit and the Track A stores (held-out links, real
features) were never run.

## 3. Plan (local; EMR only with Bilal's OK)
- **B1, baseline matrix.** partition_layout_bench and join_skipping_bench on `ogbn-arxiv-lp`, `reddit-lp`,
  `ogbn-products-lp` (Track A stores, Louvain partition). One JSONL per bench in `results/trackB/`.
- **B2, the layout rule.** Per dataset, the measured best layout, against graph statistics: community size
  distribution, share of tiny communities, halo / member ratio, feature bytes per row, unit count. A rule only counts if
  it predicts the best layout on a dataset it was not fitted on (leave-one-dataset-out over the three, plus WikiCS).
- **B3, maintenance.** Cost of keeping each layout under appends: OPTIMIZE / re-clustering after k% new rows. This links
  to LakeFresh's E4, which measures refresh I/O per layout.
- **B4, request cost.** Model S3 GET / byte cost from the measured files and row groups (the same price table as
  LakeReplay's `io_accounting`), labelled "modelled".

Order: B1 on arxiv (smoke, runner job `grl-trackB-layout-arxiv`) → reddit → products (largest; alone among big jobs).

## 4. B1 results (2026-10-13, local, Louvain units of the Track A stores, `tiny = misc`)
Per-unit reads through Phase 3b's loader: data selected by each unit's predicates, summed over units (median of 2
repeats). Rows, id sums and edges per community are identical across layouts on every dataset. Results are in
`results/trackB/partition_layout.jsonl`.

| dataset | units | range (default) | range + 1 MB row groups | hive (dir per unit) | hive_merged + row groups |
|---|---|---|---|---|---|
| ogbn-arxiv-lp | 392 (37 ≥ 100 members) | 3.75 GB, 4.7 s | 0.67 GB, 1.3 s | 0.18 GB, 0.8 s (760 files < 1 MB) | 0.49 GB, 0.9 s |
| reddit-lp | 25 | 3.30 GB, 5.6 s | 3.32 GB, 2.1 s | 3.30 GB, 4.0 s | 3.31 GB, 2.3 s |
| ogbn-products-lp | 1,063 | 79.6 GB, 84.5 s | 4.93 GB, 8.8 s | 2.68 GB, 7.1 s (2,025 files < 1 MB) | 3.76 GB, 5.3 s |

- The default range layout makes every unit read files that span many communities. That costs nothing on Reddit (25
  large units) and is 21× (arxiv) and 30× (products) more data than one directory per unit.
- Small row groups recover most of it inside the same files: 5.6× less on arxiv, 16× on products.
- One directory per unit reads the least but writes thousands of sub-MB files. Merging the minor units into shared
  directories, with small row groups, stays within 1.4× (products) to 2.7× (arxiv) of it with 20–30× fewer files.
- First shape of the layout rule: the cost of the default layout grows with the number of units and the share of small
  units. Unit count and size distribution are the inputs to test in B2 (leave-one-dataset-out, plus WikiCS).
- WikiCS (33 units, 18 tiny): range 0.18 GB vs hive 0.05 GB (3.6×).

## 5. B2 result: a row-group rule for the range layout (2026-10-13)
In Phase 2's range layout the number of files equals the number of major communities, so files are shared by many units.
A unit then reads every row group that overlaps its id range. The row-group size is therefore the knob, and the sweep
varied it (0.25 / 1 / 4 / 16 MB) on arxiv, WikiCS and products (`results/trackB/rg_sweep.jsonl`). Bytes are selected
per unit, summed over units, relative to one directory per unit (hive, the minimum):

| dataset | units | table / units | 0.25 MB | 1 MB | 4 MB | 16 MB |
|---|---|---|---|---|---|---|
| ogbn-arxiv-lp | 392 | 0.44 MB | 1.6× | 3.7× | 9.7× | 20.9× |
| WikiCS | 33 | 1.53 MB | 1.1× | 1.4× | 2.5× | 3.4× |
| ogbn-products-lp | 1,063 | 2.40 MB | 1.20× (23 s) | 1.84× (8 s) | 4.25× (12 s) | 9.75× (24 s) |
| reddit-lp (B1) | 25 | 126 MB | – | 1.0× | – | – |

- **Rule:** row-group size ≤ table bytes / number of units keeps per-unit reads within 2× of the one-directory-per-unit
  layout. It does so without that layout's thousands of sub-MB files: the files stay as many as the major communities.
  The rule held on every dataset. Every setting at or under the bound stayed within 2× (arxiv 0.25 MB, WikiCS 0.25 / 1 MB,
  products 0.25 / 1 MB, Reddit), and every setting above it exceeded 2×.
- The excess grows roughly linearly with the row-group size: each unit reads about one extra row group at its edges.
- Much smaller row groups cut bytes further but cost time in per-row-group overhead (products: 1.20× bytes at 0.25 MB
  but 23 s, against 1.84× at 1 MB and 8 s). So the setting to use is the largest row group under the bound.
- LakeGRL writes Phase 2 with Parquet's default 128 MB row groups. On products that reads 30× the necessary bytes per
  unit (B1). Setting `parquet.block.size` from the rule is a one-line change in Phase 2.
- Next (B3 / B4): measure the rule on EMR / S3, where per-request cost adds to bytes (needs AWS approval), and check it
  under appends (LakeFresh E4 layouts).

## 6. Plan from 2026-10-13: predictive storage optimization (primary LakeGRL track)
Advisor's direction: *"while keeping the current LakeGRL structure, brainstorm and test predictive storage-optimization
methods for our GRL workload."*
- **Constraint (structure kept).** Phases 0–4 and their outputs stay as they are. Methods may only change *physical*
  choices: write options (row-group and file size, sort and cluster keys, partitioning, replication of rows) and
  reader options (prefetching, read order). Every method is checked to give identical per-unit rows, id sums and edges,
  as the layout bench already does.
- **Predictive** means the storage decision is taken *before* Phase 2 writes, from what Phases 1–2 already know (the
  community / unit sizes, the halo, the Phase 3b bin manifest). It is not tuned after the fact by trying layouts.
- **Metric.** Bytes, files, row groups and modelled GET requests read by Phase 3 / 3b per unit, plus wall time; write
  time and file counts as the cost side. Regret = bytes of the predicted choice / bytes of the best choice measured.

| Id | Method | What is predicted | Test |
|---|---|---|---|
| P1 | Workload capture | — | Per-unit read traces (files, row groups, bytes, order) on more stores: + Coauthor-CS / Physics (very wide rows), + LPA partitions of the existing stores (many small units), + K (tiny threshold) variants |
| P2 | Layout advisor | best layout, row-group and file size from unit count, size distribution, tiny share, bytes / row, halo ratio | leave-one-dataset-out regret; the B2 rule (row group ≤ table / units) is the baseline, against a small learned model (linear / GBT) |
| P3 | Halo replication | each boundary vertex's reuse (how many units read it) | replicate high-reuse halo rows into each unit's partition, fetch the rest from a shared table; bytes and requests vs the full materialized halo; a per-unit halo budget R from unit stats, with Track A's AUC as the constraint |
| P4 | Workload-ordered writes | Phase 3b's bin order (known from the manifest) | write p2 files in bin order, one or a few bins per file; files and row groups per bin vs range |
| P5 | Materialize vs join | cost of a materialized partition view vs a normalized table plus join | the existing global_join_vs_mv bench on all stores; does a stats-based rule pick the cheaper one? |
| P6 | Prefetch | next bins' row groups | overlap reads with training in Phase 3b's loader; wall time |
| P7 | Cost model | bytes / requests / time of any (layout, unit) | fit on P1–P5 runs; S3 request costs modelled (LakeReplay's price table); one EMR validation run once AWS spend is approved |

### Status (2026-10-13, night)
- **P1** queued: Coauthor-CS / Physics, then LPA units of WikiCS and arxiv. They wait for RAM behind LakeFresh's
  products E2 run.
- **P2, first pass** (`experiments/trackB/advisor.py`, `results/trackB/P2_advisor.md`). Cost model: a unit read costs
  a·bytes + b·row groups, and each unit reads about one extra row group per table. That gives
  **s\* = √(β · table / (2 · units))**, where β = b / a is the per-row-group overhead in bytes, a property of the storage
  environment (local: about 0.4–0.8 MB). With β fitted on the other datasets (leave one dataset out), the √ rule and
  the B2 rule pick the same measured size on all four stores. Both are within 1.0–1.2× of the best wall time, and
  Phase 2's default 128 MB row groups are 2.6–9.9× off.
  - The data cannot yet separate the two rules. The Reddit sweep (4 / 16 / 64 MB) decides it: B2 allows up to
    126 MB there, while the √ rule predicts about 5 MB.
  - Reddit already shows an effect outside the model. 128 MB row groups take 5.6 s against 2.1 s at 1 MB, for the same
    bytes, probably because of less parallel decode.
  - Repeats of the same setting differ by up to 30% (arxiv 1 MB: 1.33 s vs 0.95 s), so wall times within about 1.3× are
    ties. The 0.5 / 2 MB bracket runs are queued.
- **P3, step 1** (`experiments/trackB/halo_reuse.py`, `results/trackB/P3_halo_reuse.md`). This **corrects the
  direction in the table above.** Phase 2 materializes the halo per unit, so a boundary vertex in r units' halos is read
  r times. A hot table shared by B readers costs B reads per vertex. So the read-optimal rule is: **replicate halo
  vertices with reuse ≤ B, and share the rest.** It is decidable before Phase 2 writes, and per-unit inputs stay
  identical. Rows read relative to the full halo, at B = 2 / 4:
  - Reddit: 0.47 / 0.66 (halo reuse median 4, max 24; Stage 3a alone is 0.19);
  - products: 0.75 / 0.88;
  - arxiv: 0.87 / 0.96.

  B is the number of readers that each load the hot table once: nodes, if it is cached on node-local disk; cores, if
  every Python worker loads it. At B ≥ 16 the saving is gone. Next: a bench that writes the hybrid layout and measures
  real bytes and wall time per unit, the identical-rows check, and the combination with Track A's halo budget R.

### Status (2026-10-14)
- The 01:34 Windows-update reboot stopped the runner. Restarted 10:51; P1 Coauthor is running again.
- **P4.** In every B1 store, each Phase 3b bin holds exactly one unit. Phase 3b makes one bin per unit up to 2000
  units (`num_bins = n if n <= 2000`). So "write in bin order" changes nothing there: hive is already
  one-directory-per-bin. P4 is measurable only above 2000 units, where 1000 bins hold several units each. It runs on
  the LPA partitions (P1) and on arxiv with `tiny = keep`. New layout `hive_bin` (layout bench): `storage_part` = the
  unit's Phase 3b bin, so a bin's task reads one directory. Compared against range_rg and hive_merged_rg at 1 MB row
  groups → `results/trackB/P4_bin_layout.jsonl`.
- **P1 Coauthor-CS / Physics done** (wide rows, 22–28 units). P2 advisor rerun on 6 stores: on both new stores, the B2
  rule and the √ rule each pick 0.25 MB with regret 1.00. Phase 2's default 128 MB is 2.1× / 3.3× slower. The rules
  still agree everywhere, and Reddit's 4 / 16 / 64 MB sweep is what can separate them. These Coauthor walls ran next
  to LakeFresh's CPU job, so they are noisy.
- **P2 decided by Reddit's sweep** (1 / 4 / 16 / 64 / 128 MB: 2.1 / 2.5 / 3.1 / 4.4 / 5.6 s). The B2 rule allows
  64 MB → regret 2.04. The √ rule picks 4 MB → 1.15. Over 8 stores (incl. 2 LPA): **√ rule max regret 1.46,
  B2 2.04, default 128 MB 1.5–9.9×.** The √ rule is the P2 predictor. Its worst case is WikiCS/LPA (0.25 vs a best of
  1 MB, at 0.14 MB per unit).
- **P3 hybrid halo bench done** (`results/trackB/halo_hybrid.jsonl`; identical unit rows everywhere). Hybrid vs full
  halo, at B = 1 / 2 / 4 readers:

  | store | bytes read | bytes stored | write time | local wall |
  |---|---|---|---|---|
  | Reddit | 0.35 / 0.48 / 0.66 | 0.35–0.46 | 0.34–0.44 | 0.93–1.07 |
  | products | 0.75 / 0.83 / 0.92 | 0.63–0.77 | 0.60–0.73 | 0.96–1.02 |
  | arxiv | 0.92 / 0.96 / 0.99 | 0.75–0.91 | 0.56–0.70 | 1.08–1.28 |

  Bytes and write time follow step 1's prediction. Local wall time does not move, because local disk plus the page
  cache hides the bytes. The saving has to show on S3, which the P7 cost model or an EMR run must check.
- **P4: the test missed its regime.** The LPA partitions have 367 (WikiCS) and 1123 (arxiv) units, and arxiv with
  `tiny = keep` has 476. All are below 2000, so `hive_bin` is again one directory per unit, i.e. hive. A side result
  from the same runs: with many small units, per-unit directories beat range_rg (1 MB): arxiv/LPA reads 234 vs
  1175 MB and takes 6.6 vs 27.5 s. hive_merged_rg takes 19.4 s. So layout choice belongs in the advisor, not only the
  row-group size. A real P4 test needs > 2000 units (products with `tiny = keep`, or LPA on products / Reddit).
- **P5 first pass** (`results/trackB/P5_join_vs_mv.jsonl`, cap none, identical checksums). MV vs global join wall
  time: WikiCS 1.35 vs 2.16 s, Coauthor-CS 5.1 vs 5.2, Coauthor-Physics 13.9 vs 11.0, arxiv 20.7 vs 14.5. Phase 2
  writes the normalized table about 1.5× faster. **Confounded:** MV used the default 128 MB row groups and read 3.95 GB
  for a 172 MB table on arxiv. Rerun queued with the √-rule row group (`--rg-mb`, `grl-p5-join-rg`), and Reddit at
  128 and 4 MB.
- **P5 with the √-rule row group** (`grl-p5-join-rg`, cap none, checksums identical). MV vs best join:
  WikiCS 1.28 vs 2.19 s, Coauthor-CS 3.50 vs 4.99, **Coauthor-Physics 5.54 vs 9.18 (flipped: it was 13.9 vs 11.0 at
  128 MB)**, arxiv 18.0 vs 14.4. With the right row group, MV wins on 3 of 4 stores; arxiv (392 units) still favours
  the join. MV there still reads 803 MB of node and edge chunks, against the join's 37 MB from pyarrow plus its Spark
  scan. The halo ratio does not predict the winner (WikiCS: materialized table 3.7× raw, and MV wins; arxiv 2.0×,
  and the join wins), so a rule needs more stores. Reddit (128 and 4 MB) runs next.
- P5 initially queued: the join-vs-MV bench (new `--tiny` flag) on WikiCS, Coauthor-CS / Physics, arxiv and Reddit (misc),
  → `results/trackB/P5_join_vs_mv.jsonl`.

Order: P1 → P2 (uses B1 / B2 data now) → P3 → P4 / P5 → P6 / P7. Output: `results/trackB/*.jsonl`, generated tables, and
an EDBT / VLDB-experiments-style write-up. Track A (link prediction, gate GO) is done and goes into the journal version
unchanged.
