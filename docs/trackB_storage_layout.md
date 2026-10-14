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
