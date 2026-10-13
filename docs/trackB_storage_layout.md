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
