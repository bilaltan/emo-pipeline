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
- **2026-10-14 22:00: LakeGRL-only period** (Bilal). Added:
  - **Stats-pruned unit reads** (`utils.partition_layout.unit_dataset`, a reader option, rows unchanged). The range
    layout has no directory to prune, so every unit's scan visits every file. The helper gives each scan only the
    files whose `community_id` statistics can hold its units; each footer is read once per worker. First probe:
    arxiv per-unit reads 2.3× faster (0.46 → 0.20 s, 37 files). The layout bench now records `unit_pruned` and
    `p3b_bins_pruned` for every layout. The suspected cause of arxiv/LPA's 27.5 s vs hive's 6.6 s is being tested
    in `grl-prune-arxiv-lpa`.
  - **S3 emulator** (`experiments/trackB/s3_emu_bench.py`, P7). Every read call on a data file is one GET: it is
    counted and pays latency (20 ms) plus bytes at 80 MB/s. pyarrow issues its own call pattern, so the GET counts
    are real pyarrow behaviour, while the latency and bandwidth are assumptions. Runs are queued on arxiv,
    arxiv/LPA, Reddit and products for range (128 MB), range_rg (√ pick), hive and hive_merged_rg, with the
    plain and pruned readers. MinIO was the first choice, but downloading its binary was not permitted in this
    session, so a real S3 or MinIO check is still to do.
  - **Observation (Reddit P5, 128 MB):** `mv_direct` read 38.6 GB of column chunks from a 4.6 GB table. Reddit's big
    units are split into blocks (cap 10,000 rows), and each block re-reads its whole unit's row groups. A
    block-ordered write (sort within a unit by `id & (n_blocks - 1)`, which Phase 3b's block predicate uses) would
    let each block skip the others' row groups. Candidate P4b.
- **P5 Reddit done** (identical checksums; 25 units). The block cap decides the winner:
  - No cap (one task per unit): MV 22.3–22.7 s vs join 30.5–30.7 s, so **MV wins by 1.35×**. The row group
    (4 vs 128 MB) makes no difference, because Reddit's units are about 126 MB each.
  - Cap 10,000 rows (units split into blocks): MV 34.2 s (4 MB) / 41.2 s (128 MB) vs join 22.5 / 24.8 s, so **the
    join wins by 1.5–1.7×**. The cause is MV reading 38.3 GB of column chunks against 3.2 GB: each block re-reads its
    unit.
  - Phase 2 writes the materialized table in 118–135 s vs 96–100 s for the normalized one.
  - So a predictive MV-vs-join rule needs the block count per unit, which Phase 2 can compute from unit sizes and the
    cap. Block-ordered writes (P4b, above) should remove MV's re-reads and would change the answer again.
- **Pruning test done** (`grl-prune-arxiv-lpa`; arxiv/LPA misc, 1,123 units, rows identical to Spark). Per-unit
  read wall, 4 processes, local disk:

  | layout | plain | pruned |
  |---|---|---|
  | range (128 MB rg) | 29.9 s | 2.3 s |
  | range_rg 1 MB | 26.2 s | 1.7 s |
  | range_rg 0.25 MB | 27.1 s | 1.4 s |
  | hive | 4.6 s | n/a |

  The 27.5 vs 6.6 s gap in P4 came from unpruned file scans, not from the layout. Pruned range_rg is 3.4× faster
  than hive here, writes faster (9.4 vs 14.7 s), and has 266 files vs 2,246.
- **S3 emulator, arxiv and Reddit** (`results/trackB/s3_emu.jsonl`; GET = 20 ms + bytes at 80 MB/s, 4 processes,
  cold pass, rows identical everywhere). Wall / GETs / bytes read:

  | store | layout | plain | pruned |
  |---|---|---|---|
  | arxiv Louvain (392 units, 0.24 GB) | range 128 MB | 27.4 s / 1,557 / 4.1 GB | 26.6 s / 1,449 / 3.8 GB |
  | | range_rg 0.25 | 9.3 s / 1,152 / 0.81 GB | 10.9 s / 1,080 / 0.75 GB |
  | | hive | 11.3 s / 1,580 / 0.19 GB | same |
  | | hive_merged_rg | 9.5 s / 866 / 0.69 GB | same |
  | arxiv LPA (1,123 units, 0.24 GB) | range 128 MB | 224 s / 68,656 / 87 GB | 30.7 s / 3,316 / 2.0 GB |
  | | range_rg 0.25 | 185 s / 74,264 / 68 GB | 28.8 s / 3,335 / 1.9 GB |
  | | hive | 31.9 s / 4,496 / 0.30 GB | same |
  | | hive_merged_rg | 147 s / 55,978 / 52 GB | same (partitioned: not pruned) |
  | Reddit Louvain (25 units, 3.3 GB) | range / range_rg 4 / hive / hive_merged_rg | 16.2 / 14.2 / 15.3 / 14.4 s, about 3.3 GB each | no change |

  - pyarrow's `isin` scan does not skip row groups by their statistics here. arxiv/LPA hive_merged_rg reads 52 GB
    from a 0.24 GB table, because each unit reads its whole merged partition, and unpruned range reads 87 GB. P3's
    byte ratios are planned bytes from the row-group index, which assume that skipping. The emulator shows the
    reader has to do the skipping itself.
  - With file pruning, range_rg ties hive under latency (28.8 vs 31.9 s cold) and is 3.4× faster than hive locally.
  - Reddit has few, big units, so it is bandwidth-bound: every layout reads about the whole table, and wall times
    differ by at most 13%.
- **Production change:** Phase 3b's loaders (`_load_communities_data_batch`, `_load_unit_edges`) now call
  `unit_dataset` on every scan. `test_partition_layout`, `test_partition_join`, `test_unit_dataset` and `test_s3_emu`
  pass. The helper now checks only the first fragment of a partitioned dataset. The 4.6 → 5.1 s cost on hive came
  from listing all 2,246 fragments on every scan; this has not been re-measured.
- **Row-group pruning** (bcc8d19). `unit_dataset` now also skips row groups (`ParquetFileFragment.subset`), and for
  hive / hive_merged it first prunes directories with the scan predicate. Rerun on arxiv/LPA through the S3
  emulator (`grl-emu-arxiv-lpa-rgprune`; cold wall / bytes, warm bytes; rows identical):

  | layout | plain | file pruning | row-group pruning |
  |---|---|---|---|
  | range_rg 0.25 | 185 s / 68 GB | 28.8 s / 1.9 GB | 24.0 s / 0.59 GB (warm 0.54) |
  | hive_merged_rg | 147 s / 52 GB | not applied | 22.1 s / 0.55 GB (warm 0.52) |
  | hive | 31.9 s / 0.30 GB | not applied | 33.2 s / 0.30 GB |

  Warm bytes now match the planned bytes from the row-group index (range_rg: 536 MB planned, 536 MB read), so
  P3's byte model holds once the reader skips row groups. Under latency both √-row-group layouts now beat hive
  (22–24 s vs 33 s).
- **P4 done in its regime** (`grl-p4-bins-arxiv-lpa-keep`, 162 min; arxiv/LPA `tiny = keep`: 16,250 units in 1,000
  Phase 3b bins of about 16; rows identical in all layouts; local disk, 4 processes). This run predates
  bcc8d19, so "pruned" is file-level only and leaves the partitioned layouts unchanged. Read wall / planned bytes:

  | layout | write | files | per unit | per bin | per bin, pruned |
  |---|---|---|---|---|---|
  | range_rg 0.25 | 8.8 s | 246 | 558 s / 5.0 GB | 38.9 s / 4.75 GB | 8.0 s |
  | hive | 102 s | 30,582 | 661 s / 0.24 GB | 47.8 s / 0.24 GB | 47.4 s |
  | hive_merged_rg | 11.8 s | 328 | 393 s / 5.24 GB | 24.8 s / 4.69 GB | 24.9 s |
  | hive_bin | 18.1 s | 2,387 | 73 s / 2.38 GB | **5.6 s / 0.23 GB** | 5.8 s |

  - When bins hold many units, the bin-aligned layout wins. Each bin reads its own directory: 0.23 GB in total vs
    4.7 GB for range_rg and hive_merged_rg. With 0.25 MB row groups and a median unit of 4 rows, every tiny unit
    costs a whole row group. hive_bin is 7× faster than range_rg per bin and 1.4× faster than file-pruned
    range_rg.
  - The costs: 2× range_rg's write time and 10× its files. Hive's 16,250 directories cost 102 s to write and are
    slowest to read.
  - Reading per unit instead of per bin is 10–14× slower in every layout, so Phase 3b's binning matters more than
    the layout here.
  - Open: per-bin reads of range_rg with row-group pruning were not measured locally. The products P4 run (now
    running) uses bcc8d19.
- **P4 products** (`grl-p4-bins-products-keep`, 39 min; Louvain `tiny = keep`: 4,389 units, median 2 rows, largest
  357,519; 1,000 bins of about 4.4 units; rows identical; with row-group pruning, bcc8d19). Wall / planned bytes:

  | layout | write | files | per unit | per bin | unit pruned | bin pruned |
  |---|---|---|---|---|---|---|
  | range_rg 1 MB | 95.5 s | 150 | 38.4 s / 9.2 GB | 12.4 s / 6.3 GB | 9.5 s | **6.3 s** |
  | hive | 245 s | 8,778 | 54.7 s / 2.7 GB | 19.0 s / 2.7 GB | 35.6 s | 14.8 s |
  | hive_merged_rg | 124 s | 156 | 13.1 s / 8.2 GB | 7.0 s / 6.1 GB | 8.9 s | 6.5 s |
  | hive_bin | 93.5 s | 2,289 | 23.8 s / 2.8 GB | 11.5 s / 2.7 GB | 20.5 s | 10.1 s |

  - Here hive_bin loses: 10.1 s against 6.3–6.5 s for pruned range_rg and hive_merged_rg. But the comparison is
    confounded, because hive_bin (like hive) kept Spark's 128 MB row groups while the other two use 1 MB.
    Products' bins are dominated by a few large units, which then decode as single big row groups. Added
    `hive_bin_rg` (a411836) and queued reruns on products and arxiv/LPA against range_rg with row-group pruning
    (`grl-p4-binrg-*`).
  - The predicate-based directory pruning in `unit_dataset` makes hive's per-unit reads 1.5× faster (54.7 →
    35.6 s): each scan no longer evaluates all 8,778 fragments.
- **P4 rerun with hive_bin_rg and row-group pruning** (a411836 + bcc8d19; rows identical). Per-bin reads, pruned:
  - Products: range_rg 1 MB **6.3 s** (write 97 s, 150 files) vs hive_bin_rg 7.8 s (write 140 s, 2,280 files).
    The 128 MB hive_bin took 10.1 s.
  - arxiv/LPA (`grl-p4-binrg-arxiv-lpa`): range_rg 0.25 MB **3.2 s** (8.0 s with file-level pruning, 34.8 s
    plain; write 8.6 s, 123 files, 4.75 GB selected) vs hive_bin_rg 4.1–4.7 s (5.4 s unpruned; write 19.0 s,
    1,191 files, 0.24 GB selected). hive_bin_rg selects 20× fewer bytes but pays 10× the files and loses.
  - P4's bin-aligned layout won only against a reader that did not skip row groups. With row-group pruning,
    range at the √ row group is fastest on both stores, with the fewest files and the fastest write.
- **S3 emulator, products** (`grl-emu-products`; Louvain misc, 1,063 units, 2.7 GB; GET = 20 ms + 80 MB/s; Phase 3b
  bins, cold; rows identical):

  | layout | plain | pruned (row groups) |
  |---|---|---|
  | range 128 MB | 389 s / 85 GB | 383 s / 79 GB |
  | range_rg 0.5 | 55.3 s / 10.1 GB | 46.0 s / 3.8 GB |
  | hive | 48.2 s / 2.7 GB | 48.9 s / 2.7 GB |
  | hive_merged_rg | 49.8 s / 8.8 GB | **44.0 s** / 3.3 GB |

  - Phase 2's default (128 MB row groups) is 8.4× slower than the √ row group with pruning (389 vs 46 s). A
    128 MB row group holds many units, so pruning cannot help it.
  - The three tuned layouts land within 10% of each other (44–49 s).
  - Gain of the best √-row-group read over the default range under latency:
    - products 8.4×
    - arxiv/LPA 9.3× (224 → 24.0 s)
    - arxiv/Louvain 2.9× (27.4 → 9.3 s)
    - Reddit 1.14× (25 big units, bandwidth-bound)
  - The best √-row-group read is within 0–18% of the best layout on every store.
- **P5 products done** (`grl-p5-join-products`, 131 min; Louvain misc, 1,063 communities, 1,815 units with the
  10,000-row cap (50 communities split); 2 repeats each; checksums match; the Phase 3b loaders use row-group
  pruning, bcc8d19). Wall s, MV vs best join, with planned MV node+edge reads:

  | row group | cap | MV | join | winner | MV planned reads |
  |---|---|---|---|---|---|
  | 0.5 MB (√) | none | 53.2–53.5 | 58.6–59.6 | **MV 1.11×** | 10.5 GB |
  | 0.5 MB (√) | 10k | 124.7–125.0 | 86.4–87.1 | join 1.44× | 84 GB |
  | 128 MB | none | 116.8–119.0 | 68.4–70.4 | join 1.7× | 88 GB |
  | 128 MB | 10k | 191.6–195.4 | 94.7–96.0 | join 2.0× | 214 GB |

  - The √ row group makes MV 2.2× faster (117 → 53 s) and flips the winner when there are no blocks. With blocks,
    the join wins at either row group, because every block re-reads its unit (84 GB of reads vs 14 GB of edges plus
    a 3.4 GB shuffle for the join).
  - Phase 2: the normalized table is written 3–8% faster (517–533 s vs 534–571 s) and stores 26 MB of nodes against
    2.26 GB for the materialized one; both also keep the 0.99 GB raw table.
  - **P5 rule so far, across 6 stores:**
    - √ row group and no blocks: MV wins on WikiCS, Coauthor-CS, Physics, Reddit (1.35×) and products (1.11×).
      arxiv is the exception: join 14.4 s vs MV 18.0 s.
    - Units split into blocks: the join wins (Reddit 1.5–1.7×, products 1.44×).
    - 128 MB row groups favour the join.
    - Block-ordered writes (P4b) are the open way to remove MV's block re-reads.
- **End to end, arxiv Louvain** (`grl-e2e-arxiv-rg`, 116 min; `run_local_pipeline.py` Phases 2–3b, SAGE 200 epochs,
  412 units, the same Phase 1 communities, Phase 3b loaders with row-group pruning; results/trackB/e2e.jsonl).
  Default 128 MB vs √ 0.25 MB row groups:

  | | Phase 2 | Phase 3 | Phase 3b | total |
  |---|---|---|---|---|
  | 128 MB | 25.3 s | 1,451 s | 1,874 s | 3,391 s |
  | 0.25 MB | 26.5 s | 1,511 s | 1,914 s | 3,489 s |

  - Same accuracy in both (Stage 3b node acc 0.6445, link AUC 0.8765). The √ run is 3% slower, within run-to-run
    noise.
  - Training dominates. The emulated cold reads of the same units take 27.4 s (128 MB) and 9.3 s (√) under S3
    latency. That is 1.5% and 0.5% of Phase 3b, so on arxiv the row group would not show end to end on S3 either.
  - So Track B's read savings matter end to end only where reads are a real share of a unit's time: many or large
    units, wide rows, few epochs, or faster (GPU) training. Next, products (1,063 units, emulated reads 389 s
    default vs 46 s √) measures that share on the store with the largest read saving (`grl-e2e-products`).
- **Why the plain scan over-reads** (`experiments/trackB/pyarrow_prune_probe.py`, pyarrow 17.0.0; read-only, seconds).
  Two separate causes:
  - **`isin` is not checked against row-group statistics when a row group holds a range of communities.** On a
    toy file, `isin` keeps 40 of 40 row groups, while `==` and a range predicate keep 1. With one community per row
    group, `isin` keeps 1 too, which is why an earlier toy check seemed to prune. Phase 2's row groups hold ranges,
    so `isin` never prunes there.
  - **Every scan opens every file.** pyarrow reads a 64 KB footer tail from each file on every scan, because a
    dataset built from paths keeps no footers between scans.
  - arxiv Louvain p2_nodes at the √ row group (37 files, 715 row groups), one median unit (4 rows):
    - `isin` plain: 39 read calls, 3.97 MB (37 footers = 2.4 MB, plus row groups 0–5 where only 1 holds the unit).
    - OR of `==` plain: 2.74 MB (the row groups are pruned, the footers are not).
    - `unit_dataset`: 1 read call, 0.25 MB, which equals the matching row group.
  - So the earlier statement stands, now with its mechanism. A fairer plain baseline is the OR-of-`==` predicate,
    and `unit_dataset` still beats it 11× on this unit, because footers are cached and only matching files are
    opened. The benches above compared against `isin`, so their plain columns include both causes.
- **P4 / P5 real runs queued:** arxiv/LPA with `tiny = keep` (16,250 units, about 16 per bin), products with
  `tiny = keep` (4,389 units), and P5 on products at the √-rule row group (0.5 MB) and at 128 MB.
- P5 initially queued: the join-vs-MV bench (new `--tiny` flag) on WikiCS, Coauthor-CS / Physics, arxiv and Reddit (misc),
  → `results/trackB/P5_join_vs_mv.jsonl`.

### Status (2026-10-16): storage-layer gains only
Bilal: end-to-end training time is not the target; a win at the storage layer is. So the metrics are the storage
layer's own: bytes and GETs read, read time under the S3 emulator, bytes stored, write time. No more full pipeline
runs (`grl-e2e-products` is on hold). Queued, one emulator job at a time:
- `grl-p3-halo-emu`: P3's hybrid halo through the emulator (its byte saving was hidden by local disk).
- `grl-p4b-blocks`: P4b, block-ordered write. A `_blk` column (a node's Phase 3b block, `id & (n-1)`) and rows
  sorted by (community, `_blk`), so a block's reads skip the rest of its unit (`rg_resize.py --block-cap`). Reddit,
  products, arxiv at the √ row group, cap 10,000, against the same copy in Phase 2's order.
- `grl-rg-emu-sweep`: row group (0.125–16 MB; Reddit 1–64) × latency (5 / 20 / 80 ms) on products, arxiv/LPA,
  arxiv, Reddit. Does the best row group move with latency as the √ rule predicts?

- **P3 hybrid halo through the emulator** (`grl-p3-halo-emu`, 30 min; `results/trackB/halo_hybrid_emu.jsonl`; 1 MB
  row groups, GET = 20 ms + 80 MB/s, two cold repeats agree within 1%; unit rows identical everywhere; measured bytes
  equal the planned bytes). Hybrid vs full halo at B = 1 / 2 / 4 readers:

  | store | read time | bytes read | bytes stored | write time |
  |---|---|---|---|---|
  | Reddit | **0.43** / 0.55 / 0.74 (72 → 31 s at B = 1) | 0.35 / 0.48 / 0.66 | 0.35 / 0.37 / 0.46 | 0.35–0.47 |
  | products | 0.82 / 0.88 / 0.95 (130 → 106 s) | 0.75 / 0.83 / 0.92 | 0.63 / 0.68 / 0.77 | 0.58–0.73 |
  | arxiv | 1.00 / 1.00 / 1.02 | 0.92 / 0.96 / 0.99 | 0.75 / 0.79 / 0.91 | 0.6–0.8 |

  - The byte saving that local disk hid shows up as read time once reads cost bandwidth: Reddit 2.3× faster at
    B = 1, 1.35× at B = 4. Read time follows bytes read.
  - arxiv does not gain: its 400 GETs for 0.8 GB are latency-bound, and the halo saving is small there.
  - The saving shrinks as B grows (each reader loads the shared table once), as P3 step 1 predicted. B is the number
    of readers that each load it: nodes if it sits on node-local disk, worker processes otherwise.
- **P4b block-ordered write** (`grl-p4b-blocks`, 32 min; `results/trackB/p4b_blocks.jsonl`). The store's Phase 2
  tables (Louvain misc) copied at the √ row group, once in Phase 2's row order and once block-ordered
  (`rg_resize.py --block-cap 10000`). Phase 3b's block work items were read through the emulator (20 ms + 80 MB/s,
  4 processes, pruned reader). Per-block checksums are identical in both orders (same md5) and equal a plain read
  without the `_blk` conjunct. Cold pass, Phase 2 order → block order:

  | store | work items (units split) | read time | bytes read | GETs | bytes stored |
  |---|---|---|---|---|---|
  | Reddit (4 MB) | 183 (22 of 25) | 162.6 → **25.0 s (6.5×)** | 43.7 → 4.95 GB (8.8×) | 1,900 → 578 | +0.3% |
  | products (0.5 MB) | 1,815 (50 of 1,063) | 387.6 → **59.9 s (6.5×)** | 102.5 → 6.6 GB (15.5×) | 6,586 → 4,302 | +0.5% |
  | arxiv (0.25 MB) | 418 (12 of 392) | 12.0 → 10.1 s (1.2×) | 1.06 → 0.51 GB (2.1×) | 1,132 → 1,136 | +0.4% |

  - Without block order every block of a split unit reads the whole unit's row groups, because min / max statistics
    cannot decide `id & (n-1) == b`. With rows sorted by block and a `_blk` column, the implied `_blk == b`
    conjunct skips them. The block layout is decided before the write from unit sizes and the cap.
  - The gain grows with the share of rows in split units: large on Reddit and products, small on arxiv.
  - Cost: one int64 column (+0.3–0.5% stored) and a sort. The single-process pyarrow copy took 123 vs 42 s
    (Reddit) and 175 vs 49 s (products).
  - **In Phase 2's own Spark write path** (`grl-p4b-spark`, 41 min; `p4b_spark_write.py`,
    `results/trackB/p4b_spark_write.jsonl`; `repartitionByRange` + `sortWithinPartitions` as `write_layout` does,
    plain and blocked writes alternated 3 times, 4 cores; the blocked time includes counting community sizes). The
    Spark-written tables were then read as above (`p4b_blocks.jsonl`, labels `spark-rg*`), md5 identical:

    | store | write, median of 3 | bytes stored | read time (cold) | bytes read | GETs |
    |---|---|---|---|---|---|
    | Reddit (4 MB) | 70.1 → 101.4 s (+31 s) | 3,232 → 3,229 MB (−0.1%) | 143.3 → **23.3 s (6.2×)** | 38.6 → 4.38 GB (8.8×) | 1,580 → 570 |
    | products (0.5 MB) | 85.4 → 145.7 s (+60 s) | 2,739 → 2,895 MB (+5.7%) | 283.0 → **48.3 s (5.9×)** | 71.9 → 3.38 GB (21×) | 5,758 → 4,418 |

    - The extra write time is below one read pass's saving (Reddit 31 vs 120 s, products 60 vs 235 s). But the write
      ran on local disk and the read through the emulator, so this compares two different settings.
    - Products stores 5.7% more in Spark (pyarrow copy: +0.5%); I have not checked why.
    - The count pass could reuse Phase 2's community sizes; not tried.
  - Rows within a unit are reordered. The per-block row sets are identical; whether training depends on row order
    is not checked.
  - This should change P5's rule (with blocks, the join won because MV re-read its units). Not rerun yet.
- **Row group × latency** (`grl-rg-emu-sweep`, 136 min; `results/trackB/rg_emu_sweep.jsonl`). The store's Phase 2
  tables were copied at each row group (`rg_resize.py`; actual sizes 0.7–1.5× the target), and Phase 3b's bins were
  read with the pruned reader at 5 / 20 / 80 ms per GET (80 MB/s). Rows were identical everywhere. Cold read time:

  | store (table / units) | latency | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 16 MB |
  |---|---|---|---|---|---|---|---|---|
  | arxiv (0.63 MB) | 5 / 20 / 80 ms | **4.9 / 9.5 / 26.7** | 5.3 / 9.9 / 26.9 | 6.2 / 10.6 / 27.8 | 7.5 / 11.9 / 29.0 | 9.2 / 13.9 / 31.2 | 12.7 / 17.3 / 34.8 | 22.2 / 28.3 / 50.5 |
  | arxiv/LPA (0.29 MB) | 5 / 20 / 80 ms | **11.4 / 24.7 / 74.2** | 12.1 / 25.5 / 74.8 | 13.7 / 27.1 / 76.5 | 17.4 / 30.7 / 79.7 | 23.2 / 36.6 / 85.5 | 23.4 / 36.6 / 85.4 | 23.4 / 35.9 / 85.7 |
  | products (3.52 MB) | 5 / 20 / 80 ms | 57.8 / 62.5 / **85.4** | 56.2 / 60.9 / 86.0 | 55.7 / 60.3 / 88.0 | 55.5 / 59.6 / 93.1 | **48.0 / 57.0** / 102.1 | 65.0 / 77.9 / 127.4 | 133.1 / 148.3 / 211.7 |

  Reddit (187 MB per unit), 1 / 4 / 16 / 64 MB: 20.5 / 15.4 / **14.5** / 15.1 s at 5 ms, 21.8 / 16.8 / **15.9** /
  16.4 at 20 ms, 28.7 / 22.4 / 22.1 / **21.9** at 80 ms.
  - **The guess that β ≈ latency × bandwidth is wrong.** The best row group does not grow with latency. It stays at
    the smallest size on arxiv and arxiv/LPA, and on products it moves down at 80 ms. The cause is in the GET counts:
    pyarrow fetches a unit's adjacent row groups in one coalesced read, so latency is paid per file and unit, not
    per row group. A row group's own cost stays decode overhead, the same as on local disk.
  - **The √ rule with the β fitted locally (about 0.6 MB) holds at every latency.** Its picks (arxiv 0.5, LPA 0.25,
    products 1, Reddit 4 MB) are within 1.27× of the best everywhere: max 1.27 (arxiv, 5 ms), otherwise 1.01–1.16.
    Using β = latency × bandwidth instead reaches regret 1.49 (products, 80 ms). So the rule does not need
    re-tuning per storage latency.
  - The cost of a too-large row group grows with latency: products at 16 MB is 2.3–2.6× slower than the best, arxiv
    1.9–4.5×.
  - The 0.125 MB end of the grid is the best on arxiv, so the optimum may lie lower; the curve is flat there
    (within 8%).
- **Stack: the changes added one at a time** (`grl-stack`, 82 min; `run_stack.sh`, `results/trackB/stack.jsonl`).
  Phase 3b's block work items (cap 10,000) through the emulator (20 ms + 80 MB/s, 4 processes, one cold pass), from
  pyarrow copies of the store's Phase 2 tables. Per-block checksums are identical in every column (same md5 per
  store). Cold read time (bytes read):

  | store | Phase 2 default (128 MB, isin reader) | + pruned reader | + √ row group | + block order | total |
  |---|---|---|---|---|---|
  | products | 732.6 s (211.7 GB) | 690.7 s (146.4 GB) | 387.2 s (102.5 GB) | **59.8 s (6.6 GB)** | **12.3×** (32× bytes) |
  | Reddit | 155.8 s (38.9 GB) | 156.6 s (38.9 GB) | 162.2 s (43.7 GB) | **24.9 s (5.0 GB)** | **6.3×** (7.9× bytes) |
  | arxiv | 31.4 s (5.0 GB) | 30.3 s (4.4 GB) | 12.0 s (1.1 GB) | **10.2 s (0.5 GB)** | **3.1×** (9.8× bytes) |

  - **The √ row group and block order pay together, not alone.** Block order at 128 MB: products 691 → 737 s, arxiv
    unchanged, Reddit 157 → 117 s. The √ row group without block order: products 1.8×, Reddit none (162 vs 157 s).
    Block order sorts each block into its own row groups, which only helps when row groups are small enough to skip.
  - With blocks, the √ row group alone gives much less than on Phase 3b's bins (products 1.8× here vs 8.4× there),
    because every block of a split unit re-reads the unit. Block order removes that.
  - **Stored bytes grow in the pyarrow copies:** 128 MB → √ +46% on products (2,671 → 3,907 MB), +13% on Reddit,
    +34% on arxiv; block order adds 0.3–0.8% on top. Phase 2's Spark writer at products' 0.5 MB target stored 2,739
    MB (`p4b_spark_write.jsonl`), so most of that growth seems to come from the pyarrow copy. Confirmed below.
- **The same stack written by Phase 2's Spark path** (`grl-p4b-spark` at the √ row group, `grl-p4b-spark-128` at
  128 MB, 85 min; `p4b_spark_write.py`, write = median of 3 alternated repeats; reads as above with the pruned
  reader, cold, warm within 2%; md5 identical in all four tables per store):

  | store | table | write | stored | read time | bytes read |
  |---|---|---|---|---|---|
  | products | 128 MB, Phase 2 order | 63.4 s | 2,677 MB | 696.7 s | 146.6 GB |
  | | 128 MB, block order | 120.6 s | 2,741 MB | 723.0 s | 150.2 GB |
  | | 0.5 MB, Phase 2 order | 85.4 s | 2,739 MB | 283.0 s | 71.9 GB |
  | | 0.5 MB, block order | 145.7 s | 2,895 MB | **48.3 s** | 3.4 GB |
  | Reddit | 128 MB, Phase 2 order | 61.4 s | 3,239 MB | 156.6 s | 38.8 GB |
  | | 128 MB, block order | 86.8 s | 3,252 MB | 112.0 s | 25.4 GB |
  | | 4 MB, Phase 2 order | 70.1 s | 3,232 MB | 143.3 s | 38.6 GB |
  | | 4 MB, block order | 101.4 s | 3,229 MB | **23.3 s** | 4.4 GB |

  - Default → √ + block order: read time **14.4×** on products (43× fewer bytes), **6.7×** on Reddit (8.9×).
  - Cost: write time 2.3× on products (+82 s) and 1.65× on Reddit (+40 s); stored bytes +8.1% and −0.3%.
  - In Spark the √ row group stores only +2.3% (products), so the pyarrow copies' +13–46% was the copy, not the row
    group. Spark's own small row groups also read faster than the pyarrow copies (products 283 vs 387 s).
  - The same pattern as the stack: block order alone at 128 MB is no gain on products and 1.4× on Reddit.
- **P4b at higher GET bandwidth** (`grl-p4b-bw`, 29 min; `p4b_blocks.jsonl`, `mbps` field; pyarrow copies at the √
  row group, 20 ms per GET). 80 MB/s is about one S3 connection; faster clients shrink the byte saving. Cold read
  time, Phase 2 order → block order:

  | store | 80 MB/s | 200 MB/s | 800 MB/s |
  |---|---|---|---|
  | products | 387.6 → 59.9 s (6.5×) | 192.8 → 46.6 s (4.1×) | 92.3 → 39.8 s (2.3×) |
  | Reddit | 162.6 → 25.0 s (6.5×) | 78.2 → 15.2 s (5.1×) | 34.9 → 10.0 s (3.5×) |

  - The gain stays above 2× at 10× the bandwidth. GETs and bytes are unchanged, so what is left is latency (products
    4,302 GETs ≈ 21.5 s at 20 ms over 4 processes) and local decode.
  - The bytes saving (15.5× products, 8.8× Reddit) does not depend on bandwidth; it matters wherever data moved is
    billed or shared (network, cross-region).

Order: P1 → P2 (uses B1 / B2 data now) → P3 → P4 / P5 → P6 / P7. Output: `results/trackB/*.jsonl`, generated tables, and
an EDBT / VLDB-experiments-style write-up. Track A (link prediction, gate GO) is done and goes into the journal version
unchanged.
