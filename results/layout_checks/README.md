# Layout checks: rows 3b, 4b and 5b of Table 3

Three tuned baselines next to the storage layout ablation (`../1_tableA_storage`), on all three graphs. They are
written from the same Phase 2 rows as rows 1-8 (`write_layouts.py --phase2-from`, read only) and read with the same
work items (the merged plan, `read_layouts.py --plan-from`), so every read returns the same rows as rows 1-8.

- `zorder_2mb` (4b), `liquid_2mb` (5b): rows 4-5 (Z-order / Liquid clustering on community_id, id for vertices and
  community_id, src, dst for edges) with `parquet.block.size` = 2 MB during the write and OPTIMIZE.
- `hive_coarse` (3b): `partitionBy(storage_part)` with storage_part = community_id mod B, B = the number of
  directories of rows 6-8 (611 on ogbn-products, 59 on ogbn-arxiv, 62 on Reddit), one file per directory, rows sorted
  by community_id, Spark's default 128 MB row groups.

Setup as Table 3: S3 cost model (20 ms + 80 MB/s per GET), cold reads (page cache dropped before every run),
coalesced schedule, 3 writes and 3 reads per layout, mean ± sample std. Writes: Spark local, 1 executor x 20 cores.

| File | Contents |
|---|---|
| `write.jsonl` | Write records of the three layouts on every graph (3 timed writes each) and the ogbn-products Phase 2 record (rows read from rows 1-8's run) |
| `read.jsonl` | ogbn-products cold reads: `hive_coarse` with 20 readers over all work items (row 3b); `zorder_2mb` and `liquid_2mb` with 3 readers over every tenth work item (as rows 4-5; not used in the paper) |
| `read_full20.jsonl` | Read in full with 20 readers: `zorder_2mb` and `liquid_2mb` on ogbn-products (rows 4b, 5b), all three layouts on ogbn-arxiv and Reddit (rows 3b-5b) |
| `rg_sizes.jsonl` | Row groups of every layout's current snapshot (files, row groups, rows and compressed MB per row group), per graph: rows 4, 5, 2, 3 and 8b of Table 3 (Z-order and Liquid: 1 file, default row groups), then the three layouts above |
| `hive_coarse_dirs.jsonl` | ogbn-arxiv and Reddit: the directory of every community in `hive_coarse` (from the layout's `meta.json`) |
| `tableA_<graph>.md`, `tableA_ogbn-products_full20.md` | `summary.py` tables of `write.jsonl` with the reads above |

Checks: the work-item manifest is byte-identical to that of rows 1-8 on every graph; every full read returns the same
rows as every row of Table 3 (`check_md5` 7c6cd7d4... on ogbn-products, 6f0bd443... on ogbn-arxiv, a90c04bc... on
Reddit), and the tenth-of-the-work-items samples the same rows as rows 4-5's samples.
