# Layout checks: rows 3b, 4b and 5b of Table 3 (ogbn-products)

Three tuned baselines next to the storage layout ablation (`../1_tableA_storage`). They are written from the same
Phase 2 rows as rows 1-8 (`write_layouts.py --phase2-from`, read only) and read with the same work items (the merged
plan, `read_layouts.py --plan-from`), so every read returns the same rows as rows 1-8.

- `zorder_2mb` (4b), `liquid_2mb` (5b): rows 4-5 (Z-order / Liquid clustering on community_id, id for vertices and
  community_id, src, dst for edges) with `parquet.block.size` = 2 MB during the write and OPTIMIZE.
- `hive_coarse` (3b): `partitionBy(storage_part)` with storage_part = community_id mod 611 (611 = the number of
  directories of rows 6-8), one file per directory, rows sorted by community_id, Spark's default 128 MB row groups.

Setup as Table 3: S3 cost model (20 ms + 80 MB/s per GET), cold reads (page cache dropped before every run),
coalesced schedule, 3 writes and 3 reads per layout, mean ± sample std. Writes: Spark local, 1 executor x 20 cores.

| File | Contents |
|---|---|
| `write.jsonl` | Write records of the three layouts (3 timed writes each) and the Phase 2 record (rows read from rows 1-8's run) |
| `read.jsonl` | Cold reads: `hive_coarse` with 20 readers over all work items (row 3b); `zorder_2mb` and `liquid_2mb` with 3 readers over every tenth work item (as rows 4-5; not used in the paper) |
| `read_full20.jsonl` | `zorder_2mb` and `liquid_2mb` read in full with 20 readers (rows 4b, 5b) |
| `rg_sizes.jsonl` | Row groups of every layout's current snapshot (files, row groups, rows and compressed MB per row group). First 10 records: rows 4, 5, 2, 3 and 8b of Table 3 (Z-order and Liquid: 1 file, ~128 MB row groups); then the three layouts above |
| `tableA_ogbn-products.md`, `tableA_ogbn-products_full20.md` | `summary.py` tables of `write.jsonl` with `read.jsonl` / `read_full20.jsonl` |

Checks: the work-item manifest is byte-identical to that of rows 1-8; every full read returns the same rows as every
row of Table 3 (`check_md5` 7c6cd7d4...), and the tenth-of-the-work-items samples the same rows as rows 4-5's samples.
