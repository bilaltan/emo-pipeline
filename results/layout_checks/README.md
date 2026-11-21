# Layout checks (advisor 2026-11-18), ogbn-products

New rows next to Table 2 (Table A, run set `rleiden_c10000_v2`). The frozen Table 2 files and tables are untouched:
the writes read its Phase 2 rows (`~/lakegrl/tableA/rleiden_c10000_v2/ogbn-products/phase2`) and the reads use its
merged plan (`.../merged`), both read-only. New tables: `~/lakegrl/tableA/layout_checks_1118/ogbn-products`.
Code: branch `layout-checks`, commit 8347d0c (`experiments/tableA/write_layouts.py`, `read_layouts.py`, `summary.py`,
`rg_sizes.py`). Runner job `grl-layout-checks` (2026-11-18 20:54-21:33, rc=0) and `grl-layout-checks-full20`.

Setup as Table 2: S3 cost model (20 ms + 80 MB/s per GET), cold reads (page cache dropped before every run), coalesced
schedule (`--coalesce-minor`: every layout reads the same work items), 3 writes and 3 reads per layout, mean ± sample
std. Writes: Spark local, 1 executor x 20 cores, 16 GB.

| File | What |
|---|---|
| `rg_sizes.jsonl` | Row groups of every layout's current snapshot: files, row groups, rows and compressed MB per row group. First 10 records: the original Table 2 layouts (zorder, liquid, mv, hive, merged_blk); then the three new ones. |
| `write.jsonl` | Write records of the three new layouts (3 timed writes each) and the Phase 2 record (rows read from the frozen run set). |
| `read.jsonl` | Cold reads, Table 2's reader setup: hive_coarse with 20 readers over all work items; zorder_2mb and liquid_2mb with 3 readers over every 10th bin (as Table 2 rows 4-5; the table scales bytes and time by 10). |
| `read_full20.jsonl` | Extra: zorder_2mb and liquid_2mb read in full with 20 readers, as rows 1-3 and 6-8. |
| `tableA_ogbn-products.md` | `summary.py` table of `write.jsonl` + `read.jsonl`. |
| `tableA_ogbn-products_full20.md` | The same with `read_full20.jsonl`. |

Layouts:
- `zorder_2mb`, `liquid_2mb`: Table 2's Z-order / Liquid clustering (keys: community_id, id for vertices; community_id,
  src, dst for edges) with `parquet.block.size` = 2 MB during the write and OPTIMIZE.
- `hive_coarse`: `partitionBy(storage_part)` with storage_part = community_id mod 611 (611 = the number of directories
  of our layout), one file per directory, rows sorted by community_id, Spark's default 128 MB row groups.

Checks: the manifest (work items) is byte-identical to the frozen run set's; every full read returns the same rows as
every Table 2 row (check_md5 7c6cd7d4), and the 1/10 samples the same rows as Table 2's Z-order / Liquid samples
(f1a5b3c3).
