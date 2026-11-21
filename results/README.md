# Result records behind the paper

Only the records that produce the numbers of Section 4 (Tables 2-7 and their text), in one folder per subsection;
`layout_checks/` and `lp_m0/` hold rows that were added to Tables 3 and 5 later; `backbones/` and `cmax_sweep/`
hold the backbone comparison (GraphSAGE, GCN, GATv2) and the C_max sweep.
`MANIFEST.txt` lists where each file was written by the scripts and which superseded records were left out.
Every number is the mean ± sample standard deviation (n - 1) over seeds 0-2 or over 3 cold runs, and every record
states its Spark executors x cores (`spark_executors`, `spark_executor_cores`) or its reader count (`procs`).

Recompute every number of Section 4 from these files (Python 3.8+, standard library only):

```bash
python results/verify_numbers.py      # prints OK / DIFF per number; ends with "0 DIFF"
```

Table paths inside the records were written on the experiment machines; the user's home folder is shown as
`/home/user` and the code folder on EMR as `/opt/lakegrl/lakegrl-code`. Nothing else in the records was changed.

Table 2 (datasets and Phase 1 units, Section 4 setup) comes from these records too: |V| and |E| from
`2_tableB_grl/baselines.jsonl` (`n_nodes`, `n_edges_undirected`; `pipeline.jsonl` `n_edges` = 2|E|, the rows of the
edge table), classes from `pipeline.jsonl`, features from `3_halo_and_preproc/halo_auc.jsonl` (ogbn-arxiv: 128, the
dataset's own), units from the Phase 2 records of `1_tableA_storage/write.jsonl` and `unit_probe.jsonl`.

## 1_tableA_storage: Section 4.1, the storage layout ablation (Table 3)

| File | Contents |
|---|---|
| `write.jsonl` | Write time, files, small files and storage of the 8 layouts on 3 datasets |
| `read_coalesce.jsonl` | Cold reads with the coalesced schedule, ogbn-arxiv and ogbn-products (9 layouts x 3 runs) |
| `read.jsonl` | Cold reads with the first (round-robin) schedule, all datasets: the Reddit column (its 63 work items are the same under both schedules) and the "before coalescing" numbers of the text |
| `read_coalesce_8a.jsonl`, `write_8a.jsonl` | Row 8a (pure square-root rule, no floor) |
| `rowgroup_floor_sweep/` | Row-group floor 0/2/4/8/16 MB on ogbn-products (GETs 30,663 -> 4,614, read 43.0 -> 30.6 s at 2 MB) |
| `unit_probe.jsonl` | Units vs C_max (5,000 to 40,000), for "Why the tail and not C_max" |
| `summary_*.csv` | Not raw: built by `experiments/tableA/summary.py` from the records above and the tables' Parquet metadata (needed for the amplification column) |

Rows 4-5 (Z-order, Liquid clustering) use 3 readers: a single reader of these layouts needs up to 7.2 GB. On
ogbn-products they read every tenth work item (`bin_sample` = 10), and their time and bytes are scaled by 10.

## 2_tableB_grl: Section 4.2, node classification (Table 4, its footnote, the community-ID paragraph)

| File | Contents |
|---|---|
| `baselines.jsonl` | Rows 1-2: full-graph GraphSAGE and Cluster-GCN (`partition_seconds` = METIS) |
| `pipeline.jsonl` | Rows 3-5: accuracy; Stage 3 wall time = `phase_seconds` `phase3` / `phase3b`. Row 3: `tiny_comm` keep; row 4: `unit_split` idmod; row 5: `unit_split` leiden, `tiny_comm` drop |
| `read_units.jsonl` | Cold read of the units of rows 3-5 (S3 cost model) |
| `read_full.jsonl` | Cold read of rows 1-2 (raw vertex, edge and split tables) and the "+ global" part of rows 4-5 (stored macro-graph, community map, unit plan) |
| `peak.jsonl` | Peak GPU memory by nvidia-smi, 8 workers sharing the GPU |
| `one_worker/` | Rows 3-5, seed 0, one worker: ogbn-products row 5 peaks at 1.6 GB |
| `full_sage_500_epochs.jsonl` | Full-graph GraphSAGE for 500 epochs (footnote) |
| `community_id/` | Full-graph GraphSAGE + a learned 16-dim embedding of the Phase 1 community, 3 seeds x 3 datasets, and its GPU peak |

## 3_halo_and_preproc: Section 4.3, one view for two tasks and the one-time cost

| File | Contents |
|---|---|
| `halo_storage_read.jsonl` | Halo sub-table (Table 5): rows, stored GB, cold read; also the leakage check (0 of the 100,000 held-out link pairs in any stored edge table) |
| `halo_auc.jsonl` | Link prediction cross AUC, 3 seeds x {Reddit, ogbn-products} x {uncapped, capped} |
| `phase12_timing_parallel_split.jsonl` | Phases 1-2 with the 8-process recursive split (ogbn-products: Phase 1 1,354.7 s, Phase 2 427.5 s) |
| `phase12_timing_sequential_split.jsonl` | The same with the sequential split |

## 4_aws_emr: Section 4.4, Amazon EMR 7.5.0 with the tables on S3

| File | Contents |
|---|---|
| `expA_and_W_sweep/expA_read.jsonl` | Cold reads on real S3 (Table 6) |
| `expA_and_W_sweep/expB_read.jsonl`, `expB_train.jsonl` | W = 1 to 16 workers on 4 nodes (text: 11.79x training, 13.23x read) |
| `expA_and_W_sweep/counts_local.jsonl`, `logs/smoke.log` | Spark on EMR counts the same rows as locally for all 27 tables |
| `node_scaling/nodes_read.jsonl`, `nodes_train.jsonl` | 1, 2, 4 and 8 core nodes x 4 workers (Table 7; N = 8 on a second cluster): cold reads (3 runs) and Stage 3, 10 epochs, seeds 0-2 |
| `*/logs/`, `*/SUMMARY.md` | Step logs, and the tables as first summarized |

## layout_checks: Table 3 rows 3b-5b

Coarse-key Hive (unit id mod B, B = the directory count of rows 6-8: 611, 59, 62) and Z-order / Liquid clustering
with 2 MB row groups on all three graphs, written from the same Phase 2 rows and read with the same work items as
rows 1-8, all read in full with 20 readers. Also the row-group sizes of rows 4-5 (1 file per table, default row
groups). `layout_checks/README.md` lists the files.

## lp_m0: the "full graph (M0)" rows of Table 5

Full-graph GraphSAGE link prediction, 3 seeds x {Reddit, ogbn-products}, with the same test links and settings as the
halo rows. `lp_m0/README.md` has the details.

## backbones: GraphSAGE, GCN and GATv2 on one stored view

Each backbone trained by Stage 3b alone on the stored Phase 2 view of Table 4 row 5 (C_max = 10,000), against the
same layers on the full graph, 3 graphs x 3 seeds on one 24 GB GPU; OOM where the card fills. `backbones/README.md`
has the setup and the table.

## cmax_sweep: C_max = 5,000 to 40,000 on ogbn-products

Table 4 row 5 at C_max = 5,000, 20,000 and 40,000 (Phases 1-2 rerun): units, stored MV_unit, cold read, Stage 3,
peak GPU and accuracy; the 10,000 row is Table 4's view. `cmax_sweep/README.md` has the setup and the table.
