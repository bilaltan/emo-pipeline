# C_max sweep on ogbn-products (advisor 2026-11-19)

Table B row 5 (recursive Leiden units, GraphSAGE, minor communities dropped, 8 workers on the GPU, Table A row 8b layout
with the 2 MB row-group floor) at C_max 5,000, 20,000 and 40,000, with Table B's own script:
`TB_OUT=results/cmax_sweep/c<C> TB_CMAX=<C> TB_ROWS=5 TB_PIPELINE_ONLY=1 bash experiments/tableB/run.sh ogbn-products`
(seed 0: Phase 1 + Phase 2 + Stage 3b; seeds 1-2: Phase 2 again + Stage 3b; after seed 0 the cold reads of its units
and of the global tables, S3 cost model 20 ms + 80 MB/s per GET, 20 readers, 3 runs; then the Delta vacuum).
C_max = 10,000 (advisor 11-20): units, storage and cold reads = Table 4 row 5 (results/tableB read_units, global
read results/tableB/v4); Stage 3, peak GPU and accuracy = the GraphSAGE `--stage3-only` run on the stored view,
results/backbones/sage (d658119, 297.9 s / 75.96%), so the row has the same code as the rest of the sweep.

| File | Contents |
|---|---|
| `c<C>/pipeline.jsonl` | Seeds 0-2: accuracy, Stage 3 wall (`phase_seconds` phase3b), `n_units_stage3b` = major units + 1 global model |
| `c<C>/peak.jsonl`, `peak_trace/` | peak.sh per seed (GPU = nvidia-smi minus idle) |
| `c<C>/read_units.jsonl`, `read_full.jsonl` | Cold reads: the run's units (row "5") and the global tables (row "5 global") |
| `c<C>/vacuum.jsonl` | Old Delta versions removed after the run (only the new tags had any) |
| `storage.jsonl` | `experiments/tableB/store_size.py`: stored MV_unit (p2_nodes + p2_edges) and MV_macro per tag, incl. leiden_c10000 |

Table: `python experiments/tableB/summary_backbones_cmax.py` -> `SUMMARY.md`. Code: branch backbones-cmax (the
sweep ran at 8a6e1e7 / d658119). Table 4's own run of row 5 (fe671e8) gave 301.1 s / 75.95%.
