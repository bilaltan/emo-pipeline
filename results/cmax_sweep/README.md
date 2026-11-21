# C_max sweep on ogbn-products

Table 4 row 5 (recursive Leiden units, GraphSAGE, minor communities dropped, 8 workers sharing the GPU, the row 8b
layout of Table 3 with the 2 MB row-group floor) at C_max = 5,000, 20,000 and 40,000. Seed 0 runs Phases 1-2 and
Stage 3b, seeds 1-2 Phase 2 and Stage 3b; after seed 0 the run's units and the global tables (macro-graph, community
map, unit plan) are read cold with the S3 cost model (20 ms + 80 MB/s per GET), 20 readers, 3 runs.

C_max = 10,000: units, storage and cold reads are Table 4 row 5 (`../2_tableB_grl/read_units.jsonl` row "5" and
`read_full.jsonl` row "5 global", ogbn-products); Stage 3, peak GPU and accuracy are the GraphSAGE run by Stage 3b
alone on its stored view (`../backbones/sage/`, ogbn-products).

| File | Contents |
|---|---|
| `c<C>/pipeline.jsonl` | Seeds 0-2: accuracy (`stage3b_node_acc`), Stage 3 (`phase_seconds` `phase3b`), `n_units_stage3b` = major units + 1 global model |
| `c<C>/peak.jsonl` | Peak GPU per seed (nvidia-smi minus the idle reading) |
| `c<C>/read_units.jsonl`, `read_full.jsonl` | Cold reads of the run's units (row "5") and of the global tables (row "5 global") |
| `storage.jsonl` | Stored MV_unit (`mv_unit_bytes`: the live files of the Phase 2 vertex and edge tables) and MV_macro per C_max, 10,000 included |

| C_max | Major units | Stored MV_unit (GB) | Cold Read (s): units + global | Stage 3 Train (s) | Peak GPU (GB) | Test Acc (%) |
|---|---|---|---|---|---|---|
| 5,000 | 1,188 | 2.14 | 11.5 (10.9 ± 0.2 + 0.58 ± 0.03) | 578.1 ± 2.7 | 4.8 | 75.66 ± 0.03 |
| 10,000 | 610 | 2.16 | 8.5 (7.9 ± 0.2 + 0.58 ± 0.02) | 297.9 ± 0.7 | 4.6 | 75.96 ± 0.06 |
| 20,000 | 317 | 2.18 | 8.3 (7.8 ± 0.1 + 0.57 ± 0.02) | 225.1 ± 3.8 | 5.2 | 76.31 ± 0.13 |
| 40,000 | 171 | 2.20 | 9.1 (8.5 ± 0.3 + 0.58 ± 0.04) | 205.6 ± 3.8 | 7.5 | 76.62 ± 0.11 |

Mean ± sample std over seeds 0-2 (Stage 3, accuracy) or 3 cold runs (reads); peak GPU = the largest over the seeds.
