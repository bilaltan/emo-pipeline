# Table A run set: id mod n split, C_max = 10,000 (2026-10-27 to 2026-10-29)

**Status:** final: Table A, the diagnostic and the halo sub-table (AUC over seeds 0-2, finished 2026-10-30). This set is now the **ablation
baseline**: Stage 3b cut every community above 10,000 rows (members + halo) into n = 2^k blocks by `id & (n - 1)`.
The next run set (`../rleiden_c10000/`) uses recursive Leiden units instead.

## Setup
- Local server only: 20 hardware threads, 31.7 GB RAM, WSL VM with 26 GB. Spark local, 1 executor x 20 cores.
- S3 emulator: every GET costs 20 ms + bytes / 80 MB/s; GETs and bytes are counted.
- Cold reads: page cache dropped and fresh reader processes before every run. 20 readers; Z-order and Liquid 3
  readers (one reader peaks at 7.2 GB).
- One Leiden partition (seed 0). Each layout written 3 times and read in 3 cold runs (labelled seeds 0, 1, 2).
- Reads follow Stage 3b's bins (commit 786abc5). Every layout's per-item checksums match.
- Units: arxiv 133 communities (13 split into 40 blocks), Reddit 24 (22 split into 172 blocks, up to 32 each),
  products 52,855 (49 split into 798 blocks, up to 64 each).

## Files
| File | What |
| --- | --- |
| `tableA.md`, `tableA.csv` | Final Table A for the 3 datasets (CSV: one row per dataset x layout, for comparing run sets) |
| `write.jsonl` | Every write run (layout write times, storage, files) and Phase 2 records |
| `read.jsonl` | Every cold read run in Stage 3b's bins (wall time, GETs, bytes, rows, checksum) |
| `read_v1_per_community_SUPERSEDED.jsonl` | First harness (one task per community); read times too high, kept only for the record |
| `inputs.jsonl` | Phase 0 / Leiden input runs, including the halo `-lp` stores |
| `halo.jsonl` | Halo sub-table: uncapped vs capped (R = 1) storage, bytes, leakage check |
| `halo_auc.jsonl` | Track A M2 link AUC, seeds 0-2; uncapped has halo_rows 1,001,048 (Reddit) / 2,947,387 (products) |
| `tableA_<ds>_from_job.md` | The tables as each job printed them |

Written tables (65 GB) stay in WSL: `~/lakegrl/tableA/idmod_c10000/<dataset>/<layout>/`, halo work dir in
`.../halo_workdir/`.

## Caveats
- Z-order and Liquid read times use 3 readers, the other rows 20: compare their bytes, not their times.
- Products Z-order and Liquid read every 10th bin (`bin_sample` 10): bytes, GETs and times x10, marked est.
- Products rows 1-6 were written on 10-27; rows 7-8 on 10-28 after the disk-full stop (Phase 2 rerun, same checksums).
- Bytes differ by < 0.1 % across runs of one layout (which reader gets which bin changes a few footer reads).
- Reddit's first 3 attempts crashed the VM in the Z-order reads; `summary.py` uses the latest write per layout and
  the latest 3 reads.

## Code
fe8f7cc, 1d54db6, c5d370b, 786abc5, 974a2c9, d2a6254, fce3980 on `icde-pipeline-fixes`.

Rebuild the table (WSL):

    python experiments/tableA/summary.py --writes results/tableA/idmod_c10000/write.jsonl \
      --reads results/tableA/idmod_c10000/read.jsonl --out-dir ~/lakegrl/tableA/idmod_c10000 \
      --md results/tableA/idmod_c10000/tableA.md --csv results/tableA/idmod_c10000/tableA.csv \
      --halo results/tableA/idmod_c10000/halo.jsonl results/tableA/idmod_c10000/halo_auc.jsonl
