# AWS validation, Section 4.4, ogbn-products

One EMR cluster (emr-7.5.0, Spark 3.5.2, us-east-1): 1 x m5.xlarge primary + 4 x m5.2xlarge core nodes, up 5 h 13 min.
Tables: the local Delta tables uploaded unchanged to s3://us-east-1-s3-gnn/LakeGRL/ (no Phase 1-2 on AWS); Spark's
Delta reader on EMR counts the same rows as locally for all 27 tables (logs/smoke.log). GETs and bytes are counted by
the reader (ranged GETs; HEAD and LIST not counted), as in Table A. Raw records: expA_read.jsonl, expB_read.jsonl,
expB_train.jsonl; step logs in logs/.

## Experiment A: cold reads on real S3 (coalesced schedule, 3 runs, mean +- std)

| Layout | Readers | Read time (s) | GETs | Bytes read (GB) | Table A, emulator (s) |
|---|---|---|---|---|---|
| 1. Runtime join | 16 | 244.1 +- 10.7 | 34,419 | 614.25 | 318.2 |
| 2. Default MV | 16 | 37.8 +- 14.1 | 7,664 | 7.42 | 28.6 |
| 3. Naive partitionBy (58,373 files) | 16 | 667.4 +- 128.5 | 116,778 | 2.40 | 303.0 |
| 5. Liquid clustering (every 10th bin) | 3 | 620.7 +- 45.6 (x10 = 6,207 est.) | 2,168 | 131.37 | 6,591 est. |
| 8b. Ours (1,222 files, 2 MB floor) | 16 | 16.1 +- 1.9 | 2,495 | 2.28 | 12.7 |

Readers = Spark tasks, one per 1-core executor (16 = 4 per core node). Same rows in every run (checksum 7c6cd7d4;
Liquid's sample f1a5b3c3). Table A used 20 local readers through the S3 emulator (20 ms + 80 MB/s per GET).

## Experiment B: cold reads of Table B rows 5 and 4 by W readers (3 runs)

| W | Row 5 LakeGRL full: s (GETs, GB) | Row 4 id mod n: s (GETs, GB) |
|---|---|---|
| 16 | 15.9 +- 2.2 (2,472, 2.24) | 18.6 +- 2.7 (2,788, 5.93) |
| 8 | 26.5 +- 1.5 (2,456, 2.24) | 29.8 +- 0.3 (2,372, 5.90) |
| 4 | 51.1 +- 1.6 (2,448, 2.24) | 53.9 +- 0.4 (2,060, 5.88) |
| 2 | 108.0 +- 1.5 (2,444, 2.23) | 107.5 +- 1.6 (1,872, 5.87) |
| 1 | 210.6 +- 5.1 (2,442, 2.23) | 204.8 +- 6.8 (1,772, 5.86) |

## Experiment B: Stage 3b alone (10 epochs, seed 0, 1 run per W; W executors x 1 core x 4 GB, 1 torch thread)

| W | Row 5 LakeGRL full (s) | T1/TW | Row 4 id mod n (s) | T1/TW |
|---|---|---|---|---|
| 1 | 2,507.4 | 1.00 | 1,971.3 | 1.00 |
| 2 | 1,271.5 | 1.97 | 1,015.9 | 1.94 |
| 4 | 658.7 | 3.81 | 523.0 | 3.77 |
| 8 | 356.3 | 7.04 | 276.1 | 7.14 |
| 16 | 212.6 | 11.79 | 172.4 | 11.44 |

Stage 3b wall time (from its start to its end; Spark start-up, ~45-55 s, is outside). Test accuracy is the same at
every W: 0.7414 (row 5) and 0.6755 (row 4), 10 epochs (Table B at 100 epochs: 0.7595 / 0.6867). CPU units use the
CSR adjacency, the same mean SAGE as Table B's GPU path. Row 4 trains faster because its local graphs
carry 68 super-nodes (47,986 macro edges) against row 5's 610 (264,668).
