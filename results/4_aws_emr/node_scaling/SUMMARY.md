# AWS node scaling, Section 4.4 node-scaling table, ogbn-products

One EMR cluster (emr-7.5.0, Spark 3.5.2, us-east-1): 1 x m5.xlarge primary + N m5.2xlarge core nodes, grown
1 -> 2 -> 4 in one cluster (2 h 54 min). Tables: the same upload on S3 as Experiments A and B, unchanged. Every worker = 1 Spark executor x 1 core x 4 GB (+1.5 GB overhead), one
torch thread, dynamic allocation off; W = 4N, so exactly 4 per node (yarn_workers' 5.5 GB containers fit at most 4);
`yarn node -list` in logs/nodes{1,2,4}.log shows N RUNNING nodes. Raw records: nodes_read.jsonl, nodes_train.jsonl.

| N (W) | Row 5 LakeGRL read (s) | train (s) | train speedup | read speedup | Row 4 id mod n read (s) | train (s) | train speedup | read speedup |
|---|---|---|---|---|---|---|---|---|
| 1 (4) | 58.8 +- 6.6 | 696.6 +- 1.2 | 1.00 | 1.00 | 61.5 +- 3.2 | 562.6 +- 1.7 | 1.00 | 1.00 |
| 2 (8) | 28.6 +- 3.2 | 368.3 +- 1.2 | 1.89 | 2.06 | 31.4 +- 2.7 | 298.6 +- 1.9 | 1.88 | 1.96 |
| 4 (16) | 15.0 +- 1.6 | 209.2 +- 1.9 | 3.33 | 3.93 | 18.1 +- 1.5 | 171.1 +- 0.7 | 3.29 | 3.39 |

Reads: 3 cold runs by W Spark readers, same rows as Table A and Experiment B (checksums 5ae3e959 row 5, 26a8be43 row 4);
2.24 GB (row 5) and 5.88-5.93 GB (row 4). Training: Stage 3b alone, 10 epochs, seeds 0-2 (mean +- sample std), wall
time of phase3b (Spark start-up outside). Test accuracy per seed is identical at every N: row 5 74.14 / 74.53 / 74.07
(74.25 +- 0.25), row 4 67.55 / 67.76 / 67.54. Seed 0 equals the Experiment B run (0.7414 / 0.6755).
On one node the 4 workers share a machine: training 6-8 % slower and reads ~14 % slower than W = 4 spread over the
4-node cluster in Experiment B (658.7 / 523.0 s, 51.1 / 53.9 s).
