# AWS validation (Section 4.4)

One EMR cluster (emr-7.5.0, us-east-1): 1 x m5.xlarge primary + 4 x m5.2xlarge core nodes, at most 6 h and $20,
idle auto-termination 30 min, terminated when the runs are done. No Phase 1-2 on AWS: the verified local
ogbn-products tables are uploaded first. Results go to `results/aws/`.

| Step | Where | Command |
|---|---|---|
| 1. Local reference counts | WSL | `bash scripts/wsl_run.sh python experiments/aws/check_upload.py --root ~/lakegrl --spark 'local[8]' > results/aws/counts_local.jsonl` |
| 2. Upload tables (~16 GB) | WSL | `bash experiments/aws/upload.sh tables` |
| 3. Check the upload | WSL | `bash scripts/wsl_run.sh python experiments/aws/check_upload.py --root s3://us-east-1-s3-gnn/LakeGRL --local ~/lakegrl` |
| 4. Upload the code | WSL | `bash experiments/aws/upload.sh code` |
| 5. Create the cluster | WSL | see `scripts/emr_cluster.sh` (MASTER_TYPE, TYPE, IDLE, CONFIGS, BOOTSTRAP, CODE, KEY, SUBNET) |
| 6. Smoke test | primary | `cd /opt/lakegrl/lakegrl-code && bash experiments/aws/run.sh smoke` |
| 7. Experiment A | primary | `bash experiments/aws/run.sh expA` |
| 8. Experiment B | primary | `bash experiments/aws/run.sh expB-read`, then `bash experiments/aws/run.sh expB-train` (W = 16, 8, 4, 2, then 1) |
| 9. Save, terminate | primary, then WSL | `bash experiments/aws/run.sh save`; `scripts/emr_cluster.sh terminate <id>`; copy `s3://.../results/aws` back |

On the primary, set `LOCAL_ROOT` to the table root of the machine that wrote and uploaded the tables (for
example `/home/<user>/lakegrl`): the Table B work files record table paths under it, and `run.sh` maps them to S3.

Exp A: Table A rows 1 (join), 2 (MV), 3 (partitionBy, 58,373 files), 5 (Liquid; Z-order uploaded as a fallback) and
8b (1,222 files), coalesced schedule, 3 cold runs, 16 readers (one Spark task each, 4 per core node), Liquid 3 readers
on every 10th bin as in Table A. Exp B: Table B rows 5 (LakeGRL full) and 4 (id mod n), W executors x 1 core x 4 GB,
1 torch thread, 10 epochs, seed 0; cold reads with W readers, 3 runs. GETs and bytes are counted by the reader as in
Table A (ranged GETs; HEAD and LIST not counted); each reader reads the footers of the files it opens, so the counts
grow a little with the number of readers.

Local checks: Spark readers read the same rows as Table A (arxiv checksum 6f0bd443, products 7c6cd7d4) with
the same GETs and bytes as the process pool at the same reader count; products row 8b through the emulator: 16 Spark
readers 9.7-10.0 s vs 16 processes 9.2-9.3 s. `--stage3-only` trains Stage 3b on the stored arxiv tables (54.4 % at
10 epochs; Phase 1-2 skipped).
