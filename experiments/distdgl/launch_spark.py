#!/usr/bin/env python3
"""
DistDGL on EMR, end to end from the Delta tables on S3, on the same worker nodes LakeGRL uses:

  1. export   the dataset's nodes / edges / masks tables -> Track A cache (experiments/trackA/export_cache.py)
  2. partition METIS, one part per worker (partition.py), uploaded to a stage prefix on S3
  3. train    one Spark barrier task per worker node (no ssh between nodes): each fetches its part, makes sure the
              DistDGL venv exists on its node (torch 2.4 CPU + DGL 2.4, as locally), and runs machine.run()

Every step's time goes into one record (results/icde/distdgl_runs.jsonl), next to train.py's own fields. Run on
the EMR primary node with the repo checked out, after scripts/emr_setup.sh:

    spark-submit --master yarn --deploy-mode client experiments/distdgl/launch_spark.py         --dataset ogbn-products --workers 4 --trainers 4 --omp 4

--master local[1] --stage <dir> runs the same path on one machine (one part) for a dry run.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VENV = "/mnt/distdgl-venv"
# the versions the local two-machine test ran (test_two_machines.sh); DGL has no build for newer torch
VENV_PKGS = [["torch==2.4.0", "--index-url", "https://download.pytorch.org/whl/cpu"],
             ["dgl==2.4.0", "-f", "https://data.dgl.ai/wheels/torch-2.4/repo.html"],
             ["numpy<2", "pandas", "pyarrow", "psutil", "pydantic", "scipy", "requests", "tqdm", "pyyaml"]]


def sh(*cmd):
    subprocess.run(cmd, check=True)


def copy_tree(src, dst):
    """src -> dst, either side a local directory or s3:// prefix (aws cli on EMR nodes)."""
    if src.startswith("s3://") or dst.startswith("s3://"):
        sh("aws", "s3", "cp", "--recursive", "--only-show-errors", src, dst)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def worker(m, ips, stage, args):
    """Body of barrier task m: runs on worker node m."""
    t0 = time.time()
    py = sys.executable if args.venv == "current" else f"{args.venv}/bin/python"
    if not os.path.exists(py):
        sh(sys.executable, "-m", "venv", args.venv)
        for pkgs in VENV_PKGS:
            sh(py, "-m", "pip", "install", "-q", *pkgs)
    venv_s = time.time() - t0
    work = os.path.join(args.local_dir, f"m{m}")
    shutil.rmtree(work, ignore_errors=True)
    t1 = time.time()
    os.makedirs(work, exist_ok=True)
    for name in ("g.json", "partition_meta.json"):   # the partition book, then this machine's part only
        if stage.startswith("s3://"):
            sh("aws", "s3", "cp", "--only-show-errors", f"{stage}/{name}", f"{work}/{name}")
        else:
            shutil.copy(f"{stage}/{name}", f"{work}/{name}")
    copy_tree(f"{stage}/part{m}", f"{work}/part{m}")
    fetch_s = time.time() - t1
    import machine
    out = os.path.join(work, "result.jsonl")
    machine.run(m, ips, f"{work}/g.json", work, trainers=args.trainers, omp=args.omp, ifname=args.ifname,
                python=py, train_args=["--tag", args.dataset, "--out", out, "--seed", str(args.seed),
                                       *(["--epochs", str(args.epochs)] if args.epochs else [])])
    rec = json.loads(open(out).read().splitlines()[-1]) if m == 0 else {}
    return [json.dumps({"machine": m, "host": socket.gethostname(), "venv_seconds": round(venv_s, 1),
                        "fetch_seconds": round(fetch_s, 1), "record": rec})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None, help="dataset root on the store (default s3://<bucket>/delta-data/<dataset>)")
    ap.add_argument("--bucket", default="us-east-1-s3-gnn")
    ap.add_argument("--workers", type=int, required=True, help="machines = Spark barrier tasks = METIS parts")
    ap.add_argument("--trainers", type=int, default=4, help="trainer processes per machine")
    ap.add_argument("--omp", type=int, default=4, help="threads per trainer")
    ap.add_argument("--executor-cores", type=int, default=None, help="cores of the one executor per node")
    ap.add_argument("--executor-memory", default="80g", help="large enough that YARN fits one executor per node")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--master", default="yarn")
    ap.add_argument("--stage", default=None, help="where the parts go (default s3://<bucket>/distdgl/<dataset>-w<W>)")
    ap.add_argument("--work-dir", default="/mnt/distdgl", help="export cache and parts on this node")
    ap.add_argument("--local-dir", default="/mnt/distdgl/run", help="each worker's copy of its part")
    ap.add_argument("--venv", default=VENV, help="DistDGL venv on every node; 'current' = this interpreter")
    ap.add_argument("--ifname", default=None)
    ap.add_argument("--reuse-cache", action="store_true", help="keep an existing export (its time is in meta.json)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "icde", "distdgl_runs.jsonl"))
    args = ap.parse_args()

    root = args.root or f"s3://{args.bucket}/delta-data/{args.dataset}"
    stage = args.stage or f"s3://{args.bucket}/distdgl/{args.dataset}-w{args.workers}"
    cache, parts = f"{args.work_dir}/{args.dataset}/cache", f"{args.work_dir}/{args.dataset}/parts-w{args.workers}"
    venv_py = sys.executable if args.venv == "current" else f"{args.venv}/bin/python"

    if not (args.reuse_cache and os.path.exists(f"{cache}/meta.json")):
        sh(sys.executable, f"{REPO}/experiments/trackA/export_cache.py", "--dataset", args.dataset, "--root", root,
           "--out", cache)
    export = json.load(open(f"{cache}/meta.json"))
    if not os.path.exists(venv_py):   # partitioning needs DGL on this node too
        sh(sys.executable, "-m", "venv", args.venv)
        for pkgs in VENV_PKGS:
            sh(venv_py, "-m", "pip", "install", "-q", *pkgs)
    shutil.rmtree(parts, ignore_errors=True)
    sh(venv_py, f"{HERE}/partition.py", "--cache", cache, "--parts", str(args.workers), "--out", parts)
    part = json.load(open(f"{parts}/partition_meta.json"))
    t_up = time.time()
    copy_tree(parts, stage)
    upload_s = time.time() - t_up

    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName(f"distdgl-{args.dataset}-w{args.workers}").master(args.master)
         .config("spark.dynamicAllocation.enabled", "false"))
    if args.master == "yarn":
        cores = args.executor_cores or os.cpu_count()
        b = (b.config("spark.executor.instances", str(args.workers)).config("spark.executor.cores", str(cores))
             .config("spark.task.cpus", str(cores)).config("spark.executor.memory", args.executor_memory))
    spark = b.getOrCreate()
    sc = spark.sparkContext
    for f in ("machine.py", "train.py"):
        sc.addPyFile(f"{HERE}/{f}")
    sc.addPyFile(f"{REPO}/experiment_config.py")

    def task(_):
        from pyspark import BarrierTaskContext
        ctx = BarrierTaskContext.get()
        hosts = [socket.gethostbyname(i.address.split(":")[0]) for i in ctx.getTaskInfos()]
        if len(set(hosts)) != len(hosts) and args.master == "yarn":
            raise RuntimeError(f"two DistDGL machines landed on one node: {hosts}")
        return worker(ctx.partitionId(), hosts, stage, args)

    t_run = time.time()
    rows = [json.loads(r) for r in sc.parallelize(range(args.workers), args.workers).barrier()
            .mapPartitions(task).collect()]
    run_s = time.time() - t_run
    rec = next(r["record"] for r in rows if r["machine"] == 0)
    sys.path.insert(0, REPO)
    from utils.common import spark_resources
    rec.update(dataset=args.dataset, root=root, workers=args.workers, hosts=[r["host"] for r in rows],
               export_seconds=export.get("export_seconds"), partition_seconds=part["partition_seconds"],
               partition_load_seconds=part["load_seconds"], upload_seconds=round(upload_s, 1),
               fetch_seconds=max(r["fetch_seconds"] for r in rows), venv_seconds=max(r["venv_seconds"] for r in rows),
               barrier_job_seconds=round(run_s, 1), **spark_resources(sc))
    # DistDGL's time from the Delta tables: export + partition + upload/fetch of parts + its own run
    rec["end_to_end_seconds"] = round(sum(rec[k] or 0 for k in ("export_seconds", "partition_seconds",
                                      "partition_load_seconds", "upload_seconds", "fetch_seconds"))
                                      + rec["init_seconds"] + rec["train_seconds"] + rec["inference_seconds"], 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
