#!/usr/bin/env python3
"""Run the Phase 3.7/3.8 scaling matrix sequentially on an EMR driver."""

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = "wikics,ogbn-products,ogbn-papers100M"
DEFAULT_EXECUTORS = "2,4,8"


def parse_csv(value, converter, label):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{label} must not be empty")
    return [converter(item) for item in values]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cached graph propagation across datasets and executor counts."
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASETS,
                        help="Comma-separated small, medium, and large datasets")
    parser.add_argument("--executor-counts", default=DEFAULT_EXECUTORS,
                        help="Comma-separated executor-instance counts")
    parser.add_argument("--experiment-prefix", default="phase37-scaling",
                        help="Prefix for unique experiment names and manifests")
    parser.add_argument("--algorithms", default="lpa",
                        help="Community tag used for experiment output paths")
    parser.add_argument("--graph-source", choices=("phase0", "phase2"), default="phase0",
                        help="Use Phase 0 for the full stored-graph benchmark")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="Cached mean-propagation hop count")
    parser.add_argument("--partitions-per-executor", type=int, default=32,
                        help="Phase 3.7 partitions per executor, with a minimum of 200")
    parser.add_argument("--s3-bucket", default="us-east-1-s3-gnn")
    parser.add_argument("--s3-prefix", default="pipeline")
    parser.add_argument("--output-dir", default="results/phase37_scaling",
                        help="Directory for manifests and per-run logs")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip executor package verification for every run")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Record a failed cell and continue with later matrix cells")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets, str, "datasets")
    executor_counts = parse_csv(args.executor_counts, int, "executor counts")
    if any(count < 1 for count in executor_counts):
        raise ValueError("executor counts must be positive")
    if args.num_hops < 1 or args.partitions_per_executor < 1:
        raise ValueError("num-hops and partitions-per-executor must be positive")

    runner = Path(__file__).with_name("run_emr.py")
    run_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.experiment_prefix}-{run_stamp}-manifest.csv"
    fields = [
        "experiment_name", "dataset", "executor_instances", "graph_source",
        "num_hops", "phase37_partitions", "started_utc", "finished_utc",
        "return_code", "log_path",
    ]

    with manifest_path.open("w", newline="", encoding="ascii") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            for executor_count in executor_counts:
                partitions = max(200, executor_count * args.partitions_per_executor)
                experiment_name = f"{args.experiment_prefix}-{dataset}-e{executor_count}-{run_stamp}"
                log_path = output_dir / f"{experiment_name}.log"
                command = [
                    sys.executable, str(runner),
                    "--experiment-name", experiment_name,
                    "--datasets", dataset,
                    "--algorithms", args.algorithms,
                    "--no-phase0",
                    "--no-phase3",
                    "--no-phase3b",
                    "--no-phase4",
                    "--run-phase37",
                    "--run-phase38",
                    "--force-rerun",
                    "--executor-instances", str(executor_count),
                    "--phase37-graph-source", args.graph_source,
                    "--phase37-num-hops", str(args.num_hops),
                    "--phase37-num-partitions", str(partitions),
                    "--s3-bucket", args.s3_bucket,
                    "--s3-prefix", args.s3_prefix,
                ]
                if args.skip_install:
                    command.append("--no-install")

                started = dt.datetime.now(dt.timezone.utc).isoformat()
                print(f"\n{'=' * 78}\nStarting {experiment_name}\nCommand: {' '.join(command)}\n{'=' * 78}")
                with log_path.open("w", encoding="utf-8") as log_file:
                    completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
                finished = dt.datetime.now(dt.timezone.utc).isoformat()
                writer.writerow({
                    "experiment_name": experiment_name,
                    "dataset": dataset,
                    "executor_instances": executor_count,
                    "graph_source": args.graph_source,
                    "num_hops": args.num_hops,
                    "phase37_partitions": partitions,
                    "started_utc": started,
                    "finished_utc": finished,
                    "return_code": completed.returncode,
                    "log_path": log_path,
                })
                manifest_file.flush()
                if completed.returncode:
                    print(f"FAILED: {experiment_name}; inspect {log_path}")
                    if not args.continue_on_error:
                        raise SystemExit(completed.returncode)
                else:
                    print(f"Completed: {experiment_name}; log: {log_path}")

    print(f"\nScaling matrix manifest: {manifest_path}")


if __name__ == "__main__":
    main()