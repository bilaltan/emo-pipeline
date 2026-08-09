#!/usr/bin/env python3
"""Collect Phase 3.7/3.8 benchmark manifests and logs into one Excel workbook."""

import argparse
import csv
import re
from pathlib import Path


INPUT_GRAPH_RE = re.compile(r"Input graph: ([\d,]+) nodes, ([\d,]+) directed propagation edges")
COVERAGE_RE = re.compile(r"Propagation coverage verified: ([\d,]+)/([\d,]+) nodes; (\d+) Delta hops materialized in ([\d.]+)s")
CLASSIFIER_RE = re.compile(
    r"Global classifier complete in ([\d.]+)s; valid accuracy=([\d.]+), test accuracy=([\d.]+); iterations=(\d+)"
)
ERROR_RE = re.compile(r"(?:Traceback|OutOfMemory|ExecutorLost|Exception|ERROR)", re.IGNORECASE)


def to_int(value):
    return int(value.replace(",", ""))


def parse_log(log_path):
    result = {
        "log_found": log_path.exists(), "input_nodes": None, "input_edges": None,
        "covered_nodes": None, "coverage_denominator": None, "coverage_pct": None,
        "observed_hops": None, "propagation_s": None, "classifier_s": None,
        "valid_acc": None, "test_acc": None, "classifier_iterations": None,
        "error_marker": None,
    }
    if not log_path.exists():
        return result
    text = log_path.read_text(encoding="utf-8", errors="replace")
    input_graph = INPUT_GRAPH_RE.search(text)
    coverage = COVERAGE_RE.search(text)
    classifier = CLASSIFIER_RE.search(text)
    error = ERROR_RE.search(text)
    if input_graph:
        result["input_nodes"] = to_int(input_graph.group(1))
        result["input_edges"] = to_int(input_graph.group(2))
    if coverage:
        result["covered_nodes"] = to_int(coverage.group(1))
        result["coverage_denominator"] = to_int(coverage.group(2))
        result["coverage_pct"] = 100.0 * result["covered_nodes"] / result["coverage_denominator"]
        result["observed_hops"] = int(coverage.group(3))
        result["propagation_s"] = float(coverage.group(4))
    if classifier:
        result["classifier_s"] = float(classifier.group(1))
        result["valid_acc"] = float(classifier.group(2))
        result["test_acc"] = float(classifier.group(3))
        result["classifier_iterations"] = int(classifier.group(4))
    if error:
        result["error_marker"] = error.group(0)
    return result


def main():
    parser = argparse.ArgumentParser(description="Create one Excel workbook from propagation benchmark artifacts.")
    parser.add_argument("--input-dir", default=".", help="Directory containing manifests and logs")
    parser.add_argument("--output-dir", default="results/phase37_scaling", help="Directory for the workbook")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit("Install pandas and openpyxl before collecting results") from error

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = sorted(input_dir.glob("phase37-scaling-*-manifest.csv"))
    if not manifest_paths:
        raise SystemExit(f"No phase37 scaling manifests found in {input_dir}")

    rows = []
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            for row in csv.DictReader(manifest_file):
                declared_path = Path(row["log_path"])
                log_path = declared_path if declared_path.exists() else input_dir / declared_path.name
                parsed = parse_log(log_path)
                row.update(parsed)
                row["manifest_name"] = manifest_path.name
                row["resolved_log_path"] = str(log_path)
                row["executor_instances"] = int(row["executor_instances"])
                row["return_code"] = int(row["return_code"])
                row["status"] = "complete" if row["return_code"] == 0 and parsed["test_acc"] is not None else "incomplete"
                rows.append(row)

    all_runs = pd.DataFrame(rows).sort_values(["dataset", "executor_instances", "started_utc"])
    successful = all_runs[all_runs["status"] == "complete"].copy()
    latest_success = (successful.sort_values("finished_utc")
        .groupby(["dataset", "executor_instances"], as_index=False).tail(1)
        .sort_values(["dataset", "executor_instances"]))
    if not latest_success.empty:
        baseline = latest_success.groupby("dataset")["propagation_s"].transform("max")
        latest_success["propagation_speedup_vs_slowest"] = baseline / latest_success["propagation_s"]
        latest_success["total_model_s"] = latest_success["propagation_s"] + latest_success["classifier_s"]
    failures = all_runs[all_runs["status"] != "complete"].copy()

    analysis_rows = []
    for dataset, group in latest_success.groupby("dataset"):
        fastest = group.loc[group["propagation_s"].idxmin()]
        slowest = group.loc[group["propagation_s"].idxmax()]
        analysis_rows.append({
            "dataset": dataset,
            "successful_configurations": len(group),
            "executor_range": f"{int(slowest['executor_instances'])}-{int(fastest['executor_instances'])}",
            "slowest_propagation_s": slowest["propagation_s"],
            "fastest_propagation_s": fastest["propagation_s"],
            "fastest_executor_instances": int(fastest["executor_instances"]),
            "best_speedup": slowest["propagation_s"] / fastest["propagation_s"],
            "test_acc": fastest["test_acc"],
            "node_coverage_pct": fastest["coverage_pct"],
            "interpretation": "Use the fastest completed configuration; accuracy is invariant across executor counts.",
        })
    analysis = pd.DataFrame(analysis_rows)

    workbook = output_dir / "phase37_scaling_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        all_runs.to_excel(writer, sheet_name="all_runs", index=False)
        latest_success.to_excel(writer, sheet_name="latest_success", index=False)
        analysis.to_excel(writer, sheet_name="analysis", index=False)
        failures.to_excel(writer, sheet_name="incomplete_runs", index=False)
    print(f"Wrote {workbook}")
    print(f"Complete runs: {len(successful)}; incomplete runs: {len(failures)}")


if __name__ == "__main__":
    main()