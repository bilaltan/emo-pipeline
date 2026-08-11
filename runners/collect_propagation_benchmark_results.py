#!/usr/bin/env python3
"""Collect Phase 3.7/3.8 benchmark manifests into workbook, figures, and paper-ready summaries."""

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

CONFIG_COLUMNS = [
    "executor_instances",
    "executor_cores",
    "executor_memory_gb",
    "executor_memory_overhead_gb",
]


def to_int(value):
    return int(value.replace(",", ""))


def to_optional_int(value):
    if value in (None, "", "None"):
        return None
    return int(value)


def is_missing(value):
    return value is None or value != value


def optional_int_text(value):
    return "--" if is_missing(value) else str(int(value))


def config_label(row):
    parts = [f"{int(row['executor_instances'])} exec"]
    if not is_missing(row.get("executor_cores")):
        parts.append(f"{int(row['executor_cores'])} cores")
    if not is_missing(row.get("executor_memory_gb")):
        parts.append(f"{int(row['executor_memory_gb'])}g heap")
    if not is_missing(row.get("executor_memory_overhead_gb")):
        parts.append(f"{int(row['executor_memory_overhead_gb'])}g ovh")
    return " / ".join(parts)


def recommendation_row(group):
    if group.empty:
        return None
    fastest = group.loc[group["propagation_s"].idxmin()]
    threshold = fastest["propagation_s"] * 1.05
    near_optimal = group[group["propagation_s"] <= threshold].sort_values([
        "executor_instances",
        "executor_cores",
        "executor_memory_gb",
        "executor_memory_overhead_gb",
    ], na_position="last")
    return near_optimal.iloc[0] if not near_optimal.empty else fastest


def write_summary(summary_path, latest_success, analysis, recommendations, figure_paths, latex_path):
    lines = [
        "# Phase 3.7 / 3.8 Scaling Summary",
        "",
        f"Successful configurations: {len(latest_success)}",
        f"Datasets covered: {latest_success['dataset'].nunique() if not latest_success.empty else 0}",
        "",
        "## Dataset recommendations",
        "",
    ]
    if recommendations.empty:
        lines.append("No successful configurations were available.")
    else:
        for _, row in recommendations.iterrows():
            lines.append(
                f"- {row['dataset']}: recommend {row['config_label']} "
                f"({row['propagation_s']:.1f}s propagation, {row['total_model_s']:.1f}s end-to-end, test acc {row['test_acc']:.4f})"
            )
    lines.extend([
        "",
        "## Paper figures",
        "",
    ])
    for label, path in figure_paths.items():
        lines.append(f"- {label}: {path.name}")
    lines.extend([
        "",
        "## LaTeX table",
        "",
        f"- {latex_path.name}",
        "",
        "## Key findings",
        "",
    ])
    if analysis.empty:
        lines.append("No completed runs to analyze.")
    else:
        for _, row in analysis.iterrows():
            lines.append(
                f"- {row['dataset']}: fastest propagation {row['fastest_propagation_s']:.1f}s, "
                f"speedup {row['best_speedup']:.2f}x, accuracy range {row['test_acc_min']:.4f}-{row['test_acc_max']:.4f}."
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path, recommendations):
    header = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Exec & Cores & Heap (GB) & Ovh (GB) & Prop (s) & Test Acc \\",
        r"\midrule",
    ]
    body = []
    for _, row in recommendations.iterrows():
        body.append(
            f"{row['dataset']} & {int(row['executor_instances'])} & "
            f"{optional_int_text(row['executor_cores'])} & "
            f"{optional_int_text(row['executor_memory_gb'])} & "
            f"{optional_int_text(row['executor_memory_overhead_gb'])} & "
            f"{row['propagation_s']:.1f} & {row['test_acc']:.4f} \\\\"
        )
    footer = [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(header + body + footer) + "\n", encoding="utf-8")


def save_figures(output_dir, latest_success):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    figure_paths = {}

    data = latest_success.copy()
    data["executor_instances"] = data["executor_instances"].astype(int)
    data = data.sort_values(["dataset", "executor_instances"])

    dataset_order = (
        data.groupby("dataset")["input_nodes"]
        .max()
        .sort_values()
        .index
        .tolist()
    )
    executor_order = sorted(data["executor_instances"].unique())
    min_exec = min(executor_order)

    baseline = (
        data.sort_values("executor_instances")
        .groupby("dataset")["propagation_s"]
        .first()
    )
    data["speedup_vs_min_exec"] = data.apply(
        lambda row: baseline[row["dataset"]] / row["propagation_s"],
        axis=1,
    )
    data["efficiency_vs_min_exec"] = data["speedup_vs_min_exec"] / (data["executor_instances"] / min_exec)
    data["classifier_share_pct"] = 100.0 * data["classifier_s"] / data["total_model_s"]

    label_map = {
        "wikics": "WikiCS",
        "ogbn-products": "ogbn-products",
        "ogbn-papers100M": "ogbn-papers100M",
    }
    colors = {
        "wikics": "#4C78A8",
        "ogbn-products": "#F58518",
        "ogbn-papers100M": "#54A24B",
    }

    # Figure 1: 1x3 publication-style summary figure (no separate accuracy panel).
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))

    ax = axes[0]
    for dataset in dataset_order:
        group = data[data["dataset"] == dataset].sort_values("executor_instances")
        ax.plot(
            group["executor_instances"],
            group["propagation_s"],
            marker="o",
            linewidth=2.2,
            color=colors.get(dataset, "#333333"),
            label=label_map.get(dataset, dataset),
        )
    ax.set_yscale("log")
    ax.set_xticks(executor_order)
    ax.set_title("A. Propagation Runtime vs Executors")
    ax.set_xlabel("Executor instances")
    ax.set_ylabel("Propagation time (s, log scale)")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    ideal = [executor / min_exec for executor in executor_order]
    ax.plot(executor_order, ideal, linestyle="--", color="#666666", linewidth=1.8, label="Ideal linear")
    for dataset in dataset_order:
        group = data[data["dataset"] == dataset].sort_values("executor_instances")
        ax.plot(
            group["executor_instances"],
            group["speedup_vs_min_exec"],
            marker="o",
            linewidth=2.2,
            color=colors.get(dataset, "#333333"),
            label=label_map.get(dataset, dataset),
        )
    ax.set_xticks(executor_order)
    ax.set_title(f"B. Speedup vs {min_exec} Executors")
    ax.set_xlabel("Executor instances")
    ax.set_ylabel("Speedup (x)")
    ax.legend(loc="upper left", frameon=True)

    ax = axes[2]
    for dataset in dataset_order:
        group = data[data["dataset"] == dataset].sort_values("executor_instances")
        ax.plot(
            group["executor_instances"],
            group["efficiency_vs_min_exec"] * 100.0,
            marker="o",
            linewidth=2.2,
            color=colors.get(dataset, "#333333"),
            label=label_map.get(dataset, dataset),
        )
    ax.axhline(100.0, linestyle="--", color="#666666", linewidth=1.6)
    ax.set_xticks(executor_order)
    ax.set_ylim(bottom=0)
    ax.set_title("C. Parallel Efficiency")
    ax.set_xlabel("Executor instances")
    ax.set_ylabel("Efficiency (%)")

    fig.suptitle("Phase 3.7/3.8 Executor Scaling (Paper View)", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    main_pdf = output_dir / "phase37_paper_figure_main.pdf"
    main_png = output_dir / "phase37_paper_figure_main.png"
    fig.savefig(main_pdf, bbox_inches="tight")
    fig.savefig(main_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["Main scaling figure (1x3)"] = main_pdf

    # Figure 2: compact heatmaps for runtime and speedup.
    runtime_matrix = (
        data.pivot(index="dataset", columns="executor_instances", values="propagation_s")
        .reindex(index=dataset_order, columns=executor_order)
    )
    speedup_matrix = (
        data.pivot(index="dataset", columns="executor_instances", values="speedup_vs_min_exec")
        .reindex(index=dataset_order, columns=executor_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    im0 = axes[0].imshow(runtime_matrix.to_numpy(), cmap="Blues", aspect="auto")
    axes[0].set_title("A. Propagation Runtime (s)")
    axes[0].set_xticks(np.arange(len(executor_order)))
    axes[0].set_xticklabels(executor_order)
    axes[0].set_yticks(np.arange(len(dataset_order)))
    axes[0].set_yticklabels([label_map.get(dataset, dataset) for dataset in dataset_order])
    axes[0].set_xlabel("Executors")
    for row in range(len(dataset_order)):
        for col in range(len(executor_order)):
            value = runtime_matrix.to_numpy()[row, col]
            axes[0].text(col, row, f"{value:.1f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(speedup_matrix.to_numpy(), cmap="Greens", aspect="auto", vmin=0.7, vmax=max(1.35, float(np.nanmax(speedup_matrix.to_numpy()))))
    axes[1].set_title(f"B. Speedup vs {min_exec} Executors")
    axes[1].set_xticks(np.arange(len(executor_order)))
    axes[1].set_xticklabels(executor_order)
    axes[1].set_yticks(np.arange(len(dataset_order)))
    axes[1].set_yticklabels([label_map.get(dataset, dataset) for dataset in dataset_order])
    axes[1].set_xlabel("Executors")
    for row in range(len(dataset_order)):
        for col in range(len(executor_order)):
            value = speedup_matrix.to_numpy()[row, col]
            axes[1].text(col, row, f"{value:.2f}x", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Phase 3.7 Scaling Matrices", fontsize=13, y=1.02)
    fig.tight_layout()
    matrix_pdf = output_dir / "phase37_paper_figure_matrix.pdf"
    matrix_png = output_dir / "phase37_paper_figure_matrix.png"
    fig.savefig(matrix_pdf, bbox_inches="tight")
    fig.savefig(matrix_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["Runtime and speedup matrix"] = matrix_pdf

    # Backward-compatible exports for previously referenced filenames.
    legacy_runtime = output_dir / "phase37_runtime_by_config.png"
    legacy_speedup = output_dir / "phase37_speedup_by_config.png"

    # Re-save legacy names without destructive renaming.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    papers = data[data["dataset"] == "ogbn-papers100M"].sort_values("executor_instances")
    ax.plot(papers["executor_instances"], papers["propagation_s"], marker="o", linewidth=2.4, color="#54A24B")
    ax.set_title("ogbn-papers100M Propagation Runtime")
    ax.set_xlabel("Executor instances")
    ax.set_ylabel("Propagation time (s)")
    ax.set_xticks(executor_order)
    fig.tight_layout()
    fig.savefig(legacy_runtime, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(papers["executor_instances"], papers["speedup_vs_min_exec"], marker="o", linewidth=2.4, color="#4C78A8")
    ax.plot(executor_order, ideal, linestyle="--", color="#666666", linewidth=1.5, label="Ideal linear")
    ax.set_title(f"ogbn-papers100M Speedup vs {min_exec} Executors")
    ax.set_xlabel("Executor instances")
    ax.set_ylabel("Speedup (x)")
    ax.set_xticks(executor_order)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(legacy_speedup, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return figure_paths


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
    parser = argparse.ArgumentParser(description="Create workbook, figures, and summary outputs from propagation benchmark artifacts.")
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
                row["executor_cores"] = to_optional_int(row.get("executor_cores"))
                row["executor_memory_gb"] = to_optional_int(row.get("executor_memory_gb"))
                row["executor_memory_overhead_gb"] = to_optional_int(row.get("executor_memory_overhead_gb"))
                row["return_code"] = int(row["return_code"])
                row["status"] = "complete" if row["return_code"] == 0 and parsed["test_acc"] is not None else "incomplete"
                row["config_label"] = config_label(row)
                rows.append(row)

    all_runs = pd.DataFrame(rows).sort_values(["dataset", "executor_instances", "started_utc"])
    successful = all_runs[all_runs["status"] == "complete"].copy()
    latest_success = (successful.sort_values("finished_utc")
        .groupby(["dataset", *CONFIG_COLUMNS], as_index=False, dropna=False).tail(1)
        .sort_values(["dataset", "executor_instances", "executor_cores", "executor_memory_gb"], na_position="last"))
    if not latest_success.empty:
        baseline = latest_success.groupby("dataset")["propagation_s"].transform("max")
        latest_success["propagation_speedup_vs_slowest"] = baseline / latest_success["propagation_s"]
        latest_success["total_model_s"] = latest_success["propagation_s"] + latest_success["classifier_s"]
    failures = all_runs[all_runs["status"] != "complete"].copy()

    analysis_rows = []
    for dataset, group in latest_success.groupby("dataset"):
        fastest = group.loc[group["propagation_s"].idxmin()]
        slowest = group.loc[group["propagation_s"].idxmax()]
        recommended = recommendation_row(group)
        analysis_rows.append({
            "dataset": dataset,
            "successful_configurations": len(group),
            "executor_range": f"{int(slowest['executor_instances'])}-{int(fastest['executor_instances'])}",
            "slowest_propagation_s": slowest["propagation_s"],
            "fastest_propagation_s": fastest["propagation_s"],
            "fastest_executor_instances": int(fastest["executor_instances"]),
            "best_speedup": slowest["propagation_s"] / fastest["propagation_s"],
            "test_acc": fastest["test_acc"],
            "test_acc_min": group["test_acc"].min(),
            "test_acc_max": group["test_acc"].max(),
            "node_coverage_pct": fastest["coverage_pct"],
            "recommended_config": recommended["config_label"] if recommended is not None else None,
            "interpretation": "Prefer the smallest configuration within 5% of the fastest runtime; accuracy should remain stable across system-only sweeps.",
        })
    analysis = pd.DataFrame(analysis_rows)
    if latest_success.empty:
        recommendations = pd.DataFrame()
    else:
        recommendation_rows = []
        for _, group in latest_success.groupby("dataset", sort=True):
            selected = recommendation_row(group)
            if selected is not None:
                recommendation_rows.append(selected)
        recommendations = pd.DataFrame(recommendation_rows)

    workbook = output_dir / "phase37_scaling_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        all_runs.to_excel(writer, sheet_name="all_runs", index=False)
        latest_success.to_excel(writer, sheet_name="latest_success", index=False)
        analysis.to_excel(writer, sheet_name="analysis", index=False)
        recommendations.to_excel(writer, sheet_name="recommended_configs", index=False)
        failures.to_excel(writer, sheet_name="incomplete_runs", index=False)
    figure_paths = save_figures(output_dir, latest_success) if not latest_success.empty else {}
    latex_path = output_dir / "phase37_scaling_recommended_configs.tex"
    write_latex_table(latex_path, recommendations if not recommendations.empty else latest_success.head(0))
    summary_path = output_dir / "phase37_scaling_summary.md"
    write_summary(summary_path, latest_success, analysis, recommendations if not recommendations.empty else latest_success.head(0), figure_paths, latex_path)
    print(f"Wrote {workbook}")
    if figure_paths:
        print("Figures:")
        for label, path in figure_paths.items():
            print(f"  {label}: {path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {latex_path}")
    print(f"Complete runs: {len(successful)}; incomplete runs: {len(failures)}")


if __name__ == "__main__":
    main()