#!/usr/bin/env python3
"""
Export a local pipeline sweep to a self-describing Excel workbook.

Reads the provenance-stamped JSONL that each run appends and produces one
workbook a reader can interpret without this conversation: the results, the
retention figures the paper actually claims, where the time went, the exact
configuration behind every row, and an explicit list of what is and is not
trustworthy in it.

    python3 scripts/export_results_excel.py
    python3 scripts/export_results_excel.py --in results/local_pipeline_runs.jsonl \
                                            --out results/lakegrl_local_results.xlsx
"""
import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    latest = {}
    for r in rows:                      # keep the newest record per configuration
        latest[(r["dataset"], r["algorithm"])] = r
    order = ["WikiCS", "Coauthor-CS", "Coauthor-Physics", "DeezerEurope", "ogbn-arxiv"]
    return [latest[k] for k in sorted(latest,
                                      key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))]


def phase_times(r):
    agg = {}
    for k, v in (r.get("phase_seconds") or {}).items():
        name = k.split("'")[1] if "'" in k else k
        agg[name] = agg.get(name, 0.0) + v
    return agg


def sheet_results(rows):
    out = []
    for r in rows:
        a3, b3 = r.get("stage3a_node_acc"), r.get("stage3b_node_acc")
        ng, nm = r.get("stage3a_n_head_gnn"), r.get("stage3a_n_head_mlp")
        out.append({
            "Dataset": r["dataset"],
            "Partitioner": r["algorithm"],
            "Backbone": r.get("model", "sage"),
            "Nodes": r.get("n_nodes"),
            "Edges (directed)": r.get("n_edges"),
            "Classes": r.get("n_classes"),
            "Communities (Phase 1)": r.get("n_communities_phase1"),
            "Training units (Phase 3)": r.get("n_units_phase3"),
            "Stage 3a node acc": a3,
            "Stage 3b node acc (LakeGRL)": b3,
            "Fusion gain": (b3 - a3) if (a3 is not None and b3 is not None) else None,
            "Stage 3a link AUC": r.get("stage3a_link_auc"),
            "Stage 3b link AUC (LakeGRL)": r.get("stage3b_link_auc"),
            "Full-graph link AUC": r.get("baseline_link_auc"),
            "Full-graph node acc": r.get("baseline_node_acc"),
            "Runtime (min)": round(r["total_seconds"] / 60, 1),
        })
    return pd.DataFrame(out)


def sheet_retention(rows):
    """The comparison the paper actually claims: LakeGRL against the single-machine
    full-graph upper bound."""
    out = []
    for r in rows:
        lk, bs = r.get("stage3b_link_auc"), r.get("baseline_link_auc")
        ln, bn = r.get("stage3b_node_acc"), r.get("baseline_node_acc")
        t = phase_times(r)
        lake_s = sum(t.get(k, 0.0) for k in ("phase1", "phase2", "phase3", "phase3b"))
        base_s = t.get("phase4", 0.0)
        out.append({
            "Dataset": r["dataset"],
            "Partitioner": r["algorithm"],
            "LakeGRL link AUC": lk,
            "Full-graph link AUC": bs,
            "Link retention %": round(100 * lk / bs, 1) if (lk and bs) else None,
            "LakeGRL node acc": ln,
            "Full-graph node acc": bn if bn else None,
            "Node retention %": round(100 * ln / bn, 1) if (ln and bn) else None,
            "LakeGRL seconds (P1+P2+P3+P3b)": round(lake_s),
            "Full-graph seconds (P4)": round(base_s) if base_s else None,
            # Retention above the single-machine upper bound is not a result: it means
            # the two sides are not solving equally hard problems. Flag it rather than
            # let a reader take it at face value.
            "Trustworthy?": ("yes" if (lk and bs and 0.5 < lk / bs <= 1.05)
                             else "no - negative-sampling mismatch"
                             if (lk and bs) else "no - baseline missing"),
        })
    return pd.DataFrame(out)


def sheet_heads(rows):
    out = []
    for r in rows:
        ng, nm = r.get("stage3a_n_head_gnn"), r.get("stage3a_n_head_mlp")
        tot = (ng or 0) + (nm or 0)
        out.append({
            "Dataset": r["dataset"],
            "Partitioner": r["algorithm"],
            "GNN head acc": r.get("stage3a_acc_gnn_head"),
            "Probe head acc": r.get("stage3a_acc_mlp_head"),
            "Selected acc (per-unit, on validation)": r.get("stage3a_node_acc"),
            "Units choosing GNN head": ng,
            "Units choosing probe head": nm,
            "GNN head win rate": round(ng / tot, 3) if tot else None,
        })
    return pd.DataFrame(out)


def sheet_timing(rows):
    out = []
    for r in rows:
        t = phase_times(r)
        total = r["total_seconds"]
        acc = sum(t.get(k, 0.0) for k in ("phase1", "phase2", "phase3", "phase3b", "phase4"))
        out.append({
            "Dataset": r["dataset"],
            "Partitioner": r["algorithm"],
            "Phase 1 community detection (s)": round(t.get("phase1", 0)),
            "Phase 2 partition + abstraction (s)": round(t.get("phase2", 0)),
            "Phase 3 local training (s)": round(t.get("phase3", 0)),
            "Phase 3b CAAN fusion (s)": round(t.get("phase3b", 0)),
            "Phase 4 full-graph baseline (s)": round(t.get("phase4", 0)),
            "Total (s)": round(total),
            "Unaccounted overhead (s)": round(total - acc),
            "Overhead share": round((total - acc) / total, 3) if total else None,
        })
    return pd.DataFrame(out)


def sheet_provenance(rows):
    out = []
    for r in rows:
        out.append({
            "Dataset": r["dataset"],
            "Partitioner": r["algorithm"],
            "Backbone": r.get("model", "sage"),
            "Max epochs (ceiling, early-stopped)": r.get("epochs"),
            "Per-unit node cap": r.get("max_nodes") if "max_nodes" in r else None,
            "Per-unit edge cap": r.get("max_edges_cap"),
            "Edges trained on": r.get("edges_retained_phase3"),
            "Run completed (UTC)": r.get("timestamp_utc"),
            "Runtime (s)": round(r["total_seconds"]),
        })
    return pd.DataFrame(out)


README = [
    ("What this is",
     "A local, single-machine run of the full LakeGRL pipeline (Phase 0 ingestion through "
     "Phase 4 baseline) on Apache Spark 3.5 with Delta Lake, across four benchmark graphs "
     "and both partitioning algorithms. Every figure is produced by the pipeline itself; "
     "nothing is copied from prior documentation."),
    ("Reproducibility",
     "All runs are deterministic: community detection, per-unit weight initialisation and "
     "Phase 3b row ordering are seeded and pinned. Repeating a configuration reproduces its "
     "numbers exactly. This was not previously the case."),
    ("Stage 3a vs Stage 3b",
     "Stage 3a is decoupled per-community training with no cross-partition information. "
     "Stage 3b adds the CAAN global abstraction and is the LakeGRL result."),
    ("Classifier head",
     "The encoder's own classifier and a probe head over frozen embeddings are both scored, "
     "and each community keeps whichever validates better. Neither wins everywhere, so the "
     "selection is measured rather than assumed. See the 'Classifier heads' sheet."),
    ("TRUSTWORTHY: node accuracy",
     "Stage 3a and Stage 3b node accuracies are sound and reproducible."),
    ("TRUSTWORTHY: link retention on WikiCS",
     "WikiCS link retention (~99% of the single-machine full-graph upper bound) is the one "
     "clean retention figure in this workbook."),
    ("NOT TRUSTWORTHY: other link retention",
     "Phase 3 draws negative edges from within a community while Phase 4 draws them from the "
     "whole graph. Within-community link prediction is the easier task, so retention above "
     "100% reflects that mismatch, not a genuine result. A shared negative pool is needed."),
    ("NOT AVAILABLE: node retention",
     "Full-graph node accuracy is 0 on every run because the baseline's neighbour sampler "
     "requires DGL, which has no wheel for this platform. Node retention therefore cannot be "
     "computed locally and needs a single-machine EC2 run."),
    ("NOT MEASURABLE LOCALLY: the speed claim",
     "On one machine the full-graph baseline is faster than LakeGRL, as expected: a "
     "distributed architecture run on four local threads pays partitioning overhead with no "
     "executors to gain from. The scalability claim requires a cluster."),
    ("Known gaps",
     "Coauthor-Physics produced no baseline. Reddit, ogbn-products, LiveJournal, Orkut and "
     "ogbn-papers100M have not been run through the current pipeline; those are the paper's "
     "headline datasets and remain the next step."),
    ("Overhead",
     "About 40% of each run's wall clock falls outside every phase timer (Spark session "
     "startup, Delta commits, job scheduling). See the 'Phase timing' sheet."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="results/local_pipeline_runs.jsonl")
    ap.add_argument("--out", dest="dst", default="results/lakegrl_local_results.xlsx")
    args = ap.parse_args()

    rows = load(args.src)
    meta = pd.DataFrame(
        [{"Item": k, "Detail": v} for k, v in README]
        + [{"Item": "Runs in this workbook", "Detail": str(len(rows))},
           {"Item": "Exported (UTC)",
            "Detail": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")},
           {"Item": "Environment",
            "Detail": "Spark 3.5.0, Delta 3.2.0, JDK 17, local[4], Apple M4 / 32GB"}])

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with pd.ExcelWriter(args.dst, engine="openpyxl") as xw:
        meta.to_excel(xw, sheet_name="Read Me", index=False)
        sheet_results(rows).to_excel(xw, sheet_name="Results", index=False)
        sheet_retention(rows).to_excel(xw, sheet_name="Retention vs full graph", index=False)
        sheet_heads(rows).to_excel(xw, sheet_name="Classifier heads", index=False)
        sheet_timing(rows).to_excel(xw, sheet_name="Phase timing", index=False)
        sheet_provenance(rows).to_excel(xw, sheet_name="Run provenance", index=False)

        for name, ws in xw.sheets.items():                # readable column widths
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 12), 60)

    print(f"wrote {args.dst}  ({len(rows)} runs, 6 sheets)")


if __name__ == "__main__":
    main()
