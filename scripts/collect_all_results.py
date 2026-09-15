#!/usr/bin/env python3
"""
One table for every run, across every cluster size.

Source of truth is the Spark history server, not the run logs. That matters:
the history server records how many DISTINCT HOSTS an application actually used,
which is the variable that turned out to govern throughput. Run logs record what
was requested, which is not always what YARN granted, and older logs do not
record node count at all.

Per-task medians come from the same place, and they are the number that
distinguishes the two failure modes seen in this project:

  * constant median across slot counts, with a fixed max  -> a real skew ceiling
  * median rising with tasks-per-node                     -> oversubscription

Stage identity is inferred from task count, which is stable per dataset:

    reddit         Phase 3 = 113 tasks,  Phase 3b = 149 tasks
    ogbn-products  Phase 3 = 575 tasks,  Phase 3b = 823 tasks

Usage
-----
    python3 scripts/collect_all_results.py
    python3 scripts/collect_all_results.py --csv results/all_runs.csv
    python3 scripts/collect_all_results.py --host localhost --port 18080
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.request

# task count -> (dataset, phase)
STAGE_SHAPES = {
    113: ("reddit", "phase3"),
    149: ("reddit", "phase3b"),
    575: ("ogbn-products", "phase3"),
    823: ("ogbn-products", "phase3b"),
}


def parse_ts(t):
    return dt.datetime.strptime(t[:23], "%Y-%m-%dT%H:%M:%S.%f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--csv", default=None, help="also write rows to this CSV")
    ap.add_argument("--json", default=None,
                    help="write a self-describing JSON: field documentation, the "
                         "constants held fixed, every run, and derived curves")
    ap.add_argument("--min-stage-seconds", type=float, default=60.0,
                    help="ignore trivial stages")
    args = ap.parse_args()

    base = "http://%s:%d/api/v1" % (args.host, args.port)

    def get(path):
        return json.load(urllib.request.urlopen(base + path, timeout=90))

    try:
        apps = get("/applications")
    except Exception as e:
        print("history server unreachable at %s (%s)" % (base, type(e).__name__))
        return 1

    rows = []
    for a in apps:
        aid = a["id"]
        name = a.get("name") or ""
        try:
            execs = [e for e in get("/applications/%s/executors" % aid)
                     if e.get("id") != "driver"]
            stages = get("/applications/%s/stages" % aid)
        except Exception:
            continue
        if not execs:
            continue

        # DISTINCT HOSTS is the governing variable, not the executor count.
        hosts = {(e.get("hostPort") or "").split(":")[0] for e in execs}
        hosts.discard("")
        nodes = len(hosts)
        cores = sum(e.get("totalCores", 0) for e in execs)

        omp = ""
        try:
            env = get("/applications/%s/environment" % aid)
            for k, v in env.get("sparkProperties", []):
                if k.endswith("OMP_NUM_THREADS"):
                    omp = v
                    break
        except Exception:
            pass

        by_phase = {}
        for s in stages:
            shape = STAGE_SHAPES.get(s.get("numTasks"))
            if not shape:
                continue
            dataset, phase = shape
            wall = None
            if s.get("submissionTime") and s.get("completionTime"):
                wall = (parse_ts(s["completionTime"])
                        - parse_ts(s["submissionTime"])).total_seconds()
            if not wall or wall < args.min_stage_seconds:
                continue
            total = (s.get("executorRunTime") or 0) / 1000.0
            med = mx = None
            try:
                q = get("/applications/%s/stages/%d/%d/taskSummary?quantiles=0.5,1.0"
                        % (aid, s["stageId"], s.get("attemptId", 0)))
                v = q.get("executorRunTime") or []
                if v:
                    med, mx = v[0] / 1000.0, v[-1] / 1000.0
            except Exception:
                pass
            # keep the most expensive attempt of each phase
            prev = by_phase.get(phase)
            if prev is None or total > prev["sum"]:
                by_phase[phase] = {"dataset": dataset, "wall": wall, "sum": total,
                                   "median": med, "max": mx,
                                   "tasks": s.get("numTasks")}

        if not by_phase:
            continue
        dataset = next(iter(by_phase.values()))["dataset"]
        p3 = by_phase.get("phase3", {})
        p3b = by_phase.get("phase3b", {})
        # YARN does not always place an executor on every node: one 8-node run
        # landed on 7 hosts, so its tasks-per-node is not what the sweep assumed.
        # Flag it rather than let it read as a clean point.
        rows.append({
            "app": aid,
            "dataset": dataset,
            "nodes": nodes,
            "execs": len(execs),
            "cores": cores,
            "omp": omp or "1",
            "slots": cores,
            "tasks_per_node": round(cores / nodes, 1) if nodes else 0,
            "p3_stage_s": round(p3.get("wall"), 1) if p3.get("wall") else "",
            "p3_median": round(p3["median"], 1) if p3.get("median") else "",
            "p3_max": round(p3["max"], 1) if p3.get("max") else "",
            "p3_sum": round(p3["sum"], 0) if p3.get("sum") else "",
            "usable_par": (round(p3["sum"] / p3["max"], 1)
                           if p3.get("sum") and p3.get("max") else ""),
            "p3b_stage_s": round(p3b.get("wall"), 1) if p3b.get("wall") else "",
            "p3b_median": round(p3b["median"], 1) if p3b.get("median") else "",
        })

    if not rows:
        print("no matching runs found")
        return 1

    rows.sort(key=lambda r: (r["dataset"], -r["nodes"], -r["cores"]))

    hdr = ["dataset", "nodes", "execs", "cores", "omp", "tasks_per_node",
           "p3_stage_s", "p3_median", "p3_max", "usable_par", "p3b_stage_s"]
    widths = [15, 6, 6, 6, 5, 9, 11, 10, 9, 11, 11]
    print()
    print("  " + "".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("  " + "-" * sum(widths))
    last = None
    for r in rows:
        if last and last != r["dataset"]:
            print()
        last = r["dataset"]
        print("  " + "".join(str(r.get(h, "")).ljust(w) for h, w in zip(hdr, widths)))

    odd = [r for r in rows if r["nodes"] not in (1, 2, 4, 8, 12, 16)]
    if odd:
        print()
        print("  CAVEAT: %d run(s) landed on an unexpected host count "
              "(YARN placement)." % len(odd))
        for r in odd:
            print("     %s: %d hosts, %d executors -> tasks/node %.1f"
                  % (r["dataset"], r["nodes"], r["execs"], r["tasks_per_node"]))
        print("  Those points are not clean node-scaling measurements.")
    print()
    print("  p3_stage_s     = Spark STAGE wall clock. The pipeline's own")
    print("                   'Wall time:' line is ~10%% higher because it also")
    print("                   counts setup. Do not mix the two in one table.")
    print("  usable_par     = sum/max from Spark executorRunTime, which includes")
    print("                   per-task overhead; UDF-internal timings give a")
    print("                   higher figure for products. Same metric, different")
    print("                   basis -- keep one basis per table.")
    print("  nodes          = distinct hosts the application actually ran on.")
    print("                   YARN does not guarantee one executor per node: two")
    print("                   8-executor runs on the same 8-node cluster reached")
    print("                   8 and 7 hosts. Use executor counts well above the")
    print("                   node count, or verify coverage per run.")
    print("  tasks_per_node = cores / nodes; per-task cost tracks this closely")
    print("  p3_median      = per-task median; rising with tasks_per_node means")
    print("                   oversubscription rather than a genuine ceiling")

    if args.json:
        # Self-describing on purpose: anyone opening this file later should not
        # need the conversation that produced it.
        doc = {
            "generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "what_this_is":
                "Every measured run of the LakeGRL pipeline, read from the Spark "
                "history server rather than from run logs.",
            "cluster": {
                "instance_type": "r6id.8xlarge",
                "vcpus_per_node": 32,
                "physical_cores_per_node": 16,
                "note": "32 vCPUs are 16 physical cores with 2 threads each. "
                        "Reporting efficiency against vCPUs overstates it 2x.",
            },
            "held_constant": {
                "dataset_partitioning": "phases 0-2 reused from S3, so every run "
                                        "trains the identical unit set",
                "epochs": 10,
                "executor_cores": 4,
                "note": "Only node count and slot count vary across the scaling runs.",
            },
            "field_documentation": {
                "nodes": "Distinct hosts the application actually ran on. Measured, "
                         "not requested: YARN does not guarantee one executor per "
                         "node, and one nominally 8-node run landed on 7 hosts.",
                "cores": "Total executor cores = slots = concurrent tasks.",
                "omp": "OMP_NUM_THREADS per Python worker. Tasks are otherwise "
                       "single-threaded.",
                "tasks_per_node": "cores / nodes. Per-task cost tracks this more "
                                  "closely than it tracks total core count.",
                "p3_stage_s": "Spark STAGE wall clock for Phase 3. The pipeline's "
                              "own 'Wall time:' line runs about 10% higher because "
                              "it also counts setup. Do not mix the two.",
                "p3_median": "Median per-task runtime. Rising with tasks_per_node "
                             "indicates oversubscription; flat with a fixed max "
                             "indicates a genuine straggler ceiling.",
                "p3_max": "Slowest single task. A stage cannot finish before this.",
                "usable_par": "sum(task time) / max(task time). The most slots the "
                              "stage could ever benefit from. Computed here from "
                              "Spark executorRunTime, which includes per-task "
                              "overhead; UDF-internal timings give a higher number "
                              "for products. Keep one basis per table.",
                "p3b_stage_s": "Same, for Phase 3b (CaaN fusion).",
            },
            "key_findings": [
                "Throughput scales with NODES, not cores. Doubling nodes at a fixed "
                "slot count roughly halved per-task time; doubling slots on fixed "
                "nodes bought about 1.10x.",
                "Per-task median is a function of tasks-per-node: ~13.5s at 8/node, "
                "~21s at 16/node, ~37s at 32/node.",
                "The cost driver reverses by dataset: corr(n_train, time) was +0.52 "
                "on reddit and -0.02 on products, while corr(n_edges, time) was "
                "-0.01 on reddit and +0.62 on products. No single balancing "
                "heuristic is correct for both.",
                "reddit cannot demonstrate scalability at any cluster size: usable "
                "parallelism is 36 under every partitioner tested (louvain, leiden, "
                "lpa, igraph_lpa all give 22-24 on member counts).",
                "Phase 3 carries a serial floor of roughly 40-55s (load, broadcast, "
                "collect) that never parallelises.",
            ],
            "runs": rows,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=2)
        print("\n  wrote %s (%d runs)" % (args.json, len(rows)))

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n  wrote %s (%d rows)" % (args.csv, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
