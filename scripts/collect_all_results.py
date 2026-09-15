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
        rows.append({
            "app": aid,
            "dataset": dataset,
            "nodes": nodes,
            "execs": len(execs),
            "cores": cores,
            "omp": omp or "1",
            "slots": cores,
            "tasks_per_node": round(cores / nodes, 1) if nodes else 0,
            "phase3_s": round(p3.get("wall"), 1) if p3.get("wall") else "",
            "phase3_median": round(p3["median"], 1) if p3.get("median") else "",
            "phase3_max": round(p3["max"], 1) if p3.get("max") else "",
            "phase3_sum": round(p3["sum"], 0) if p3.get("sum") else "",
            "usable_par": (round(p3["sum"] / p3["max"], 1)
                           if p3.get("sum") and p3.get("max") else ""),
            "phase3b_s": round(p3b.get("wall"), 1) if p3b.get("wall") else "",
            "phase3b_median": round(p3b["median"], 1) if p3b.get("median") else "",
        })

    if not rows:
        print("no matching runs found")
        return 1

    rows.sort(key=lambda r: (r["dataset"], -r["nodes"], -r["cores"]))

    hdr = ["dataset", "nodes", "execs", "cores", "omp", "tasks_per_node",
           "phase3_s", "phase3_median", "phase3_max", "usable_par", "phase3b_s"]
    widths = [14, 6, 6, 6, 4, 8, 9, 9, 9, 10, 9]
    print()
    print("  " + "".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("  " + "-" * sum(widths))
    last = None
    for r in rows:
        if last and last != r["dataset"]:
            print()
        last = r["dataset"]
        print("  " + "".join(str(r.get(h, "")).ljust(w) for h, w in zip(hdr, widths)))

    print()
    print("  nodes          = distinct hosts the application actually ran on")
    print("  tasks_per_node = cores / nodes; per-task cost tracks this closely")
    print("  usable_par     = sum(task time) / longest task = the speedup ceiling")
    print("  phase3_median  = per-task median; rising with tasks_per_node means")
    print("                   oversubscription rather than a genuine ceiling")

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
