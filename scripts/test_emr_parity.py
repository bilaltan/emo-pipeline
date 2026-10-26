#!/usr/bin/env python3
"""
EMR (runners/run_emr_step.py, settings from experiment_config.py) must run the local driver's configuration
(scripts/run_local_pipeline.py defaults), so AWS and local numbers come from the same method.

    python scripts/test_emr_parity.py
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import experiment_config as C  # noqa: E402


def local_defaults():
    """--flag -> default of every ap.add_argument in the local driver, read from its source."""
    tree = ast.parse(open(os.path.join(ROOT, "scripts", "run_local_pipeline.py"), encoding="utf-8").read())
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            kw = {k.arg: k.value for k in n.keywords}
            if "default" in kw:
                try:
                    out[n.args[0].value] = ast.literal_eval(kw["default"])
                except ValueError:
                    out[n.args[0].value] = ast.unparse(kw["default"])   # a name such as TRAIN["epochs"]
    return out


def main():
    d = local_defaults()
    checks = {
        "model": (d["--model"], C.GNN_MODELS, d["--model"] in C.GNN_MODELS and len(C.GNN_MODELS) == 1),
        "algorithm": (d["--alg"], C.ALGORITHMS_TO_RUN, C.ALGORITHMS_TO_RUN == [d["--alg"]]),
        "min community size": (d["--min-size"], C.MIN_COMMUNITY_SIZE, d["--min-size"] == C.MIN_COMMUNITY_SIZE),
        "minor communities": (d["--tiny-comm"], C.TINY_COMM_HANDLING, d["--tiny-comm"] == C.TINY_COMM_HANDLING),
        "merge minor (off locally)": (False, C.MERGE_MINOR_COMMUNITIES, C.MERGE_MINOR_COMMUNITIES is False),
        "unit cap": (d["--max-nodes"], C.PHASE3_MAX_NODES_PER_COMMUNITY,
                     d["--max-nodes"] == "PHASE3_MAX_NODES_PER_COMMUNITY"),
        "task": (d["--task"], C.TASK_TYPE, d["--task"] == C.TASK_TYPE),
        "Stage 3a off": (False, C.RUN_PHASE3, C.RUN_PHASE3 is False),
        "training seed": (d["--seed"], C.GCN_CFG.get("seed"), d["--seed"] == C.GCN_CFG.get("seed")),
        "epochs / lr / dropout / wd / hidden": (
            "TRAIN", C.GCN_CFG,
            (C.GCN_CFG["num_epochs"], C.GCN_CFG["lr"], C.GCN_CFG["dropout"], C.GCN_CFG["weight_decay"],
             C.GCN_CFG["hidden_dim"]) == (C.TRAIN["epochs"], C.TRAIN["lr"], C.TRAIN["dropout"],
                                          C.TRAIN["weight_decay"], C.TRAIN["hidden_dim"])),
        "official splits": (True, C.USE_OGB_SPLITS, C.USE_OGB_SPLITS is True),
    }
    bad = [k for k, (_, _, ok) in checks.items() if not ok]
    for k, (loc, emr, ok) in checks.items():
        print(f"  {'ok ' if ok else 'BAD'} {k}: local {loc!r}, EMR {emr!r}")
    assert not bad, f"EMR differs from the local driver: {bad}"
    print("ok: EMR runs the local driver's configuration")


if __name__ == "__main__":
    main()
