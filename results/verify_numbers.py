#!/usr/bin/env python3
"""Recompute every number of Section 4 (setup and 4.1-4.4) from the files in results/ and compare it with the paper.

    python verify_numbers.py        (Python 3.8+, standard library only)

Each line prints OK or DIFF, the paper's value and the value recomputed from the records. Means are over the
runs or seeds of a cell, +- is the sample standard deviation (n - 1).
"""
import csv
import json
import os
import statistics as st

R = os.path.dirname(os.path.abspath(__file__))  # results/
A, B, H, W = (os.path.join(R, d) for d in ("1_tableA_storage", "2_tableB_grl", "3_halo_and_preproc", "4_aws_emr"))
L, M0 = os.path.join(R, "layout_checks"), os.path.join(R, "lp_m0")
fails = []


def load(*p):
    return [json.loads(l) for l in open(os.path.join(*p), encoding="utf-8") if l.strip()]


def ms(v, d=1):
    return f"{st.mean(v):.{d}f} ± {st.stdev(v):.{d}f}"


def check(what, paper, got):
    ok = str(paper) == str(got)
    print(f"{'OK  ' if ok else 'DIFF'} {what}: paper {paper} | records {got}")
    if not ok:
        fails.append(what)


def section(t):
    print(f"\n== {t}")


# ---------------------------------------------------------------- 4 setup, Table 2
section("4 setup, Table 2 (datasets and their Phase 1 units at C_max = 10,000)")
g = {r["dataset"]: r for r in load(B, "baselines.jsonl") if r["baseline"] == "full_graph_sage"}
gp = {r["dataset"]: r for r in load(B, "pipeline.jsonl")}
fd = {r["dataset"].replace("-lp", ""): r["feat_dim"] for r in load(H, "halo_auc.jsonl")}
p2s = {r["dataset"]: r for r in load(A, "write.jsonl") if r["kind"] == "phase2"}
u10 = {r["dataset"]: r["units"] for r in load(A, "unit_probe.jsonl") if r["mode"] == "core" and r["c_max"] == 10000}
paper_stats = {"ogbn-arxiv": ("169,343", "1.16M", "--", "40", "58", "99", "0"),  # 128 features: not in these records
               "reddit": ("232,965", "57.3M", "602", "41", "61", "2", "0"),
               "ogbn-products": ("2,449,029", "61.9M", "100", "47", "610", "4,366", "48,421")}
for ds, (pv, pe, pf, pc, pmaj, pmin, piso) in paper_stats.items():
    iso = p2s[ds]["communities"] - u10[ds]          # every isolated vertex is its own Phase 2 community
    check(f"{ds} |V|", pv, f"{g[ds]['n_nodes']:,}")
    check(f"{ds} |E| (undirected)", pe, f"{g[ds]['n_edges_undirected'] / 1e6:.3g}M")
    check(f"{ds} T_E rows = 2|E|", True, gp[ds]["n_edges"] == 2 * g[ds]["n_edges_undirected"])
    if pf != "--":
        check(f"{ds} features", pf, f"{fd[ds]}")
    check(f"{ds} classes", pc, f"{gp[ds]['n_classes']}")
    check(f"{ds} major units", pmaj, f"{p2s[ds]['communities'] - p2s[ds]['minor']:,}")
    check(f"{ds} minor communities (without isolated vertices)", pmin, f"{p2s[ds]['minor'] - iso:,}")
    check(f"{ds} isolated vertices", piso, f"{iso:,}")

# ---------------------------------------------------------------- 4.1 Table A
section("4.1 Table A (ogbn-products in full; ogbn-arxiv and Reddit read time)")
csvs = {}
for f in ("summary_coalesce.csv", "summary_reddit.csv"):
    for r in csv.DictReader(open(os.path.join(A, f))):
        csvs[(r["dataset"], r["layout"])] = r
w = {(r["dataset"], r["layout"]): r for r in load(A, "write.jsonl") if r["kind"] == "layout"}
reads = {}
for f, dss in (("read_coalesce.jsonl", ("ogbn-arxiv", "ogbn-products")), ("read.jsonl", ("reddit",))):
    for r in load(A, f):
        if r["dataset"] in dss:
            reads.setdefault((r["dataset"], r["layout"]), []).append(r)
paper_products = {  # layout: write, files (<1 MB), read GB (amplification), read s
    "join": ("12.8 ± 0.3", "400 (248)", "614.4 (280x)", "318.2 ± 1.9"),
    "mv": ("21.0 ± 0.6", "400 (46)", "7.53 (3.4x)", "28.6 ± 0.9"),
    "hive": ("508.7 ± 18.2", "58,373 (57,792)", "2.40 (1.1x)", "303.0 ± 1.7"),
    "zorder": ("143.5 ± 3.2", "2 (0)", "1,305 (579x)", "~6,552"),
    "liquid": ("143.3 ± 2.0", "2 (0)", "1,314 (583x)", "~6,591"),
    "merged": ("40.9 ± 0.7", "1,222 (640)", "2.42 (1.1x)", "13.5 ± 0.3"),
    "merged_rg": ("45.7 ± 0.7", "1,222 (640)", "2.35 (1.1x)", "12.3 ± 0.2"),
    "merged_blk": ("36.4 ± 1.4", "1,222 (640)", "2.28 (1.0x)", "12.7 ± 0.1"),
}
for lay, (pw, pf, pg, pr) in paper_products.items():
    ds = "ogbn-products"
    c, rr = csvs[(ds, lay)], reads[(ds, lay)]
    k = rr[0].get("bin_sample", 1)       # Z-order / Liquid: every 10th bin read, time and bytes scaled by 10
    sto = w[(ds, lay)]["storage"]
    files = sum(s["files"] for s in sto.values())
    small = sum(s["files_lt_1mb"] for s in sto.values())
    gb = st.mean(r["bytes"] for r in rr) * k / 1e9
    amp = float(c["amplification"])
    gbs = f"{gb:,.0f}" if gb > 100 and k > 1 else (f"{gb:.1f}" if gb > 100 else f"{gb:.2f}")
    amps = f"{amp:.0f}x" if amp > 10 else f"{amp:.1f}x"
    check(f"products {lay} write (s)", pw, ms(w[(ds, lay)]["write_s"]))
    check(f"products {lay} files (<1 MB)", pf, f"{files:,} ({small:,})")
    check(f"products {lay} read GB (ampl.)", pg, f"{gbs} ({amps})")
    rs = [r["wall_s"] * k for r in rr]
    check(f"products {lay} read (s)", pr, f"~{st.mean(rs):,.0f}" if k > 1 else ms(rs))
for ds, vals in (("ogbn-arxiv", ("8.6 ± 0.1", "6.5 ± 0.1", "6.2 ± 0.0", "49.3 ± 0.2", "49.5 ± 0.0", "0.8 ± 0.0",
                                 "0.9 ± 0.0", "0.9 ± 0.0")),
                 ("reddit", ("40.5 ± 1.6", "10.0 ± 0.3", "5.8 ± 0.1", "348.5 ± 0.4", "346.9 ± 0.6", "6.5 ± 0.5",
                             "6.4 ± 0.2", "6.8 ± 0.1"))):
    for lay, p in zip(paper_products, vals):
        check(f"{ds} {lay} read (s)", p, ms([r["wall_s"] for r in reads[(ds, lay)]]))
r8a = {}
for r in load(A, "read_coalesce_8a.jsonl"):
    r8a.setdefault(r["dataset"], []).append(r)
c8a = {r["dataset"]: r for r in csv.DictReader(open(os.path.join(A, "summary_row8a.csv"))) if r["layout"] == "merged_blk"}
check("row 8a products read GB (ampl.)", "2.42 (1.1x)",
      f"{st.mean(r['bytes'] for r in r8a['ogbn-products']) / 1e9:.2f} ({float(c8a['ogbn-products']['amplification']):.1f}x)")
for ds, p in (("ogbn-products", "13.5 ± 0.9"), ("ogbn-arxiv", "0.9 ± 0.0"), ("reddit", "7.0 ± 0.3")):
    check(f"row 8a {ds} read (s)", p, ms([r["wall_s"] for r in r8a[ds]]))
check("row 8a pure sqrt row groups on products (MB)", "0.25",
      next(r["rg_mb"] for r in load(A, "rowgroup_floor_sweep", "write.jsonl")
           if r.get("layout") == "merged_blk" and r["rg_floor_mb"] == 0.0))

section("4.1 text")
m = lambda ds, lay: st.mean(r["wall_s"] for r in reads[(ds, lay)])  # noqa: E731
check("join slower than 8b", "25x", f"{m('ogbn-products', 'join') / m('ogbn-products', 'merged_blk'):.0f}x")
check("partitionBy slower than 8b", "24x", f"{m('ogbn-products', 'hive') / m('ogbn-products', 'merged_blk'):.0f}x")
check("partitionBy GETs", "116,786", f"{reads[('ogbn-products', 'hive')][0]['gets']:,}")
check("Z-order estimated read (h)", "over 1.8", f"over {6552.1 / 3600:.1f}" if m('ogbn-products', 'zorder') * 10 > 1.8 * 3600 else "no")
check("8b needed-bytes ratio", "1.04", f"{float(csvs[('ogbn-products', 'merged_blk')]['amplification']):.2f}")
rr = {r["layout"]: r for r in csv.DictReader(open(os.path.join(A, "summary_products_roundrobin.csv")))}
check("round-robin rows 6-8 GB", "27-28", f"{min(float(rr[l]['bytes_gb']) for l in ('merged', 'merged_rg', 'merged_blk')):.0f}-"
      f"{max(float(rr[l]['bytes_gb']) for l in ('merged', 'merged_rg', 'merged_blk')):.0f}")
check("round-robin rows 6-8 amplification", "12-13x", f"{min(float(rr[l]['amplification']) for l in ('merged', 'merged_rg', 'merged_blk')):.0f}-"
      f"{max(float(rr[l]['amplification']) for l in ('merged', 'merged_rg', 'merged_blk')):.0f}x")
check("round-robin rows 6-8 read (s)", "about 31",
      f"about {st.mean(float(rr[l]['read_s_mean']) for l in ('merged', 'merged_rg', 'merged_blk')):.0f}")
fl = {f: [r for r in load(A, "rowgroup_floor_sweep", f"read_f{f}.jsonl") if r["layout"] == "merged_blk"] for f in (0, 2, 4, 8, 16)}
check("floor sweep GETs 0 MB -> 2 MB", "30,663 -> 4,614", f"{fl[0][0]['gets']:,} -> {fl[2][0]['gets']:,}")
check("floor sweep read 0 MB -> 2 MB (s)", "43.0 -> 30.6",
      f"{st.mean(r['wall_s'] for r in fl[0]):.1f} -> {st.mean(r['wall_s'] for r in fl[2]):.1f}")
print("     (4, 8, 16 MB floors: " + ", ".join(f"{st.mean(r['wall_s'] for r in fl[f]):.1f} s" for f in (4, 8, 16)) + ")")
p2 = {r["dataset"]: r for r in load(A, "write.jsonl") if r["kind"] == "phase2"}
check("minor communities arxiv / Reddit", "99 / 2", f"{p2['ogbn-arxiv']['minor']} / {p2['reddit']['minor']}")
check("Reddit partitionBy vs ours (s)", "5.8 vs 6.8", f"{m('reddit', 'hive'):.1f} vs {m('reddit', 'merged_blk'):.1f}")
check("arxiv ours vs partitionBy (s)", "0.9 vs 6.2", f"{m('ogbn-arxiv', 'merged_blk'):.1f} vs {m('ogbn-arxiv', 'hive'):.1f}")
check("Reddit partitionBy files", "126", f"{sum(s['files'] for s in w[('reddit', 'hive')]['storage'].values())}")
check("arxiv: ours faster than partitionBy", "7x", f"{m('ogbn-arxiv', 'hive') / m('ogbn-arxiv', 'merged_blk'):.0f}x")
amps = [float(csvs[(d, l)]["amplification"]) for d in ("ogbn-arxiv", "reddit") for l in ("join", "zorder", "liquid")]
check("small graphs: join and clustered layouts read", "28-51x", f"{min(amps):.0f}-{max(amps):.0f}x")
check("small graphs: default view slower than ours (Reddit, arxiv)", "1.5x, 7.6x",
      f"{m('reddit', 'mv') / m('reddit', 'merged_blk'):.1f}x, {m('ogbn-arxiv', 'mv') / m('ogbn-arxiv', 'merged_blk'):.1f}x")
check("readers for rows 4-5 (Z-order, Liquid) on all graphs", "3",
      ",".join(sorted({str(r["procs"]) for (d, l), v in reads.items() if l in ("zorder", "liquid") for r in v})))
check("readers for all other rows", "20",
      ",".join(sorted({str(r["procs"]) for (d, l), v in reads.items() if l not in ("zorder", "liquid") for r in v})))
up = {r["c_max"]: r for r in load(A, "unit_probe.jsonl") if r["dataset"] == "ogbn-products" and r["mode"] == "core"}
check("units at C_max 5,000 / 40,000", "5,555 / 4,537", f"{up[5000]['units']:,} / {up[40000]['units']:,}")
check("units under 100 core vertices", "about 4,370", f"about {round(st.mean(u['units_lt100_core'] for u in up.values()), -1):,.0f}")
check("isolated vertices (Phase 2 communities - probe units at 10,000)", "48,421",
      f"{p2['ogbn-products']['communities'] - up[10000]['units']:,}")
check("products communities / minor in Phase 2", "53,397 / 52,787",
      f"{p2['ogbn-products']['communities']:,} / {p2['ogbn-products']['minor']:,}")

section("4.1 Table A rows 3b-5b (ogbn-products, results/layout_checks)")
lw = {r["layout"]: r for r in load(L, "write.jsonl") if r["kind"] == "layout"}
lr = {}
for f, lays in (("read.jsonl", ("hive_coarse",)), ("read_full20.jsonl", ("zorder_2mb", "liquid_2mb"))):
    for r in load(L, f):
        if r["layout"] in lays:
            lr.setdefault(r["layout"], []).append(r)
c8b = csvs[("ogbn-products", "merged_blk")]
need = float(c8b["bytes_gb"]) / float(c8b["amplification"])   # GB the work items need, the same rows in every layout
paper_new = {"hive_coarse": ("33.4 ± 6.4", "1,222 (676)", "17.29 (7.9x)", "49.7 ± 0.1"),
             "zorder_2mb": ("205.8 ± 14.5", "2 (0)", "151.7 (69x)", "164.9 ± 0.2"),
             "liquid_2mb": ("211.6 ± 2.7", "2 (0)", "152.5 (69x)", "164.9 ± 0.1")}
for lay, (pw, pf, pg, pr) in paper_new.items():
    rr, sto = lr[lay], lw[lay]["storage"]
    assert len(rr) == 3 and len(lw[lay]["write_s"]) == 3
    gb = st.mean(r["bytes"] for r in rr) / 1e9
    check(f"products {lay} write (s)", pw, ms(lw[lay]["write_s"]))
    check(f"products {lay} files (<1 MB)", pf,
          f"{sum(s['files'] for s in sto.values()):,} ({sum(s['files_lt_1mb'] for s in sto.values()):,})")
    check(f"products {lay} read GB (ampl.)", pg,
          f"{gb:.1f} ({gb / need:.0f}x)" if gb > 100 else f"{gb:.2f} ({gb / need:.1f}x)")
    check(f"products {lay} read (s)", pr, ms([r["wall_s"] for r in rr]))
    check(f"products {lay}: 20 readers, every work item", "20, 1",
          f"{','.join(sorted({str(r['procs']) for r in rr}))}, {','.join(sorted({str(r.get('bin_sample', 1)) for r in rr}))}")
    check(f"products {lay}: same rows as rows 1-8", c8b["check_md5"], ",".join(sorted({r["check_md5"] for r in rr})))
rg = {(r["dir"].rstrip("/").split("/")[-2], r["layout"], r["table"]): r for r in load(L, "rg_sizes.jsonl")}
med = lambda lays: [r["compressed_mb_per_rg"]["median"] for (_, l, _), r in rg.items() if l in lays]  # noqa: E731
print("     (median MiB per row group: rows 4-5 " + ", ".join(f"{v:.1f}" for v in med(("zorder", "liquid")))
      + "; rows 4b-5b " + ", ".join(f"{v:.2f}" for v in med(("zorder_2mb", "liquid_2mb"))) + ")")
check("rows 4-5 have the default 128 MB row groups (1 file per table)", True,
      all(127 <= v <= 130 for v in med(("zorder", "liquid")))
      and all(r["files"] == 1 for (_, l, _), r in rg.items() if l in ("zorder", "liquid")))
check("rows 4b-5b have ~2 MB row groups", True, all(1.5 <= v <= 2.5 for v in med(("zorder_2mb", "liquid_2mb"))))
check("row 3b: one row group per file", True,
      all(r["files"] == r["row_groups"] == 611 for (_, l, _), r in rg.items() if l == "hive_coarse"))
lm = lambda lay: st.mean(r["wall_s"] for r in lr[lay])  # noqa: E731
check("rows 4b-5b read (s), slower than 8b", "165, 13x",
      f"{st.mean([lm('zorder_2mb'), lm('liquid_2mb')]):.0f}, "
      f"{max(lm('zorder_2mb'), lm('liquid_2mb')) / m('ogbn-products', 'merged_blk'):.0f}x")
check("rows 4-5 amplification, about", "580x",
      f"{round(st.mean(float(csvs[('ogbn-products', l)]['amplification']) for l in ('zorder', 'liquid')), -1):.0f}x")
check("row 3b: communities per directory, about", "87", f"{p2['ogbn-products']['communities'] / 611:.0f}")
check("row 3b slower than 8b", "3.9x", f"{lm('hive_coarse') / m('ogbn-products', 'merged_blk'):.1f}x")

# ---------------------------------------------------------------- 4.2 Table B
section("4.2 Table B accuracy (%)")
base = load(B, "baselines.jsonl")
pipe = load(B, "pipeline.jsonl")


def row_of(r):
    return 3 if r["tiny_comm"] == "keep" else (4 if r["unit_split"] == "idmod" else 5)


acc = {}
for r in base:
    acc.setdefault((r["dataset"], 1 if r["baseline"] == "full_graph_sage" else 2), []).append(r["test_acc"] * 100)
for r in pipe:
    a = r["stage3a_node_acc"] if row_of(r) == 3 else r["stage3b_node_acc"]
    acc.setdefault((r["dataset"], row_of(r)), []).append(a * 100)
paper_acc = {1: ("64.02 ± 0.25", "95.15 ± 0.08", "73.17 ± 0.14"), 2: ("64.34 ± 0.66", "94.85 ± 0.18", "70.75 ± 0.09"),
             3: ("68.38 ± 0.19", "94.52 ± 0.05", "75.51 ± 0.04"), 4: ("65.47 ± 0.00", "92.29 ± 0.09", "68.67 ± 0.14"),
             5: ("68.49 ± 0.07", "94.62 ± 0.01", "75.95 ± 0.08")}
DS = ("ogbn-arxiv", "reddit", "ogbn-products")
for row, vals in paper_acc.items():
    for ds, p in zip(DS, vals):
        assert len(acc[(ds, row)]) == 3
        check(f"row {row} {ds}", p, ms(acc[(ds, row)], 2))
mean = lambda ds, row: st.mean(acc[(ds, row)])  # noqa: E731
check("row 5 - row 1 (arxiv, products, Reddit)", "4.5, 2.8, -0.5",
      ", ".join(f"{mean(d, 5) - mean(d, 1):.1f}" for d in ("ogbn-arxiv", "ogbn-products", "reddit")))
check("row 5 - row 4 (arxiv, Reddit, products)", "3.0, 2.3, 7.3", ", ".join(f"{mean(d, 5) - mean(d, 4):.1f}" for d in DS))
check("row 5 - row 3 (macro-graph)", "0.1-0.4", f"{min(mean(d, 5) - mean(d, 3) for d in DS):.1f}-{max(mean(d, 5) - mean(d, 3) for d in DS):.1f}")

section("4.2 Table B ogbn-products cost")
ds = "ogbn-products"
rf = {}
for r in load(B, "read_full.jsonl"):
    rf.setdefault((r["dataset"], r["row"]), []).append(r["wall_s"])
ru = {}
for r in load(B, "read_units.jsonl"):
    ru.setdefault((r["dataset"], r["row"]), []).append(r["wall_s"])
check("rows 1-2 read (s)", "2.3", f"{st.mean(rf[(ds, '1-2')]):.1f}")
check("row 3 read (s)", "12.7", f"{st.mean(ru[(ds, '3')]):.1f}")
check("row 4 read (s)", "12.6 + 0.6", f"{st.mean(ru[(ds, '4')]):.1f} + {st.mean(rf[(ds, '4 global')]):.1f}")
check("row 5 read (s)", "7.9 + 0.6", f"{st.mean(ru[(ds, '5')]):.1f} + {st.mean(rf[(ds, '5 global')]):.1f}")
check("row 5 global read (s), the paper's 0.58", "0.58 ± 0.02", ms(rf[(ds, "5 global")], 2))
bt = {}
for r in base:
    if r["dataset"] == ds:
        bt.setdefault(r["baseline"], []).append(r)
check("row 1 train (s)", "114.3", f"{st.mean(r['train_seconds'] for r in bt['full_graph_sage']):.1f}")
check("row 2 train (s)", "1,453.0", f"{st.mean(r['train_seconds'] for r in bt['cluster_gcn']):,.1f}")
check("row 2 METIS (s)", "265.7", f"{bt['cluster_gcn'][0]['partition_seconds']:.1f}")
wall = {}
for r in pipe:
    if r["dataset"] == ds:
        k = next(k for k in r["phase_seconds"] if k.startswith("('phase3"))
        wall.setdefault(row_of(r), []).append(r["phase_seconds"][k])
for row, p in ((3, "242.1"), (4, "286.2"), (5, "301.1")):
    check(f"row {row} train = Stage 3 wall (s)", p, f"{st.mean(wall[row]):.1f}")
pk = {r["label"]: r["peak_gpu_mb"] for r in load(B, "peak.jsonl")}
gpu = {1: pk[f"{ds} full"], 2: pk[f"{ds} cluster"]}
for row in (3, 4, 5):
    gpu[row] = max(pk[f"{ds} row{row} seed{s}"] for s in range(3))
for row, p in zip(range(1, 6), ("20.9", "14.2", "3.1", "3.9", "4.6")):
    check(f"row {row} peak GPU (GB, max over seeds)", p, f"{gpu[row] / 1000:.1f}")
one = {r["label"]: r["peak_gpu_mb"] for r in load(B, "one_worker", "peak.jsonl")}
check("row 5 one worker peak GPU (GB)", "1.6", f"{one[f'{ds} row5 seed0'] / 1000:.1f}")
check("GPU memory: full vs 8 workers / vs 1 worker", "4.5x / 13x",
      f"{gpu[1] / gpu[5]:.1f}x / {gpu[1] / one[f'{ds} row5 seed0']:.0f}x")
check("GPU per worker with 8 workers (GB)", "about 0.6", f"about {gpu[5] / 8 / 1000:.1f}")
check("units + global read of row 5 (s)", "8.5", f"{st.mean(ru[(ds, '5')]) + st.mean(rf[(ds, '5 global')]):.1f}")
check("Cluster-GCN train + METIS (s)", "1,453 + 266", f"{st.mean(r['train_seconds'] for r in bt['cluster_gcn']):,.0f} + {bt['cluster_gcn'][0]['partition_seconds']:.0f}")
check("row 5 workers at once", "8", f"{pipe[-1]['spark_executors'] * pipe[-1]['spark_executor_cores']}")

section("4.2 500-epoch full-graph GraphSAGE and community IDs")
b5 = {}
for r in load(B, "full_sage_500_epochs.jsonl"):
    assert r["train"]["epochs"] == 500
    b5.setdefault(r["dataset"], []).append(r["test_acc"] * 100)
for ds_, p in zip(DS, ("67.63 ± 0.17", "95.56 ± 0.06", "74.61 ± 0.15")):
    check(f"500 epochs {ds_}", p, ms(b5[ds_], 2))
check("row 5 - 500-epoch SAGE (arxiv, products, Reddit)", "0.9, 1.3, -0.9",
      ", ".join(f"{mean(d, 5) - st.mean(b5[d]):.1f}" for d in ("ogbn-arxiv", "ogbn-products", "reddit")))
ci = {}
for r in load(B, "community_id", "baselines.jsonl"):
    assert r["comm_dim"] == 16 and r["min_size"] == 100 and r["train"]["epochs"] == 100
    ci.setdefault(r["dataset"], []).append(r["test_acc"] * 100)
for ds_, p in zip(DS, ("68.18 ± 0.17", "95.36 ± 0.02", "75.58 ± 0.16")):
    check(f"full-graph + community ID {ds_}", p, ms(ci[ds_], 2))
cp = {r["label"]: r["peak_gpu_mb"] for r in load(B, "community_id", "peak.jsonl")}
check("full-graph + community ID products peak GPU (GB)", "20.7", f"{cp['ogbn-products commid'] / 1000:.1f}")
check("community ID - row 5 on Reddit", "0.7", f"{st.mean(ci['reddit']) - mean('reddit', 5):.1f}")

# ---------------------------------------------------------------- 4.3 halo and one-time cost
section("4.3 halo sub-table")
hs = {(r["dataset"], r["halo"]): r for r in load(H, "halo_storage_read.jsonl")}
ha = {}
for r in load(H, "halo_auc.jsonl"):
    ha.setdefault((r["dataset"], "capped" if r["halo_rows"] < r["subgraph_rows"] / 2 else "uncapped"), []).append(r)
paper_halo = {("reddit-lp", "uncapped"): ("2.37M", "6.29", "10.8", "0.941", "0.818"),
              ("reddit-lp", "capped"): ("0.23M", "1.12", "4.4", "0.825", "0.748"),
              ("ogbn-products-lp", "uncapped"): ("9.63M", "4.88", "14.1", "0.954", "0.877"),
              ("ogbn-products-lp", "capped"): ("2.36M", "1.94", "9.6", "0.902", "0.823")}
for key, (pr, pg, pt, pa, pm) in paper_halo.items():
    s, a = hs[key], ha[key]
    assert len(a) == 3 and s["halo_rows"] == a[0]["halo_rows"]
    check(f"{key} halo rows", pr, f"{s['halo_rows'] / 1e6:.2f}M")
    check(f"{key} stored GB", pg, f"{s['bytes_stored'] / 1e9:.2f}")
    check(f"{key} read (s)", pt, f"{s['read_wall_s']:.1f}")
    check(f"{key} cross AUC random", pa, f"{st.mean(r['auc_global_cross'] for r in a):.3f}")
    check(f"{key} cross AUC same-unit", pm, f"{st.mean(r['auc_matched_cross'] for r in a):.3f}")
check("largest AUC std <= 0.01", True, max(st.stdev(r[m] for r in a) for a in ha.values() for m in ('auc_global_cross', 'auc_matched_cross')) <= 0.01)
check("cross links share of test links", "29-43%",
      f"{min(a[0]['n_cross'] / a[0]['n_pos'] for a in ha.values()) * 100:.0f}-{max(a[0]['n_cross'] / a[0]['n_pos'] for a in ha.values()) * 100:.0f}%")
check("Reddit uncapped halo rows / vertices", "10", f"{hs[('reddit-lp', 'uncapped')]['halo_rows'] / 232965:.0f}")
for d, p in (("reddit-lp", "5.6x storage, 2.5x read"), ("ogbn-products-lp", "2.5x storage")):
    u, c = hs[(d, "uncapped")], hs[(d, "capped")]
    got = f"{u['bytes_stored'] / c['bytes_stored']:.1f}x storage" + (f", {u['read_wall_s'] / c['read_wall_s']:.1f}x read" if d == "reddit-lp" else "")
    check(f"{d} cap saves", p, got)
for d, p in (("reddit-lp", "0.07-0.12"), ("ogbn-products-lp", "0.05-0.05")):
    u, c = ha[(d, "uncapped")], ha[(d, "capped")]
    loss = [st.mean(r[m] for r in u) - st.mean(r[m] for r in c) for m in ("auc_global_cross", "auc_matched_cross")]
    check(f"{d} cross AUC loss from the cap", p, f"{min(loss):.2f}-{max(loss):.2f}")
m0 = {}
for r in load(M0, "m0.jsonl"):
    m0.setdefault(r["dataset"], []).append(r)
for d, (pa, pm) in (("reddit-lp", ("0.927", "0.725")), ("ogbn-products-lp", ("0.976", "0.761"))):
    a, ref = m0[d], ha[(d, "uncapped")][0]
    assert sorted(r["seed"] for r in a) == [0, 1, 2] and all(r["method"] == "M0" for r in a)
    check(f"{d} full graph (M0) cross AUC random", pa, f"{st.mean(r['auc_global_cross'] for r in a):.3f}")
    check(f"{d} full graph (M0) cross AUC same-unit", pm, f"{st.mean(r['auc_matched_cross'] for r in a):.3f}")
    check(f"{d} M0: same test links and settings as the halo rows", True,
          all(r["n_pos"] == ref["n_pos"] and r["n_cross"] == ref["n_cross"] and r["cfg"] == ref["cfg"] for r in a))
    check(f"{d} M0 same-unit AUC below both halo rows", True,
          all(st.mean(r["auc_matched_cross"] for r in a) < st.mean(r["auc_matched_cross"] for r in ha[(d, h)])
              for h in ("uncapped", "capped")))
check("M0 AUC std <= 0.01", True,
      max(st.stdev(r[k] for r in a) for a in m0.values() for k in ("auc_global_cross", "auc_matched_cross")) <= 0.01)
check("M0 ogbn-products peak GPU (GB, PyTorch allocation)", "22.1",
      f"{max(r['peak_gpu_mb'] for r in m0['ogbn-products-lp']) / 1000:.1f}")
lk = [v for r in load(H, "halo_storage_read.jsonl") for k, v in r["leakage"].items() if k != "held_out_pairs"]
check("held-out pairs found in any stored table", "0 of 100,000",
      f"{sum(lk)} of {load(H, 'halo_storage_read.jsonl')[0]['leakage']['held_out_pairs']:,}")

section("4.3 one-time cost of Phases 1-2")
t2 = {r["dataset"]: r for r in load(H, "phase12_timing_parallel_split.jsonl")}
t1 = {r["dataset"]: r for r in load(H, "phase12_timing_sequential_split.jsonl")}
ph = lambda r, p: next(v for k, v in r["phase_seconds"].items() if k.startswith(f"('{p}'"))  # noqa: E731
pr_ = t2["ogbn-products"]
check("products Phases 1-2 (s)", "1,782", f"{ph(pr_, 'phase1') + ph(pr_, 'phase2'):,.0f}")
check("products Phase 1 / Phase 2 (s)", "1,354.7 / 427.5", f"{ph(pr_, 'phase1'):,.1f} / {ph(pr_, 'phase2'):.1f}")
check("macro-graph write (s)", "21.9", f"{pr_['phase2_steps']['macro_s']:.1f}")
check("first Leiden pass (s)", "1,145", f"{pr_['phase1_steps']['leiden_s']:,.0f}")
check("recursive split 1 worker -> 8 (s)", "584 -> 111",
      f"{t1['ogbn-products']['phase1_steps']['split_s']:.0f} -> {pr_['phase1_steps']['split_s']:.0f}")
check("split processes", 8, pr_["split_units"]["processes"])
check("arxiv / Reddit Phases 1-2 (s)", "60 / 911",
      " / ".join(f"{ph(t2[d], 'phase1') + ph(t2[d], 'phase2'):.0f}" for d in ("ogbn-arxiv", "reddit")))
check("one-time cost / METIS", "6.7x", f"{(ph(pr_, 'phase1') + ph(pr_, 'phase2')) / bt['cluster_gcn'][0]['partition_seconds']:.1f}x")

# ---------------------------------------------------------------- 4.4 AWS
section("4.4 Exp A: cold reads on S3 (16 readers, 3 runs)")
ea = {}
for r in load(W, "expA_and_W_sweep", "expA_read.jsonl"):
    ea.setdefault(r["layout"], []).append(r)
paper_a = {"join": ("244.1 ± 10.7", "34,419", "614.2"), "mv": ("37.8 ± 14.1", "7,664", "7.42"),
           "hive": ("667.4 ± 128.5", "116,778", "2.40"), "merged_blk": ("16.1 ± 1.9", "2,495", "2.28")}
for lay, (pt, pgets, pgb) in paper_a.items():
    v = ea[lay]
    check(f"S3 {lay} read (s)", pt, ms([r["wall_s"] for r in v]))
    check(f"S3 {lay} GETs", pgets, f"{round(st.mean(r['gets'] for r in v)):,}")
    check(f"S3 {lay} GB", pgb, f"{st.mean(r['bytes'] for r in v) / 1e9:.{1 if lay == 'join' else 2}f}")
lq = ea["liquid"]
k = lq[0].get("bin_sample", 10)  # every k-th work item read; the table row shows k x the measured values
check("S3 Liquid sample: read (s), GETs, GB", "620.7, 2,168, 131.4",
      f"{st.mean(r['wall_s'] for r in lq):.1f}, {round(st.mean(r['gets'] for r in lq)):,}, "
      f"{st.mean(r['bytes'] for r in lq) / 1e9:.1f}")
check("S3 Liquid row: read (s), GETs, GB", "~6,207, ~21,680, ~1,314",
      f"~{st.mean(r['wall_s'] for r in lq) * k:,.0f}, ~{round(st.mean(r['gets'] for r in lq)) * k:,}, "
      f"~{st.mean(r['bytes'] for r in lq) * k / 1e9:,.0f}")
sm = lambda lay: st.mean(r["wall_s"] for r in ea[lay])  # noqa: E731
check("8b vs MV / join / partitionBy", "2.3x / 15x / 41x",
      f"{sm('mv') / sm('merged_blk'):.1f}x / {sm('join') / sm('merged_blk'):.0f}x / {sm('hive') / sm('merged_blk'):.0f}x")
cl = load(W, "expA_and_W_sweep", "counts_local.jsonl")
check("tables counted locally (EMR counts in logs/smoke.log)", "27", f"{len(cl)}")

section("4.4 node scaling (LakeGRL, rows = N = 1, 2, 4 core nodes)")
nr, nt = {}, {}
for r in load(W, "node_scaling", "nodes_read.jsonl"):
    nr.setdefault((r["row"], r["spark_executors"]), []).append(r)
for r in load(W, "node_scaling", "nodes_train.jsonl"):
    r["t3"] = next(v for k, v in r["phase_seconds"].items() if k.startswith("('phase3b'"))
    nt.setdefault((str(row_of(r)), r["spark_executors"]), []).append(r)
tkey = "t3"   # Stage 3 wall time (phase3b; Spark start-up is outside it)
paper_n = {4: ("58.8 ± 6.6", "1.00", "696.6 ± 1.2", "1.00", "74.25 ± 0.25"),
           8: ("28.6 ± 3.2", "2.06", "368.3 ± 1.2", "1.89", "74.25 ± 0.25"),
           16: ("15.0 ± 1.6", "3.93", "209.2 ± 1.9", "3.33", "74.25 ± 0.25")}
r1 = st.mean(r["wall_s"] for r in nr[("5", 4)])
t1_ = st.mean(r[tkey] for r in nt[("5", 4)])
for wk, (prd, prs, ptr, pts, pac) in paper_n.items():
    rd = [r["wall_s"] for r in nr[("5", wk)]]
    tr = [r[tkey] for r in nt[("5", wk)]]
    ac = [r["stage3b_node_acc"] * 100 for r in nt[("5", wk)]]
    assert len(rd) == len(tr) == 3
    check(f"W={wk} read (s)", prd, ms(rd))
    check(f"W={wk} read speedup", prs, f"{r1 / st.mean(rd):.2f}")
    check(f"W={wk} train (s)", ptr, ms(tr))
    check(f"W={wk} train speedup", pts, f"{t1_ / st.mean(tr):.2f}")
    check(f"W={wk} accuracy (%)", pac, ms(ac, 2))
check("speedup 1 -> 4 nodes, share of linear", "83%", f"{t1_ / st.mean(r[tkey] for r in nt[('5', 16)]) / 4 * 100:.0f}%")
ebt = {}
for r in load(W, "expA_and_W_sweep", "expB_train.jsonl"):
    r["t3"] = next(v for k, v in r["phase_seconds"].items() if k.startswith("('phase3b'"))
    ebt[(str(row_of(r)), r["spark_executors"])] = r
ebk = "t3"
check("1 node vs 4 nodes at W=4, training slower by", "about 6%", f"about {(t1_ / ebt[('5', 4)][ebk] - 1) * 100:.0f}%")
check("id mod n vs LakeGRL bytes at N=4 (GB)", "5.93 vs 2.24 (2.7x)",
      f"{st.mean(r['bytes'] for r in nr[('4', 16)]) / 1e9:.2f} vs {st.mean(r['bytes'] for r in nr[('5', 16)]) / 1e9:.2f} "
      f"({st.mean(r['bytes'] for r in nr[('4', 16)]) / st.mean(r['bytes'] for r in nr[('5', 16)]):.1f}x)")
check("id mod n vs LakeGRL read at N=4 (s)", "18.1 vs 15.0",
      f"{st.mean(r['wall_s'] for r in nr[('4', 16)]):.1f} vs {st.mean(r['wall_s'] for r in nr[('5', 16)]):.1f}")
ebr = {}
for r in load(W, "expA_and_W_sweep", "expB_read.jsonl"):
    ebr.setdefault((r["row"], r["spark_executors"]), []).append(r["wall_s"])
check("W = 1 -> 16 on 4 nodes, train (s)", "2,507.4 -> 212.6 (11.79x)",
      f"{ebt[('5', 1)][ebk]:,.1f} -> {ebt[('5', 16)][ebk]:.1f} ({ebt[('5', 1)][ebk] / ebt[('5', 16)][ebk]:.2f}x)")
check("W = 1 -> 16 on 4 nodes, read (s)", "210.6 -> 15.9 (13.23x)",
      f"{st.mean(ebr[('5', 1)]):.1f} -> {st.mean(ebr[('5', 16)]):.1f} ({st.mean(ebr[('5', 1)]) / st.mean(ebr[('5', 16)]):.2f}x)")

print(f"\n{len(fails)} DIFF" + (": " + "; ".join(fails) if fails else ""))
