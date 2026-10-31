#!/usr/bin/env bash
# Row-group floor test (advisor 2026-10-31): rows 6-8 of one dataset with the bounded √ rule max(√ rule, F MB) for
# each floor F, 3 writes + 3 cold reads each, as run.sh. Then the floor where row 8 reads fewer bytes AND is faster
# than row 7 (the fastest row 8 among them) goes to results/tableA/rgfloor/floor_mb.txt for the Table A rerun; rc 3
# when no floor does. F = 0 is the pure √ rule.
#   bash experiments/tableA/rg_floor.sh ogbn-products 0 2 4 8 16
set -eo pipefail
ds=$1; shift
cd "$(dirname "$0")/../.."
res=results/tableA/rgfloor
mkdir -p "$res"
for f in "$@"; do
  tab=$HOME/lakegrl/tableA/rgfloor/f$f
  bash scripts/wsl_run.sh python experiments/tableA/write_layouts.py --dataset "$ds" --alg leiden --repeats 3 \
    --layouts mv merged merged_rg merged_blk --cap 10000 --unit-split leiden --rg-floor-mb "$f" --out-dir "$tab" \
    --out "$res/write.jsonl"
  bash scripts/wsl_run.sh python experiments/tableA/read_layouts.py --dataset "$ds" --runs 3 --out-dir "$tab" \
    --out "$res/read_f$f.jsonl" --layouts merged merged_rg merged_blk
  rm -rf "$tab"   # only the jsonl records are kept
done
python3 - "$res" "$ds" "$@" <<'EOF'
import json, sys
res, ds, floors = sys.argv[1], sys.argv[2], sys.argv[3:]
rows, ok = {}, []
for f in floors:
    rd = [r for r in map(json.loads, open(f"{res}/read_f{f}.jsonl")) if r["dataset"] == ds]
    m = {k: [r for r in rd if r["layout"] == k][-3:] for k in ("merged", "merged_rg", "merged_blk")}
    rows[f] = {k: (sum(r["bytes"] for r in v) / len(v) / 1e9, sum(r["wall_s"] for r in v) / len(v),
                   sum(r["gets"] for r in v) / len(v)) for k, v in m.items()}
    print(f"floor {f:>3} MB | " + " | ".join(f"{k} {b:.2f} GB {t:.1f} s {g:,.0f} GETs" for k, (b, t, g) in rows[f].items()))
    (b7, t7, _), (b8, t8, _) = rows[f]["merged_rg"], rows[f]["merged_blk"]
    if b8 < b7 and t8 < t7:
        ok.append(f)
if not ok:
    print("no floor gives row 8 fewer bytes and a faster read than row 7")
    sys.exit(3)
best = min(ok, key=lambda f: rows[f]["merged_blk"][1])
open(f"{res}/floor_mb.txt", "w").write(best + "\n")
print(f"floors where row 8 beats row 7 on bytes and time: {ok}; chosen (fastest row 8): {best} MB")
EOF
