# Full-graph link prediction reference M0 (advisor 2026-11-18)

`m0.jsonl`: `experiments/trackA/harness.py --methods M0 --seeds 0,1,2 --device cuda --alg leiden_c10000` on reddit-lp
and ogbn-products-lp (runner job `grl-lp-m0`, 2026-11-18 21:33-22:04, rc=0, GPU lock, git 8347d0c; the harness is
unchanged since Table 4's fe671e8). M0 is full-graph GraphSAGE on every training edge, with the same settings as
Table 4's M2 runs (100 epochs, batch 65,536 positives, hidden 256, lr 0.01, dropout 0.5) and the same test links:
`--alg leiden_c10000` gives Table 4's units, so the cross-community positives (21,312 Reddit, 14,538 products) and
the same-unit negatives are the ones of `results/tableA/rleiden_c10000/halo_auc.jsonl`.

Columns as Table 4: `auc_global_cross` = Cross AUC (random negatives), `auc_matched_cross` = Cross AUC (same-unit
negatives). No Hits@K (advisor). Table 4's own records are untouched.
