# Full-graph link prediction reference: the "full graph (M0)" rows of Table 5

`m0.jsonl`: `experiments/trackA/harness.py --methods M0 --seeds 0,1,2 --device cuda --alg leiden_c10000` on
`reddit-lp` and `ogbn-products-lp`. M0 is full-graph GraphSAGE trained on every training edge, with the same settings
as the halo rows (`../3_halo_and_preproc/halo_auc.jsonl`: 100 epochs, batch 65,536 positives, hidden 256, lr 0.01,
dropout 0.5) and the same test links: `--alg leiden_c10000` gives the same units, so the cross-community positives
(21,312 on Reddit, 14,538 on ogbn-products) and the same-unit negatives are the ones of the halo rows.

Columns as Table 5: `auc_global_cross` = Cross AUC (random negatives), `auc_matched_cross` = Cross AUC (same-unit
negatives). `peak_gpu_mb` = `torch.cuda.max_memory_allocated()` in MiB (ogbn-products: 22,063, the paper's 22.1 GB).
M0 draws its training negatives uniformly from all vertices; the unit models of the halo rows draw them from their
own unit.
