# Community-ID feature ablation (advisor 2026-11-13, reviewer 3)

Full-graph GraphSAGE (Table B row 1's model and settings: 2-layer mean SAGE, hidden 256, dropout 0.5, Adam lr 0.01,
wd 5e-4, 100 epochs, best-validation checkpoint, official splits) with a learned 16-dim embedding of each vertex's
Phase 1 community concatenated to its features: x~ = [x || E[c(v)]]. Communities: `communities/leiden_c10000` (the
row 5 store); major = >= 100 members (58 / 61 / 610 on arxiv / Reddit / products, the row 5 unit counts) get rows
0..K-1, every minor or isolated vertex shares row K. Local RTX A5000, seeds 0-2. Tables A and B are untouched.

    experiments/baselines/node_baselines.py --method full --communities <store>/<ds>/communities/leiden_c10000 --device cuda
    runner job grl-commid rev 6 (2026-11-13 22:53-23:11)

| Dataset | Row 1 full-graph SAGE | + community ID | Row 5 LakeGRL | Train (s) | Peak GPU, nvidia-smi (GB) |
|---|---|---|---|---|---|
| ogbn-arxiv | 64.02 +- 0.25 | 68.18 +- 0.17 | 68.49 +- 0.07 | 5.5 | 1.2 |
| Reddit | 95.15 +- 0.08 | 95.36 +- 0.02 | 94.62 +- 0.01 | 153.3 | 10.3 |
| ogbn-products | 73.17 +- 0.14 | 75.58 +- 0.16 | 75.95 +- 0.08 | 131.9 | 20.7 |

Files: `baselines.jsonl` (one record per run), `peak.jsonl` + `peak_trace/` (experiments/tableB/peak.sh, GPU minus
idle). Code 2b54d54 for products; arxiv and Reddit record cfedf91, which only changed scripts/emr_cluster.sh.

## Fitting ogbn-products on the 24 GB card

The model is the encoder on [x || E[c]], but computed so that products fits next to row 1's ~20 GB: layer 1's
weights apply to the x and embedding columns apart, A @ E[c] is M @ E with M = A @ onehot(c) (12.0M entries),
each layer sums its products in one [N, 256] buffer with an in-place ReLU, and layer 1 is recomputed in backward
(torch checkpoint, same dropout masks). test_node_baselines.py checks the forward equals the encoder on the
concatenated input. The earlier forms all stopped at the card limit (peak.sh kills at 23.5 GB); their partial
outputs are kept:
- `attempt1_concat/` (128bd40): plain concatenation; backward builds [N, 116] gradients for the constant features.
- `attempt2_split/` (7aee18f): split weights; the gradient of A @ E[c] goes through A's transpose.
- `attempt3_M/` (531b72c, 0f7d7ae): M @ E; four separate [N, 256] products.
- `attempt4_probe/`: 3-epoch probe of 0f7d7ae, same failure. A one-step probe by hand (scratchpad commid_mem_probe.py,
  under gpu_lock) showed allocator fragmentation at the edge of the card: 19.18 GiB allocated vs 18.89 for row 1's
  model; `expandable_segments` fails under WSL.
- `attempt5_alloc/` (f7da77e): PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128; still
  23.5 GB in the training loop. The final form (2b54d54) needs no allocator setting (one step: 15.05 GiB allocated).
arxiv and Reddit also finished in attempts 1 and 2; their means agree with the final run within 0.3 points (GPU
sparse ops are not deterministic).
