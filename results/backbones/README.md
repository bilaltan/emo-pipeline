# Backbones: GraphSAGE, GCN and GATv2 on one stored view

LakeGRL trains each backbone by Stage 3b alone (`run_local_pipeline.py --stage3-only --model <backbone>`) on the same
stored Phase 2 view: the units of Table 4 row 5 at C_max = 10,000 (recursive Leiden, minor communities dropped),
written once by Phases 1-2 and read unchanged by every backbone. The full-graph baseline trains the same layers on
the whole graph.

- Layers (units, their global model and the full-graph baseline): conv(f, 256) -> activation -> dropout ->
  conv(256, 256) -> linear head. GraphSAGE as Table 4; GCN: GCNConv on D^-1/2 (A + I) D^-1/2, computed once per
  graph, ReLU; GATv2: GATv2Conv with 8 heads x 32 (concatenated), then 1 head x 256, ELU, attention dropout.
- As Table 4: official split, 100 epochs, lr 0.01, best-validation checkpoint, seeds 0-2, one 24 GB GPU; LakeGRL
  runs 8 workers sharing the GPU (Spark local, 1 executor x 8 cores).
- Peak GPU: `peak_gpu_mb` of `peak.jsonl` (nvidia-smi minus the idle reading), the largest over the seeds.
- OOM: the run filled the 24 GB card and was stopped (`gpu_full`: 1, rc 137), or CUDA refused one allocation (`oom`
  in the baseline record; Reddit's full graph asked for 109.5 GiB). Reddit's densest unit holds ~4.2M edges and
  GATv2 keeps ~8 KB per edge, so it does not fit even as the only worker (`gatv2_one_worker/`).

| File | Contents |
|---|---|
| `gcn/baselines.jsonl`, `pipeline.jsonl`, `peak.jsonl` | GCN: full graph (`full_graph_gcn`) and LakeGRL, 3 graphs x 3 seeds; peaks labelled `<graph> full` and `<graph> row5 seed<s>` |
| `gatv2/baselines.jsonl`, `pipeline.jsonl`, `peak.jsonl` | GATv2: the `oom` records of the Reddit and ogbn-products full graphs, LakeGRL on ogbn-arxiv, and the peaks (ogbn-arxiv full graph, Reddit and ogbn-products LakeGRL: `gpu_full`) |
| `gatv2_one_worker/peak.jsonl` | Reddit GATv2 LakeGRL with one worker, seed 0: the card fills on the second unit |
| `sage/pipeline.jsonl`, `peak.jsonl` | GraphSAGE LakeGRL by Stage 3b alone; its full graph is Table 4's (`../2_tableB_grl/baselines.jsonl`, `peak.jsonl`) |

| Graph | Backbone | Full-graph Test Acc (%) | Full-graph Peak GPU (GB) | LakeGRL Test Acc (%) | LakeGRL Stage 3 Train (s) | LakeGRL Peak GPU (GB) |
|---|---|---|---|---|---|---|
| ogbn-arxiv | GraphSAGE | 64.02 ± 0.25 | 1.7 | 68.48 ± 0.02 | 36.4 ± 1.9 | 3.4 |
| ogbn-arxiv | GCN | 64.79 ± 0.26 | 1.5 | 68.49 ± 0.02 | 36.8 ± 0.1 | 3.2 |
| ogbn-arxiv | GATv2 | OOM | OOM | 68.60 ± 0.11 | 93.4 ± 0.2 | 17.0 |
| Reddit | GraphSAGE | 95.15 ± 0.08 | 11.4 | 94.60 ± 0.02 | 96.2 ± 1.1 | 5.7 |
| Reddit | GCN | 92.16 ± 0.01 | 10.6 | 93.49 ± 0.02 | 89.9 ± 0.4 | 5.5 |
| Reddit | GATv2 | OOM | OOM | OOM | – | OOM |
| ogbn-products | GraphSAGE | 73.17 ± 0.14 | 20.9 | 75.96 ± 0.06 | 297.9 ± 0.7 | 4.6 |
| ogbn-products | GCN | 68.20 ± 0.11 | 22.4 | 75.28 ± 0.11 | 327.9 ± 1.3 | 4.1 |
| ogbn-products | GATv2 | OOM | OOM | OOM | – | OOM |

Accuracy = `test_acc` (full graph) and `stage3b_node_acc` (LakeGRL); Stage 3 = `phase_seconds` `phase3b`; mean ±
sample std over seeds 0-2.
