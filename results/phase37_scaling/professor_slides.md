# Phase 3.7/3.8: Three-Slide Professor Update

## Slide 1 - Why a New Global Graph Path?

**Title:** Full-Graph Learning Without Loading Papers100M on One Machine

- The existing pipeline trains bounded local GNNs per community. It is useful for community experiments, but each local model sees only a bounded subgraph.
- The new path uses the same EMO data model for every dataset: distributed Delta tables for `nodes`, `edges`, and official `splits`.
- Spark computes two fixed neighborhood-mean propagation hops over the full stored graph, then checkpoints each hop in Delta Lake.
- Each node representation is $[x^{(0)} \Vert x^{(1)} \Vert x^{(2)}]$: original features plus one-hop and two-hop neighborhood means.
- A single global Spark ML multinomial logistic-regression classifier is trained on the official training split.

**Say:** “This is not a replacement claim for GAT or GraphSAGE. It is a scalable global graph-learning execution path that avoids driver collection and unbounded Python adjacency lists.”

```mermaid
flowchart LR
  A[Delta nodes, edges, splits] --> B[Distributed hop 1 mean]
  B --> C[Delta checkpoint]
  C --> D[Distributed hop 2 mean]
  D --> E[Cached x0, x1, x2]
  E --> F[One global logistic classifier]
```

## Slide 2 - Evaluation Protocol and Coverage

**Title:** Same General Pipeline Tested from Small to Papers100M Scale

| Dataset | Scale | Executor settings | Coverage | Test accuracy |
|---|---:|---|---:|---:|
| WikiCS | 11,701 nodes; 431,206 propagation edges | 8, 16, 32, 64 | 100% nodes | 0.7746 |
| ogbn-products | 2,449,029 nodes; 123,718,024 propagation edges | 8, 16, 32, 64 | 100% nodes | 0.7068 |
| ogbn-papers100M | 111,059,956 nodes; 3,228,124,712 propagation edges | 8, 16, 32, 64 | 100% nodes | 0.6327 |

- All datasets used two cached mean-propagation hops and one global classifier.
- Official OGB train/validation/test splits were used for OGB datasets.
- Accuracy remains unchanged across executor counts because the model and graph computation are the same; executor count measures systems performance only.
- The driver never collects the graph; Spark processes distributed edge and feature partitions, and Delta stores reusable intermediate features.

**Say:** “Papers100M is the stress test, but the method is general because it operates on the same nodes-edges-splits abstraction for every dataset.”

## Slide 3 - Scaling Result and Research Takeaway

**Title:** Scaling Helps Large Graphs, Then Saturates

| Dataset | 8 executors | Best result | Practical setting |
|---|---:|---:|---:|
| WikiCS propagation | 41.1 s | 41.0 s at 16 | 8 executors |
| Products propagation | 110.4 s | 81.7 s at 32 | 32 executors |
| Papers100M propagation | 3,199.1 s | 1,912.0 s at 32 | 32 executors |

- Papers100M improves by $1.67\times$ from 8 to 32 executors while retaining full node coverage and the same 0.6327 test accuracy.
- Increasing Papers100M from 32 to 64 executors only reduces propagation time from 1,912.0 s to 1,890.7 s, so coordination and I/O overhead begin to dominate.
- The classifier is much cheaper than propagation; future systems optimization should target graph aggregation, shuffle, and Delta I/O.
- Next paper experiment: compare this global cached-propagation path against the bounded community-GNN path and a specialized GNN baseline using accuracy, coverage, runtime, memory, and cost.

**Say:** “The result is not linear scaling. It shows the system has a useful operating point: 32 executors gives nearly all observed Papers100M benefit without paying for 64.”