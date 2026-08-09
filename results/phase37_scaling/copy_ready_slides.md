# Copy-Ready Slides: EMO Global Graph Path

## Slide 1: Global Graph Learning at Papers100M Scale

**Problem**

Conventional GNN training repeatedly performs graph message passing every epoch. At Papers100M scale, this can require large graph transfers and unbounded worker memory.

**Our approach**

- Store `nodes`, `edges`, and official data splits as distributed Delta tables.
- Use Spark to compute two neighborhood-mean propagation hops across the graph.
- Cache each propagation hop in Delta Lake.
- Train one shared global multinomial logistic-regression classifier on the cached features.

$$
z_v = [x_v^{(0)} \Vert x_v^{(1)} \Vert x_v^{(2)}]
$$

`x0`: original node features; `x1`: one-hop neighborhood mean; `x2`: two-hop neighborhood mean.

**Speaker line:** “We separate the expensive graph-wide operation from classifier training, so graph features are computed once, cached, and reused.”

---

## Slide 2: General Evaluation and Full Coverage

| Dataset | Nodes | Propagation edges | Coverage | Global test accuracy |
|---|---:|---:|---:|---:|
| WikiCS | 11,701 | 431,206 | 100% | 0.7746 |
| ogbn-products | 2,449,029 | 123,718,024 | 100% | 0.7068 |
| ogbn-papers100M | 111,059,956 | 3,228,124,712 | 100% | 0.6327 |

- Same `nodes`-`edges`-`splits` abstraction across small, medium, and very large graphs.
- Two cached propagation hops for every experiment.
- Official OGB splits used for Products and Papers100M.
- No driver-side graph collection and no unbounded Python adjacency lists.
- Accuracy is unchanged across executor counts because the model computation is identical.

**Speaker line:** “Papers100M is the scale stress test, but the implementation is general: every dataset enters through the same Delta graph tables and Spark propagation operators.”

---

## Slide 3: Scaling Result and Practical Operating Point

| Dataset | Propagation at 8 executors | Best propagation result | Recommended setting |
|---|---:|---:|---:|
| WikiCS | 41.1 s | 41.0 s at 16 | 8 executors |
| ogbn-products | 110.4 s | 81.7 s at 32 | 32 executors |
| ogbn-papers100M | 3,199.1 s | 1,912.0 s at 32 | 32 executors |

**Papers100M result**

- $1.67\times$ faster propagation from 8 to 32 executors.
- 100% node coverage and unchanged test accuracy of 0.6327.
- 64 executors only improves 1,912.0 s to 1,890.7 s, so scaling saturates after 32 executors.

**Research takeaway**

EMO provides a scalable, full-data graph feature-processing path. The next comparison is against bounded community GNN training and specialized distributed GNN systems using accuracy, runtime, memory, and cost.

**Speaker line:** “The important result is not linear scaling. It is that we identify an efficient operating point for a 111-million-node graph while preserving full node coverage.”