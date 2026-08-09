# Phase 3.7/3.8 Scaling Analysis

## Scope

This experiment measures the full-graph Phase 3.7/3.8 path with two cached mean-propagation hops and one global Spark ML multinomial logistic-regression classifier. The graph source is Phase 0, and each successful run verifies output-node coverage before classifier fitting.

The combined raw data and parsed metrics are in `phase37_scaling_results.xlsx`.

## Verified Coverage and Accuracy

| Dataset | Input nodes | Symmetrized propagation edges | Verified node coverage | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|
| WikiCS | 11,701 | 431,206 | 100% | 0.7829 | 0.7746 |
| ogbn-products | 2,449,029 | 123,718,024 | 100% | 0.8720 | 0.7068 |
| ogbn-papers100M | 111,059,956 | 3,228,124,712 | 100% | 0.6314 | 0.6327 |

Accuracy is invariant across executor counts because each configuration runs the same deterministic two-hop propagation and the same global classifier. Executor count is therefore evaluated as a systems-performance parameter, not a model hyperparameter.

## Scaling Results

| Dataset | Executors | Propagation time (s) | Classifier time (s) | Total model time (s) | Propagation speedup vs. 8 executors |
|---|---:|---:|---:|---:|---:|
| WikiCS | 8 | 41.1 | 8.9 | 50.0 | 1.00x |
| WikiCS | 16 | 41.0 | 8.4 | 49.4 | 1.00x |
| WikiCS | 32 | 45.5 | 7.9 | 53.4 | 0.90x |
| WikiCS | 64 | 58.0 | 10.2 | 68.2 | 0.71x |
| ogbn-products | 8 | 110.4 | 16.7 | 127.1 | 1.00x |
| ogbn-products | 16 | 86.7 | 11.7 | 98.4 | 1.27x |
| ogbn-products | 32 | 81.7 | 10.3 | 92.0 | 1.35x |
| ogbn-products | 64 | 82.5 | 12.0 | 94.5 | 1.34x |
| ogbn-papers100M | 8 | 3199.1 | 237.8 | 3436.9 | 1.00x |
| ogbn-papers100M | 16 | 2459.0 | 118.6 | 2577.6 | 1.30x |
| ogbn-papers100M | 32 | 1912.0 | 67.1 | 1979.1 | 1.67x |
| ogbn-papers100M | 64 | 1890.7 | 70.3 | 1961.0 | 1.69x |

## Interpretation

- WikiCS is too small to benefit from more executors. Scheduling, shuffle, and cluster coordination overhead dominate; use 8 executors or fewer for this scale.
- `ogbn-products` improves substantially from 8 to 16 executors, reaches its best observed propagation time at 32, and shows no meaningful gain at 64. Use 32 executors.
- Papers100M improves materially from 8 to 32 executors. Increasing to 64 yields only 21.3 seconds, or about 1.1%, additional propagation improvement. Use 32 executors unless the cost of 64 executors is effectively identical.
- The classifier is much cheaper than propagation at all scales, especially on Papers100M. The system optimization target should therefore be graph propagation, shuffle, and Delta I/O rather than the classifier.
- One early bootstrap run is incomplete and is retained in the workbook's `incomplete_runs` sheet. It should not be used in tables or figures.

## Paper-Safe Statement

Across three graph sizes, the Phase 3.7/3.8 path preserved 100% of input nodes through two cached propagation hops and produced identical global test accuracy across executor configurations. Scaling benefits depend on graph size: executor growth hurts the small graph, saturates near 32 executors for `ogbn-products`, and reduces Papers100M propagation time from 3,199.1 seconds at 8 executors to 1,912.0 seconds at 32 executors.

Do not claim linear scaling or a cost improvement until repeated trials and per-run cluster cost are recorded.