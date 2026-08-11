# Phase 3.7/3.8 Scaling Analysis (Run1)

## Scope

This run evaluates Phase 3.7/3.8 using Phase 0 graph tables, two cached mean-propagation hops, and one global Spark ML multinomial classifier.

- Run folder: `results/phase37_scaling/phase37_paper_run1`
- Executor sweep: 16, 32, 64
- Datasets: WikiCS, ogbn-products, ogbn-papers100M

## Verified Coverage and Accuracy

| Dataset | Input nodes | Symmetrized propagation edges | Verified node coverage | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|
| WikiCS | 11,701 | 431,206 | 100% | 0.7829 | 0.7746 |
| ogbn-products | 2,449,029 | 123,718,024 | 100% | 0.8720 | 0.7068 |
| ogbn-papers100M | 111,059,956 | 3,228,124,712 | 100% | 0.6314 | 0.6327 |

Accuracy remains invariant across executor settings in this matrix because the model path is fixed and only system parallelism changes.

## Scaling Results

| Dataset | Executors | Propagation time (s) | Classifier time (s) | Total model time (s) | Propagation speedup vs. 16 executors |
|---|---:|---:|---:|---:|---:|
| WikiCS | 16 | 41.6 | 9.0 | 50.6 | 1.00x |
| WikiCS | 32 | 43.7 | 7.6 | 51.3 | 0.95x |
| WikiCS | 64 | 55.3 | 9.8 | 65.1 | 0.75x |
| ogbn-products | 16 | 84.5 | 11.1 | 95.6 | 1.00x |
| ogbn-products | 32 | 80.8 | 9.9 | 90.7 | 1.05x |
| ogbn-products | 64 | 82.3 | 11.3 | 93.6 | 1.03x |
| ogbn-papers100M | 16 | 2478.3 | 125.1 | 2603.4 | 1.00x |
| ogbn-papers100M | 32 | 1940.5 | 73.5 | 2014.0 | 1.28x |
| ogbn-papers100M | 64 | 1892.1 | 71.5 | 1963.6 | 1.31x |

## Interpretation

- WikiCS is too small for aggressive executor scaling; coordination overhead dominates.
- `ogbn-products` benefits slightly at 32 executors but largely saturates.
- Papers100M gains substantially from 16 to 32 executors; 64 adds only marginal additional runtime improvement.
- Classifier time is consistently smaller than propagation time, so optimization priority remains propagation and Delta I/O.

## Generated Paper Artifacts

- `phase37_runtime_by_config.png`
- `phase37_speedup_by_config.png`
- `phase38_accuracy_stability.png`
- `phase37_scaling_recommended_configs.tex`
- `phase37_scaling_results.xlsx`

## Paper-Safe Statement

Across all three datasets, the Phase 3.7/3.8 pipeline preserved full node coverage and fixed test accuracy while sweeping executor count. The largest graph (`ogbn-papers100M`) improved from 2478.3s propagation at 16 executors to 1940.5s at 32 executors, with diminishing gains at 64 executors.