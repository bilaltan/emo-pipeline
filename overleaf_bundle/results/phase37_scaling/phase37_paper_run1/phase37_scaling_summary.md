# Phase 3.7 / 3.8 Scaling Summary

Successful configurations: 9
Datasets covered: 3

## Dataset recommendations

- ogbn-papers100M: recommend 32 exec (1940.5s propagation, 2014.0s end-to-end, test acc 0.6327)
- ogbn-products: recommend 16 exec (84.5s propagation, 95.6s end-to-end, test acc 0.7068)
- wikics: recommend 16 exec (41.6s propagation, 50.6s end-to-end, test acc 0.7746)

## Paper figures

- Main scaling figure (1x3): phase37_paper_figure_main.pdf
- Runtime and speedup matrix: phase37_paper_figure_matrix.pdf

## LaTeX table

- phase37_scaling_recommended_configs.tex

## Key findings

- ogbn-papers100M: fastest propagation 1892.1s, speedup 1.31x, accuracy range 0.6327-0.6327.
- ogbn-products: fastest propagation 80.8s, speedup 1.05x, accuracy range 0.7068-0.7068.
- wikics: fastest propagation 41.6s, speedup 1.33x, accuracy range 0.7746-0.7746.
