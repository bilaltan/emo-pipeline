**ogbn-products** (53,397 communities, 52,787 minor; 53,397 work items at cap 10,000; Phase 2 0 s; 1 executor x 20 cores)

| Layout | Write time (s) | Storage (GB), files (< 1 MB) | Bytes read (GB), amplification | Stage 3b read time (s) |
| --- | --- | --- | --- | --- |
| Z-order (2 MB RG) | 205.8 ± 14.5 | 2.21, 2 (0) | 146.87, 65.2x est. | 1089.2 ± 2.6 (n=3, 3 readers, est. from 1/10 of the bins) |
| Liquid (2 MB RG) | 211.6 ± 2.7 | 2.21, 2 (0) | 148.15, 65.8x est. | 1095.2 ± 1.8 (n=3, 3 readers, est. from 1/10 of the bins) |
| Coarse-key Hive | 33.4 ± 6.4 | 2.19, 1,222 (676) | 17.29, 7.9x | 49.7 ± 0.1 (n=3, 20 readers) |
