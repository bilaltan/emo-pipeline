**ogbn-products** (53,397 communities, 52,787 minor; 53,397 work items at cap 10,000; Phase 2 0 s; 1 executor x 20 cores)

| Layout | Write time (s) | Storage (GB), files (< 1 MB) | Bytes read (GB), amplification | Stage 3b read time (s) |
| --- | --- | --- | --- | --- |
| Z-order (2 MB RG) | 205.8 ± 14.5 | 2.21, 2 (0) | 151.72, 69.1x | 164.9 ± 0.2 (n=3, 20 readers) |
| Liquid (2 MB RG) | 211.6 ± 2.7 | 2.21, 2 (0) | 152.46, 69.4x | 164.9 ± 0.1 (n=3, 20 readers) |
| Coarse-key Hive | 33.4 ± 6.4 | 2.19, 1,222 (676) |  |  |
