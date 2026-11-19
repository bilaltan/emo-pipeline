**ogbn-arxiv** (157 communities, 99 minor; 157 work items at cap 10,000; Phase 2 0 s; 1 executor x 20 cores)

| Layout | Write time (s) | Storage (GB), files (< 1 MB) | Bytes read (GB), amplification | Stage 3b read time (s) |
| --- | --- | --- | --- | --- |
| Z-order (2 MB RG) | 17.5 ± 2.6 | 0.17, 2 (0) | 2.19, 13.0x | 3.2 ± 0.0 (n=3, 20 readers) |
| Liquid (2 MB RG) | 15.8 ± 0.2 | 0.17, 2 (0) | 2.09, 12.4x | 3.2 ± 0.1 (n=3, 20 readers) |
| Coarse-key Hive | 3.1 ± 0.4 | 0.17, 118 (82) | 0.45, 2.7x | 4.8 ± 0.1 (n=3, 20 readers) |
