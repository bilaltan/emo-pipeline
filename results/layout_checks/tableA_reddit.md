**reddit** (63 communities, 2 minor; 63 work items at cap 10,000; Phase 2 0 s; 1 executor x 20 cores)

| Layout | Write time (s) | Storage (GB), files (< 1 MB) | Bytes read (GB), amplification | Stage 3b read time (s) |
| --- | --- | --- | --- | --- |
| Z-order (2 MB RG) | 150.3 ± 5.4 | 1.29, 2 (0) | 5.89, 4.6x | 12.2 ± 0.4 (n=3, 20 readers) |
| Liquid (2 MB RG) | 140.2 ± 3.6 | 1.29, 2 (0) | 5.69, 4.4x | 11.2 ± 0.1 (n=3, 20 readers) |
| Coarse-key Hive | 20.7 ± 1.5 | 1.29, 124 (13) | 1.32, 1.0x | 7.3 ± 0.5 (n=3, 20 readers) |
