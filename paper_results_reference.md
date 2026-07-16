# GRL Papers Benchmark Reference Catalog

This document is a consolidated reference catalog of the experimental results reported in **ML-GRL (SIGMOD 2025)**, **CaaN 2L-GRL (IEEE)**, and **SaaN 2L-GRL (IEEE)**. 

---

## 1. Node Classification Performance (Accuracy & Runtime)

| Paper | Dataset | Base Model | Original Baseline Acc | Original Time (s) | EMO / ML-GRL Acc | GRL Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CaaN** | Flickr | GraphSAGE | 64.12% | 3027.0s | 64.65% | 730.3s |
| **CaaN** | Flickr | GAT | 78.28% | 904.1s | 79.57% | 747.1s |
| **CaaN** | PubMed | GraphSAGE | 77.86% | 871.3s | 77.96% | 402.2s |
| **CaaN** | PubMed | GAT | 77.90% | 880.5s | 78.54% | 400.8s |
| **ML-GRL** | WikiCS | GAT | 63.51% | 56.0s | 81.29% | 10.0s |
| **ML-GRL** | WikiCS | GraphTransformer | 53.54% | 45.0s | 75.44% | 16.0s |
| **ML-GRL** | WikiCS | ClusterSCL | 81.03% | 51.0s | 83.74% | 21.0s |
| **ML-GRL** | Coauthor-Phys | GAT | 79.54% | 118.0s | 94.11% | 98.0s |
| **ML-GRL** | Coauthor-Phys | GraphTransformer | 90.83% | 508.0s | 94.57% | 383.0s |
| **ML-GRL** | Coauthor-Phys | ClusterSCL | 94.36% | 922.0s | 96.44% | 391.0s |
| **ML-GRL** | Coauthor-CS | GAT | 61.61% | 84.0s | 86.09% | 78.0s |
| **ML-GRL** | Coauthor-CS | GraphTransformer | 73.38% | 225.0s | 90.40% | 186.0s |
| **ML-GRL** | Coauthor-CS | ClusterSCL | 94.38% | 336.0s | 96.03% | 111.0s |
| **ML-GRL** | DeezerEurope | GAT | 54.98% | 40.0s | 61.52% | 35.0s |
| **ML-GRL** | DeezerEurope | GraphTransformer | 60.22% | 66.0s | 68.61% | 58.0s |
| **ML-GRL** | DeezerEurope | ClusterSCL | 57.25% | 181.0s | 59.33% | 49.0s |

---

## 2. Link Prediction Performance (AUC & Runtime)

| Paper | Dataset | Base Model | Original Baseline AUC | Original Time (s) | EMO / ML-GRL AUC | GRL Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CaaN** | Foursquare | GraphSAGE | 84.78% | 6270.0s | 86.11% | 3270.0s |
| **CaaN** | Foursquare | GCN | 83.53% | 7041.0s | 84.91% | 3569.0s |
| **CaaN** | Foursquare | GAT | 90.80% | 667.0s | 92.50% | 508.0s |
| **ML-GRL** | WikiCS | GraphSAGE | 65.17% | 42.0s | 77.68% | 29.0s |
| **ML-GRL** | WikiCS | ARMA | 86.34% | 265.0s | 91.02% | 34.0s |
| **ML-GRL** | WikiCS | ASAP | 87.73% | 64.0s | 90.96% | 33.0s |
| **ML-GRL** | Coauthor-Phys | GraphSAGE | 61.34% | 306.0s | 65.49% | 256.0s |
| **ML-GRL** | Coauthor-Phys | ARMA | 75.74% | 581.0s | 85.89% | 376.0s |
| **ML-GRL** | Coauthor-Phys | ASAP | 74.89% | 654.0s | 79.60% | 512.0s |
