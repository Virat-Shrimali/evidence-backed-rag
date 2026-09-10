| Retriever Strategy | Recall@5 | Precision@5 | MRR | Refusal Accuracy | p50 Latency (ms) | p95 Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **BM25-only** | 77.27% | 15.45% | 0.599 | 0.00% | 1.1 | 2.5 |
| **Dense-only** | 81.82% | 16.36% | 0.529 | 100.00% | 69.5 | 81.8 |
| **Hybrid (RRF)** | 86.36% | 17.27% | 0.654 | 0.00% | 76.2 | 88.5 |
| **Hybrid + Cross-Encoder** | 95.45% | 19.09% | 0.705 | 100.00% | 2896.7 | 3211.1 |
