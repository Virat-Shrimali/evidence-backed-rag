| Retriever Strategy | Recall@5 | Precision@5 | MRR | Refusal Accuracy | p50 Latency (ms) | p95 Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **BM25-only** | 77.27% | 15.45% | 0.599 | 0.00% | 0.7 | 0.8 |
| **Dense-only** | 81.82% | 16.36% | 0.529 | 100.00% | 36.9 | 46.0 |
| **Hybrid (RRF)** | 86.36% | 17.27% | 0.654 | 0.00% | 51.0 | 69.8 |
| **Hybrid + Cross-Encoder** | 95.45% | 19.09% | 0.705 | 100.00% | 2237.9 | 2464.7 |
