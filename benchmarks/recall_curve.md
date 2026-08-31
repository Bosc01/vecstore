Recall@10 vs p50 latency, n=10000, dim=128, M=16, efConstruction=50, 200 queries, ground truth = IndexFlatIP.

| ef_search | HNSW (Python) recall@10 | HNSW (Python) p50 (ms) | FAISS HNSW recall@10 | FAISS HNSW p50 (ms) |
|---|---|---|---|---|
| 10 | 0.239 | 0.129 | 0.215 | 0.008 |
| 25 | 0.426 | 0.253 | 0.384 | 0.016 |
| 50 | 0.623 | 0.451 | 0.562 | 0.028 |
| 100 | 0.803 | 0.761 | 0.768 | 0.055 |
| 200 | 0.933 | 1.341 | 0.928 | 0.115 |
| 400 | 0.987 | 2.302 | 0.986 | 0.249 |
