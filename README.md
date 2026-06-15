# Vector Retrieval Engine

A vector search engine built from scratch in Python, benchmarked against FAISS at 1 million vectors.

## What this is

I implemented brute force vector search and HNSW (Hierarchical Navigable Small World) from scratch
to understand what production vector databases like FAISS, Pinecone, and Weaviate do under the hood.
The project came out of my published RAG paper (IEEE MIT URTC 2025) where I used FAISS as a black box.
This is the answer to the question: do I actually know what FAISS is doing?

## Benchmark results

All results on Apple M5 Air, 24GB RAM, dim=128, 200 queries, top_k=10.

| Algorithm            | n=10k p50  | n=100k p50 | n=1M p50   |
|----------------------|------------|------------|------------|
| Python Brute Force   | 0.49ms     | 6.59ms     | 80.39ms    |
| FAISS Flat (C++)     | 0.12ms     | 0.91ms     | 7.94ms     |
| FAISS HNSW (C++)     | 0.06ms     | 0.10ms     | 0.23ms     |

FAISS HNSW speedup over Python brute force: 8x at 10k, 65x at 100k, 347x at 1M.

## What the numbers show

Brute force scales linearly. 10x more vectors means roughly 10x slower queries.

FAISS Flat is the same exact algorithm as Python brute force but written in C++ with
vectorized CPU instructions. The 10x speedup at 1M vectors is purely an implementation
difference, not an algorithmic one.

FAISS HNSW scales logarithmically. 100x more vectors (10k to 1M) produces only a 4x
latency increase (0.06ms to 0.23ms). That is the algorithmic win of HNSW: it navigates
a layered graph to find approximate nearest neighbors in O(log n) instead of scanning
every vector in O(n).

## Why FAISS wins

FAISS is written in C++ with AVX2 and AVX-512 vectorized instructions. A single FAISS dot
product over 128 dimensions executes in a handful of CPU cycles. The same operation in
Python requires interpreter overhead on every multiplication. At 1 million vectors that
difference compounds into the 347x gap shown above.

## The tradeoff HNSW makes

HNSW is approximate. It does not guarantee returning the true nearest neighbor. It returns
a very good approximation by navigating a layered graph: higher layers have sparse long-range
connections for fast navigation, lower layers have dense local connections for precision.
The ef_search parameter controls the recall vs speed tradeoff at query time without
rebuilding the index. Higher ef_search finds better neighbors at the cost of exploring
more of the graph.

## Where my implementation differs from FAISS

My Python HNSW is correct but slow to build. At 5000 vectors it takes roughly 4 seconds.
FAISS builds the same index in milliseconds because the inner loop runs in C++ with
cache-optimized memory access patterns. The graph traversal logic is identical. The
performance gap is entirely implementation, not algorithm.

## The five questions I can answer cold

1. The brute force index is a numpy matrix. Search is a single matrix multiply followed
   by argpartition. O(n) query time, exact results.

2. HNSW builds a layered directed graph. It trades recall accuracy for O(log n) query
   time. ef_search controls the tradeoff at query time without rebuilding.

3. The benchmarking harness times each query individually with time.perf_counter and
   reports p50, p95, and p99 percentiles. Percentiles matter because averages hide tail
   latency that real users experience.

4. Brute force latency grows linearly with corpus size. HNSW latency grows logarithmically.
   At 1M vectors the gap is 347x.

5. FAISS wins on every performance metric. The gap is C++ vectorized instructions vs
   Python interpreter overhead, plus cache-optimized memory layout vs Python dicts.
   My implementation exists to prove the algorithm is understood, not to replace FAISS.

## Project structure

    vecstore/store.py         Brute force index with numpy matrix search
    vecstore/hnsw.py          HNSW approximate nearest neighbor index
    benchmarks/harness.py     p50/p95/p99 latency measurement
    evaluation/metrics.py     Precision@K, Recall@K, MRR
    compare.py                Full benchmark runner
    smoke_test.py             Correctness tests

## How to run

    pip install faiss-cpu numpy
    python smoke_test.py
    python compare.py --n 100000
