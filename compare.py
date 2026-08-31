import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import faiss
from vecstore.store import VectorStore
from vecstore.hnsw import HNSWIndex
from benchmarks.harness import generate_random_vectors, run_benchmark, print_comparison_table
from evaluation.metrics import evaluate, build_ground_truth, print_eval_table

def run_comparison(corpus_size=10000, dim=128, num_queries=2000, top_k=10, seed=42):
    faiss.omp_set_num_threads(1)  # single thread so FAISS latencies are reproducible

    print(f"\nVector Retrieval Comparison")
    print(f"corpus_size={corpus_size}  dim={dim}  num_queries={num_queries}  top_k={top_k}")

    # Brute force (mine)
    bf_store = VectorStore(dim=dim)
    bf_result = run_benchmark(
        label="BruteForce (Python)",
        build_fn=lambda vecs: bf_store.add_batch(list(enumerate(vecs))),
        search_fn=bf_store.search,
        corpus_size=corpus_size, dim=dim, num_queries=num_queries, top_k=top_k, seed=seed,
    )

    # HNSW (mine)
    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=50, seed=seed)
    def build_hnsw(vecs):
        for i, v in enumerate(vecs):
            hnsw.add(i, v)
    hnsw_result = run_benchmark(
        label="HNSW (Python)",
        build_fn=build_hnsw,
        search_fn=hnsw.search,
        corpus_size=corpus_size, dim=dim, num_queries=num_queries, top_k=top_k, seed=seed,
    )

    # FAISS flat (exact, C++)
    faiss_flat = faiss.IndexFlatIP(dim)
    def faiss_flat_search(q, k):
        D, I = faiss_flat.search(np.asarray(q, dtype=np.float32).reshape(1, -1), k)
        return list(zip(I[0].tolist(), D[0].tolist()))
    faiss_flat_result = run_benchmark(
        label="FAISS Flat (exact, C++)",
        build_fn=lambda vecs: faiss_flat.add(np.asarray(vecs, dtype=np.float32)),
        search_fn=faiss_flat_search,
        corpus_size=corpus_size, dim=dim, num_queries=num_queries, top_k=top_k, seed=seed,
    )

    # FAISS HNSW (C++), matched to mine: M=16, efConstruction=50, inner product.
    # The IndexHNSWFlat default is METRIC_L2; on unit-norm vectors the rankings
    # match, but the scores would be ascending L2 distances while every other
    # search function here returns descending cosine similarity.
    faiss_hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    faiss_hnsw.hnsw.efConstruction = 50
    faiss_hnsw.hnsw.efSearch = 50
    def faiss_hnsw_search(q, k):
        D, I = faiss_hnsw.search(np.asarray(q, dtype=np.float32).reshape(1, -1), k)
        return list(zip(I[0].tolist(), D[0].tolist()))
    faiss_hnsw_result = run_benchmark(
        label="FAISS HNSW (C++)",
        build_fn=lambda vecs: faiss_hnsw.add(np.asarray(vecs, dtype=np.float32)),
        search_fn=faiss_hnsw_search,
        corpus_size=corpus_size, dim=dim, num_queries=num_queries, top_k=top_k, seed=seed,
    )

    print_comparison_table([bf_result, hnsw_result, faiss_flat_result, faiss_hnsw_result])

    # recall@10 for every index against exact inner-product ground truth,
    # over the same queries the latency runs used
    queries = generate_random_vectors(num_queries, dim, seed=seed + 9999)
    ground_truth = build_ground_truth(faiss_flat_search, list(queries), k=top_k)
    print_eval_table([
        evaluate("BruteForce (Python)", bf_store.search, ground_truth, k=top_k),
        evaluate("HNSW (Python)", hnsw.search, ground_truth, k=top_k),
        evaluate("FAISS Flat (exact, C++)", faiss_flat_search, ground_truth, k=top_k),
        evaluate("FAISS HNSW (C++)", faiss_hnsw_search, ground_truth, k=top_k),
    ])

    print(f"\nFAISS Flat speedup over Python brute force at p50: {bf_result.p50_ms / faiss_flat_result.p50_ms:.1f}x")
    print(f"FAISS HNSW speedup over my Python HNSW at p50: {hnsw_result.p50_ms / faiss_hnsw_result.p50_ms:.1f}x")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=2000)
    args = parser.parse_args()
    run_comparison(corpus_size=args.n, dim=args.dim, num_queries=args.queries)
