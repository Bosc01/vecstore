import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import faiss
from vecstore.store import VectorStore
from benchmarks.harness import generate_random_vectors, run_benchmark, print_comparison_table
from evaluation.metrics import evaluate, build_ground_truth, print_eval_table

def run_comparison(corpus_size=10000, dim=128, num_queries=200, top_k=10):
    print(f"\nVector Retrieval Comparison")
    print(f"corpus_size={corpus_size}  dim={dim}  num_queries={num_queries}  top_k={top_k}")

    corpus = generate_random_vectors(corpus_size, dim, seed=42)
    queries = generate_random_vectors(num_queries, dim, seed=9999)

    # Brute force
    bf_store = VectorStore(dim=dim)
    bf_result = run_benchmark(
        label="BruteForce (Python)",
        build_fn=lambda vecs: bf_store.add_batch(list(enumerate(vecs))),
        search_fn=bf_store.search,
        corpus_size=corpus_size, dim=dim, num_queries=num_queries, top_k=top_k,
    )

    # FAISS flat (exact, C++)
    faiss_flat = faiss.IndexFlatIP(dim)
    corpus_np = np.array(corpus, dtype=np.float32)
    faiss_flat_result = run_benchmark(
        label="FAISS Flat (exact, C++)",
        build_fn=lambda vecs: faiss_flat.add(np.array(vecs, dtype=np.float32)),
        search_fn=lambda q, k: list(zip(faiss_flat.search(np.array([q], dtype=np.float32), k)[1][0], faiss_flat.search(np.array([q], dtype=np.float32), k)[0][0])),
        corpus_size=corpus_size, dim=dim, num_queries=num_queries, top_k=top_k,
    )

    # FAISS HNSW
    faiss_hnsw = faiss.IndexHNSWFlat(dim, 16)
    faiss_hnsw.hnsw.efSearch = 50
    faiss_hnsw_result = run_benchmark(
        label="FAISS HNSW (C++)",
        build_fn=lambda vecs: faiss_hnsw.add(np.array(vecs, dtype=np.float32)),
        search_fn=lambda q, k: list(zip(faiss_hnsw.search(np.array([q], dtype=np.float32), k)[1][0], faiss_hnsw.search(np.array([q], dtype=np.float32), k)[0][0])),
        corpus_size=corpus_size, dim=dim, num_queries=num_queries, top_k=top_k,
    )

    print_comparison_table([bf_result, faiss_flat_result, faiss_hnsw_result])

    print(f"\nFAISS Flat speedup over Python brute force at p50: {bf_result.p50_ms / faiss_flat_result.p50_ms:.1f}x")
    print(f"FAISS HNSW speedup over Python brute force at p50: {bf_result.p50_ms / faiss_hnsw_result.p50_ms:.1f}x")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    run_comparison(corpus_size=args.n, dim=args.dim)
