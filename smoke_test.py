import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vecstore.store import VectorStore, cosine_similarity
from vecstore.hnsw import HNSWIndex
from benchmarks.harness import generate_random_vectors, run_benchmark
from evaluation.metrics import evaluate, build_ground_truth

def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9
    assert abs(cosine_similarity(a, [0.0, 1.0, 0.0])) < 1e-9
    assert abs(cosine_similarity(a, [-1.0, 0.0, 0.0]) - (-1.0)) < 1e-9
    print("  cosine_similarity: PASS")

def test_vector_store():
    store = VectorStore(dim=3)
    store.add("doc_a", [1.0, 0.0, 0.0])
    store.add("doc_b", [0.0, 1.0, 0.0])
    store.add("doc_c", [0.0, 0.0, 1.0])
    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "doc_a"
    assert len(store) == 3
    print("  VectorStore: PASS")

def test_hnsw():
    # use 200 vectors so the graph has enough connections to navigate correctly
    dim = 32
    corpus = generate_random_vectors(200, dim, seed=1)
    index = HNSWIndex(dim=dim, M=16, ef_construction=50)
    for i, v in enumerate(corpus):
        index.add(i, v)

    bf = VectorStore(dim=dim)
    bf.add_batch(list(enumerate(corpus)))

    queries = generate_random_vectors(10, dim, seed=9999)
    hits = 0
    for q in queries:
        hnsw_top = index.search(q, top_k=1)[0][0]
        bf_top = bf.search(q, top_k=1)[0][0]
        if hnsw_top == bf_top:
            hits += 1

    # expect at least 80% agreement with brute force on small corpus
    assert hits >= 8, f"HNSW matched brute force on only {hits}/10 queries"
    print(f"  HNSWIndex: PASS ({hits}/10 queries matched brute force)")

def test_benchmark_harness():
    store = VectorStore(dim=32)
    result = run_benchmark(
        label="smoke", build_fn=lambda vecs: store.add_batch(list(enumerate(vecs))),
        search_fn=store.search, corpus_size=100, dim=32, num_queries=20, top_k=5,
    )
    assert result.p99_ms >= result.p95_ms >= result.p50_ms
    print("  Benchmarking harness: PASS")

def test_evaluation_metrics():
    store = VectorStore(dim=32)
    store.add_batch(list(enumerate(generate_random_vectors(100, 32))))
    queries = generate_random_vectors(10, 32, seed=9999)
    gt = build_ground_truth(store.search, queries, k=5)
    result = evaluate("smoke", store.search, gt, k=5)
    assert abs(result.precision_at_k - 1.0) < 1e-9
    assert abs(result.recall_at_k - 1.0) < 1e-9
    print("  Evaluation metrics: PASS")

if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_cosine_similarity()
    test_vector_store()
    test_hnsw()
    test_benchmark_harness()
    test_evaluation_metrics()
    print("\nAll checks passed.")
