"""Edge cases with no prior coverage: empty index, single vector, top_k > n,
zero vectors, duplicate vectors, duplicate doc_ids, dim mismatch, and the
benchmark harness on an empty latency list."""
import math

import numpy as np
import pytest

from vecstore.store import VectorStore, cosine_similarity
from vecstore.hnsw import HNSWIndex
from benchmarks.harness import BenchmarkResult, generate_random_vectors

DIM = 8


def both_indexes():
    return [VectorStore(dim=DIM), HNSWIndex(dim=DIM, seed=3)]


def test_cosine_similarity_axioms():
    a = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9
    assert abs(cosine_similarity(a, [0.0, 1.0, 0.0])) < 1e-9
    assert abs(cosine_similarity(a, [-1.0, 0.0, 0.0]) + 1.0) < 1e-9
    assert cosine_similarity(a, [0.0, 0.0, 0.0]) == 0.0


def test_empty_index():
    for index in both_indexes():
        assert len(index) == 0
        assert index.search(np.ones(DIM, dtype=np.float32), top_k=5) == []


def test_single_vector():
    vec = np.arange(1, DIM + 1, dtype=np.float32)
    for index in both_indexes():
        index.add("only", vec)
        results = index.search(vec, top_k=5)
        assert len(results) == 1
        doc_id, score = results[0]
        assert doc_id == "only"
        assert abs(score - 1.0) < 1e-5


def test_top_k_exceeds_corpus_size():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((7, DIM)).astype(np.float32)
    for index in both_indexes():
        for i in range(7):
            index.add(i, vectors[i])
        results = index.search(vectors[0], top_k=50)
        assert len(results) == 7
        assert results[0][0] == 0


def test_top_k_zero():
    for index in both_indexes():
        index.add("a", np.ones(DIM, dtype=np.float32))
        assert index.search(np.ones(DIM, dtype=np.float32), top_k=0) == []


def test_zero_vectors_never_nan():
    rng = np.random.default_rng(1)
    normal = rng.standard_normal(DIM).astype(np.float32)
    zero = np.zeros(DIM, dtype=np.float32)
    for index in both_indexes():
        index.add("zero", zero)
        index.add("normal", normal)
        for query in (normal, zero):
            results = index.search(query, top_k=2)
            assert len(results) == 2
            assert all(not math.isnan(score) for _, score in results)
        # a zero vector has cosine 0 against everything
        zero_score = dict(index.search(normal, top_k=2))["zero"]
        assert abs(zero_score) < 1e-6


def test_duplicate_vectors_distinct_ids():
    vec = np.arange(1, DIM + 1, dtype=np.float32)
    rng = np.random.default_rng(2)
    for index in both_indexes():
        index.add("first", vec)
        index.add("second", vec)
        for i in range(20):
            index.add(i, rng.standard_normal(DIM).astype(np.float32))
        results = index.search(vec, top_k=2)
        assert {doc_id for doc_id, _ in results} == {"first", "second"}
        assert all(abs(score - 1.0) < 1e-5 for _, score in results)


def test_hnsw_duplicate_doc_id_replaces():
    rng = np.random.default_rng(3)
    index = HNSWIndex(dim=DIM, seed=3)
    filler = rng.standard_normal((30, DIM)).astype(np.float32)
    for i in range(30):
        index.add(i, filler[i])
    old_vec = rng.standard_normal(DIM).astype(np.float32)
    index.add("dup", old_vec)
    assert len(index) == 31

    new_vec = -old_vec  # as far from the old vector as possible
    index.add("dup", new_vec)
    assert len(index) == 31

    # every search returns the doc at most once, scored against the new vector
    results = index.search(new_vec, top_k=31)
    assert len(results) == 31
    doc_ids = [doc_id for doc_id, _ in results]
    assert doc_ids.count("dup") == 1
    assert abs(dict(results)["dup"] - 1.0) < 1e-5

    results_near_old = index.search(old_vec, top_k=31)
    assert [d for d, _ in results_near_old].count("dup") == 1
    assert abs(dict(results_near_old)["dup"] + 1.0) < 1e-5


def test_dim_mismatch_raises():
    for index in both_indexes():
        with pytest.raises(ValueError):
            index.add("bad", np.ones(DIM + 3, dtype=np.float32))
        index.add("ok", np.ones(DIM, dtype=np.float32))
        with pytest.raises(ValueError):
            index.search(np.ones(DIM - 2, dtype=np.float32), top_k=1)


def test_harness_empty_latencies():
    result = BenchmarkResult(label="empty", corpus_size=0, num_queries=0,
                             latencies_ms=[]).compute()
    assert math.isnan(result.p50_ms) and math.isnan(result.p99_ms)


def test_harness_percentiles_interpolate():
    result = BenchmarkResult(label="known", corpus_size=4, num_queries=4,
                             latencies_ms=[10.0, 20.0, 30.0, 40.0]).compute()
    assert result.p50_ms == 25.0
    assert result.min_ms == 10.0 and result.max_ms == 40.0


def test_generate_random_vectors_contract():
    a = generate_random_vectors(50, DIM, seed=5)
    b = generate_random_vectors(50, DIM, seed=5)
    c = generate_random_vectors(50, DIM, seed=6)
    assert a.shape == (50, DIM) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
