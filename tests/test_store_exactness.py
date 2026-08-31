"""VectorStore.search against a naive pure-Python cosine loop.

The store computes cosine as a float32 matmul over vectors it normalized at
insert time. The reference below recomputes cosine from the raw vectors in
plain Python floats, sharing no code with the store. If insert-time
normalization (or the cached-norm shortcut it enables) is ever wrong, the
rankings diverge and this fails -- it turns the norm bug into a correctness
failure instead of a silent performance regression.
"""
import math

import numpy as np

from vecstore.store import VectorStore

N = 1000
DIM = 64
NUM_QUERIES = 200
TOP_K = 10
SEED = 20260831


def naive_top_ids(corpus, query, k):
    q_norm = math.sqrt(sum(x * x for x in query))
    scored = []
    for doc_id, vec in enumerate(corpus):
        dot = sum(a * b for a, b in zip(vec, query))
        v_norm = math.sqrt(sum(a * a for a in vec))
        denom = v_norm * q_norm
        scored.append((dot / denom if denom > 0 else 0.0, doc_id))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [doc_id for _, doc_id in scored[:k]]


def test_search_matches_naive_cosine():
    rng = np.random.default_rng(SEED)
    corpus32 = rng.standard_normal((N, DIM)).astype(np.float32)
    queries32 = rng.standard_normal((NUM_QUERIES, DIM)).astype(np.float32)
    # .tolist() hands the naive side the exact same float values the store saw
    corpus = corpus32.tolist()

    store = VectorStore(dim=DIM)
    store.add_batch(list(enumerate(corpus32)))

    for qi in range(NUM_QUERIES):
        expected = naive_top_ids(corpus, queries32[qi].tolist(), TOP_K)
        got = [doc_id for doc_id, _ in store.search(queries32[qi], top_k=TOP_K)]
        assert got == expected, f"query {qi}: store {got} != naive {expected}"


def test_add_matches_add_batch():
    # one-by-one inserts go through the doubling buffer; results must be
    # identical to a single batch insert of the same vectors
    rng = np.random.default_rng(SEED + 1)
    vecs = rng.standard_normal((300, DIM)).astype(np.float32)
    one_by_one = VectorStore(dim=DIM)
    for i, v in enumerate(vecs):
        one_by_one.add(i, v)
    batched = VectorStore(dim=DIM)
    batched.add_batch(list(enumerate(vecs)))

    queries = rng.standard_normal((20, DIM)).astype(np.float32)
    for q in queries:
        assert one_by_one.search(q, top_k=TOP_K) == batched.search(q, top_k=TOP_K)


def test_add_batch_appends():
    store = VectorStore(dim=3)
    store.add_batch([("a", [1.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0])])
    store.add_batch([("c", [0.0, 0.0, 1.0])])
    assert len(store) == 3
    assert store.search([0.0, 0.0, 1.0], top_k=1)[0][0] == "c"


def test_top_k_clamped_to_corpus_size():
    store = VectorStore(dim=4)
    store.add("only", [1.0, 0.0, 0.0, 0.0])
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10)
    assert [doc_id for doc_id, _ in results] == ["only"]
