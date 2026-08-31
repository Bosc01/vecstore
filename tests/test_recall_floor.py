"""Recall regression gate. A refactor that quietly damages the graph -- wrong
neighbor wiring, broken layer descent, bad distance math -- lands well below
this floor even when every structural invariant still passes.

The shipped config measures ~0.61-0.62 recall@10 on this setup; 0.55 leaves
room for topology drift across seeds without letting a real regression
through. Runtime is dominated by the ~10s index build.
"""
import numpy as np

from vecstore.hnsw import HNSWIndex

N = 10_000
DIM = 128
NUM_QUERIES = 200
FLOOR = 0.55


def test_recall_at_10_floor():
    rng = np.random.default_rng(42)
    corpus = rng.standard_normal((N, DIM)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = np.random.default_rng(42 + 9999).standard_normal((NUM_QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    sims = corpus @ queries.T
    ground_truth = []
    for qi in range(NUM_QUERIES):
        col = sims[:, qi]
        top = np.argpartition(col, -10)[-10:]
        ground_truth.append(set(top.tolist()))

    index = HNSWIndex(dim=DIM, M=16, ef_construction=50, seed=42)
    for i in range(N):
        index.add(i, corpus[i])

    hits = 0
    for qi in range(NUM_QUERIES):
        results = index.search(queries[qi], top_k=10, ef_search=50)
        hits += len({doc_id for doc_id, _ in results} & ground_truth[qi])
    recall = hits / (NUM_QUERIES * 10)

    assert recall >= FLOOR, f"recall@10 = {recall:.4f} fell below the {FLOOR} gate"
