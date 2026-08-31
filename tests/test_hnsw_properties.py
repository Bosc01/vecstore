"""Hypothesis property tests for the HNSW graph invariants and the search
contract, run across random corpora, both neighbor-selection modes, and both
layer-0 link policies."""
import numpy as np
from hypothesis import given, settings, strategies as st

from vecstore.hnsw import HNSWIndex

COMMON = dict(max_examples=25, deadline=None)


@st.composite
def built_index_params(draw):
    dim = draw(st.integers(min_value=2, max_value=16))
    n = draw(st.integers(min_value=1, max_value=48))
    data_seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    heuristic = draw(st.booleans())
    full_l0 = draw(st.booleans())
    return dim, n, data_seed, heuristic, full_l0


def build(dim, n, data_seed, heuristic, full_l0):
    rng = np.random.default_rng(data_seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    idx = HNSWIndex(dim=dim, M=16, ef_construction=50, seed=7,
                    heuristic_select=heuristic, layer0_full_links=full_l0)
    for i in range(n):
        idx.add(i, vectors[i])
    return idx, vectors


@given(built_index_params())
@settings(**COMMON)
def test_graph_structure_invariants(params):
    idx, _ = build(*params)
    n = len(idx)
    graph = {layer: dict(adj) for layer, adj in idx._graph.items()}

    for layer, adjacency in graph.items():
        cap = idx.M_max0 if layer == 0 else idx.M
        members = set(adjacency.keys())
        for node, neighbors in adjacency.items():
            assert len(neighbors) <= cap, f"layer {layer} degree {len(neighbors)} > {cap}"
            assert len(set(neighbors)) == len(neighbors), "duplicate edge"
            assert node not in neighbors, "self edge"
            assert set(neighbors) <= members, "edge to a node absent from this layer"

    # the entry point lives at the top layer, and no layer sits above it
    assert idx._entry_point in graph[idx._max_layer]
    assert max(graph.keys()) == idx._max_layer

    # each layer's node set nests inside the layer below; layer 0 holds everyone
    for layer in graph:
        if layer > 0:
            assert set(graph[layer].keys()) <= set(graph[layer - 1].keys())
    assert set(graph[0].keys()) == set(range(n))


@given(built_index_params())
@settings(**COMMON)
def test_layer0_reachable_from_entry_point(params):
    idx, _ = build(*params)
    adjacency = idx._graph[0]
    seen = {idx._entry_point}
    frontier = [idx._entry_point]
    while frontier:
        nxt = []
        for node in frontier:
            for neighbor in adjacency.get(node, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    nxt.append(neighbor)
        frontier = nxt
    assert seen == set(adjacency.keys()), "layer 0 not fully reachable by BFS from entry"


@given(built_index_params(),
       st.integers(min_value=1, max_value=96),
       st.sampled_from([1, 10, 50]),
       st.integers(min_value=0, max_value=2**31 - 1))
@settings(**COMMON)
def test_search_contract(params, top_k, ef_search, query_seed):
    idx, _ = build(*params)
    dim = params[0]
    q = np.random.default_rng(query_seed).standard_normal(dim).astype(np.float32)

    results = idx.search(q, top_k=top_k, ef_search=ef_search)

    assert len(results) == min(top_k, len(idx))
    scores = [score for _, score in results]
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    doc_ids = [doc_id for doc_id, _ in results]
    assert len(set(doc_ids)) == len(doc_ids)


@given(built_index_params(), st.integers(min_value=1, max_value=96))
@settings(**COMMON)
def test_search_contract_survives_readds(params, top_k):
    # re-adding doc_ids tombstones old nodes; arity, ordering, and uniqueness
    # must hold on the resulting graph too
    idx, vectors = build(*params)
    dim, n = params[0], params[1]
    rng = np.random.default_rng(1234)
    for doc_id in range(0, n, 2):
        idx.add(doc_id, rng.standard_normal(dim).astype(np.float32))
    assert len(idx) == n

    q = rng.standard_normal(dim).astype(np.float32)
    results = idx.search(q, top_k=top_k, ef_search=50)
    assert len(results) == min(top_k, n)
    doc_ids = [doc_id for doc_id, _ in results]
    assert len(set(doc_ids)) == len(doc_ids)
    scores = [score for _, score in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
