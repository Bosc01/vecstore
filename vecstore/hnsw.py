import random
import heapq
import numpy as np
from collections import defaultdict


class HNSWIndex:
    """HNSW over cosine similarity. Vectors are unit-normalized at insert, so
    every internal distance is 1 - dot.

    Constructor flags, both benchmarked in the README:
      heuristic_select: use the paper's Algorithm 4 neighbor heuristic instead
        of plain nearest-M when wiring nodes.
      layer0_full_links: give a new node M_max0 links at layer 0 (this repo's
        original behavior) instead of the paper's M; M_max0 stays the prune
        cap on existing nodes either way.

    Re-adding an existing doc_id tombstones the old node: HNSW has no clean
    delete, so the dead node stays in the graph for routing and is filtered
    out of results. len(self) counts live docs only.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 50, seed: int = 42,
                 heuristic_select: bool = False, layer0_full_links: bool = True):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self.heuristic_select = heuristic_select
        self.layer0_full_links = layer0_full_links
        # own RNG instance: harness code reseeds the global random module for
        # corpus generation, which would otherwise couple index topology to it
        self._rng = random.Random(seed)
        # one contiguous matrix, node_id == row index, amortized-doubling
        # capacity; rows [0, _next_id) are valid
        self._mat = np.empty((0, dim), dtype=np.float32)
        self._graph = defaultdict(lambda: defaultdict(list))
        self._entry_point = None
        self._max_layer = 0
        self._next_id = 0
        self._doc_to_node = {}
        self._node_to_doc = {}
        self._dead = set()

    def _random_level(self):
        # P(level >= l) = M^-l, the same geometric distribution as the paper's
        # floor(-ln(U) / ln(M)) draw (i.e. -ln(U) * mL with mL = 1/ln(M)).
        # Verified equivalent; a closed-form rewrite would change nothing.
        level = 0
        while self._rng.random() < (1.0 / self.M) and level < 32:
            level += 1
        return level

    def _ensure_capacity(self, rows_needed):
        capacity = self._mat.shape[0]
        if rows_needed <= capacity:
            return
        new_capacity = max(capacity, 16)
        while new_capacity < rows_needed:
            new_capacity *= 2
        grown = np.empty((new_capacity, self.dim), dtype=np.float32)
        grown[: self._next_id] = self._mat[: self._next_id]
        self._mat = grown

    def _batch_dist(self, query_vec, node_ids, dist_cache):
        # rows are unit-norm at insert and callers normalize the query once,
        # so cosine distance is exactly 1 - dot; no per-call norms
        missing = [n for n in node_ids if n not in dist_cache]
        if not missing:
            return
        dists = 1.0 - self._mat[missing] @ query_vec
        for n, d in zip(missing, dists.tolist()):
            dist_cache[n] = d

    def _search_layer(self, query_vec, entry_points, ef, layer, dist_cache):
        self._batch_dist(query_vec, entry_points, dist_cache)

        visited = set(entry_points)

        # use a counter to break ties without comparing node_ids
        counter = 0
        candidates = []
        result = []

        for ep in entry_points:
            d = dist_cache[ep]
            heapq.heappush(candidates, (d, counter, ep))
            heapq.heappush(result, (-d, counter, ep))
            counter += 1

        while candidates:
            curr_dist, _, current = heapq.heappop(candidates)
            worst_dist = -result[0][0]

            if curr_dist > worst_dist and len(result) >= ef:
                break

            neighbors = self._graph[layer].get(current, [])
            new_neighbors = [n for n in neighbors if n not in visited]
            visited.update(new_neighbors)

            self._batch_dist(query_vec, new_neighbors, dist_cache)

            for neighbor in new_neighbors:
                nd = dist_cache[neighbor]
                worst_dist = -result[0][0]
                if nd < worst_dist or len(result) < ef:
                    heapq.heappush(candidates, (nd, counter, neighbor))
                    heapq.heappush(result, (-nd, counter, neighbor))
                    counter += 1
                    if len(result) > ef:
                        heapq.heappop(result)

        return [node_id for _, _, node_id in result]

    def _select_neighbors(self, query_vec, candidates, M, dist_cache):
        if not self.heuristic_select:
            return sorted(candidates, key=lambda n: dist_cache[n])[:M]
        # Algorithm 4: walk candidates nearest-first, keep c only if no
        # already-selected p is closer to c than the query is, then refill
        # from the pruned pile (keepPrunedConnections) if short of M
        ordered = sorted(candidates, key=lambda n: dist_cache[n])
        if len(ordered) <= M:
            return ordered
        rows = self._mat[ordered]
        pairwise = 1.0 - rows @ rows.T  # one gemm for all candidate pairs
        d_query = [dist_cache[n] for n in ordered]
        selected, pruned = [], []
        for i in range(len(ordered)):
            if len(selected) == M:
                break
            keep = not selected or bool(np.all(pairwise[i, selected] >= d_query[i]))
            (selected if keep else pruned).append(i)
        for i in pruned:
            if len(selected) >= M:
                break
            selected.append(i)
        return [ordered[i] for i in selected]

    def add(self, doc_id, vector):
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vec.shape}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if doc_id in self._doc_to_node:
            self._dead.add(self._doc_to_node[doc_id])

        node_id = self._next_id
        self._ensure_capacity(node_id + 1)
        self._mat[node_id] = vec
        self._next_id += 1
        self._doc_to_node[doc_id] = node_id
        self._node_to_doc[node_id] = doc_id
        node_level = self._random_level()

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_layer = node_level
            for layer in range(node_level + 1):
                self._graph[layer][node_id] = []
            return

        dist_cache = {}
        entry_points = [self._entry_point]

        for layer in range(self._max_layer, node_level, -1):
            entry_points = self._search_layer(vec, entry_points, ef=1, layer=layer, dist_cache=dist_cache)

        for layer in range(min(node_level, self._max_layer), -1, -1):
            candidates = self._search_layer(vec, entry_points, ef=self.ef_construction, layer=layer, dist_cache=dist_cache)
            cap = self.M_max0 if layer == 0 else self.M
            # the paper's Algorithm 1 hands the new node M links even at layer
            # 0 and uses M_max0 only as the prune cap on existing nodes; with
            # layer0_full_links the new node takes M_max0 links there instead
            select_count = cap if (layer == 0 and self.layer0_full_links) else self.M
            neighbors = self._select_neighbors(vec, candidates, select_count, dist_cache)

            self._graph[layer][node_id] = list(neighbors)
            for neighbor in neighbors:
                self._graph[layer][neighbor].append(node_id)
                if len(self._graph[layer][neighbor]) > cap:
                    neighbor_vec = self._mat[neighbor]
                    neighbor_cache = {}
                    self._batch_dist(neighbor_vec, self._graph[layer][neighbor], neighbor_cache)
                    self._graph[layer][neighbor] = self._select_neighbors(
                        neighbor_vec, self._graph[layer][neighbor], cap, neighbor_cache
                    )

            entry_points = candidates

        # register the node at layers above the old top, so layer membership
        # is always readable off the graph keys
        for layer in range(min(node_level, self._max_layer) + 1, node_level + 1):
            self._graph[layer][node_id] = []

        if node_level > self._max_layer:
            self._max_layer = node_level
            self._entry_point = node_id

    def search(self, query, top_k=10, ef_search=50):
        """Return exactly min(top_k, len(self)) (doc_id, score) pairs, sorted
        by descending cosine similarity, scores in [-1, 1], no doc repeated.

        The layer-0 walk runs at ef = max(ef_search, top_k). If it still
        comes back short (disconnected region, heavy tombstoning), the
        remainder is filled by brute force over the leftover live nodes so
        the arity contract always holds.
        """
        live = len(self._doc_to_node)
        if top_k <= 0 or live == 0:
            return []
        vec = np.asarray(query, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"expected query of shape ({self.dim},), got {vec.shape}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        expected = min(top_k, live)
        ef = max(ef_search, top_k)
        dist_cache = {}
        entry_points = [self._entry_point]

        for layer in range(self._max_layer, 0, -1):
            entry_points = self._search_layer(vec, entry_points, ef=1, layer=layer, dist_cache=dist_cache)

        candidates = self._search_layer(vec, entry_points, ef=ef, layer=0, dist_cache=dist_cache)
        candidates.sort(key=lambda n: dist_cache[n])

        picked = []
        for node_id in candidates:
            if node_id in self._dead:
                continue
            picked.append(node_id)
            if len(picked) == expected:
                break

        if len(picked) < expected:
            picked_set = set(picked)
            rest = [n for n in self._doc_to_node.values() if n not in picked_set]
            rest_dists = 1.0 - self._mat[rest] @ vec
            for n, d in zip(rest, rest_dists.tolist()):
                dist_cache[n] = d
            rest.sort(key=lambda n: dist_cache[n])
            picked.extend(rest[: expected - len(picked)])

        # clamp: float32 rounding can push a cosine a hair past +/-1
        results = [(self._node_to_doc[n], min(1.0, max(-1.0, 1.0 - dist_cache[n]))) for n in picked]
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    def __len__(self):
        return len(self._doc_to_node)

    def __repr__(self):
        return f"HNSWIndex(dim={self.dim}, n={len(self._doc_to_node)}, M={self.M}, layers={self._max_layer + 1})"
