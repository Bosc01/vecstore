import random
import heapq
import numpy as np
from collections import defaultdict


class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 50):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self._vectors = {}
        self._graph = defaultdict(lambda: defaultdict(list))
        self._entry_point = None
        self._max_layer = 0
        self._next_id = 0
        self._doc_to_node = {}
        self._node_to_doc = {}

    def _random_level(self):
        level = 0
        while random.random() < (1.0 / self.M) and level < 32:
            level += 1
        return level

    def _batch_dist(self, query_vec, node_ids, dist_cache):
        missing = [n for n in node_ids if n not in dist_cache]
        if not missing:
            return
        matrix = np.stack([self._vectors[n] for n in missing])
        dots = matrix @ query_vec
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = float(np.linalg.norm(query_vec))
        denom = norms * q_norm
        denom = np.where(denom == 0, 1e-10, denom)
        dists = 1.0 - dots / denom
        for n, d in zip(missing, dists.tolist()):
            dist_cache[n] = float(d)

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

    def _select_neighbors(self, candidates, M, dist_cache):
        return sorted(candidates, key=lambda n: dist_cache[n])[:M]

    def add(self, doc_id, vector):
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        node_id = self._next_id
        self._next_id += 1
        self._vectors[node_id] = vec
        self._doc_to_node[doc_id] = node_id
        self._node_to_doc[node_id] = doc_id
        node_level = self._random_level()

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_layer = node_level
            return

        dist_cache = {}
        entry_points = [self._entry_point]

        for layer in range(self._max_layer, node_level, -1):
            entry_points = self._search_layer(vec, entry_points, ef=1, layer=layer, dist_cache=dist_cache)

        for layer in range(min(node_level, self._max_layer), -1, -1):
            candidates = self._search_layer(vec, entry_points, ef=self.ef_construction, layer=layer, dist_cache=dist_cache)
            M_layer = self.M_max0 if layer == 0 else self.M
            neighbors = self._select_neighbors(candidates, M_layer, dist_cache)

            self._graph[layer][node_id] = neighbors
            for neighbor in neighbors:
                self._graph[layer][neighbor].append(node_id)
                if len(self._graph[layer][neighbor]) > M_layer:
                    neighbor_cache = {}
                    self._batch_dist(self._vectors[neighbor], self._graph[layer][neighbor], neighbor_cache)
                    self._graph[layer][neighbor] = self._select_neighbors(
                        self._graph[layer][neighbor], M_layer, neighbor_cache
                    )

            entry_points = candidates

        if node_level > self._max_layer:
            self._max_layer = node_level
            self._entry_point = node_id

    def search(self, query, top_k=10, ef_search=50):
        if self._entry_point is None:
            return []
        vec = np.array(query, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        ef_search = max(ef_search, top_k)
        dist_cache = {}
        entry_points = [self._entry_point]

        for layer in range(self._max_layer, 0, -1):
            entry_points = self._search_layer(vec, entry_points, ef=1, layer=layer, dist_cache=dist_cache)

        candidates = self._search_layer(vec, entry_points, ef=ef_search, layer=0, dist_cache=dist_cache)

        results = []
        candidates.sort(key=lambda n: dist_cache.get(n, 1.0))
        for node_id in candidates[:top_k]:
            doc_id = self._node_to_doc[node_id]
            score = 1.0 - dist_cache[node_id]
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def __len__(self):
        return len(self._vectors)

    def __repr__(self):
        return f"HNSWIndex(dim={self.dim}, n={len(self._vectors)}, M={self.M}, layers={self._max_layer + 1})"
