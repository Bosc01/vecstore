import numpy as np


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids = []
        # amortized-doubling capacity buffer; rows [0, _size) are valid
        self._matrix = np.empty((0, dim), dtype=np.float32)
        self._size = 0

    def _ensure_capacity(self, additional: int) -> None:
        needed = self._size + additional
        capacity = self._matrix.shape[0]
        if needed <= capacity:
            return
        new_capacity = max(capacity, 16)
        while new_capacity < needed:
            new_capacity *= 2
        grown = np.empty((new_capacity, self.dim), dtype=np.float32)
        grown[: self._size] = self._matrix[: self._size]
        self._matrix = grown

    def _normalize_rows(self, block: np.ndarray) -> np.ndarray:
        # vectors are stored unit-norm so search is a bare dot product;
        # zero vectors stay zero and score 0 against everything
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        return block / np.where(norms == 0, 1.0, norms)

    def add(self, doc_id, vector) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vec.shape}")
        self._ensure_capacity(1)
        self._matrix[self._size] = self._normalize_rows(vec[np.newaxis, :])[0]
        self._ids.append(doc_id)
        self._size += 1

    def add_batch(self, items) -> None:
        items = list(items)
        if not items:
            return
        ids, vecs = zip(*items)
        block = np.array(vecs, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape ({self.dim},), got {block.shape[1:]}")
        self._ensure_capacity(len(ids))
        self._matrix[self._size : self._size + len(ids)] = self._normalize_rows(block)
        self._ids.extend(ids)
        self._size += len(ids)

    def search(self, query, top_k: int = 10):
        if top_k <= 0 or self._size == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"expected query of shape ({self.dim},), got {q.shape}")
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        scores = self._matrix[: self._size] @ q
        k = min(top_k, self._size)
        if k < self._size:
            top_indices = np.argpartition(scores, -k)[-k:]
        else:
            top_indices = np.arange(self._size)
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        # clamp: float32 rounding can push a cosine a hair past +/-1
        return [(self._ids[i], min(1.0, max(-1.0, float(scores[i])))) for i in top_indices]

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"VectorStore(dim={self.dim}, n={self._size})"
