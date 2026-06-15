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
        self._matrix = np.empty((0, dim), dtype=np.float32)

    def add(self, doc_id, vector) -> None:
        self._ids.append(doc_id)
        self._matrix = np.vstack([self._matrix, np.array(vector, dtype=np.float32)])

    def add_batch(self, items) -> None:
        ids, vecs = zip(*items)
        self._ids = list(ids)
        self._matrix = np.array(vecs, dtype=np.float32)

    def search(self, query, top_k: int = 10):
        q = np.array(query, dtype=np.float32)
        norms = np.linalg.norm(self._matrix, axis=1)
        q_norm = np.linalg.norm(q)
        scores = self._matrix @ q / (norms * q_norm + 1e-10)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(self._ids[i], float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f"VectorStore(dim={self.dim}, n={len(self._ids)})"
