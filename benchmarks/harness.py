import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class BenchmarkResult:
    label: str
    corpus_size: int
    num_queries: int
    latencies_ms: list[float] = field(repr=False)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def compute(self) -> "BenchmarkResult":
        if not self.latencies_ms:
            # no queries ran; NaN keeps "unknown" distinct from "0 ms"
            self.p50_ms = self.p95_ms = self.p99_ms = float("nan")
            self.mean_ms = self.min_ms = self.max_ms = float("nan")
            return self
        lat = np.asarray(self.latencies_ms)
        self.p50_ms = float(np.percentile(lat, 50))
        self.p95_ms = float(np.percentile(lat, 95))
        self.p99_ms = float(np.percentile(lat, 99))
        self.mean_ms = float(lat.mean())
        self.min_ms = float(lat.min())
        self.max_ms = float(lat.max())
        return self

    def __str__(self) -> str:
        return (
            f"{self.label:<35} "
            f"n={self.corpus_size:<8} "
            f"p50={self.p50_ms:6.2f}ms  "
            f"p95={self.p95_ms:6.2f}ms  "
            f"p99={self.p99_ms:6.2f}ms  "
            f"mean={self.mean_ms:6.2f}ms"
        )


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    # vectorized: the old per-component random.gauss loop was 128M interpreter
    # calls for a 1M x 128 corpus, and reseeding the global random module here
    # coupled every other consumer of it to corpus generation
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def time_queries(search_fn: Callable, queries, top_k: int = 10) -> list[float]:
    latencies = []
    for query in queries:
        start = time.perf_counter()
        search_fn(query, top_k)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
    return latencies


def run_benchmark(label, build_fn, search_fn, corpus_size, dim=128, num_queries=2000, top_k=10, seed=42):
    corpus = generate_random_vectors(corpus_size, dim, seed=seed)
    queries = generate_random_vectors(num_queries, dim, seed=seed + 9999)
    build_fn(corpus)
    for q in queries[:5]:
        search_fn(q, top_k)
    latencies = time_queries(search_fn, queries[:num_queries], top_k)
    result = BenchmarkResult(label=label, corpus_size=corpus_size, num_queries=num_queries, latencies_ms=latencies)
    return result.compute()


def print_comparison_table(results):
    print("\n" + "=" * 90)
    print(f"{'Label':<35} {'n':<8} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8}")
    print("=" * 90)
    for r in results:
        print(f"{r.label:<35} {r.corpus_size:<8} {r.p50_ms:>7.2f}ms {r.p95_ms:>7.2f}ms {r.p99_ms:>7.2f}ms {r.mean_ms:>7.2f}ms")
    print("=" * 90)
