import time
import statistics
import random
import math
from dataclasses import dataclass, field
from typing import Callable


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
        sorted_latencies = sorted(self.latencies_ms)
        n = len(sorted_latencies)
        self.p50_ms = sorted_latencies[int(n * 0.50)]
        self.p95_ms = sorted_latencies[int(n * 0.95)]
        self.p99_ms = sorted_latencies[int(n * 0.99)]
        self.mean_ms = statistics.mean(sorted_latencies)
        self.min_ms = sorted_latencies[0]
        self.max_ms = sorted_latencies[-1]
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


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> list[list[float]]:
    random.seed(seed)
    vectors = []
    for _ in range(n):
        v = [random.gauss(0, 1) for _ in range(dim)]
        mag = math.sqrt(sum(x * x for x in v))
        v = [x / mag for x in v]
        vectors.append(v)
    return vectors


def time_queries(search_fn: Callable, queries: list[list[float]], top_k: int = 10) -> list[float]:
    latencies = []
    for query in queries:
        start = time.perf_counter()
        search_fn(query, top_k)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
    return latencies


def run_benchmark(label, build_fn, search_fn, corpus_size, dim=128, num_queries=200, top_k=10, seed=42):
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
