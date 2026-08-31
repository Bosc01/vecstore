"""Sweep ef_search for my HNSWIndex and faiss.IndexHNSWFlat at matched
parameters (M=16, efConstruction=50, inner product), n=10k dim=128, with
ground truth from IndexFlatIP. Emits a markdown table (stdout and
benchmarks/recall_curve.md) and a recall-vs-latency plot
(benchmarks/recall_vs_latency.png).

Latency per sweep point is the p50 of the fastest of 3 passes, which damps
scheduler noise from whatever else the machine is doing.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecstore.hnsw import HNSWIndex
from benchmarks.harness import generate_random_vectors

EF_SWEEP = [10, 25, 50, 100, 200, 400]
N = 10_000
DIM = 128
NUM_QUERIES = 200
TOP_K = 10
M = 16
EF_CONSTRUCTION = 50
SEED = 42
REPS = 3

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def sweep_point(search_fn, ground_truth, queries):
    p50s = []
    hits = 0
    for rep in range(REPS):
        lat = []
        rep_hits = 0
        for qi in range(len(queries)):
            t0 = time.perf_counter()
            ids = search_fn(queries[qi])
            lat.append((time.perf_counter() - t0) * 1e3)
            rep_hits += len(set(ids) & ground_truth[qi])
        p50s.append(float(np.percentile(lat, 50)))
        hits = rep_hits  # deterministic across reps
    recall = hits / (len(queries) * TOP_K)
    return recall, min(p50s)


def main():
    faiss.omp_set_num_threads(1)
    corpus = generate_random_vectors(N, DIM, seed=SEED)
    queries = generate_random_vectors(NUM_QUERIES, DIM, seed=SEED + 9999)

    flat = faiss.IndexFlatIP(DIM)
    flat.add(corpus)
    _, gt_ids = flat.search(queries, TOP_K)
    ground_truth = [set(row.tolist()) for row in gt_ids]

    print(f"building HNSWIndex (Python), n={N}...", flush=True)
    mine = HNSWIndex(dim=DIM, M=M, ef_construction=EF_CONSTRUCTION, seed=SEED)
    t0 = time.perf_counter()
    for i in range(N):
        mine.add(i, corpus[i])
    print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    theirs = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    theirs.hnsw.efConstruction = EF_CONSTRUCTION
    theirs.add(corpus)

    rows = []
    for ef in EF_SWEEP:
        my_recall, my_p50 = sweep_point(
            lambda q: [d for d, _ in mine.search(q, TOP_K, ef_search=ef)],
            ground_truth, queries,
        )
        theirs.hnsw.efSearch = ef
        their_recall, their_p50 = sweep_point(
            lambda q: theirs.search(q.reshape(1, -1), TOP_K)[1][0].tolist(),
            ground_truth, queries,
        )
        rows.append((ef, my_recall, my_p50, their_recall, their_p50))
        print(f"ef={ef}: mine recall={my_recall:.3f} p50={my_p50:.3f}ms | "
              f"faiss recall={their_recall:.3f} p50={their_p50:.3f}ms", flush=True)

    lines = [
        f"Recall@{TOP_K} vs p50 latency, n={N}, dim={DIM}, M={M}, "
        f"efConstruction={EF_CONSTRUCTION}, {NUM_QUERIES} queries, "
        f"ground truth = IndexFlatIP.",
        "",
        "| ef_search | HNSW (Python) recall@10 | HNSW (Python) p50 (ms) | FAISS HNSW recall@10 | FAISS HNSW p50 (ms) |",
        "|---|---|---|---|---|",
    ]
    for ef, mr, mp, tr, tp in rows:
        lines.append(f"| {ef} | {mr:.3f} | {mp:.3f} | {tr:.3f} | {tp:.3f} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(OUT_DIR, "recall_curve.md"), "w") as f:
        f.write(table + "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, recalls, p50s, color in [
        ("HNSW (Python)", [r[1] for r in rows], [r[2] for r in rows], "tab:blue"),
        ("FAISS HNSW (C++)", [r[3] for r in rows], [r[4] for r in rows], "tab:orange"),
    ]:
        ax.plot(p50s, recalls, marker="o", color=color, label=label)
        for ef, x, y in zip(EF_SWEEP, p50s, recalls):
            ax.annotate(f"ef={ef}", (x, y), textcoords="offset points",
                        xytext=(6, -4), fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("p50 query latency (ms, log scale)")
    ax.set_ylabel(f"recall@{TOP_K}")
    ax.set_title(f"Recall vs latency, n={N}, dim={DIM}, M={M}, efC={EF_CONSTRUCTION}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "recall_vs_latency.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_png} and recall_curve.md")


if __name__ == "__main__":
    main()
