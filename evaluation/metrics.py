from dataclasses import dataclass

@dataclass
class EvalResult:
    label: str
    num_queries: int
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float

    def __str__(self) -> str:
        return (f"{self.label:<35} P@{self.k}={self.precision_at_k:.3f}  R@{self.k}={self.recall_at_k:.3f}  MRR={self.mrr:.3f}")

def precision_at_k(retrieved, relevant, k):
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k if k > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def evaluate(label, search_fn, ground_truth, k=10):
    precision_scores, recall_scores, rr_scores = [], [], []
    for query_vector, relevant_ids in ground_truth:
        relevant_set = set(relevant_ids)
        results = search_fn(query_vector, k)
        retrieved_ids = [doc_id for doc_id, _ in results]
        precision_scores.append(precision_at_k(retrieved_ids, relevant_set, k))
        recall_scores.append(recall_at_k(retrieved_ids, relevant_set, k))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant_set))
    return EvalResult(
        label=label, num_queries=len(ground_truth), k=k,
        precision_at_k=sum(precision_scores)/len(precision_scores),
        recall_at_k=sum(recall_scores)/len(recall_scores),
        mrr=sum(rr_scores)/len(rr_scores),
    )

def build_ground_truth(oracle_search_fn, queries, k=10):
    return [(q, [doc_id for doc_id, _ in oracle_search_fn(q, k)]) for q in queries]

def print_eval_table(results):
    print("\n" + "=" * 75)
    print(f"{'Label':<35} {'P@K':>8} {'R@K':>8} {'MRR':>8}")
    print("=" * 75)
    for r in results:
        print(f"{r.label:<35} {r.precision_at_k:>8.3f} {r.recall_at_k:>8.3f} {r.mrr:>8.3f}")
    print("=" * 75)
