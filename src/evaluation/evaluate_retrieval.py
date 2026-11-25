import json 
import time
from pathlib import Path
from collections.abc import Callable

from src.retrieval.search import search as dense_search
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import NeuralReranker

EVALUATION_FILE = Path("data/evaluation/retrieval_eval.json")

TOP_K = 5
RETRIEVAL_K=10

def load_evaluation_questions() -> list[dict]:
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    for item in questions:
        if "question" not in item or "expected_documents" not in item:
            raise ValueError(
                f"Invalid evaluation entry: {item.get('id', 'unknown')}"
            )

    return questions


def get_unique_documents(results: list[dict]) -> list[str]:
    """Return unique document names in retrieval ranking order."""
    documents = []

    for result in results:
        name = result["document_name"]

        if name not in documents:
            documents.append(name)

    return documents

def calculate_metrics(
        retrieved_documents: list[str],
        expected_documents: list[str],
        k: int = TOP_K
) -> dict:
    relevant = set(expected_documents)
    top_results= retrieved_documents[:k]

    relevant_ranks = [
        rank 
        for rank, document in enumerate(top_results, start=1) 
        if document in relevant
    ]

    precision = len(relevant_ranks)/k
    recall = len(relevant_ranks) / len(relevant) if relevant else 0.0
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0

    return {
        f"precision@{k}": precision,
        f"recall@{k}":recall,
        "mrr": reciprocal_rank
    }


def evaluate_method(
    method_name: str,
    retrieval_function: Callable,
    questions: list[dict]
) -> dict:
    question_results = []
    total_latency = 0.0

    print(f"\nEvaluating {method_name}...")

    for item in questions:
        question = item["question"]
        expected_documents = item["expected_documents"]

        start_time = time.perf_counter()
        results = retrieval_function(question)
        latency = time.perf_counter() - start_time

        total_latency += latency

        retrieved_documents = get_unique_documents(results)
        metrics = calculate_metrics(
            retrieved_documents,
            expected_documents
        )

        question_results.append(metrics)

        print(
            f"  {item.get('id', 'unknown')}: "
            f"Recall@{TOP_K}={metrics[f'recall@{TOP_K}']:.3f}, "
            f"MRR={metrics['mrr']:.3f}, "
            f"Latency={latency:.3f}s"
        )

    count = len(question_results)

    summary = {
        "method": method_name,
        f"precision@{TOP_K}": sum(
            result[f"precision@{TOP_K}"]
            for result in question_results
        ) / count,
        f"recall@{TOP_K}": sum(
            result[f"recall@{TOP_K}"]
            for result in question_results
        ) / count,
        "mrr": sum(
            result["mrr"] for result in question_results
        ) / count,
        "avg_latency": total_latency / count,
    }

    return summary


def print_report(summaries: list[dict]) -> None:
    """Print a comparison table for all retrieval methods."""
    print("\n" + "=" * 78)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 78)

    print(
        f"{'Method':<25}"
        f"{'Precision@5':>13}"
        f"{'Recall@5':>12}"
        f"{'MRR':>10}"
        f"{'Avg Time':>12}"
    )
    print("-" * 78)

    for result in summaries:
        print(
            f"{result['method']:<25}"
            f"{result['precision@5']:>13.3f}"
            f"{result['recall@5']:>12.3f}"
            f"{result['mrr']:>10.3f}"
            f"{result['avg_latency']:>10.3f}s"
        )

    print("=" * 78)


def main() -> None:
    questions = load_evaluation_questions()

    print(f"Loaded {len(questions)} evaluation questions.")
    print("Initializing retrieval components...")

    # Initialize components once, outside the benchmark timers.
    bm25 = BM25Search()
    hybrid = HybridSearch(dense_search_fn=dense_search)
    reranker = NeuralReranker()

    methods = [
        (
            "Dense",
            lambda query: dense_search(query, top_k=TOP_K),
        ),
        (
            "BM25",
            lambda query: bm25.search(query, top_k=TOP_K),
        ),
        (
            "Hybrid",
            lambda query: hybrid.search(
                query,
                top_k=TOP_K,
                retrieval_k=RETRIEVAL_K,
            ),
        ),
        (
            "Hybrid + Reranker",
            lambda query: reranker.rerank(
                query=query,
                candidates=hybrid.search(
                    query,
                    top_k=RETRIEVAL_K,
                    retrieval_k=RETRIEVAL_K,
                ),
                top_k=TOP_K,
            ),
        ),
    ]

    summaries = []

    for method_name, retrieval_function in methods:
        summary = evaluate_method(
            method_name,
            retrieval_function,
            questions,
        )
        summaries.append(summary)

    print_report(summaries)


if __name__ == "__main__":
    main()
