from src.retrieval.search import search as dense_search
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import NeuralReranker

from src.generation.answer_generator import generate_answer

from src.verification.citation_validator import validate_citations
from src.verification.semantic_verifier import verify_claims

bm25_searcher = BM25Search()
hybrid_searcher = HybridSearch(dense_search_fn=dense_search)
reranker = NeuralReranker()

def planner_node(state: dict) -> dict:
    question = state["question"]

    research_plan = [
        f"Find relevant research evidence for: {question}",
        f"Identify key findings related to: {question}",
        f"verify the findings against their original sources."
    ]

    print("\n[Planner] Research plan created.")

    return {
        "research_plan": research_plan,
        "revision_count": state.get("revision_count", 0),
        "errors": state.get("errors", [])
    }

def retrieval_node(state: dict) -> dict:

    question = state["question"]

    print("\n[Retrieval] Searching research papers...")

    candidates = hybrid_searcher.search(
        question,
        top_k = 10,
        retrieval_k=10
    )

    retrieved_chunks = reranker.rerank(
        query=question,
        candidates=candidates,
        top_k=5
    )

    print(f"[Retrieval] Retrieved {len(retrieved_chunks)} passages.")

    return {
        "retrieved_chunks": retrieved_chunks,
    }

def generation_node(state: dict) -> dict:
    """
    Generate an evidence-grounded answer using Gemini.
    """
    print("\n[Generation] Generating research answer...")

    answer = generate_answer(
        question=state["question"],
        chunks=state.get("retrieved_chunks", []),
    )

    return {
        "answer": answer,
    }


def verification_node(state: dict) -> dict:
    """
    Validate citations and semantically verify generated claims.
    """
    print("\n[Verification] Validating citations...")

    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])

    citation_result = validate_citations(answer, chunks)

    print("[Verification] Checking semantic support...")

    semantic_result = verify_claims(answer, chunks)

    semantic_results = semantic_result.get("results", [])

    semantic_passed = (
        semantic_result.get("total_claims", 0) > 0
        and all(
            item.get("verdict") == "SUPPORTED"
            for item in semantic_results
        )
    )

    verification_passed = (
        citation_result.get("all_valid", False)
        and semantic_passed
    )

    print(
        f"[Verification] Passed: {verification_passed}"
    )

    return {
        "citation_validation": citation_result,
        "semantic_verification": semantic_result,
        "verification_passed": verification_passed,
    }


def final_report_node(state: dict) -> dict:
    """
    Prepare the final research report.
    """
    answer = state.get("answer", "")

    status = (
        "Verified"
        if state.get("verification_passed", False)
        else "Verification requires review"
    )

    final_report = (
        f"RESEARCH REPORT\n"
        f"{'=' * 60}\n\n"
        f"Question: {state['question']}\n\n"
        f"Status: {status}\n\n"
        f"{answer}\n"
    )

    return {
        "final_report": final_report,
    }
