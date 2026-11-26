from src.retrieval.search import search as dense_search
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import NeuralReranker

from google.genai.errors import APIError

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
    Handle API failures without crashing the graph.
    """
    print("\n[Generation] Generating research answer...")

    try:
        answer = generate_answer(
            question=state["question"],
            chunks=state.get("retrieved_chunks", []),
        )

        return {
            "answer": answer,
        }

    except APIError as error:
        message = f"Gemini API error: {error}"

        print(f"\n[Generation] {message}")

        errors = list(state.get("errors", []))
        errors.append(message)

        return {
            "answer": "",
            "errors": errors,
        }


def verification_node(state: dict) -> dict:
    """
    Validate citations and verify whether generated claims
    are supported by retrieved evidence.
    """
    print("\n[Verification] Validating citations...")

    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])

    citation_result = validate_citations(answer, chunks)

    print("[Verification] Checking semantic support...")

    errors = list(state.get("errors", []))

    try:
        semantic_result = verify_claims(answer, chunks)

        claims = semantic_result.get("claims", [])

        semantic_passed = (
            len(claims) > 0
            and all(
                claim.get("verdict") == "SUPPORTED"
                for claim in claims
            )
        )

        verification_passed = (
            citation_result.get("all_valid", False)
            and semantic_passed
        )

    except APIError as error:
        message = f"Semantic verification API error: {error}"
        print(f"\n[Verification] {message}")

        errors.append(message)

        semantic_result = {
            "claims": [],
            "error": message,
        }

        verification_passed = False

    return {
        "citation_validation": citation_result,
        "semantic_verification": semantic_result,
        "verification_passed": verification_passed,
        "errors": errors,
    }



def final_report_node(state: dict) -> dict:
    """
    Prepare the final research report, including failure status.
    """
    answer = state.get("answer", "")
    errors = state.get("errors", [])

    if not answer:
        status = "Generation Failed"
        report_content = (
            "An answer could not be generated because the LLM "
            "service was unavailable.\n\n"
            f"Retrieved evidence passages: "
            f"{len(state.get('retrieved_chunks', []))}\n\n"
            "Please retry the research request later."
        )
    else:
        status = (
            "Verified"
            if state.get("verification_passed", False)
            else "Verification requires review"
        )
        report_content = answer

    final_report = (
        f"RESEARCH REPORT\n"
        f"{'=' * 60}\n\n"
        f"Question: {state['question']}\n\n"
        f"Status: {status}\n\n"
        f"{report_content}\n"
    )

    if errors:
        final_report += "\nErrors:\n"
        final_report += "\n".join(errors)

    return {
        "final_report": final_report,
    }
