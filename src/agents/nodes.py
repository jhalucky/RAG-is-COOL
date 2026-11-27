from src.retrieval.search import search as dense_search
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import NeuralReranker

from google.genai.errors import APIError

import json

from google.genai import types
from src.generation.answer_generator import (
    client,
    MODEL,
    generate_answer,
)

from src.generation.answer_generator import generate_answer

from src.verification.citation_validator import validate_citations
from src.verification.semantic_verifier import verify_claims

bm25_searcher = BM25Search()
hybrid_searcher = HybridSearch(dense_search_fn=dense_search)
reranker = NeuralReranker()


def planner_node(state: dict) -> dict:
    """
    Decompose a research question into focused subquestions
    using Gemini. Fall back to a deterministic plan if needed.
    """
    question = state["question"]

    print("\n[Planner] Creating intelligent research plan...")

    errors = list(state.get("errors", []))

    fallback_plan = [
        question,
        f"What are the key concepts and background needed to answer: {question}",
        f"What evidence and research findings support the answer to: {question}",
    ]

    prompt = f"""
You are an expert academic research planner.

Break the user's research question into 3 to 5 focused,
independent research subquestions.

Each subquestion should:
- Investigate one specific aspect of the main question.
- Be suitable for searching academic research papers.
- Help build a comprehensive answer to the original question.
- Avoid repeating another subquestion.

Return ONLY valid JSON in this exact format:

{{
    "subquestions": [
        "First research subquestion",
        "Second research subquestion",
        "Third research subquestion"
    ]
}}

Research question:
{question}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )

        raw_response = response.text or ""

        print("\n[Planner] Raw Gemini response:")
        print(repr(raw_response))

        result = json.loads(raw_response)

        print("\n[Planner] Raw Gemini response:")
        print(repr(raw_response))

        result = json.loads(raw_response)
        research_plan = result.get("subquestions", [])

        if (
            not isinstance(research_plan, list)
            or not 3 <= len(research_plan) <= 5
            or not all(
                isinstance(item, str) and item.strip()
                for item in research_plan
            )
        ):
            raise ValueError("Planner returned an invalid research plan.")

        research_plan = [item.strip() for item in research_plan]

        print(f"[Planner] Generated {len(research_plan)} research subquestions.")

    except Exception as error:
        message = f"LLM planning failed; using fallback plan: {error}"
        print(f"[Planner] {message}")

        errors.append(message)
        research_plan = fallback_plan

    print("\nResearch Plan:")
    for index, subquestion in enumerate(research_plan, start=1):
        print(f"{index}. {subquestion}")

    return {
        "research_plan": research_plan,
        "revision_count": state.get("revision_count", 0),
        "errors": errors,
    }



def retrieval_node(state: dict) -> dict:
    """
    Retrieve evidence using every research subquestion,
    merge duplicate passages, and rerank against the
    original research question.
    """
    question = state["question"]
    research_plan = state.get("research_plan") or [question]

    print("\n[Retrieval] Starting multi-query retrieval...")
    print(f"[Retrieval] Research queries: {len(research_plan)}")

    all_candidates = {}

    for index, subquestion in enumerate(research_plan, start=1):
        print(f"\n[Retrieval] Query {index}/{len(research_plan)}:")
        print(subquestion)

        candidates = hybrid_searcher.search(
            subquestion,
            top_k=10,
            retrieval_k=10
        )

        print(
            f"[Retrieval] Retrieved {len(candidates)} "
            f"candidates."
        )

        for candidate in candidates:
            chunk_id = (
                candidate.get("document_id")
                or candidate.get("document_name"),
                candidate.get("page_number"),
                candidate.get("chunk_index")
            )

            if chunk_id not in all_candidates:
                all_candidates[chunk_id] = candidate

    merged_candidates = list(all_candidates.values())

    print(
        f"\n[Retrieval] Total unique candidates: "
        f"{len(merged_candidates)}"
    )

    if not merged_candidates:
        return {"retrieved_chunks": []}

    print("\n[Retrieval] Reranking combined evidence...")

    retrieved_chunks = reranker.rerank(
        query=question,
        candidates=merged_candidates,
        top_k=5
    )

    print(
        f"[Retrieval] Final evidence passages: "
        f"{len(retrieved_chunks)}"
    )

    return {
        "retrieved_chunks": retrieved_chunks
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
