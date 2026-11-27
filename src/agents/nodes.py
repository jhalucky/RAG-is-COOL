from src.retrieval.search import search as dense_search
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import NeuralReranker
from src.verification.evidence_sufficiency import evaluate_evidence


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



def parse_planner_response(raw_response: str) -> list[str]:
    """Parse and validate Gemini's research plan."""

    if not raw_response or not raw_response.strip():
        raise ValueError("Gemini returned an empty response.")

    cleaned = raw_response.strip()

    # Handle Markdown code fences.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    result = json.loads(cleaned)

    if not isinstance(result, dict):
        raise ValueError("Planner response must be a JSON object.")

    research_plan = result.get("subquestions")

    if (
        not isinstance(research_plan, list)
        or not 3 <= len(research_plan) <= 5
        or not all(
            isinstance(item, str) and item.strip()
            for item in research_plan
        )
    ):
        raise ValueError("Invalid research subquestions.")

    return [item.strip() for item in research_plan]


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
            response_mime_type="application/json",
        ),
        )

        raw_response = response.text or ""

        print("\n[Planner] Raw Gemini response:")
        print(repr(raw_response))

        research_plan = parse_planner_response(raw_response)

        print(
            f"[Planner] Generated "
            f"{len(research_plan)} research subquestions."
        )

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
    Perform multi-query retrieval and preserve evidence
    collected across adaptive retrieval attempts.
    """

    question = state["question"]

    queries = (
        state.get("retrieval_queries")
        or state.get("research_plan")
        or [question]
    )

    print("\n[Retrieval] Starting retrieval...")
    print(f"[Retrieval] Queries: {len(queries)}")

    # Preserve evidence from previous attempts.
    all_candidates = {}

    for chunk in state.get("retrieved_chunks", []):
        chunk_id = (
            chunk.get("document_id") or chunk.get("document_name"),
            chunk.get("page_number"),
            chunk.get("chunk_index"),
        )
        all_candidates[chunk_id] = chunk

    # Retrieve evidence for each query.
    for index, query in enumerate(queries, start=1):

        print(f"\n[Retrieval] Query {index}/{len(queries)}")
        print(query)

        candidates = hybrid_searcher.search(
            query,
            top_k=10,
            retrieval_k=10,
        )

        print(f"[Retrieval] Candidates: {len(candidates)}")

        for candidate in candidates:
            chunk_id = (
                candidate.get("document_id")
                or candidate.get("document_name"),
                candidate.get("page_number"),
                candidate.get("chunk_index"),
            )

            if chunk_id not in all_candidates:
                all_candidates[chunk_id] = candidate

    merged_candidates = list(all_candidates.values())

    print(
        f"\n[Retrieval] Total unique evidence: "
        f"{len(merged_candidates)}"
    )

    if not merged_candidates:
        retrieved_chunks = []
    else:
        retrieved_chunks = reranker.rerank(
            query=question,
            candidates=merged_candidates,
            top_k=5,
        )

    attempt = state.get("retrieval_attempts", 0) + 1

    print(f"[Retrieval] Attempt: {attempt}")

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieval_attempts": attempt,
    }



def evidence_evaluation_node(state: dict) -> dict:
    """
    Evaluate whether the accumulated evidence sufficiently
    covers the research question.
    """

    print("\n[Evidence Evaluation] Evaluating evidence quality...")

    try:
        result = evaluate_evidence(
            question=state["question"],
            research_plan=state.get("research_plan", []),
            retrieved_chunks=state.get("retrieved_chunks", []),
        )

        print(
            f"[Evidence Evaluation] Coverage: "
            f"{result['coverage_score']:.2f}"
        )

        print(
            f"[Evidence Evaluation] Sufficient: "
            f"{result['sufficient']}"
        )

        print(
            f"[Evidence Evaluation] Missing aspects: "
            f"{result['missing_aspects']}"
        )

        return {
            "evidence_evaluation": result,
            "evidence_sufficient": result["sufficient"],
            "evidence_evaluation_failed": False,
        }

    except Exception as error:
        message = f"Evidence evaluation failed: {error}"

        print(f"\n[Evidence Evaluation] {message}")

        errors = list(state.get("errors", []))
        errors.append(message)

        return {
            "evidence_sufficient": False,
            "evidence_evaluation_failed": True,
            "errors": errors,
        }



def query_refinement_node(state: dict) -> dict:
    """
    Prepare focused retrieval queries based on evidence gaps.
    """

    evaluation = state.get("evidence_evaluation", {})

    suggested_queries = evaluation.get("suggested_queries", [])
    missing_aspects = evaluation.get("missing_aspects", [])

    # Prefer evaluator-generated queries.
    refined_queries = suggested_queries or missing_aspects

    # Remove empty queries and duplicates.
    refined_queries = list(dict.fromkeys(
        query.strip()
        for query in refined_queries
        if isinstance(query, str) and query.strip()
    ))

    # Safety fallback.
    if not refined_queries:
        refined_queries = [
            f"Additional evidence about {aspect}"
            for aspect in missing_aspects
            if isinstance(aspect, str) and aspect.strip()
        ]

    if not refined_queries:
        print(
            "[Query Refinement] No useful additional queries. "
            "Proceeding to generation."
        )

    print("\n[Query Refinement] New research queries:")

    for index, query in enumerate(refined_queries, start=1):
        print(f"{index}. {query}")

    return {
        "retrieval_queries": refined_queries,
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


    evaluation = state.get("evidence_evaluation", {})

    if evaluation:
        report_content += (
            "\n\nEVIDENCE EVALUATION\n"
            f"Coverage score: {evaluation.get('coverage_score', 0):.2f}\n"
            f"Evidence sufficient: {evaluation.get('sufficient', False)}\n"
            f"Reasoning: {evaluation.get('reasoning', '')}\n"
        )

        missing = evaluation.get("missing_aspects", [])

        if missing:
            report_content += "\nUnresolved evidence gaps:\n"
            report_content += "\n".join(
                f"- {aspect}" for aspect in missing
            )

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
