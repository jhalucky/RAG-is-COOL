import math
import re
from typing import Any

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "from", "by", "with",
    "as", "at", "this", "that", "these", "those", "it",
    "its", "into", "about", "than", "their", "they",
    "them", "we", "our", "you", "your", "which", "what",
    "when", "where", "who", "how", "why", "can", "could",
    "would", "should", "may", "might", "will", "also",
}


def _tokens(text: str) -> set[str]:
    return {
        token 
        for token in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }

def _relevance_score(chunk_tokens: set[str], query_tokens: set[str]) -> float:
    if not chunk_tokens or not query_tokens:
        return 0.0

    overlap = len(chunk_tokens & query_tokens)

    return overlap/math.sqrt(len(chunk_tokens) * len(query_tokens))


def _document_key(chunk: dict[str, Any], index: int) -> str:
    document = (
        chunk.get("document_id")
        or chunk.get("document_name")
        or chunk.get("source")
    )

    return str(document) if document else f"unknown-document-{index}"


def select_diverse_evidence(
        question: str,
        subquestions: list[str],
        chunks: list[dict[str, Any]],
        max_chunks: int = 8,
        max_per_document: int = 2
) -> dict[str, Any]:

    if max_chunks < 1:
        raise ValueError("max_chunks must be atleast 1.")

    if max_per_document < 1:
        raise ValueError("max_per_document must be atleast 1.")

    if not chunks:
        return {
            "chunks": [],
            "source_count": 0,
            "subquestions_covered": [],
            "subquestion_coverage": 0.0
        }

    research_questions = [
        item.strip()
        for item in subquestions
        if isinstance(item, str) and item.strip()
    ]

    if not research_questions:
        research_questions = [question]

    query_tokens = _tokens(question)
    subquestion_tokens = [_tokens(item) for item in research_questions]

    candidates = []

    for index, chunk in enumerate(chunks):
        chunk_tokens = _tokens(str(chunk.get("text", "")))

        coverage = set()
        for sub_index, tokens in enumerate(subquestion_tokens):
            if not tokens:
                continue

            overlap = chunk_tokens & tokens
            minimum_overlap = max(1, math.ceil(len(tokens) * 0.2))

            if len(overlap) >= minimum_overlap:
                coverage.add(sub_index)

        relevance = _relevance_score(chunk_tokens, query_tokens)

        # Fall back to subquestion relevance when the main question
        # has little lexical overlap with the passage.
        if relevance == 0.0:
            relevance = max(
                (
                    _relevance_score(chunk_tokens, tokens)
                    for tokens in subquestion_tokens
                ),
                default=0.0,
            )

        candidates.append({
            "index": index,
            "chunk": chunk,
            "document": _document_key(chunk, index),
            "coverage": coverage,
            "relevance": relevance,
        })

    selected = []
    selected_indices = set()
    document_counts: dict[str, int] = {}
    covered_subquestions: set[int] = set()

    while len(selected) < max_chunks:
        best_candidate = None
        best_score = -1.0

        for candidate in candidates:
            index = candidate["index"]
            document = candidate["document"]

            if index in selected_indices:
                continue

            if document_counts.get(document, 0) >= max_per_document:
                continue

            new_coverage = candidate["coverage"] - covered_subquestions

            coverage_gain = (
                len(new_coverage) / len(subquestion_tokens)
                if subquestion_tokens
                else 0.0
            )

            score = (
                0.55 * candidate["relevance"]
                + 0.45 * coverage_gain
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate["chunk"])
        selected_indices.add(best_candidate["index"])

        document = best_candidate["document"]
        document_counts[document] = document_counts.get(document, 0) + 1
        covered_subquestions.update(best_candidate["coverage"])

    covered_questions = [
        research_questions[index]
        for index in sorted(covered_subquestions)
    ]

    return {
        "chunks": selected,
        "source_count": len(document_counts),
        "subquestions_covered": covered_questions,
        "subquestion_coverage": round(
            len(covered_subquestions) / len(subquestion_tokens), 4
        ) if subquestion_tokens else 0.0,
    }

    