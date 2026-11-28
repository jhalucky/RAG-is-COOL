
import pytest

from src.retrieval.evidence_selector import select_diverse_evidence


def test_limits_chunks_per_document():
    chunks = [
        {"document_id": "paper-a", "text": "RAG improves retrieval accuracy."},
        {"document_id": "paper-a", "text": "RAG improves factual reliability."},
        {"document_id": "paper-a", "text": "RAG retrieves relevant evidence."},
        {"document_id": "paper-b", "text": "RAG supports factual verification."},
    ]

    result = select_diverse_evidence(
        "RAG factual reliability",
        ["RAG retrieval accuracy", "factual verification"],
        chunks,
        max_chunks=4,
        max_per_document=2,
    )

    document_ids = [chunk["document_id"] for chunk in result["chunks"]]

    assert document_ids.count("paper-a") <= 2
    assert document_ids.count("paper-b") <= 2


def test_selects_evidence_from_multiple_sources():
    chunks = [
        {"document_name": "Paper A", "text": "Hybrid retrieval improves RAG search."},
        {"document_name": "Paper A", "text": "RAG search combines multiple retrieval methods."},
        {"document_name": "Paper B", "text": "Citation verification checks factual claims."},
        {"document_name": "Paper C", "text": "Context compression reduces token usage."},
    ]

    result = select_diverse_evidence(
        "RAG retrieval and citation reliability",
        ["hybrid retrieval methods", "citation verification"],
        chunks,
        max_chunks=3,
        max_per_document=1,
    )

    assert result["source_count"] >= 2


def test_balances_subquestion_coverage():
    chunks = [
        {
            "document_id": "paper-a",
            "text": "Hybrid retrieval combines dense search and BM25 methods.",
        },
        {
            "document_id": "paper-b",
            "text": "Citation verification checks claims against source evidence.",
        },
    ]

    result = select_diverse_evidence(
        "Reliable RAG research systems",
        ["hybrid retrieval methods", "citation verification"],
        chunks,
        max_chunks=2,
    )

    assert result["subquestion_coverage"] == 1.0
    assert len(result["subquestions_covered"]) == 2


def test_respects_max_chunks():
    chunks = [
        {"document_id": f"paper-{i}", "text": "RAG retrieval improves accuracy."}
        for i in range(10)
    ]

    result = select_diverse_evidence(
        "RAG retrieval accuracy",
        ["RAG retrieval"],
        chunks,
        max_chunks=3,
    )

    assert len(result["chunks"]) <= 3


def test_preserves_original_chunk_metadata():
    chunks = [
        {
            "document_id": "paper-a",
            "document_name": "Research Paper",
            "page_number": 12,
            "chunk_index": 4,
            "text": "RAG retrieval improves factual accuracy.",
            "custom_metadata": "preserve-me",
        }
    ]

    result = select_diverse_evidence(
        "RAG factual accuracy",
        ["retrieval accuracy"],
        chunks,
    )

    assert result["chunks"][0] == chunks[0]
    assert result["chunks"][0]["page_number"] == 12
    assert result["chunks"][0]["custom_metadata"] == "preserve-me"


def test_handles_empty_chunks():
    result = select_diverse_evidence(
        "RAG reliability",
        ["retrieval"],
        [],
    )

    assert result["chunks"] == []
    assert result["source_count"] == 0
    assert result["subquestion_coverage"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chunks": 0},
        {"max_per_document": 0},
    ],
)
def test_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        select_diverse_evidence("RAG", ["retrieval"], [], **kwargs)
