
from src.compression.context_compressor import compress_context


def test_compression_preserves_source_metadata():
    chunks = [
        {
            "document_name": "rag.pdf",
            "page_number": 5,
            "chunk_index": 10,
            "text": (
                "RAG combines retrieval with language generation. "
                "It can improve factual accuracy. "
                "Unrelated information concerns image classification."
            ),
        }
    ]

    result = compress_context(
        question="How does RAG improve factual accuracy?",
        chunks=chunks,
        max_tokens=100,
    )

    compressed = result["chunks"][0]

    assert compressed["document_name"] == "rag.pdf"
    assert compressed["page_number"] == 5
    assert compressed["chunk_index"] == 10
    assert "factual accuracy" in compressed["text"].lower()


def test_duplicate_sentences_are_removed():
    sentence = "RAG retrieves relevant documents for generation."

    chunks = [
        {"document_name": "a.pdf", "text": sentence},
        {"document_name": "b.pdf", "text": sentence},
    ]

    result = compress_context(
        question="How does RAG retrieve documents?",
        chunks=chunks,
        max_tokens=100,
    )

    assert len(result["chunks"]) == 1


def test_compression_respects_token_budget():
    chunks = [
        {
            "document_name": "research.pdf",
            "page_number": 1,
            "text": (
                "Retrieval improves factual accuracy in language models. "
                "Hybrid retrieval combines dense and sparse search. "
                "Reranking improves the relevance of retrieved passages."
            ),
        }
    ]

    result = compress_context(
        question="How does retrieval improve factual accuracy?",
        chunks=chunks,
        max_tokens=12,
    )

    assert result["compressed_tokens"] <= 12


def test_empty_evidence_returns_empty_result():
    result = compress_context(
        question="Research question",
        chunks=[],
    )

    assert result["chunks"] == []
    assert result["original_tokens"] == 0
    assert result["compressed_tokens"] == 0


def test_original_chunks_are_not_modified():
    chunks = [
        {
            "document_name": "paper.pdf",
            "page_number": 2,
            "text": (
                "RAG improves factual accuracy. "
                "It uses external information."
            ),
        }
    ]

    original_text = chunks[0]["text"]

    compress_context(
        question="RAG factual accuracy",
        chunks=chunks,
        max_tokens=100,
    )

    assert chunks[0]["text"] == original_text


def test_invalid_token_budget_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        compress_context(
            question="Research question",
            chunks=[],
            max_tokens=0,
        )
