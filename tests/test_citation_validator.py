from src.verification.citation_validator import validate_citations

def test_validate_citation():
    chunks = [
        {
            "document_name": "self_rag.pdf",
            "page_number":3,
            "text":"RAG retrieves external passages."
        }
    ]

    answer = (
        "RAG uses retrieved passages"
        "[self_rag.pdf, page 3]."
    )

    result = validate_citations(answer, chunks)

    assert result["total_citations"] == 1
    assert result["all_valid"] is True
    assert len(result["invalid_citations"]) == 0


def test_invalid_citation():
    chunks = [
        {
            "document_name": "self_rag.pdf",
            "page_number":3,
            "text":"RAG retrieves external passages."
        }
    ]


    answer = (
        "RAG uses retrieved passages"
        "[unknown.pdf, page 4]."
    )

    result = validate_citations(answer, chunks)

    assert result["all_valid"] is False
    assert len(result["invalid_citations"]) == 1