
from src.verification.semantic_verifier import (
    extract_claims,
    verify_claims
)


def test_extract_claims():
    answer = (
        "RAG uses retrieved passages "
        "[self_rag.pdf, page 3]. "
        "SELF-RAG evaluates retrieved information "
        "[self_rag.pdf, page 2]."
    )

    claims = extract_claims(answer)

    assert len(claims) == 2
    assert claims[0]["document"] == "self_rag.pdf"
    assert claims[0]["page"] == "3"
    assert claims[1]["page"] == "2"


def test_missing_source():
    answer = (
        "RAG improves knowledge-intensive tasks "
        "[unknown.pdf, page 99]."
    )

    chunks = [
        {
            "document_name": "self_rag.pdf",
            "page_number": 3,
            "text": "RAG retrieves external passages."
        }
    ]

    result = verify_claims(answer, chunks)

    assert result["total_claims"] == 1
    assert result["results"][0]["status"] == "SOURCE_NOT_RETRIEVED"
