import re


def validate_citations(answer: str, chunks: list[dict]) -> dict:
    """
    Checks whether the citations in the answer are valid based on the provided chunks.
    """

    citation_pattern = r"\[([^,\]]+),\s*page\s+(\d+)\]"

    citations = re.findall(
        citation_pattern,
        answer,
        flags=re.IGNORECASE,
    )

    # Build the set of valid document page combinations.
    valid_sources = {
        (
            str(chunk.get("document_name", "")).strip(),
            str(chunk.get("page_number", "")).strip(),
        )
        for chunk in chunks
    }

    valid_citations = []
    invalid_citations = []

    for document, page in citations:
        citation = (document.strip(), page.strip())

        if citation in valid_sources:
            valid_citations.append(citation)
        else:
            invalid_citations.append(citation)

    return {
        "total_citations": len(citations),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "all_valid": len(citations) > 0 and len(invalid_citations) == 0,
    }
