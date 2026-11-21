import re

from torch import chunk

def validate_citations(answer: str, chunks: list[dict]) -> dict:
    """
    Checks whether the citations in the answer are valid based on the provided chunks.
    """

    citation_pattern = r"\[([^,\]]+),\s*page\s+(\d+)\]"

    citations = re.findall[
        citation_pattern,
        answer,
        flags=re.IGNORECASE
    ]

    # Build the set of valid document page combinations.

    valid_sources = {
        (
            str(chunk.get("document_name","")).strip(),
            str(chunk.get("page_number","")).strip()

        )

        for chunk in chunks
    }

    valid_citations = []
    invalid_citations = []

    for document, page in citations:
        citation = (document.strip(), page.strip())

        if citation in valid_sources:
            
    