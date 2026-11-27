
from typing import TypedDict, NotRequired, Any


class ResearchState(TypedDict):
    """Shared state for the adaptive research workflow."""

    question: str

    # Planning
    research_plan: NotRequired[list[str]]

    # Retrieval
    retrieval_queries: NotRequired[list[str]]
    retrieved_chunks: NotRequired[list[dict[str, Any]]]
    retrieval_attempts: NotRequired[int]
    max_retrieval_attempts: NotRequired[int]

    # Evidence evaluation
    evidence_evaluation: NotRequired[dict[str, Any]]
    evidence_sufficient: NotRequired[bool]
    evidence_evaluation_failed: NotRequired[bool]

    # Generation
    answer: NotRequired[str]

    # Verification
    citation_validation: NotRequired[dict[str, Any]]
    semantic_verification: NotRequired[dict[str, Any]]
    verification_passed: NotRequired[bool]
    revision_count: NotRequired[int]

    # Error handling
    errors: NotRequired[list[str]]

    # Final output
    final_report: NotRequired[str]
