
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
    evidence_selection_stats: NotRequired[dict[str, Any]]


    # Generation
    answer: NotRequired[str]


    #compression
    compressed_chunks: NotRequired[list[dict[str, Any]]]
    compression_stats: NotRequired[dict[str, Any]]

    #revision
    revision_feedback: NotRequired[dict[str, Any]]


    # Verification
    citation_validation: NotRequired[dict[str, Any]]
    semantic_verification: NotRequired[dict[str, Any]]
    verification_passed: NotRequired[bool]
    revision_count: NotRequired[int]

    # Error handling
    errors: NotRequired[list[str]]
    llm_unavailable: NotRequired[bool]
    llm_error: NotRequired[str]

    # Final output
    final_report: NotRequired[str]
