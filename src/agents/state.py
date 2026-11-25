from typing import TypedDict, NotRequired

class ResearchState(TypedDict):
    """
    Shared state passed between nodes in the LangGraph research workflow.
    """

    question: str

    research_plan: NotRequired[list[str]]
    retrieved_chunks: NotRequired[list[dict]]
    answer: NotRequired[str]
    citation_validation: NotRequired[str]
    semantic_verification: NotRequired[dict]

    verification_passed: NotRequired[bool]
    revision_count: NotRequired[int]

    errors: NotRequired[list[str]]
    final_agent: NotRequired[str]


