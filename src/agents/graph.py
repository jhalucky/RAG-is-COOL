
from langgraph.graph import StateGraph, START, END

from src.agents.state import ResearchState

from src.agents.nodes import (
    planner_node,
    retrieval_node,
    evidence_evaluation_node,
    query_refinement_node,
    generation_node,
    verification_node,
    final_report_node,
)

MAX_REVISIONS = 1
MAX_RETRIEVAL_ATTEMPTS = 2


def should_continue_retrieval(state: ResearchState) -> str:
    """Decide whether to refine retrieval or stop."""

    if state.get("llm_unavailable", False):
        return "finish"

    if state.get("evidence_evaluation_failed", False):
        return "generate"

    if state.get("evidence_sufficient", False):
        return "generate"

    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get(
        "max_retrieval_attempts",
        MAX_RETRIEVAL_ATTEMPTS,
    )

    if attempts < max_attempts:
        return "refine"

    print("\n[Adaptive Retrieval] Maximum attempts reached.")

    return "generate"


def after_retrieval(state: ResearchState) -> str:
    """Skip Gemini evaluation if the quota is exhausted."""

    if state.get("llm_unavailable", False):
        return "finish"

    return "evaluate"


def should_revise(state: ResearchState) -> str:

    if state.get("llm_unavailable", False):
        return "finish"

    errors = state.get("errors", [])

    if any(
        "Semantic verification API error" in error
        for error in errors
    ):
        return "finish"

    if state.get("verification_passed", False):
        return "finish"

    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "revise"

    return "finish"


def revision_node(state: ResearchState) -> dict:

    revision_count = state.get("revision_count", 0) + 1

    print(
        f"\n[Revision] Preparing revision "
        f"{revision_count}/{MAX_REVISIONS}"
    )

    return {
        "revision_count": revision_count
    }


def after_generation(state: ResearchState) -> str:

    if state.get("answer"):
        return "verify"

    return "finish"


def build_research_graph():

    workflow = StateGraph(ResearchState)

    # Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)

    workflow.add_node(
        "evidence_evaluation",
        evidence_evaluation_node,
    )

    workflow.add_node(
        "query_refinement",
        query_refinement_node,
    )

    workflow.add_node("generation", generation_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("final_report", final_report_node)

    # Initial workflow
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retrieval")

    # Adaptive retrieval
    workflow.add_conditional_edges(
    "retrieval",
    after_retrieval,
    {
        "evaluate": "evidence_evaluation",
        "finish": "final_report",
    },
)
    workflow.add_edge("query_refinement", "retrieval")

    # Answer generation
    workflow.add_conditional_edges(
    "evidence_evaluation",
    should_continue_retrieval,
    {
        "refine": "query_refinement",
        "generate": "generation",
        "finish": "final_report",
    },
    )

    # Answer generation
    workflow.add_conditional_edges(
        "generation",
        after_generation,
        {
            "verify": "verification",
            "finish": "final_report",
        },
    )

    # Answer verification and revision
    workflow.add_conditional_edges(
        "verification",
        should_revise,
        {
            "revise": "revision",
            "finish": "final_report",
        },
    )

    workflow.add_edge("revision", "generation")

    workflow.add_edge("final_report", END)

    return workflow.compile()


research_graph = build_research_graph()


if __name__ == "__main__":
    print("LangGraph research workflow compiled successfully!")
    print(research_graph.get_graph().draw_ascii())