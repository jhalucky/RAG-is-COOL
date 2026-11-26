from langgraph.graph import StateGraph, START, END

from src.agents.state import ResearchState
from src.agents.nodes import (
    planner_node,
    retrieval_node,
    generation_node,
    verification_node,
    final_report_node
)

MAX_REVISIONS = 1

def should_revise(state: ResearchState) -> str:
    if state.get("verification_passed", False):
        return "Finish"

    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "revise"

    return "finish"



def revision_node(state: ResearchState) -> dict:
    revision_count = state.get("revision_count", 0) + 1

    print(
        f"\n[Revision] Preparing revision"
        f"{revision_count}/{MAX_REVISIONS}"
    )

    return {
        "revision_count": revision_count
    }

def build_research_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("final_report", final_report_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retrieval")
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", "verification")

     # Conditional verification
    workflow.add_conditional_edges(
        "verification",
        should_revise,
        {
            "revise": "revision",
            "finish": "final_report",
        },
    )

    # Revision returns to generation
    workflow.add_edge("revision", "generation")

    workflow.add_edge("final_report", END)

    return workflow.compile()


research_graph = build_research_graph()


if __name__ == "__main__":
    print("LangGraph research workflow compiled successfully!")
    print(research_graph.get_graph().draw_ascii())
