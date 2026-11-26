
from src.agents.graph import research_graph


def run_research_agent(question: str) -> dict:
    """
    Execute the LangGraph research workflow.
    """
    initial_state = {
        "question": question,
        "revision_count": 0,
        "errors": [],
    }

    print("\n" + "=" * 65)
    print("DEEP RESEARCH AI")
    print("=" * 65)
    print(f"\nResearch question: {question}")

    result = research_graph.invoke(initial_state)

    print("\n" + "=" * 65)
    print("FINAL RESEARCH REPORT")
    print("=" * 65)

    print(result.get("final_report", "No report generated."))

    return result


if __name__ == "__main__":
    question = input("\nEnter your research question: ").strip()

    if not question:
        print("Please enter a valid research question.")
    else:
        run_research_agent(question)
