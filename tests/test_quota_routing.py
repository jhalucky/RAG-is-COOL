
import importlib

graph_module = importlib.import_module("src.agents.graph")


def test_quota_failure_skips_generation_and_verification(monkeypatch):
    calls = []

    def fake_planner(state):
        calls.append("planner")
        return {"research_plan": ["Research subquestion"]}

    def fake_retrieval(state):
        calls.append("retrieval")
        return {
            "retrieved_chunks": [
                {
                    "document_name": "test.pdf",
                    "page_number": 1,
                    "text": "Sample retrieved evidence.",
                }
            ],
            "retrieval_attempts": 1,
        }

    def fake_evaluation(state):
        calls.append("evaluation")
        return {
            "evidence_evaluation_failed": True,
            "evidence_sufficient": False,
            "llm_unavailable": True,
            "llm_error": "429 RESOURCE_EXHAUSTED",
            "errors": ["Gemini quota exhausted"],
        }

    def fake_generation(state):
        calls.append("generation")
        raise AssertionError("Generation must not execute.")

    def fake_verification(state):
        calls.append("verification")
        raise AssertionError("Verification must not execute.")

    def fake_final_report(state):
        calls.append("final_report")
        return {
            "final_report": "LLM quota exhausted — partial research preserved."
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "retrieval_node", fake_retrieval)
    monkeypatch.setattr(
        graph_module,
        "evidence_evaluation_node",
        fake_evaluation,
    )
    monkeypatch.setattr(
        graph_module,
        "generation_node",
        fake_generation,
    )
    monkeypatch.setattr(
        graph_module,
        "verification_node",
        fake_verification,
    )
    monkeypatch.setattr(
        graph_module,
        "final_report_node",
        fake_final_report,
    )

    workflow = graph_module.build_research_graph()

    result = workflow.invoke({
        "question": "Test research question",
        "retrieval_attempts": 0,
        "max_retrieval_attempts": 2,
        "revision_count": 0,
        "retrieved_chunks": [],
        "errors": [],
    })

    assert "generation" not in calls
    assert "verification" not in calls
    assert calls[-1] == "final_report"

    assert result["final_report"] == (
        "LLM quota exhausted — partial research preserved."
    )
