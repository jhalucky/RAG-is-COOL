
from src.agents.graph import should_continue_retrieval


def test_sufficient_evidence_proceeds_to_generation():
    state = {
        "evidence_sufficient": True,
        "retrieval_attempts": 1,
        "max_retrieval_attempts": 2,
    }

    assert should_continue_retrieval(state) == "generate"


def test_insufficient_evidence_triggers_refinement():
    state = {
        "evidence_sufficient": False,
        "retrieval_attempts": 1,
        "max_retrieval_attempts": 2,
    }

    assert should_continue_retrieval(state) == "refine"


def test_max_attempts_proceeds_to_generation():
    state = {
        "evidence_sufficient": False,
        "retrieval_attempts": 2,
        "max_retrieval_attempts": 2,
    }

    assert should_continue_retrieval(state) == "generate"


def test_evaluator_failure_does_not_trigger_retry():
    state = {
        "evidence_sufficient": False,
        "evidence_evaluation_failed": True,
        "retrieval_attempts": 1,
        "max_retrieval_attempts": 2,
    }

    assert should_continue_retrieval(state) == "generate"

from src.agents.graph import after_retrieval


def test_retrieval_skips_evaluation_when_quota_exhausted():
    state = {"llm_unavailable": True}

    assert after_retrieval(state) == "finish"


def test_retrieval_evaluates_when_llm_available():
    state = {"llm_unavailable": False}

    assert after_retrieval(state) == "evaluate"
