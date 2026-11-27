
import pytest

from src.verification.evidence_sufficiency import (
    parse_sufficiency_response,
    evaluate_evidence,
)


def valid_response():
    return """
    {
        "sufficient": true,
        "coverage_score": 0.9,
        "covered_subquestions": [
            "What is RAG?"
        ],
        "missing_aspects": [],
        "suggested_queries": [],
        "reasoning": "Evidence covers the research question."
    }
    """


def test_parse_valid_response():
    result = parse_sufficiency_response(valid_response())

    assert result["sufficient"] is True
    assert result["coverage_score"] == 0.9


def test_parse_markdown_json():
    response = "```json\n" + valid_response() + "\n```"

    result = parse_sufficiency_response(response)

    assert result["sufficient"] is True


def test_reject_empty_response():
    with pytest.raises(ValueError):
        parse_sufficiency_response("")


def test_reject_invalid_json():
    with pytest.raises(ValueError):
        parse_sufficiency_response("invalid json")


def test_reject_invalid_score():
    response = valid_response().replace("0.9", "1.5")

    with pytest.raises(ValueError):
        parse_sufficiency_response(response)


def test_reject_missing_fields():
    with pytest.raises(ValueError):
        parse_sufficiency_response(
            '{"sufficient": true}'
        )


def test_empty_evidence():
    result = evaluate_evidence(
        question="What is RAG?",
        research_plan=["What is RAG?"],
        retrieved_chunks=[],
    )

    assert result["sufficient"] is False
    assert result["coverage_score"] == 0.0
    assert len(result["missing_aspects"]) == 1
