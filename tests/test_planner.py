import pytest

from src.agents.nodes import parse_planner_response

def test_parse_valid_plan():
    response = """
    {
        "subquestions": [
            "What is RAG?",
            "How does retrieval work?",
            "How is retrieved context used by LLMs?"
        ]
    }
    """

    result = parse_planner_response(response)

    assert len(result) == 3
    assert result[0] == "What is RAG?"

def test_parse_markdown_json():
    response = '''```json
    {
        "subquestions": [
            "Question one",
            "Question two",
            "Question three"
        ]
    }
    ```'''

    result = parse_planner_response(response)

    assert len(result) == 3


def test_reject_empty_response():
    with pytest.raises(ValueError):
        parse_planner_response("")


def test_reject_invalid_json():
    with pytest.raises(ValueError):
        parse_planner_response("This is not JSON")


def test_reject_insufficient_questions():
    response = '{"subquestions": ["Only one"]}'

    with pytest.raises(ValueError):
        parse_planner_response(response)
