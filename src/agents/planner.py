import json


def parse_planner_response(raw_response: str) -> list[str]:
    """Parse and validate a Gemini research plan."""
    if not raw_response or not raw_response.strip():
        raise ValueError("Gemini returned an empty response.")

    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) == 2 else parts[0][3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError("Planner returned invalid JSON.") from error

    if not isinstance(result, dict):
        raise ValueError("Planner response must be a JSON object.")

    research_plan = result.get("subquestions")

    if (
        not isinstance(research_plan, list)
        or not 3 <= len(research_plan) <= 5
        or not all(isinstance(item, str) and item.strip() for item in research_plan)
    ):
        raise ValueError("Invalid research subquestions.")

    return [item.strip() for item in research_plan]
