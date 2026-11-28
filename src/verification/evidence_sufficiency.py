import json
from typing import Any

from google.genai import types

from src.generation.answer_generator import client, MODEL




def parse_sufficiency_response(raw_response: str) -> dict:
    """Extract, parse, and validate Gemini's evidence evaluation JSON."""

    if not raw_response or not raw_response.strip():
        raise ValueError("Empty evidence evaluation response.")

    cleaned = raw_response.strip()

    # Remove Markdown code fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    # Find the beginning of the JSON object, even if Gemini
    # includes introductory text before it.
    json_start = cleaned.find("{")

    if json_start == -1:
        raise ValueError(
            f"No JSON object found. Response: {cleaned[:300]!r}"
        )

    try:
        result, _ = json.JSONDecoder().raw_decode(cleaned[json_start:])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON response: {error}. "
            f"Response: {cleaned[:300]!r}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    required = {
        "sufficient",
        "coverage_score",
        "covered_subquestions",
        "missing_aspects",
        "suggested_queries",
        "reasoning",
    }

    if not required.issubset(result):
        missing = required - result.keys()
        raise ValueError(f"Missing required fields: {missing}")

    if not isinstance(result["sufficient"], bool):
        raise ValueError("sufficient must be a boolean.")

    score = result["coverage_score"]

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0 <= score <= 1
    ):
        raise ValueError("coverage_score must be between 0 and 1.")

    for field in (
        "covered_subquestions",
        "missing_aspects",
        "suggested_queries",
    ):
        if (
            not isinstance(result[field], list)
            or not all(
                isinstance(item, str) and item.strip()
                for item in result[field]
            )
        ):
            raise ValueError(f"Invalid field: {field}")

    if not isinstance(result["reasoning"], str):
        raise ValueError("reasoning must be a string.")

    return result




def evaluate_evidence(
    question: str,
    research_plan: list[str],
    retrieved_chunks: list[dict[str, Any]],
) -> dict:
    """
    Evaluate whether retrieved evidence sufficiently covers
    the research question and its subquestions.
    """

    if not retrieved_chunks:
        return {
            "sufficient": False,
            "coverage_score": 0.0,
            "covered_subquestions": [],
            "missing_aspects": research_plan,
            "suggested_queries": research_plan,
            "reasoning": "No evidence passages were retrieved.",
        }

    evidence = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        evidence.append({
            "evidence_id": index,
            "document": chunk.get("document_name", "Unknown"),
            "page": chunk.get("page_number", "Unknown"),
            "text": chunk.get("text", "")[:1500],
        })

    prompt = f"""
You are an academic research evidence evaluator.

Research question:
{question}

Research subquestions:
{json.dumps(research_plan, indent=2)}

Retrieved evidence:
{json.dumps(evidence, indent=2)}

Evaluate:
1. Whether each subquestion is covered.
2. Whether evidence is relevant and substantive.
3. What important information is missing.
4. What additional queries could retrieve that information.

Return ONLY valid JSON:

{{
    "sufficient": true,
    "coverage_score": 0.85,
    "covered_subquestions": [],
    "missing_aspects": [],
    "suggested_queries": [],
    "reasoning": "Explanation."
}}
"""

    
    for attempt in range(2):
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )

        raw_response = response.text or ""

        print(
            f"\n[Evidence Evaluation Debug] "
            f"Attempt: {attempt + 1}"
        )

        print(
            f"[Evidence Evaluation Debug] "
            f"Raw response: {raw_response!r}"
        )

        print(
            f"[Evidence Evaluation Debug] "
            f"Candidates: {response.candidates!r}"
        )

        print(
            f"[Evidence Evaluation Debug] "
            f"Prompt feedback: "
            f"{getattr(response, 'prompt_feedback', None)!r}"
        )

        try:
            return parse_sufficiency_response(raw_response)

        except ValueError as error:
            print(
                f"[Evidence Evaluation] Invalid response "
                f"(attempt {attempt + 1}/2): {error}"
            )

            if attempt == 0:
                prompt += """

    IMPORTANT: Your previous response could not be parsed.
    Return a single valid JSON object only.
    Do not include Markdown fences, introductory text,
    or explanations outside the JSON object.
    Include every required field with the correct data type.
    """

    raise ValueError(
        "Evidence evaluation failed after two invalid responses."
    )

