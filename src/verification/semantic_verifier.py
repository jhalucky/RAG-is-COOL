
import json
import re

from google.genai import types

from src.generation.answer_generator import client, MODEL


CITATION_PATTERN = re.compile(
    r"\[([^,\]]+),\s*page\s+(\d+)\]",
    flags=re.IGNORECASE,
)


def extract_claims(answer: str) -> list[dict]:
    """
    Extract sentences containing citations.
    Each citation becomes a separate claim-evidence pair.
    """

    claims = []

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    for sentence in sentences:
        matches = list(CITATION_PATTERN.finditer(sentence))

        if not matches:
            continue

        claim_text = CITATION_PATTERN.sub("", sentence).strip()

        if not claim_text:
            continue

        for match in matches:
            claims.append({
                "claim": claim_text,
                "document": match.group(1).strip(),
                "page": match.group(2).strip(),
            })

    return claims


def verify_claims(answer: str, chunks: list[dict]) -> dict:
    """
    Verify whether cited research passages support
    the claims made in the generated answer.
    """

    claims = extract_claims(answer)

    if not claims:
        return {
            "total_claims": 0,
            "results": [],
            "message": "No cited claims found."
        }

    verification_inputs = []

    for claim in claims:

        matching_chunks = [
            chunk for chunk in chunks
            if (
                str(chunk.get("document_name", "")).strip().lower()
                == claim["document"].lower()
                and
                str(chunk.get("page_number", "")).strip()
                == claim["page"]
            )
        ]

        if not matching_chunks:
            verification_inputs.append({
                **claim,
                "evidence": None,
                "status": "SOURCE_NOT_RETRIEVED",
                "explanation": (
                    "The cited source was not found in retrieved chunks."
                )
            })
            continue

        evidence = "\n".join(
            chunk.get("text", "")
            for chunk in matching_chunks
        )

        verification_inputs.append({
            **claim,
            "evidence": evidence[:3000],
            "status": "PENDING"
        })

    pending = [
        item for item in verification_inputs
        if item["status"] == "PENDING"
    ]

    if not pending:
        return {
            "total_claims": len(claims),
            "results": verification_inputs
        }

    prompt = f"""
You are a strict research evidence evaluator.

Evaluate whether each research claim is supported by
its associated evidence.

Use ONLY the supplied evidence.

Classification rules:

SUPPORTED:
The evidence directly supports the claim.

CONTRADICTED:
The evidence directly conflicts with the claim.

INSUFFICIENT:
The evidence does not provide enough information
to support or contradict the claim.

Important:
- Do not use outside knowledge.
- Do not assume facts that are not stated.
- Do not treat missing information as contradiction.
- Be conservative when evaluating evidence.
- Keep explanations short.

Return ONLY valid JSON in this exact structure:

{{
  "results": [
    {{
      "claim_index": 0,
      "status": "SUPPORTED",
      "explanation": "Short explanation based on the evidence."
    }}
  ]
}}

CLAIMS AND EVIDENCE:

{json.dumps(pending, ensure_ascii=False, indent=2)}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="LOW"
            ),
            max_output_tokens=2048,
            response_mime_type="application/json"
        )
    )

    try:
        model_result = json.loads(response.text or "{}")
        evaluations = model_result.get("results", [])

    except (json.JSONDecodeError, TypeError):
        return {
            "total_claims": len(claims),
            "results": verification_inputs,
            "error": (
                "Gemini returned invalid JSON. "
                "Verification could not be completed."
            )
        }

    # Attach Gemini's evaluations to the original claims.
    for evaluation in evaluations:

        index = evaluation.get("claim_index")

        if not isinstance(index, int) or not 0 <= index < len(pending):
            continue

        status = evaluation.get("status", "").upper()

        if status not in {
            "SUPPORTED",
            "CONTRADICTED",
            "INSUFFICIENT"
        }:
            status = "INSUFFICIENT"

        pending[index]["status"] = status
        pending[index]["explanation"] = evaluation.get(
            "explanation",
            "No explanation provided."
        )

    # Mark claims Gemini did not evaluate.
    for item in pending:
        if item["status"] == "PENDING":
            item["status"] = "UNVERIFIED"
            item["explanation"] = (
                "Gemini did not return an evaluation."
            )

    return {
        "total_claims": len(claims),
        "results": verification_inputs
    }
