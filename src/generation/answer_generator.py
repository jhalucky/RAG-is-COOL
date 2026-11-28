
import os
import time

import json

from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

client = genai.Client(api_key=API_KEY)


def generate_answer(question: str, chunks: list[dict], revision_feedback: dict | None = None) -> str:
    if not chunks:
        return "No relevant passages found."

    context = []

    for i, chunk in enumerate(chunks[:5], start=1):
        document = chunk.get("document_name") or chunk.get(
            "document_id", "Unknown"
        )
        page = chunk.get("page_number", "Unknown")

        context.append(
            f"[Source {i} | Document: {document} | Page: {page}]\n"
            f"{chunk.get('text', '')[:1200]}"
        )

    prompt = f"""
You are a research assistant.

Answer the user's question using ONLY the research passages.

Requirements:
- Answer the exact question asked.
- Prioritize information directly relevant to the question.
- If asked about limitations, weaknesses, or failure cases,
  focus specifically on those aspects.
- Do not introduce unrelated background information.
- If the provided evidence does not discuss an aspect,
  explicitly state that evidence is insufficient.
- Do not invent limitations or unsupported conclusions.
- Cite factual claims using [document_name, page number].
- Use the exact document filename provided in the passages.
- Never cite using Source numbers.
- Never invent facts or citations.
- Mention when evidence is insufficient.

RESEARCH PASSAGES:
{"".join(chr(10) + p for p in context)}

QUESTION:
{question}
"""

    if revision_feedback:
        failed_claims = revision_feedback.get("failed_claims", [])

        failed_claim_text = "\n".join(
            (
                f"- Claim: {item.get('claim', '')}\n"
                f"  Verdict: {item.get('verdict', 'UNKNOWN')}\n"
                f"  Reason: {item.get('reason', '')}"
            )
            for item in failed_claims
        )

        citation_results = json.dumps(
            revision_feedback.get("citation_validation", {}),
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        revision_section = f"""

REVISION MODE

Previous answer:
{revision_feedback.get('previous_answer', '')}

Verification feedback:
{failed_claim_text or "No individually unsupported claims were identified."}

Citation validation results:
{citation_results}

Revision instructions:
{revision_feedback.get('instruction', '')}

Additional requirements:
- Re-examine the original research passages.
- Correct or remove claims that lack sufficient support.
- Correct invalid citations using only the supplied passages.
- Do not introduce new facts or sources.
- Retain accurate and well-supported information.
- Return the complete corrected answer, not just the changes.
"""

        prompt += revision_section

    start_time = time.perf_counter()

    print("\nGenerating research answer...\n")

    response = None
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="LOW"
                    ),
                    max_output_tokens=1024,
                ),
            )
            break

        except APIError as error:
            # Retry only temporary service-unavailable errors.
            if getattr(error, "code", None) != 503:
                raise

            if attempt == max_attempts - 1:
                print("Gemini remained unavailable after all retries.")
                raise

            wait_time = 2 ** (attempt + 1)

            print(
                f"Gemini returned HTTP 503. "
                f"Retrying in {wait_time} seconds "
                f"(attempt {attempt + 2}/{max_attempts})..."
            )

            time.sleep(wait_time)

    total_time = time.perf_counter() - start_time
    answer = response.text or ""

    candidate = (
        response.candidates[0]
        if response.candidates else None
    )

    print("\nGeneration metrics")
    print("------------------")
    print(f"Model: {MODEL}")
    print(
        f"Finish reason: "
        f"{candidate.finish_reason if candidate else 'None'}"
    )
    print(f"Generation time: {total_time:.2f}s")
    print(f"Answer length: {len(answer)} characters")

    if not answer:
        print("WARNING: Gemini returned no text.")

    return answer
