
import os
import time

from google.genai import types
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

client = genai.Client(api_key=API_KEY)



def generate_answer(question: str, chunks: list[dict]) -> str:
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

    start_time = time.perf_counter()

    print("\nGenerating research answer...\n")

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

    total_time = time.perf_counter() - start_time
    answer = response.text or ""

    candidate = (
        response.candidates[0]
        if response.candidates else None
    )

    print("\nGeneration metrics")
    print("------------------")
    print(f"Model: {MODEL}")
    print(f"Finish reason: {candidate.finish_reason if candidate else 'None'}")
    print(f"Generation time: {total_time:.2f}s")
    print(f"Answer length: {len(answer)} characters")

    if not answer:
        print("WARNING: Gemini returned no text.")

    return answer
