
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

print("Testing Gemini API...")
print("Model:", model)

start = time.perf_counter()

try:
    response = client.models.generate_content(
        model=model,
        contents="Explain RAG in exactly two sentences."
    )

    print("\nResponse:")
    print(response.text)

    print(
        f"\nTotal time: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

except Exception as e:
    print("\nGemini API error:")
    print(type(e).__name__, str(e))

    print(
        f"Time before error: "
        f"{time.perf_counter() - start:.2f} seconds"
    )
