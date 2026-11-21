
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from src.generation.answer_generator import generate_answer

import time

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "research_papers"

model = SentenceTransformer(MODEL_NAME, device="cpu")
client = QdrantClient(url="http://localhost:6333")


def search(query: str, top_k: int = 3):
    query_embedding = model.encode(
        "Represent this sentence for searching relevant passages: "
        + query,
        normalize_embeddings=True
    )

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )

    chunks = []

    for rank, point in enumerate(results.points, start=1):
        payload = point.payload or {}

        chunk = {
            "document_id": payload.get("document_id", "Unknown"),
            "document_name": payload.get("document_name", "Unknown"),
            "page_number": payload.get("page_number", "Unknown"),
            "chunk_index": payload.get("chunk_index", -1),
            "text": payload.get("text", ""),
            "score": point.score,
        }

        chunks.append(chunk)

        print(
            f"Retrieved {rank}: "
            f"{chunk['document_name']}, "
            f"Page {chunk['page_number']}, "
            f"Score: {point.score:.4f}"
        )

    return chunks



if __name__ == "__main__":
    question = input("Enter your research question: ")

    # Measure retrieval time
    start = time.perf_counter()

    print("\nSearching research papers...\n")
    retrieved_chunks = search(question, top_k=3)

    retrieval_time = time.perf_counter() - start
    print(f"\nRetrieval time: {retrieval_time:.2f} seconds")

    # Measure generation time
    start = time.perf_counter()

    print("\nGenerating answer with Gemini...\n")
    answer = generate_answer(question, retrieved_chunks)

    generation_time = time.perf_counter() - start

    print("\n" + "=" * 65)
    print("FINAL RESEARCH ANSWER")
    print("=" * 65)
    print(answer)
    print("=" * 65)

    print(f"\nRetrieval time: {retrieval_time:.2f} seconds")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(
        f"Total time: "
        f"{retrieval_time + generation_time:.2f} seconds"
    )
