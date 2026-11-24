
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from src.generation.answer_generator import generate_answer
from src.verification.citation_validator import validate_citations
from src.verification.semantic_verifier import verify_claims

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

        print(f"Passage: {chunk['text'][:500]}")
        print("-" * 65)

    return chunks



if __name__ == "__main__":
    question = input("Enter your research question: ")

    # Measure retrieval time
    start = time.perf_counter()

    print("\nSearching research papers...\n")
    retrieved_chunks = search(question, top_k=5)

    retrieval_time = time.perf_counter() - start
    print(f"\nRetrieval time: {retrieval_time:.2f} seconds")

    # Measure generation time
    start = time.perf_counter()

    print("\nGenerating answer with Gemini...\n")
    answer = generate_answer(question, retrieved_chunks)
    

    generation_time = time.perf_counter() - start

    verification = validate_citations(answer, retrieved_chunks)

    print("\nVerifying claims against research evidence...\n")

    semantic_start = time.perf_counter()
    semantic_report = verify_claims(answer, retrieved_chunks)
    semantic_time = time.perf_counter() - semantic_start

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

    print("\n" + "=" * 65)
    print("CITATION VERIFICATION REPORT")
    print("=" * 65)

    print(f"Total citations: {verification['total_citations']}")
    print(f"Valid citations: {len(verification['valid_citations'])}")
    print(f"Invalid citations: {len(verification['invalid_citations'])}")

    if verification["total_citations"] == 0:
        print("\nCitation status: NO CITATIONS FOUND")
    elif verification["all_valid"]:
        print("\nCitation status: ALL CITATIONS VALID")
    else:
        print("\nCitation status: INVALID CITATIONS FOUND")

    if verification["invalid_citations"]:
        print("\nInvalid citations:")

        for document, page in verification["invalid_citations"]:
            print(f"- {document}, page {page}")

    print("=" * 65)

    print("\n" + "=" * 65)
    print("SEMANTIC VERIFICATION REPORT")
    print("=" * 65)

    results = semantic_report.get("results", [])

    print(f"Total claims: {semantic_report.get('total_claims', 0)}")

    statuses = [
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT",
        "UNVERIFIED",
        "SOURCE_NOT_RETRIEVED",
    ]

    for status in statuses:
        count = sum(
            1 for result in results
            if result.get("status", "").upper() == status
        )
        print(f"{status.replace('_', ' ').title()}: {count}")

    print("\nIndividual claim results:")

    for index, result in enumerate(results, start=1):
        print(f"\nClaim {index}: {result.get('claim', 'Unknown')}")
        print(
            f"Source: {result.get('document', 'Unknown')}, "
            f"Page {result.get('page', 'Unknown')}"
        )
        print(f"Status: {result.get('status', 'Unknown')}")
        print(
            f"Explanation: "
            f"{result.get('explanation', 'No explanation provided.')}"
        )

    print(f"\nSemantic verification time: {semantic_time:.2f} seconds")
    print("=" * 65)
