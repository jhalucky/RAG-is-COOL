from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from google.genai.errors import APIError

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

    # Import locally to avoid circular imports
    
    from src.retrieval.hybrid_search import HybridSearch
    from src.retrieval.reranker import NeuralReranker

    # Initialize hybrid retrieval
    hybrid_search = HybridSearch(dense_search_fn=search)

    # Step 1: Retrieve 10 candidates using Dense + BM25 + RRF
    retrieval_start = time.perf_counter()

    candidate_chunks = hybrid_search.search(
        question,
        top_k=10,
        retrieval_k=10
    )

    retrieval_time = time.perf_counter() - retrieval_start

    # Step 2: Rerank candidates using BGE Cross-Encoder
    reranker = NeuralReranker()

    rerank_start = time.perf_counter()

    retrieved_chunks = reranker.rerank(
        query=question,
        candidates=candidate_chunks,
        top_k=5
    )

    print("\nRERANKED RESULTS")
    print("=" * 70)

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nRank: {rank}")
        print(f"Document: {chunk['document_name']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Rerank Score: {chunk['rerank_score']:.4f}")
        print(f"Text: {chunk['text'][:300]}")
        print("-" * 70)

    rerank_time = time.perf_counter() - rerank_start

    print(f"\nHybrid retrieval time: {retrieval_time:.2f} seconds")
    print(f"Reranking time: {rerank_time:.2f} seconds")


    
    # Measure generation time
    generation_start = time.perf_counter()

    print("\nGenerating answer with Gemini...\n")

    answer = None
    generation_error = None

    try:
        answer = generate_answer(question, retrieved_chunks)

    except APIError as error:
        generation_error = str(error)

        print("\nAnswer generation unavailable.")
        print(f"Reason: {generation_error}")

    generation_time = time.perf_counter() - generation_start

    # Initialize verification reports
    verification = None
    semantic_report = None
    semantic_time = 0.0

    # Run verification only if generation succeeds
    if answer is not None:

        verification = validate_citations(
            answer,
            retrieved_chunks
        )

        print("\nVerifying claims against research evidence...\n")

        semantic_start = time.perf_counter()

        try:
            semantic_report = verify_claims(
                answer,
                retrieved_chunks
            )

        except APIError as error:
            print("\nSemantic verification unavailable.")
            print(f"Reason: {error}")

        semantic_time = time.perf_counter() - semantic_start

    # Final research answer
    print("\n" + "=" * 65)
    print("FINAL RESEARCH ANSWER")
    print("=" * 65)

    if answer is not None:
        print(answer)
    else:
        print("Answer generation failed. Retrieved passages are available above.")

    print("=" * 65)

    # Performance metrics
    print(f"\nHybrid retrieval time: {retrieval_time:.2f} seconds")
    print(f"Reranking time: {rerank_time:.2f} seconds")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Semantic verification time: {semantic_time:.2f} seconds")

    total_time = (
        retrieval_time
        + rerank_time
        + generation_time
        + semantic_time
    )

    print(f"Total pipeline time: {total_time:.2f} seconds")

    # Citation verification report
    print("\n" + "=" * 65)
    print("CITATION VERIFICATION REPORT")
    print("=" * 65)

    if verification is not None:

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

    else:
        print("Citation verification skipped: no generated answer.")

    print("=" * 65)

    # Semantic verification report
    print("\n" + "=" * 65)
    print("SEMANTIC VERIFICATION REPORT")
    print("=" * 65)

    results = (
        semantic_report.get("results", [])
        if semantic_report is not None
        else []
    )

    if semantic_report is not None:

        print(f"Total claims: {semantic_report.get('total_claims', 0)}")

    elif answer is None:
        print("Semantic verification skipped: no generated answer.")

    else:
        print("Semantic verification unavailable.")

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

    if semantic_report is not None:

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

    print("=" * 65)
