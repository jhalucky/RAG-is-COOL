
from src.retrieval.bm25_search import BM25Search

RRF_K = 60

def get_chunk_id(chunk: dict) -> tuple:
    "Create a stable identity for a retrieved chunk."
    return (
        chunk["document_id"],
        chunk["page_number"],
        chunk["chunk_index"]
    )



def reciprocal_rank_fusion(
        dense_results: list[dict],
        bm25_results: list[dict],
        top_k: int = 5,
        k: int = RRF_K
) -> list[dict]:

    "Combine dense and BM25 ranking using RRF."

    fused_results = {}

    retrieval_lists = [
        dense_results,
        bm25_results
    ]

    for results in retrieval_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = get_chunk_id(chunk)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = chunk.copy()
                fused_results[chunk_id]["rrf_score"] = 0.0

            fused_results[chunk_id]["rrf_score"] += (
                1.0/(k + rank)
            )

    ranked_results = sorted(
        fused_results.values(),
        key = lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:top_k]


class HybridSearch:
    def __init__(self, dense_search_fn):
        print("Initiating hybrid retrieval...")

        self.dense_search = dense_search_fn
        self.bm25_search = BM25Search()

        print("Hybrid retrieval initialized successfully.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 10
    ) -> list[dict]:
        "Retrieve and fuse dense and BM25 results."

        
        print("\nRunning dense retrieval...")
        dense_results = self.dense_search(
            query,
            top_k=retrieval_k
        )

        print("\nRunning BM25 retrieval...")
        bm25_results = self.bm25_search.search(
            query,
            top_k=retrieval_k
        )

        print("\nCombining results using RRF...")

        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            top_k=top_k
        )

        return hybrid_results


if __name__ == "__main__":
    from src.retrieval.search import search as dense_search

    hybrid_search = HybridSearch(dense_search_fn=dense_search)

    query = input("\nEnter your research question: ")

    results = hybrid_search.search(
        query,
        top_k=5,
        retrieval_k=10
    )

    # Keep your existing results-printing loop.

    print("\nHYBRID SEARCH RESULTS")
    print("="*70)

    for rank, chunk in enumerate(results, start=1):
         print(f"\nRank: {rank}")
         print(f"Document: {chunk['document_name']}")
         print(f"Page: {chunk['page_number']}")
         print(f"Chunk: {chunk['chunk_index']}")
         print(f"RRF Score: {chunk['rrf_score']:.6f}")
         print(f"Text: {chunk['text'][:300]}")
         print("-" * 70) 
        
