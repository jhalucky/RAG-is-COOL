from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"

class NeuralReranker:
    def __init__(self):
        print("Loading BGE reranker model...")

        self.model = CrossEncoder(
            MODEL_NAME,
            device="cpu",
            max_length=512
        )

        print("BGE reranker loaded successfully.")


    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5
    ) -> list[dict]:
        "Rerank retrieved messages using cross-encoder scores."

        if not candidates:
            return []

        query_passage_pairs = [
            (query, chunk["text"])
            for chunk in candidates
        ]

        scores = self.model.predict(
            query_passage_pairs,
            batch_size=8,
            show_progress_bar=False
        )

        reranked_results = []

        for chunk, score in zip(candidates, scores):
            result = chunk.copy()
            result["rerank_score"] = float(score)
            reranked_results.append(result)


        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]
