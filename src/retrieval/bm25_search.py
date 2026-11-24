
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from src.indexing.chunker import chunk_text


DATA_DIR = Path("data/processed")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks() -> list[dict]:
    chunks = []

    for file_path in DATA_DIR.glob("*.jsonl"):
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                document = json.loads(line)

                for index, chunk in enumerate(
                    chunk_text(document["text"])
                ):
                    chunks.append({
                        "text": chunk,
                        "document_id": document["document_id"],
                        "document_name": document["document_name"],
                        "page_number": document["page_number"],
                        "chunk_index": index,
                    })

    return chunks


class BM25Search:
    def __init__(self):
        print("Loading BM25 corpus...")

        self.chunks = load_chunks()

        if not self.chunks:
            raise ValueError("No chunks found for BM25 indexing.")

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"BM25 indexed chunks: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:
            if scores[index] <= 0:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(scores[index])

            results.append(result)

        return results
