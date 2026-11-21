import json
import uuid
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from src.indexing.chunker import chunk_text

DATA_DIR = Path("data/processed")
COLLECTION = "research_papers"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH_SIZE = 32

def load_documents():
    records = []

    for file_path in DATA_DIR.glob("*.jsonl"):
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
    
    return records


def build_index():
    print("Loading documents...")

    documents = load_documents()

    if not documents:
        raise ValueError("No documents found in the data/processed directory.")

    chunks = []

    for document in documents:
        for index, chunk in enumerate(
            chunk_text(document["text"])
        ):
            chunks.append({
                "text": chunk,
                "document_id": document["document_id"],
                "document_name": document["document_name"],
                "page_number": document["page_number"],
                "chunk_index": index
            })


    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings)

    client = QdrantClient(
        url="http://localhost:6333",
        prefer_grpc=True
    )

    if not client.collection_exists(COLLECTION):
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

    points = []

    for item, embedding in zip(chunks, embeddings):
        unique_key = (
            f'{item["document_id"]}:'
            f'{item["page_number"]}:'
            f'{item["chunk_index"]}:'
        )

        points_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, unique_key)
        )

        points.append(
            models.PointStruct(
                id=points_id,
                vector=embedding.tolist(),
                payload=item
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True
    )


    print("indexing completed successfully.")
    print("Collection:", COLLECTION)
    print("Indexed chunks:", len(points))

    print(
        "Qdrant Client: ",
        client.count(
            collection_name=COLLECTION,
            exact=True
        ).count
    )


if __name__ == "__main__":
    build_index()