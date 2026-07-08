import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")
COLLECTION_NAME = "hoi4_wiki"


def load_chunks() -> list[dict]:
    records = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main() -> None:
    model = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(url="http://localhost:6333")

    chunks = load_chunks()

    vector_size = model.get_sentence_embedding_dimension()

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    batch_size = 32
    point_id = 0

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        texts = [item["text"] for item in batch]

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = []
        for item, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "title": item["title"],
                        "url": item["url"],
                        "text": item["text"],
                        "page_id": item.get("page_id"),
                        "revision_id": item.get("revision_id"),
                    },
                )
            )
            point_id += 1

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Indexados {point_id} chunks.")


if __name__ == "__main__":
    main()