import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
import json
from pathlib import Path
import threading

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")
DEFAULT_COLLECTION_NAME = "hoi4_wiki"
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"

DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_UPLOAD_WORKERS = 2
DEFAULT_MAX_PENDING_UPLOADS = 4

DEFAULT_QDRANT_URL = "http://localhost:6333"

_thread_local = threading.local()


def get_qdrant_client(qdrant_url: str) -> QdrantClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = QdrantClient(
            url=qdrant_url,
            prefer_grpc=True,
        )

    return _thread_local.client


def iter_chunks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def batched(iterable, batch_size: int):
    iterator = iter(iterable)

    while batch := list(islice(iterator, batch_size)):
        yield batch


def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


def format_passages(texts: list[str], model_name: str) -> list[str]:
    if not needs_e5_prefix(model_name):
        return texts
    return [f"passage: {text}" for text in texts]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_points(batch: list[dict], vectors, start_id: int) -> list[PointStruct]:
    points = []

    for offset, (item, vector) in enumerate(zip(batch, vectors)):
        points.append(
            PointStruct(
                id=start_id + offset,
                vector=vector.tolist(),
                payload={
                    "chunk_id": item["chunk_id"],
                    "title": item["title"],
                    "url": item["url"],
                    "page_id": item.get("page_id"),
                    "revision_id": item.get("revision_id"),
                    "section": item.get("section"),
                    "subsection": item.get("subsection"),
                    "chunk_index": item.get("chunk_index"),
                    "source_path": item.get("source_path"),
                },
            )
        )

    return points


def upload_points(points: list[PointStruct], collection_name: str, qdrant_url: str) -> None:
    client = get_qdrant_client(qdrant_url)

    client.upload_points(
        collection_name=collection_name,
        points=points,
    )


def recreate_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--upload-batch-size", type=int, default=None)
    parser.add_argument("--upload-workers", type=int, default=DEFAULT_MAX_UPLOAD_WORKERS)
    parser.add_argument("--max-pending-uploads", type=int, default=DEFAULT_MAX_PENDING_UPLOADS)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encode_batch_size = args.encode_batch_size or args.batch_size
    upload_batch_size = args.upload_batch_size or args.batch_size

    model = SentenceTransformer(args.model, device=args.device)
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length

    client = QdrantClient(
        url=args.qdrant_url,
        prefer_grpc=True,
    )

    vector_size = model.get_embedding_dimension()

    recreate_collection(
        client=client,
        collection_name=args.collection_name,
        vector_size=vector_size,
    )

    total_chunks = count_lines(args.chunks_path)
    total_batches = (total_chunks + upload_batch_size - 1) // upload_batch_size

    point_id = 0
    pending_futures = set()

    with ThreadPoolExecutor(max_workers=args.upload_workers) as executor:
        for batch in tqdm(
            batched(iter_chunks(args.chunks_path), upload_batch_size),
            total=total_batches,
        ):
            texts = format_passages([item["text"] for item in batch], args.model)

            vectors = model.encode(
                texts,
                batch_size=encode_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            points = build_points(
                batch=batch,
                vectors=vectors,
                start_id=point_id,
            )

            point_id += len(points)

            future = executor.submit(
                upload_points,
                points,
                args.collection_name,
                args.qdrant_url,
            )
            pending_futures.add(future)

            if len(pending_futures) >= args.max_pending_uploads:
                done, pending_futures = wait(
                    pending_futures,
                    return_when=FIRST_COMPLETED,
                )

                for completed in done:
                    completed.result()

        for future in pending_futures:
            future.result()

    print(f"Indexados {point_id} chunks.")


if __name__ == "__main__":
    main()
