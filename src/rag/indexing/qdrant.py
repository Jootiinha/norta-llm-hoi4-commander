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

from src.config import PIPELINE_CONFIG_PATH, get_config_section, require_config_key, require_config_value

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
    parser.add_argument("--config", type=Path, default=PIPELINE_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config_section("index", args.config)

    chunks_path = Path(require_config_value(config, "chunks_path", "index"))
    collection_name = str(require_config_value(config, "collection_name", "index"))
    model_name = str(require_config_value(config, "model", "index"))
    device = require_config_value(config, "device", "index")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(require_config_value(config, "batch_size", "index"))
    encode_batch_size = int(require_config_key(config, "encode_batch_size", "index") or batch_size)
    upload_batch_size = int(require_config_key(config, "upload_batch_size", "index") or batch_size)
    upload_workers = int(require_config_value(config, "upload_workers", "index"))
    max_pending_uploads = int(require_config_value(config, "max_pending_uploads", "index"))
    max_seq_length = require_config_key(config, "max_seq_length", "index")
    qdrant_url = str(require_config_value(config, "qdrant_url", "index"))

    model = SentenceTransformer(model_name, device=str(device))
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)

    client = QdrantClient(
        url=qdrant_url,
        prefer_grpc=True,
    )

    vector_size = model.get_embedding_dimension()

    recreate_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    total_chunks = count_lines(chunks_path)
    total_batches = (total_chunks + upload_batch_size - 1) // upload_batch_size

    point_id = 0
    pending_futures = set()

    with ThreadPoolExecutor(max_workers=upload_workers) as executor:
        for batch in tqdm(
            batched(iter_chunks(chunks_path), upload_batch_size),
            total=total_batches,
        ):
            texts = format_passages([item["text"] for item in batch], model_name)

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
                collection_name,
                qdrant_url,
            )
            pending_futures.add(future)

            if len(pending_futures) >= max_pending_uploads:
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
