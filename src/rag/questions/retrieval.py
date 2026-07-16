from pathlib import Path
from typing import Any

from pydantic import ConfigDict
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .chunks import exit_with_message


def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


def format_query(query: str, model_name: str) -> str:
    if not needs_e5_prefix(model_name):
        return query
    return f"query: {query}"


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    qdrant_url: str,
    collection_name: str,
    chunk_lookup: dict[str, dict],
    chunks_path: Path,
    embedding_model_name: str,
    top_k: int,
    candidate_multiplier: int,
) -> list[dict]:
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection_name):
        exit_with_message(
            f"Colecao Qdrant '{collection_name}' nao encontrada em {qdrant_url}.\n"
            "Indexe os chunks antes da consulta com:\n"
            "  make index-data\n"
            "Se voce usou outro nome de colecao ao indexar, informe o mesmo nome com "
            "--collection-name."
        )

    vector = embedder.encode(
        format_query(query, embedding_model_name),
        normalize_embeddings=True,
    ).tolist()

    candidate_limit = max(top_k, top_k * candidate_multiplier)
    results = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=candidate_limit,
    ).points

    contexts = []
    seen_sources: set[tuple[str, str]] = set()

    for hit in results:
        payload = dict(hit.payload or {})
        chunk_id = payload.get("chunk_id")
        chunk = chunk_lookup.get(chunk_id) if chunk_id else None

        if chunk is None:
            raise KeyError(
                f"Chunk '{chunk_id}' nao encontrado em {chunks_path}."
            )

        source_key = (
            str(payload.get("url") or ""),
            str(payload.get("title") or ""),
        )
        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        payload["score"] = hit.score
        payload["text"] = chunk["text"]
        payload["section"] = chunk.get("section")
        payload["subsection"] = chunk.get("subsection")
        contexts.append(payload)
        if len(contexts) >= top_k:
            break

    return contexts


class Hoi4QdrantRetriever(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedder: Any
    qdrant_url: str
    collection_name: str
    chunk_lookup: dict[str, dict]
    chunks_path: Path
    embedding_model_name: str
    top_k: int
    candidate_multiplier: int

    def invoke(self, query: str) -> list[dict]:
        return retrieve(
            query,
            embedder=self.embedder,
            qdrant_url=self.qdrant_url,
            collection_name=self.collection_name,
            chunk_lookup=self.chunk_lookup,
            chunks_path=self.chunks_path,
            embedding_model_name=self.embedding_model_name,
            top_k=self.top_k,
            candidate_multiplier=self.candidate_multiplier,
        )
