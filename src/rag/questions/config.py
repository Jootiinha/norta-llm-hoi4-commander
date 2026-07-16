from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True)
class RagConfig:
    model_path: str
    embedding_model: str
    qdrant_url: str
    collection_name: str
    chunks_path: Path
    top_k: int
    candidate_multiplier: int
    max_new_tokens: int
    enable_thinking: bool


class RagGraphState(TypedDict, total=False):
    question: str
    contexts: list[dict]
    prompt: str
    answer: str
