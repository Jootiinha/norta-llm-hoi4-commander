from pathlib import Path

from qdrant_client import QdrantClient

from src.config import get_config_section, require_config_value


def ask_defaults() -> dict:
    return get_config_section("ask")


def web_defaults() -> dict:
    return get_config_section("web")


def web_generation_defaults() -> dict:
    config = require_config_value(web_defaults(), "generation", "web")
    if not isinstance(config, dict):
        raise SystemExit("Configure 'web.generation' como mapa em configs/pipeline.yaml.")
    return config


def default_qdrant_url() -> str:
    return str(require_config_value(ask_defaults(), "qdrant_url", "ask"))


def first_query_value(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    if not values:
        return default

    return values[0] or default


def is_local_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]
    return any((path / filename).exists() for filename in tokenizer_files)


def list_local_models(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []

    return [
        str(path)
        for path in sorted(models_dir.iterdir(), key=lambda item: item.name.lower())
        if is_local_model_dir(path)
    ]


def list_chunk_files(chunks_dir: Path) -> list[str]:
    if not chunks_dir.exists():
        return []

    return [
        str(path)
        for path in sorted(chunks_dir.rglob("*.jsonl"), key=lambda item: str(item).lower())
        if path.is_file() and path.name == "chunks.jsonl"
    ]


def list_qdrant_collections(qdrant_url: str) -> tuple[list[str], str | None]:
    try:
        client = QdrantClient(url=qdrant_url, timeout=2.0)
        response = client.get_collections()
    except Exception as exc:
        return [], str(exc)

    return sorted(collection.name for collection in response.collections), None


def get_available_options(qdrant_url: str) -> dict:
    defaults = ask_defaults()
    web_config = web_defaults()
    generation = web_generation_defaults()
    collections, collections_error = list_qdrant_collections(qdrant_url)
    models = list_local_models(Path(require_config_value(web_config, "models_dir", "web")))
    chunk_files = list_chunk_files(Path(require_config_value(web_config, "chunks_dir", "web")))

    return {
        "defaults": {
            "model_path": str(require_config_value(defaults, "model", "ask")),
            "embedding_model": str(require_config_value(defaults, "embedding_model", "ask")),
            "chunks_path": str(require_config_value(defaults, "chunks_path", "ask")),
            "collection_name": str(require_config_value(defaults, "collection_name", "ask")),
            "qdrant_url": str(require_config_value(defaults, "qdrant_url", "ask")),
            "top_k": int(require_config_value(defaults, "top_k", "ask")),
            "candidate_multiplier": int(require_config_value(defaults, "candidate_multiplier", "ask")),
            "max_new_tokens": int(require_config_value(defaults, "max_new_tokens", "ask")),
            "do_sample": bool(require_config_value(generation, "do_sample", "web.generation")),
            "enable_thinking": bool(require_config_value(defaults, "enable_thinking", "ask")),
            "temperature": float(require_config_value(generation, "temperature", "web.generation")),
            "top_p": float(require_config_value(generation, "top_p", "web.generation")),
            "repetition_penalty": float(require_config_value(generation, "repetition_penalty", "web.generation")),
            "no_repeat_ngram_size": int(require_config_value(generation, "no_repeat_ngram_size", "web.generation")),
            "show_context": False,
        },
        "models": models,
        "embedding_models": list(require_config_value(web_config, "embedding_models", "web")),
        "chunk_files": chunk_files,
        "collections": collections,
        "collections_error": collections_error,
    }
