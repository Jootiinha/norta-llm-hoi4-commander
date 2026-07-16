import argparse
import math
from pathlib import Path
from typing import Any

from src.config import PIPELINE_CONFIG_PATH, get_config_section, require_config_value
from tqdm import tqdm

from .front_matter import load_pages, load_unique_pages
from .io import page_slug, write_chunks_and_manifest
from .splitters import build_semantic_units, split_into_sections, word_count


def load_embedder(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embed_units(units: list[str], embedder: Any, batch_size: int) -> list[Any]:
    if not units:
        return []
    return list(
        embedder.encode(
            units,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )


def cosine_similarity(left: Any, right: Any) -> float:
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for left_value, right_value in zip(left, right):
        left_float = float(left_value)
        right_float = float(right_value)
        dot += left_float * right_float
        norm_left += left_float * left_float
        norm_right += right_float * right_float
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / math.sqrt(norm_left * norm_right)


def semantic_chunks(
    units: list[str],
    embeddings: list[Any],
    max_chars: int,
    overlap_units: int,
    threshold: float,
    min_chunk_units: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_embeddings: list[Any] = []

    def current_text_with(unit: str) -> str:
        return "\n".join([*current, unit]).strip()

    def flush() -> None:
        nonlocal current, current_embeddings
        if current:
            chunks.append("\n".join(current).strip())
        if overlap_units > 0:
            overlap_count = min(overlap_units, len(current))
            current = current[-overlap_count:]
            current_embeddings = current_embeddings[-overlap_count:]
        else:
            current = []
            current_embeddings = []

    for unit, embedding in zip(units, embeddings):
        unit = unit.strip()
        if not unit:
            continue

        if not current:
            current = [unit]
            current_embeddings = [embedding]
            continue

        if len(current_text_with(unit)) > max_chars:
            flush()
            current = [unit]
            current_embeddings = [embedding]
            continue

        similarity = cosine_similarity(current_embeddings[-1], embedding)
        if len(current) >= min_chunk_units and similarity < threshold:
            flush()
            current = [unit]
            current_embeddings = [embedding]
            continue

        current.append(unit)
        current_embeddings.append(embedding)

    if current:
        chunks.append("\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def build_chunks_for_page(
    page_path: Path,
    metadata: dict[str, Any],
    body: str,
    max_chars: int,
    overlap_units: int,
    semantic_model: str,
    semantic_threshold: float,
    min_chunk_units: int,
    batch_size: int,
    device: str,
    structured_group_lines: int,
    min_unit_chars: int,
    embedder: Any,
) -> list[dict[str, Any]]:
    title = str(metadata.get("title") or page_path.stem)
    page_key = page_slug(page_path, metadata)
    chunks: list[dict[str, Any]] = []
    chunk_number = 0
    prepared_sections: list[dict[str, Any]] = []

    for section_index, section in enumerate(split_into_sections(body), start=1):
        units = build_semantic_units(
            section["lines"],
            max_chars=max_chars,
            structured_group_lines=structured_group_lines,
            min_unit_chars=min_unit_chars,
        )
        prepared_sections.append(
            {
                "section_index": section_index,
                "section_path": section["section_path"],
                "units": units,
            }
        )

    page_units = [unit for section in prepared_sections for unit in section["units"]]
    page_embeddings = embed_units(page_units, embedder, batch_size=batch_size)
    embedding_offset = 0

    for section in prepared_sections:
        units = section["units"]
        section_embeddings = page_embeddings[embedding_offset : embedding_offset + len(units)]
        embedding_offset += len(units)
        section_chunks = semantic_chunks(
            units=units,
            embeddings=section_embeddings,
            max_chars=max_chars,
            overlap_units=overlap_units,
            threshold=semantic_threshold,
            min_chunk_units=min_chunk_units,
        )
        section_path = section["section_path"]
        section_index = section["section_index"]

        for section_chunk_index, text in enumerate(section_chunks, start=1):
            text = text.strip()
            if not text:
                continue

            chunk_number += 1
            chunk_id = f"{page_key}-{chunk_number:04d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "page_id": metadata.get("page_id"),
                    "revision_id": metadata.get("revision_id"),
                    "title": metadata.get("title", title),
                    "display_title": metadata.get("display_title", title),
                    "section": section_path[0] if section_path else None,
                    "subsection": " > ".join(section_path[1:]) if len(section_path) > 1 else None,
                    "url": metadata.get("url"),
                    "source_path": str(page_path),
                    "chunk_index": chunk_number,
                    "section_index": section_index,
                    "section_chunk_index": section_chunk_index,
                    "section_path": section_path,
                    "section_depth": len(section_path),
                    "char_count": len(text),
                    "word_count": word_count(text),
                    "token_count": word_count(text),
                    "chunking_strategy": "semantic",
                    "semantic_model": semantic_model,
                    "semantic_threshold": semantic_threshold,
                    "embedding_device": device,
                    "embedding_batch_size": batch_size,
                    "structured_group_lines": structured_group_lines,
                    "min_unit_chars": min_unit_chars,
                    "text": text,
                }
            )

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria chunks semanticos a partir das paginas Markdown da HOI4 Wiki.")
    parser.add_argument("--config", type=Path, default=PIPELINE_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config_section("chunk", args.config)

    input_dir = Path(require_config_value(config, "input_dir", "chunk"))
    output = Path(require_config_value(config, "output", "chunk"))
    manifest = Path(require_config_value(config, "manifest", "chunk"))
    max_chars = int(require_config_value(config, "max_chars", "chunk"))
    overlap_units = int(require_config_value(config, "overlap_units", "chunk"))
    semantic_model = str(require_config_value(config, "semantic_model", "chunk"))
    semantic_threshold = float(require_config_value(config, "semantic_threshold", "chunk"))
    min_chunk_units = int(require_config_value(config, "min_chunk_units", "chunk"))
    batch_size = int(require_config_value(config, "batch_size", "chunk"))
    device = str(require_config_value(config, "device", "chunk"))
    structured_group_lines = int(require_config_value(config, "structured_group_lines", "chunk"))
    min_unit_chars = int(require_config_value(config, "min_unit_chars", "chunk"))
    deduplicate_pages = bool(require_config_value(config, "deduplicate_pages", "chunk"))

    page_paths = sorted(input_dir.glob("*.md"))
    if deduplicate_pages:
        pages, duplicate_page_count = load_unique_pages(page_paths)
    else:
        pages = load_pages(page_paths)
        duplicate_page_count = 0

    embedder = load_embedder(semantic_model, device=device)

    def build_page_chunks(page_path: Path, metadata: dict[str, Any], body: str) -> list[dict[str, Any]]:
        return build_chunks_for_page(
            page_path=page_path,
            metadata=metadata,
            body=body,
            max_chars=max_chars,
            overlap_units=overlap_units,
            semantic_model=semantic_model,
            semantic_threshold=semantic_threshold,
            min_chunk_units=min_chunk_units,
            batch_size=batch_size,
            device=device,
            structured_group_lines=structured_group_lines,
            min_unit_chars=min_unit_chars,
            embedder=embedder,
        )

    total_chunks = write_chunks_and_manifest(
        pages=pages,
        output=output,
        manifest=manifest,
        build_chunks=build_page_chunks,
        progress=tqdm,
    )

    tqdm.write(f"Paginas encontradas: {len(page_paths)}")
    tqdm.write(f"Paginas duplicadas ignoradas: {duplicate_page_count}")
    tqdm.write(f"Paginas processadas: {len(pages)}")
    tqdm.write(f"Chunks gerados: {total_chunks}")
    tqdm.write(f"Arquivo de chunks: {output}")
    tqdm.write(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
