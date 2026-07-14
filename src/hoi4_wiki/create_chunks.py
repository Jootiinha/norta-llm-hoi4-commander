import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items=None, **_kwargs):  # type: ignore[no-redef]
        if items is None:
            return []
        return items

    tqdm.write = print  # type: ignore[attr-defined]

DEFAULT_INPUT_DIR = Path("data/interim/hoi4_wiki/pages")
DEFAULT_OUTPUT = Path("data/processed/chunks/chunks.jsonl")
DEFAULT_MANIFEST = Path("data/processed/chunks/manifest.jsonl")
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_UNITS = 1
DEFAULT_SEMANTIC_MODEL = "BAAI/bge-m3"
DEFAULT_SEMANTIC_THRESHOLD = 0.31
DEFAULT_MIN_CHUNK_UNITS = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = "cpu"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WORD_RE = re.compile(r"\w+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
LIST_OR_TABLE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\|)")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].replace(r"\"", '"')
    if value.isdigit():
        return int(value)
    return value


def parse_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    """Read the simple YAML-like metadata written by convert_to_markdown.py."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown

    metadata: dict[str, Any] = {}
    for index, line in enumerate(lines[1:], start=1):
        line = line.strip()
        if line == "---":
            body = "\n".join(lines[index + 1 :]).lstrip("\n")
            return metadata, body
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = parse_scalar(value)

    return metadata, markdown


def page_dedup_key(metadata: dict[str, Any], body: str) -> tuple[str, str]:
    page_id = metadata.get("page_id")
    revision_id = metadata.get("revision_id")
    if page_id is not None and revision_id is not None:
        return "page_revision", f"{page_id}:{revision_id}"
    if page_id is not None:
        return "page", str(page_id)
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return "body_hash", body_hash


def load_unique_pages(page_paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    pages: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_count = 0

    for page_path in page_paths:
        metadata, body = parse_front_matter(page_path.read_text(encoding="utf-8"))
        dedup_key = page_dedup_key(metadata, body)
        if dedup_key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(dedup_key)
        pages.append(
            {
                "page_path": page_path,
                "metadata": metadata,
                "body": body,
                "dedup_key": f"{dedup_key[0]}:{dedup_key[1]}",
            }
        )

    return pages, duplicate_count


def parse_heading(line: str) -> tuple[int, str] | None:
    match = HEADING_RE.match(line.strip())
    if not match:
        return None
    level = len(match.group(1))
    title = match.group(2).strip()
    return level, title


def clean_markdown_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"^#{1,6}\s+", "", line)
    line = re.sub(r"^[-*+]\s+", "", line)
    line = re.sub(r"^\d+[.)]\s+", "", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"\*([^*]+)\*", r"\1", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def split_into_sections(markdown: str) -> list[dict[str, Any]]:
    """Group markdown lines by heading, keeping the heading path as metadata."""
    sections: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []

    def save_current_section() -> None:
        if current_lines:
            sections.append(
                {
                    "section_path": current_path.copy(),
                    "lines": current_lines.copy(),
                }
            )

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        heading = parse_heading(line)

        if heading is not None:
            save_current_section()
            current_lines = [line]

            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = [heading_title for _level, heading_title in heading_stack]
            continue

        if not current_lines and not line.strip():
            continue
        current_lines.append(line)

    save_current_section()
    return sections


def split_into_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.strip():
            current.append(line.rstrip())
            continue

        if current:
            blocks.append("\n".join(current).strip())
            current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks


def split_block_into_units(block: str) -> list[str]:
    """Create small semantic units without losing markdown list/table structure."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) > 1 and any(LIST_OR_TABLE_RE.match(line) for line in lines):
        return lines

    units = [sentence.strip() for sentence in SENTENCE_RE.split(block) if sentence.strip()]
    return units or [block.strip()]


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split one oversized semantic unit as a safety limit."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    if len(sentences) <= 1:
        return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars)]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def build_semantic_units(lines: list[str], max_chars: int) -> list[str]:
    units: list[str] = []
    for block in split_into_blocks(lines):
        for unit in split_block_into_units(block):
            cleaned = clean_markdown_line(unit)
            units.extend(split_long_text(cleaned, max_chars))
    return [unit for unit in units if unit.strip()]


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


def slugify(value: str, fallback: str) -> str:
    slug = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or fallback


def page_slug(page_path: Path, metadata: dict[str, Any]) -> str:
    page_id = metadata.get("page_id")
    if page_id is not None:
        return f"p{page_id}-{slugify(str(metadata.get('title') or page_path.stem), page_path.stem)}"
    return slugify(page_path.stem, "page")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


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
    embedder: Any,
) -> list[dict[str, Any]]:
    title = str(metadata.get("title") or page_path.stem)
    page_key = page_slug(page_path, metadata)
    chunks: list[dict[str, Any]] = []
    chunk_number = 0
    prepared_sections: list[dict[str, Any]] = []

    for section_index, section in enumerate(split_into_sections(body), start=1):
        units = build_semantic_units(section["lines"], max_chars=max_chars)
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
                    "text": text,
                }
            )

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria chunks semanticos a partir das paginas Markdown da HOI4 Wiki.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-units", type=int, default=DEFAULT_OVERLAP_UNITS)
    parser.add_argument("--semantic-model", type=str, default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--semantic-threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--min-chunk-units", type=int, default=DEFAULT_MIN_CHUNK_UNITS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--no-deduplicate-pages", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    page_paths = sorted(args.input_dir.glob("*.md"))
    if args.no_deduplicate_pages:
        pages = []
        for page_path in page_paths:
            metadata, body = parse_front_matter(page_path.read_text(encoding="utf-8"))
            pages.append(
                {
                    "page_path": page_path,
                    "metadata": metadata,
                    "body": body,
                    "dedup_key": None,
                }
            )
        duplicate_page_count = 0
    else:
        pages, duplicate_page_count = load_unique_pages(page_paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    embedder = load_embedder(args.semantic_model, device=args.device)

    total_chunks = 0

    with args.output.open("w", encoding="utf-8") as chunk_file, args.manifest.open("w", encoding="utf-8") as manifest_file:

        for page in tqdm(pages, desc="Gerando chunks", unit="pagina", dynamic_ncols=True):
            page_path = page["page_path"]
            metadata = page["metadata"]
            body = page["body"]

            chunks = build_chunks_for_page(
                page_path=page_path,
                metadata=metadata,
                body=body,
                max_chars=args.max_chars,
                overlap_units=args.overlap_units,
                semantic_model=args.semantic_model,
                semantic_threshold=args.semantic_threshold,
                min_chunk_units=args.min_chunk_units,
                batch_size=args.batch_size,
                device=args.device,
                embedder=embedder,
            )

            for chunk in chunks:
                chunk_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            manifest_file.write(
                json.dumps(
                    {
                        "source_path": str(page_path),
                        "title": metadata.get("title"),
                        "display_title": metadata.get("display_title"),
                        "page_id": metadata.get("page_id"),
                        "revision_id": metadata.get("revision_id"),
                        "dedup_key": page["dedup_key"],
                        "chunk_count": len(chunks),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            total_chunks += len(chunks)

    tqdm.write(f"Paginas encontradas: {len(page_paths)}")
    tqdm.write(f"Paginas duplicadas ignoradas: {duplicate_page_count}")
    tqdm.write(f"Paginas processadas: {len(pages)}")
    tqdm.write(f"Chunks gerados: {total_chunks}")
    tqdm.write(f"Arquivo de chunks: {args.output}")
    tqdm.write(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
