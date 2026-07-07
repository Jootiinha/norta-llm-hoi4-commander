import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_INPUT_DIR = Path("data/interim/hoi4_wiki/pages")
DEFAULT_OUTPUT = Path("data/processed/chunks/chunks.jsonl")
DEFAULT_MANIFEST = Path("data/processed/chunks/manifest.jsonl")
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_UNITS = 1
DEFAULT_CHUNKING_STRATEGY = "semantic"
DEFAULT_SEMANTIC_THRESHOLD = 0.31
DEFAULT_MIN_SEMANTIC_SENTENCES = 3
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_TRANSFORMER_CLASS: Any | None = None
_EMBEDDER_CACHE: dict[str, Any] = {}
_AUTO_TOKENIZER_CLASS: Any | None = None
_TOKENIZER_CACHE: dict[str, Any] = {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].replace(r"\"", '"')
    if value.isdigit():
        try:
            return int(value)
        except ValueError:
            return value
    return value


def parse_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown

    metadata: dict[str, Any] = {}
    for index in range(1, len(lines)):
        line = lines[index].strip()
        if line == "---":
            body = "\n".join(lines[index + 1 :]).lstrip("\n")
            return metadata, body
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = parse_scalar(value)

    return metadata, markdown


def parse_heading(line: str) -> tuple[int, str] | None:
    match = HEADING_RE.match(line.strip())
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def sectionize(markdown: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        heading = parse_heading(line)
        if heading:
            if current_lines:
                sections.append(
                    {
                        "section_path": current_path.copy(),
                        "lines": current_lines.copy(),
                    }
                )
                current_lines = []

            level, title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = [item[1] for item in stack]
            current_lines = [line.strip()]
            continue

        if not current_lines and not line.strip():
            continue
        current_lines.append(line.rstrip())

    if current_lines:
        sections.append(
            {
                "section_path": current_path.copy(),
                "lines": current_lines.copy(),
            }
        )

    return sections


def split_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if buffer:
                paragraphs.append("\n".join(buffer).strip())
                buffer = []
            continue
        buffer.append(stripped)

    if buffer:
        paragraphs.append("\n".join(buffer).strip())

    return [paragraph for paragraph in paragraphs if paragraph]


def split_long_unit(unit: str, max_chars: int) -> list[str]:
    unit = unit.strip()
    if len(unit) <= max_chars:
        return [unit]

    if "\n" in unit:
        lines = [line.rstrip() for line in unit.splitlines() if line.strip()]
        packed: list[str] = []
        current: list[str] = []
        for line in lines:
            candidate = "\n".join(current + [line]).strip()
            if current and len(candidate) > max_chars:
                packed.extend(split_long_unit("\n".join(current), max_chars))
                current = [line]
            else:
                current.append(line)
        if current:
            packed.extend(split_long_unit("\n".join(current), max_chars))
        return packed

    sentences = re.split(r"(?<=[.!?])\s+", unit)
    if len(sentences) > 1:
        packed: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if current and len(candidate) > max_chars:
                packed.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current:
            packed.append(current.strip())
        if any(len(chunk) > max_chars for chunk in packed):
            return [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)]
        return packed

    return [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)]


def normalize_sentence_value(value: str) -> str:
    return value.strip().replace("\n", " ")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # Keep one fallback path for headings/lines without ponto final.
    raw = [segment.strip() for segment in SENTENCE_RE.split(text) if segment.strip()]
    if raw:
        return [normalize_sentence_value(segment) for segment in raw]
    return [normalize_sentence_value(text)]


def sentence_tokens(sentence: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(sentence)}


def semantic_similarity(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 1.0

    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        dot = sum(a.get(key, 0.0) * b.get(key, 0.0) for key in keys)
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # Dense list/tuple fallback.
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def get_embedder(model_name: str | None) -> Any | None:
    global _SENTENCE_TRANSFORMER_CLASS

    if not model_name:
        return None

    embedder = _EMBEDDER_CACHE.get(model_name)
    if embedder is not None:
        return embedder

    if _SENTENCE_TRANSFORMER_CLASS is None:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_TRANSFORMER_CLASS = SentenceTransformer

    embedder = _SENTENCE_TRANSFORMER_CLASS(model_name)
    _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def get_tokenizer(model_name: str | None) -> Any | None:
    global _AUTO_TOKENIZER_CLASS

    if not model_name:
        return None

    tokenizer = _TOKENIZER_CACHE.get(model_name)
    if tokenizer is not None:
        return tokenizer

    if _AUTO_TOKENIZER_CLASS is None:
        from transformers import AutoTokenizer

        _AUTO_TOKENIZER_CLASS = AutoTokenizer

    tokenizer = _AUTO_TOKENIZER_CLASS.from_pretrained(model_name)
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def count_tokens(text: str, tokenizer: Any | None) -> int:
    if not text:
        return 0

    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            # Mantem fluxo operacional sem quebra caso tokenizer nao carregue.
            pass

    return len(WORD_RE.findall(text))


def embed_sentences(
    sentences: list[str],
    embedder: Any | None,
) -> tuple[str, list[Any]]:
    if not sentences:
        return "counter", []

    if embedder is not None:
        try:
            encoded = embedder.encode(
                sentences,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors: list[Any] = list(encoded)
            return "dense", vectors
        except Exception:
            # Mantem fluxo operacional sem quebra caso modelo nao carregue.
            pass

    return "counter", [CounterVec(sentence_tokens(sentence)) for sentence in sentences]


class CounterVec(dict[str, float]):
    """Simple sparse vector representation (bag of words)."""

    @classmethod
    def from_text(cls, text: str) -> "CounterVec":
        return cls({token: 1.0 for token in sentence_tokens(text)})


def chunk_units_semantic(
    sentences: list[str],
    max_chars: int,
    overlap_units: int,
    similarity_threshold: float,
    min_chunk_sentences: int,
    embeddings: list[Any] | None = None,
    embedder: Any | None = None,
) -> list[str]:
    if not sentences:
        return []

    if embeddings is None:
        _, embeddings = embed_sentences(sentences, embedder=embedder)
    chunks: list[str] = []
    current_units: list[str] = []
    current_vectors: list[Any] = []

    def append_chunk() -> None:
        nonlocal current_units, current_vectors
        if not current_units:
            return
        text = " ".join(current_units).strip()
        if text:
            chunks.append(text)

        if overlap_units <= 0:
            current_units = []
            current_vectors = []
            return

        overlap_units_count = min(overlap_units, len(current_units))
        current_units = current_units[-overlap_units_count:]
        current_vectors = current_vectors[-overlap_units_count:]

    def current_len_with(candidate: str) -> int:
        if not current_units:
            return len(candidate)
        return len(" ".join(current_units + [candidate]).strip())

    for sentence, vector in zip(sentences, embeddings):
        if len(sentence) > max_chars:
            append_chunk()
            chunks.extend(split_long_unit(sentence, max_chars))
            continue

        if not current_units:
            current_units = [sentence]
            current_vectors = [vector]
            continue

        candidate_len = current_len_with(sentence)
        if candidate_len > max_chars:
            append_chunk()
            current_units = [sentence]
            current_vectors = [vector]
            continue

        previous_vector = current_vectors[-1]
        if len(current_units) >= min_chunk_sentences and (
            semantic_similarity(previous_vector, vector) < similarity_threshold
        ):
            append_chunk()
            current_units = [sentence]
            current_vectors = [vector]
            continue

        current_units.append(sentence)
        current_vectors.append(vector)

    if current_units:
        chunks.append(" ".join(current_units).strip())
        current_units = []
        current_vectors = []

    return chunks


def pack_units(units: list[str], max_chars: int, overlap_units: int) -> list[str]:
    chunks: list[str] = []
    current_units: list[str] = []

    for unit in units:
        for piece in split_long_unit(unit, max_chars):
            piece = piece.strip()
            if not piece:
                continue

            if not current_units:
                current_units = [piece]
                continue

            candidate = "\n\n".join(current_units + [piece]).strip()
            if len(candidate) <= max_chars:
                current_units.append(piece)
                continue

            chunks.append("\n\n".join(current_units).strip())

            if overlap_units > 0:
                overlap = current_units[-overlap_units:]
                while overlap and len("\n\n".join(overlap + [piece])) > max_chars and len(overlap) > 1:
                    overlap = overlap[1:]
                current_units = overlap + [piece] if overlap else [piece]
            else:
                current_units = [piece]

    if current_units:
        chunks.append("\n\n".join(current_units).strip())

    return [chunk for chunk in chunks if chunk]


def slugify(value: str, fallback: str) -> str:
    slug = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or fallback


def read_page(path: Path) -> tuple[dict[str, Any], str]:
    metadata, body = parse_front_matter(read_text(path))
    return metadata, body


def build_chunks_for_page(
    page_path: Path,
    metadata: dict[str, Any],
    body: str,
    max_chars: int,
    overlap_units: int,
    semantic: bool,
    similarity_threshold: float,
    min_chunk_sentences: int,
    embedder: Any | None = None,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    title = str(metadata.get("title") or page_path.stem)
    section_rows = sectionize(body)
    chunks: list[dict[str, Any]] = []
    chunk_number = 0
    prepared_sections: list[dict[str, Any]] = []

    for section in section_rows:
        raw_units = split_paragraphs(section["lines"])
        sentence_units: list[str] = []
        for unit in raw_units:
            sentence_units.extend(split_sentences(unit))
        prepared_sections.append(
            {
                "section_path": section["section_path"],
                "raw_units": raw_units,
                "sentence_units": sentence_units,
            }
        )

    page_embeddings: list[Any] | None = None
    if semantic:
        page_sentences = [
            sentence
            for section in prepared_sections
            for sentence in section["sentence_units"]
        ]
        _, page_embeddings = embed_sentences(page_sentences, embedder=embedder)
        embedding_offset = 0

    for section_index, section in enumerate(prepared_sections, start=1):
        if semantic:
            section_sentence_count = len(section["sentence_units"])
            section_embeddings = page_embeddings[
                embedding_offset : embedding_offset + section_sentence_count
            ]
            embedding_offset += section_sentence_count
            section_chunks = chunk_units_semantic(
                sentences=section["sentence_units"],
                max_chars=max_chars,
                overlap_units=overlap_units,
                similarity_threshold=similarity_threshold,
                min_chunk_sentences=min_chunk_sentences,
                embeddings=section_embeddings,
            )
        else:
            section_chunks = pack_units(
                section["raw_units"],
                max_chars=max_chars,
                overlap_units=overlap_units,
            )

        for section_chunk_index, text in enumerate(section_chunks, start=1):
            chunk_number += 1
            text = text.strip()
            if not text:
                continue
            chunk_id = f"{slugify(title, page_path.stem)}-{chunk_number:04d}"
            section_path = section["section_path"]
            section_title = section_path[1] if len(section_path) > 1 else None
            subsection_title = " > ".join(section_path[2:]) if len(section_path) > 2 else None
            chunks.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "page_id": metadata.get("page_id"),
                    "revision_id": metadata.get("revision_id"),
                    "title": metadata.get("title", title),
                    "display_title": metadata.get("display_title", title),
                    "section": section_title,
                    "subsection": subsection_title,
                    "url": metadata.get("url"),
                    "source_path": str(page_path),
                    "chunk_index": chunk_number,
                    "section_index": section_index,
                    "section_chunk_index": section_chunk_index,
                    "section_path": section_path,
                    "section_depth": len(section_path),
                    "char_count": len(text),
                    "word_count": len(re.findall(r"\w+", text)),
                    "token_count": count_tokens(text, tokenizer),
                    "text": text,
                }
            )

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-units", type=int, default=DEFAULT_OVERLAP_UNITS)
    parser.add_argument(
        "--chunking-strategy",
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=("semantic", "paragraph"),
        help="semantic: usa agrupamento por semantica entre sentencas; paragraph: comportamento anterior",
    )
    parser.add_argument("--semantic-threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--min-chunk-sentences", type=int, default=DEFAULT_MIN_SEMANTIC_SENTENCES)
    parser.add_argument("--semantic-model", type=str, default=DEFAULT_SEMANTIC_MODEL)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_path: Path = args.output
    manifest_path: Path = args.manifest

    page_paths = sorted(input_dir.glob("*.md"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    use_semantic = args.chunking_strategy == "semantic"
    embedder: Any | None = None
    tokenizer: Any | None = None

    if use_semantic:
        try:
            embedder = get_embedder(args.semantic_model)
        except Exception:
            embedder = None

    try:
        tokenizer = get_tokenizer(args.semantic_model)
    except Exception:
        tokenizer = None

    total_chunks = 0
    with output_path.open("w", encoding="utf-8") as chunk_file, manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest_file:
        with tqdm(
            total=len(page_paths),
            desc="Gerando chunks",
            unit="pagina",
            dynamic_ncols=True,
        ) as progress:
            for page_path in page_paths:
                metadata, body = read_page(page_path)
                chunks = build_chunks_for_page(
                    page_path=page_path,
                    metadata=metadata,
                    body=body,
                    max_chars=args.max_chars,
                    overlap_units=args.overlap_units,
                    semantic=use_semantic,
                    similarity_threshold=args.semantic_threshold,
                    min_chunk_sentences=args.min_chunk_sentences,
                    embedder=embedder,
                    tokenizer=tokenizer,
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
                            "chunk_count": len(chunks),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_chunks += len(chunks)
                progress.set_postfix(chunks=total_chunks)
                progress.update(1)

    tqdm.write(f"Paginas lidas: {len(page_paths)}")
    tqdm.write(f"Chunks gerados: {total_chunks}")
    tqdm.write(f"Arquivo de chunks: {output_path}")
    tqdm.write(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
