import json
import re
from pathlib import Path
from typing import Any, Callable


ChunkBuilder = Callable[[Path, dict[str, Any], str], list[dict[str, Any]]]


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


def write_chunks_and_manifest(
    pages: list[dict[str, Any]],
    output: Path,
    manifest: Path,
    build_chunks: ChunkBuilder,
    progress,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with output.open("w", encoding="utf-8") as chunk_file, manifest.open("w", encoding="utf-8") as manifest_file:
        for page in progress(pages, desc="Gerando chunks", unit="pagina", dynamic_ncols=True):
            page_path = page["page_path"]
            metadata = page["metadata"]
            body = page["body"]

            chunks = build_chunks(page_path, metadata, body)

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

    return total_chunks

