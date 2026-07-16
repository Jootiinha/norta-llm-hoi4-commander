import hashlib
from pathlib import Path
from typing import Any


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


def load_pages(page_paths: list[Path]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
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

    return pages


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

