import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PIPELINE_CONFIG_PATH, get_config_section, require_config_value

DROP_SELECTORS = [
    ".toc",
    ".mw-editsection",
    ".navbox",
    ".vertical-navbox",
    ".metadata",
    ".ambox",
    ".hatnote",
    ".stub",
    ".printfooter",
    ".catlinks",
    ".mw-references-wrap",
    ".reference",
    ".reflist",
    "#siteNotice",
    "#jump-to-nav",
    "nav",
    "script",
    "style",
    "noscript",
]

DROP_SECTION_TITLES = {
    "navigation menu",
    "personal tools",
    "namespaces",
    "variants",
    "views",
    "more",
    "search",
    "navigation",
    "tools",
    "print/export",
    "in other projects",
    "languages",
    "references",
    "notes",
    "notes and references",
    "footnotes",
    "citations",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido em {path}:{line_number}") from exc
    return rows


def display_text(value: str | None) -> str:
    if not value:
        return ""
    return BeautifulSoup(value, "lxml").get_text(" ", strip=True)


def slugify(title: str, fallback: str) -> str:
    base = title or fallback
    base = BeautifulSoup(base, "lxml").get_text(" ", strip=True)
    base = base.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    base = base.strip("-")
    return base or fallback


def extract_source(record: dict[str, Any]) -> tuple[str, bool]:
    parse = record.get("parse")
    if isinstance(parse, dict):
        text = parse.get("text")
        if isinstance(text, str):
            return text, True

    for key in ("html", "raw_html", "content"):
        value = record.get(key)
        if isinstance(value, str) and looks_like_html(value):
            return value, True

    text = record.get("text")
    if isinstance(text, str):
        return text, looks_like_html(text)

    raise ValueError(f"Registro sem campo de conteudo reconhecido: {record.get('title')}")


def looks_like_html(value: str) -> bool:
    sample = value[:500].lower()
    return "<html" in sample or "<div" in sample or "<p" in sample or "<table" in sample


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one(".mw-parser-output") or soup

    for selector in DROP_SELECTORS:
        for tag in content.select(selector):
            tag.decompose()

    return md(
        str(content),
        heading_style="ATX",
        bullets="*",
        strip=["img"],
    )


def markdown_heading_level(line: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if not match:
        return None
    return len(match.group(1))


def markdown_heading_title(line: str) -> str:
    line = re.sub(r"^#{1,6}\s+", "", line).strip()
    line = re.sub(r"\s+#*$", "", line).strip()
    return line.lower()


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower()
    if not stripped:
        return False
    if stripped.startswith("[edit]"):
        return True
    if stripped.startswith("![") or "thumb.php" in stripped:
        return True
    if re.match(r"^\[\[?file:", lower):
        return True
    if lower.startswith("retrieved from "):
        return True
    if lower.startswith("category:") or lower.startswith("categories:"):
        return True
    if "this page was last edited" in lower:
        return True
    return False


def content_start_index(lines: list[str], title: str) -> int:
    normalized_title = title.lower().strip()
    title_markers = {
        f"**{normalized_title}**",
        f"# {normalized_title}",
        f"## {normalized_title}",
    }

    for index, line in enumerate(lines):
        lower = line.lower().strip()
        if lower in title_markers:
            return index
        if markdown_heading_level(line) is not None:
            return index

    for index, line in enumerate(lines):
        if likely_content_line(line):
            return index

    return 0


def likely_content_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 80:
        return False
    link_count = stripped.count("](")
    word_count = len(re.findall(r"\w+", stripped))
    return word_count >= 12 and link_count <= 3 and any(char in stripped for char in ".:")


def remove_drop_sections(lines: list[str]) -> list[str]:
    cleaned = []
    skip_level: int | None = None

    for line in lines:
        level = markdown_heading_level(line)
        if level is not None:
            title = markdown_heading_title(line)
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if title in DROP_SECTION_TITLES:
                skip_level = level
                continue

        if skip_level is not None:
            continue
        cleaned.append(line)

    return cleaned


def normalize_markdown(markdown: str, title: str) -> str:
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    markdown = re.sub(r"\[\[[^\]]+\]\]\(#cite_(?:note|ref)-[^)]+\)", "", markdown)
    markdown = re.sub(r"\[\^\]\(#cite_(?:note|ref)-[^)]+\)", "", markdown)

    lines = []
    previous_blank = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        line = re.sub(r"\s{2,}", " ", line)
        if is_noise_line(line):
            continue
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False

    start = content_start_index(lines, title)
    lines = lines[start:]
    lines = remove_drop_sections(lines)

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def front_matter(record: dict[str, Any], markdown: str) -> str:
    title = display_text(record.get("title")) or "Untitled"
    display_title = display_text(record.get("display_title")) or title
    metadata = {
        "page_id": record.get("page_id"),
        "revision_id": record.get("revision_id"),
        "title": title,
        "display_title": display_title,
        "url": record.get("url"),
    }
    lines = ["---"]
    for key, value in metadata.items():
        if value is None:
            continue
        escaped = str(value).replace('"', '\\"')
        lines.append(f'{key}: "{escaped}"')
    lines.append("---")
    lines.append("")
    lines.append(markdown)
    lines.append("")
    return "\n".join(lines)


def convert_record(record: dict[str, Any]) -> tuple[str, str]:
    title = display_text(record.get("title")) or display_text(record.get("display_title"))
    if not title:
        title = f"page-{record.get('page_id', 'unknown')}"

    source, is_html = extract_source(record)
    markdown = html_to_markdown(source) if is_html else source
    markdown = normalize_markdown(markdown, title)
    return title, front_matter(record, markdown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PIPELINE_CONFIG_PATH)
    args = parser.parse_args()
    config = get_config_section("convert", args.config)

    input_path = Path(require_config_value(config, "input", "convert"))
    output_dir = Path(require_config_value(config, "output_dir", "convert"))
    manifest_path = Path(require_config_value(config, "manifest", "convert"))
    min_chars = int(require_config_value(config, "min_chars", "convert"))

    records = read_jsonl(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, record in enumerate(tqdm(records, desc="Convertendo paginas"), start=1):
            title, markdown = convert_record(record)
            content_without_meta = markdown.split("---", 2)[-1].strip()
            if len(content_without_meta) < min_chars:
                continue

            filename = f"{index:05d}-{slugify(title, f'page-{index}')}.md"
            output_path = output_dir / filename
            output_path.write_text(markdown, encoding="utf-8")

            manifest.write(
                json.dumps(
                    {
                        "title": title,
                        "page_id": record.get("page_id"),
                        "revision_id": record.get("revision_id"),
                        "url": record.get("url"),
                        "path": str(output_path),
                        "chars": len(content_without_meta),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Paginas lidas: {len(records)}")
    print(f"Markdown gerado: {written}")
    print(f"Diretorio: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
