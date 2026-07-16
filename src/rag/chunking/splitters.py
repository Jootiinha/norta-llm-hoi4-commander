import re
from typing import Any


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WORD_RE = re.compile(r"\w+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
LIST_OR_TABLE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\|)")


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


def pack_structured_lines(lines: list[str], group_lines: int, max_chars: int) -> list[str]:
    if group_lines <= 1:
        return lines

    groups: list[str] = []
    current: list[str] = []

    for line in lines:
        candidate = "\n".join([*current, line]).strip()
        if current and (len(current) >= group_lines or len(candidate) > max_chars):
            groups.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        groups.append("\n".join(current).strip())

    return groups


def split_block_into_units(block: str, structured_group_lines: int, max_chars: int) -> list[str]:
    """Create small semantic units without losing markdown list/table structure."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) > 1 and any(LIST_OR_TABLE_RE.match(line) for line in lines):
        return pack_structured_lines(lines, group_lines=structured_group_lines, max_chars=max_chars)

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


def normalize_unit_for_dedup(unit: str) -> str:
    return re.sub(r"\s+", " ", unit).strip().lower()


def preprocess_semantic_units(units: list[str], min_unit_chars: int) -> list[str]:
    processed: list[str] = []
    previous_key = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) < min_unit_chars and not processed:
            continue

        dedup_key = normalize_unit_for_dedup(unit)
        if dedup_key == previous_key:
            continue

        processed.append(unit)
        previous_key = dedup_key

    return processed


def build_semantic_units(
    lines: list[str],
    max_chars: int,
    structured_group_lines: int,
    min_unit_chars: int,
) -> list[str]:
    units: list[str] = []
    for block in split_into_blocks(lines):
        for unit in split_block_into_units(block, structured_group_lines=structured_group_lines, max_chars=max_chars):
            cleaned = clean_markdown_line(unit)
            units.extend(split_long_text(cleaned, max_chars))
    return preprocess_semantic_units(units, min_unit_chars=min_unit_chars)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))

