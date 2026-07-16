import json
from pathlib import Path


def exit_with_message(message: str) -> None:
    raise SystemExit(message)


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    if not path.exists():
        exit_with_message(
            f"Arquivo de chunks nao encontrado: {path}\n"
            "Gere os chunks antes da consulta com:\n"
            "  make chunk-data"
        )

    lookup: dict[str, dict] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                lookup[chunk_id] = chunk

    if not lookup:
        exit_with_message(
            f"Nenhum chunk encontrado em {path}.\n"
            "Gere novamente os chunks antes da consulta com:\n"
            "  make chunk-data"
        )

    return lookup
