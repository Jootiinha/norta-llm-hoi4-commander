import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, request

from web_service.metrics import (
    RunStats,
    append_metrics_sample,
    collect_metrics,
    read_cpu_snapshot,
    write_summary,
)


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs" / "runs"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
CHUNK_SUMMARY_PATH = ROOT / "data" / "processed" / "chunks" / "summary.json"
DEFAULT_MODEL = "models/qwen3-0.6b"
DEFAULT_PROMPT = (
    "Monte uma build inicial para o Brasil em Hearts of Iron IV "
    "focada em industria e exercito."
)
app = Flask(__name__)
CHUNK_ANALYSIS_CACHE: dict[str, object] = {
    "path": None,
    "mtime_ns": None,
    "size": None,
    "payload": None,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return default


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def write_status(run_dir: Path, status: str) -> None:
    (run_dir / "status.txt").write_text(status + "\n", encoding="utf-8")


def tail_jsonl(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in read_text(path).splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def list_models() -> list[dict[str, str | bool]]:
    models_dir = ROOT / "models"
    if not models_dir.exists():
        return []

    models = []
    for path in sorted(models_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        has_config = any(
            (path / name).exists()
            for name in (
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            )
        )
        models.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)),
                "has_config": has_config,
            }
        )
    return models


class RunManager:
    def __init__(self) -> None:
        self.active: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        model: str,
        prompt: str,
        max_new_tokens: int,
        metrics_enabled: bool,
        metrics_interval: float,
    ) -> dict:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = LOG_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        console_file = run_dir / "console.log"
        metrics_file = run_dir / "metrics.jsonl"
        summary_file = run_dir / "summary.json"
        run_env = run_dir / "run.env"
        run_env.write_text(
            "\n".join(
                [
                    f"RUN_ID={run_id}",
                    f"MODEL={model}",
                    f"PROMPT={prompt}",
                    f"MAX_NEW_TOKENS={max_new_tokens}",
                    f"METRICS_ENABLED={str(metrics_enabled).lower()}",
                    f"METRICS_INTERVAL={metrics_interval}",
                    f"STARTED_AT={utc_now()}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        write_status(run_dir, "running")

        output = console_file.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-B",
                "src/norta_llm/run_model.py",
                "--model",
                model,
                "--prompt",
                prompt,
                "--max-new-tokens",
                str(max_new_tokens),
            ],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        with self.lock:
            self.active[run_id] = process

        thread = threading.Thread(
            target=self._monitor_process,
            args=(
                run_id,
                process,
                output,
                metrics_file,
                summary_file,
                run_dir,
                metrics_enabled,
                metrics_interval,
            ),
            daemon=True,
        )
        thread.start()

        return run_payload(run_id)

    def stop(self, run_id: str) -> bool:
        with self.lock:
            process = self.active.get(run_id)
        if process is None or process.poll() is not None:
            return False
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def _monitor_process(
        self,
        run_id: str,
        process: subprocess.Popen,
        output,
        metrics_file: Path,
        summary_file: Path,
        run_dir: Path,
        metrics_enabled: bool,
        metrics_interval: float,
    ) -> None:
        stats = RunStats(run_id)
        started = time.monotonic()
        previous_cpu = read_cpu_snapshot() if metrics_enabled else None
        previous_pid_ticks = None
        last_summary_write = 0.0

        while process.poll() is None and metrics_enabled and previous_cpu is not None:
            metrics, previous_cpu, previous_pid_ticks = collect_metrics(
                previous_cpu,
                previous_pid_ticks,
                process.pid,
                metrics_interval,
            )
            stats.update(metrics)
            now = time.monotonic()
            append_metrics_sample(metrics_file, run_id, metrics, now - started)
            if now - last_summary_write >= 5.0:
                write_summary(summary_file, stats)
                last_summary_write = now

        if not metrics_enabled:
            process.wait()

        ended_at = datetime.now(timezone.utc)
        write_summary(summary_file, stats, ended_at)
        output.close()
        return_code = process.returncode
        write_status(run_dir, "completed" if return_code == 0 else f"failed:{return_code}")
        with self.lock:
            self.active.pop(run_id, None)


def run_payload(run_id: str, metrics_limit: int = 180) -> dict:
    run_dir = LOG_ROOT / run_id
    env = read_env(run_dir / "run.env")
    summary = read_json(run_dir / "summary.json")
    status = read_text(run_dir / "status.txt", "unknown").strip() or "unknown"
    metrics = tail_jsonl(run_dir / "metrics.jsonl", metrics_limit)
    return {
        "id": run_id,
        "status": status,
        "env": env,
        "summary": summary,
        "metrics": metrics,
        "paths": {
            "dir": str(run_dir),
            "console": str(run_dir / "console.log"),
            "metrics": str(run_dir / "metrics.jsonl"),
            "summary": str(run_dir / "summary.json"),
        },
    }


def list_runs() -> list[dict]:
    runs = []
    if not LOG_ROOT.exists():
        return runs
    for run_dir in sorted(LOG_ROOT.iterdir(), key=lambda path: path.name, reverse=True):
        if not run_dir.is_dir():
            continue
        payload = run_payload(run_dir.name, metrics_limit=1)
        runs.append(
            {
                "id": payload["id"],
                "status": payload["status"],
                "env": payload["env"],
                "summary": payload["summary"],
                "paths": payload["paths"],
            }
        )
    return runs


def quantile(sorted_values: list[int], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def summarize_word_counts(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": float(ordered[0]),
        "q1": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.5),
        "q3": quantile(ordered, 0.75),
        "max": float(ordered[-1]),
    }


def section_label(value: object) -> str:
    if isinstance(value, list) and value:
        return " > ".join(str(item) for item in value)
    return "Sem secao"


def chunk_file_signature() -> dict[str, object]:
    if not CHUNKS_PATH.exists():
        return {
            "available": False,
            "path": str(CHUNKS_PATH),
            "summary_path": str(CHUNK_SUMMARY_PATH),
            "message": "Arquivo de chunks nao encontrado.",
            "signature": None,
        }

    try:
        stat = CHUNKS_PATH.stat()
    except OSError:
        return {
            "available": False,
            "path": str(CHUNKS_PATH),
            "summary_path": str(CHUNK_SUMMARY_PATH),
            "message": "Nao foi possivel ler o arquivo de chunks.",
            "signature": None,
        }

    return {
        "available": True,
        "path": str(CHUNKS_PATH),
        "summary_path": str(CHUNK_SUMMARY_PATH),
        "signature": {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        },
    }


def read_chunk_summary(signature: dict[str, object] | None) -> dict | None:
    if not signature or not CHUNK_SUMMARY_PATH.exists():
        return None

    summary = read_json(CHUNK_SUMMARY_PATH)
    source = summary.get("source")
    payload = summary.get("payload")
    if not isinstance(source, dict) or not isinstance(payload, dict):
        return None
    if (
        source.get("path") != str(CHUNKS_PATH)
        or source.get("mtime_ns") != signature.get("mtime_ns")
        or source.get("size") != signature.get("size")
    ):
        return None
    return payload


def write_chunk_summary(signature: dict[str, object], payload: dict) -> None:
    CHUNK_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": utc_now(),
        "source": {
            "path": str(CHUNKS_PATH),
            "mtime_ns": signature.get("mtime_ns"),
            "size": signature.get("size"),
        },
        "payload": payload,
    }
    CHUNK_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def chunk_analysis_payload() -> dict:
    metadata = chunk_file_signature()
    if not metadata.get("available"):
        return {
            "available": False,
            "path": str(CHUNKS_PATH),
            "summary_path": str(CHUNK_SUMMARY_PATH),
            "message": str(metadata.get("message") or "Arquivo de chunks indisponivel."),
            "signature": None,
        }

    signature = metadata.get("signature")
    if not isinstance(signature, dict):
        return {
            "available": False,
            "path": str(CHUNKS_PATH),
            "summary_path": str(CHUNK_SUMMARY_PATH),
            "message": "Assinatura do arquivo de chunks indisponivel.",
            "signature": None,
        }

    cache_hit = (
        CHUNK_ANALYSIS_CACHE.get("path") == str(CHUNKS_PATH)
        and CHUNK_ANALYSIS_CACHE.get("mtime_ns") == signature.get("mtime_ns")
        and CHUNK_ANALYSIS_CACHE.get("size") == signature.get("size")
        and isinstance(CHUNK_ANALYSIS_CACHE.get("payload"), dict)
    )
    if cache_hit:
        return dict(CHUNK_ANALYSIS_CACHE["payload"])

    summary_payload = read_chunk_summary(signature)
    if summary_payload is not None:
        CHUNK_ANALYSIS_CACHE.update(
            {
                "path": str(CHUNKS_PATH),
                "mtime_ns": signature.get("mtime_ns"),
                "size": signature.get("size"),
                "payload": summary_payload,
            }
        )
        return dict(summary_payload)

    total_chunks = 0
    titles: dict[str, int] = {}
    page_word_counts: dict[str, list[int]] = {}
    section_counts: dict[str, int] = {}
    histogram_bins = [
        {"label": "0-99", "min": 0, "max": 99, "count": 0},
        {"label": "100-199", "min": 100, "max": 199, "count": 0},
        {"label": "200-399", "min": 200, "max": 399, "count": 0},
        {"label": "400-699", "min": 400, "max": 699, "count": 0},
        {"label": "700-999", "min": 700, "max": 999, "count": 0},
        {"label": "1000+", "min": 1000, "max": None, "count": 0},
    ]
    scatter_points: list[dict[str, object]] = []
    word_sum = 0
    below_100 = 0
    above_1000 = 0
    word_counts: list[int] = []

    for chunk_index, row in enumerate(iter_jsonl(CHUNKS_PATH) or []):
        text = str(row.get("text") or "")
        word_count = row.get("word_count")
        if not isinstance(word_count, int):
            word_count = len(text.split())

        title = str(row.get("title") or "Sem titulo")
        section = section_label(row.get("section_path"))

        total_chunks += 1
        word_sum += word_count
        word_counts.append(word_count)
        titles[title] = titles.get(title, 0) + 1
        page_word_counts.setdefault(title, []).append(word_count)
        section_counts[section] = section_counts.get(section, 0) + 1

        if word_count < 100:
            below_100 += 1
        if word_count > 1000:
            above_1000 += 1

        for bucket in histogram_bins:
            upper = bucket["max"]
            if word_count >= bucket["min"] and (upper is None or word_count <= upper):
                bucket["count"] += 1
                break

        scatter_points.append(
            {
                "chunk_index": chunk_index,
                "word_count": word_count,
                "title": title,
                "section": section,
            }
        )

    if total_chunks == 0:
        return {
            "available": False,
            "path": str(CHUNKS_PATH),
            "summary_path": str(CHUNK_SUMMARY_PATH),
            "message": "Arquivo de chunks existe, mas nao possui registros validos.",
            "signature": signature,
        }

    step = max(1, len(scatter_points) // 4000)
    scatter_sample = scatter_points[::step]

    top_pages = sorted(titles.items(), key=lambda item: item[1], reverse=True)[:20]
    top_page_names = [title for title, _ in top_pages[:15]]
    top_page_count = top_pages[0][1] if top_pages else 0
    boxplot = [
        {
            "title": title,
            **summarize_word_counts(page_word_counts[title]),
        }
        for title in top_page_names
        if page_word_counts.get(title)
    ]
    top_sections = sorted(section_counts.items(), key=lambda item: item[1], reverse=True)[:8]

    payload = {
        "available": True,
        "path": str(CHUNKS_PATH),
        "summary_path": str(CHUNK_SUMMARY_PATH),
        "signature": signature,
        "summary": {
            "total_chunks": total_chunks,
            "total_pages": len(titles),
            "avg_words": round(word_sum / total_chunks, 2),
            "median_words": round(quantile(sorted(word_counts), 0.5), 2),
            "p90_words": round(quantile(sorted(word_counts), 0.9), 2),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "below_100": below_100,
            "below_100_ratio": round((below_100 / total_chunks) * 100, 2),
            "above_1000": above_1000,
            "above_1000_ratio": round((above_1000 / total_chunks) * 100, 2),
            "avg_chunks_per_page": round(total_chunks / max(len(titles), 1), 2),
            "top_page_share": round((top_page_count / total_chunks) * 100, 2),
        },
        "histogram": histogram_bins,
        "scatter": scatter_sample,
        "top_pages": [{"title": title, "count": count} for title, count in top_pages],
        "boxplot": boxplot,
        "top_sections": [{"section": name, "count": count} for name, count in top_sections],
    }
    CHUNK_ANALYSIS_CACHE.update(
        {
            "path": str(CHUNKS_PATH),
            "mtime_ns": signature.get("mtime_ns"),
            "size": signature.get("size"),
            "payload": payload,
        }
    )
    write_chunk_summary(signature, payload)
    return payload


INDEX_HTML = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>norta-llm-hoi4-commander</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #647184;
      --line: #d9dee7;
      --accent: #176b87;
      --accent-2: #5f6f52;
      --danger: #a43f3f;
      --warn: #a36b12;
      --ok: #2f7d4f;
      --code: #101820;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    button, input, textarea, select {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    button.danger {
      color: white;
      background: var(--danger);
      border-color: var(--danger);
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    main {
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      min-height: calc(100vh - 56px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      padding: 14px;
      overflow: auto;
    }
    .content {
      padding: 16px 18px 22px;
      overflow: auto;
    }
    .tabs {
      display: inline-flex;
      gap: 6px;
      margin-bottom: 14px;
    }
    .tab {
      border: 1px solid var(--line);
      background: white;
      color: var(--muted);
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 650;
    }
    .tab.active {
      color: white;
      background: var(--accent);
      border-color: var(--accent);
    }
    .form {
      display: grid;
      gap: 10px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 14px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      color: var(--text);
      background: white;
    }
    textarea {
      min-height: 88px;
      resize: vertical;
    }
    .runs {
      display: grid;
      gap: 8px;
    }
    .run {
      width: 100%;
      text-align: left;
      display: grid;
      gap: 3px;
    }
    .run.active {
      border-color: var(--accent);
      outline: 2px solid color-mix(in srgb, var(--accent) 20%, transparent);
    }
    .row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .muted { color: var(--muted); }
    .status {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: white;
    }
    .status.running { color: var(--warn); border-color: #dfc289; }
    .status.completed { color: var(--ok); border-color: #9cc9ad; }
    .status.failed { color: var(--danger); border-color: #d7a1a1; }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .chunk-grid {
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 6px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 112px;
    }
    .card h2 {
      margin: 0 0 7px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
    }
    .chunk-card {
      padding: 7px 8px;
      min-height: 72px;
    }
    .chunk-card h2 {
      margin-bottom: 4px;
      font-size: 10px;
      letter-spacing: 0.02em;
    }
    .value {
      font-size: 27px;
      font-weight: 700;
      letter-spacing: 0;
      margin-bottom: 8px;
    }
    .chunk-card .value {
      font-size: 18px;
      margin-bottom: 2px;
      line-height: 1.1;
    }
    .chunk-card .muted {
      font-size: 10px;
      line-height: 1.2;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-top: 12px;
    }
    .section h2 {
      margin: 0 0 10px;
      font-size: 14px;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .section-head h2 {
      margin: 0;
    }
    canvas {
      width: 100%;
      display: block;
      cursor: crosshair;
    }
    .chart-wrap {
      position: relative;
      padding-top: 2px;
    }
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      background: white;
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--text);
      cursor: pointer;
      user-select: none;
      font-size: 12px;
      font-weight: 650;
    }
    .legend-item.off {
      opacity: 0.42;
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }
    .chart-tooltip {
      position: absolute;
      min-width: 210px;
      max-width: 300px;
      pointer-events: none;
      background: #ffffff;
      border: 1px solid var(--line);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      color: var(--text);
      z-index: 2;
    }
    .chart-tooltip[hidden] {
      display: none;
    }
    .tooltip-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-top: 5px;
    }
    .inline-field {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-size: 13px;
      font-weight: 500;
    }
    .inline-field input {
      width: auto;
    }
    pre {
      margin: 0;
      min-height: 260px;
      max-height: 420px;
      overflow: auto;
      background: var(--code);
      color: #d8e0e8;
      border-radius: 6px;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .empty {
      padding: 40px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: white;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chunk-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 620px) {
      .grid { grid-template-columns: 1fr; }
      .chunk-grid { grid-template-columns: 1fr; }
      header { padding: 0 12px; }
      .content { padding: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>norta-llm-hoi4-commander</h1>
    <div class="row">
      <span id="serverStatus" class="muted">sincronizando</span>
      <button id="refreshBtn">Atualizar</button>
    </div>
  </header>
  <main>
    <aside>
      <form id="runForm" class="form">
        <label>Modelo
          <select id="modelInput"></select>
        </label>
        <label>Prompt
          <textarea id="promptInput">Monte uma build inicial para o Brasil em Hearts of Iron IV focada em industria e exercito.</textarea>
        </label>
        <label>Max tokens
          <input id="tokensInput" type="number" min="1" max="8192" value="300">
        </label>
        <label>Intervalo de metricas
          <select id="metricsIntervalInput">
            <option value="1">1 segundo</option>
            <option value="2" selected>2 segundos</option>
            <option value="5">5 segundos</option>
          </select>
        </label>
        <label class="inline-field">
          <input id="metricsEnabledInput" type="checkbox" checked>
          Metricas em tempo real
        </label>
        <button class="primary" type="submit">Iniciar execucao</button>
      </form>
      <div class="row" style="margin-bottom: 10px;">
        <strong>Execucoes</strong>
        <span id="runCount" class="muted">0</span>
      </div>
      <div id="runs" class="runs"></div>
    </aside>
    <section class="content">
      <div class="tabs">
        <button id="runsTab" class="tab active" type="button">Execucoes</button>
        <button id="chunksTab" class="tab" type="button">Chunks</button>
      </div>
      <div id="runsView">
        <div id="empty" class="empty">Inicie ou selecione uma execucao.</div>
        <div id="detail" hidden>
          <div class="row" style="margin-bottom: 12px;">
            <div>
              <strong id="runTitle"></strong>
              <div id="runMeta" class="muted"></div>
            </div>
            <button id="stopBtn" class="danger">Parar</button>
          </div>
          <div id="cards" class="grid"></div>
          <div class="section">
            <h2>Historico de uso</h2>
            <div class="chart-wrap">
              <div id="chartLegend" class="chart-legend"></div>
              <canvas id="chart" data-chart-height="340"></canvas>
              <div id="chartTooltip" class="chart-tooltip" hidden></div>
            </div>
          </div>
          <div class="section">
            <h2>Arquivos</h2>
            <div id="paths" class="muted"></div>
          </div>
        </div>
      </div>
      <div id="chunksView" hidden>
        <div id="chunkEmpty" class="empty">Carregando analise dos chunks.</div>
        <div id="chunkDetail" hidden>
          <div id="chunkCards" class="grid chunk-grid"></div>
          <div class="section">
            <div class="section-head">
              <h2>Distribuicao do tamanho dos chunks</h2>
              <span class="muted">Abaixo de 100 indica chunks curtos; acima de 1000 indica chunks grandes.</span>
            </div>
            <div class="muted" style="margin-bottom: 10px;">Use este histograma para ver a concentracao de tamanhos. Se houver muitos chunks nas faixas extremas, o corte esta agressivo demais ou permissivo demais.</div>
            <div class="chart-wrap">
              <canvas id="chunkHistogram" data-chart-height="340"></canvas>
              <div id="chunkHistogramTooltip" class="chart-tooltip" hidden></div>
            </div>
          </div>
          <div class="section">
            <h2>Tamanho dos chunks por pagina</h2>
            <div class="muted" style="margin-bottom: 10px;">O boxplot mostra a variacao por pagina. Caixas muito diferentes entre si sugerem que algumas paginas estao sendo quebradas de forma inconsistente.</div>
            <div class="chart-wrap">
              <canvas id="chunkBoxplot" data-chart-height="380"></canvas>
              <div id="chunkBoxplotTooltip" class="chart-tooltip" hidden></div>
            </div>
          </div>
          <div class="section">
            <h2>Tamanho dos chunks ao longo do dataset</h2>
            <div class="muted" style="margin-bottom: 10px;">A dispersao ajuda a detectar mudancas de comportamento ao longo da extracao. Bandas muito espalhadas ou blocos com padroes distintos indicam chunking irregular.</div>
            <div class="chart-wrap">
              <canvas id="chunkScatter" data-chart-height="320"></canvas>
              <div id="chunkScatterTooltip" class="chart-tooltip" hidden></div>
            </div>
          </div>
          <div class="section">
            <h2>Top 20 paginas com mais chunks</h2>
            <div class="muted" style="margin-bottom: 10px;">Este ranking mostra onde o volume de chunks se concentra. Paginas com quantidade muito alta merecem revisao porque podem estar fragmentadas demais.</div>
            <div class="chart-wrap">
              <canvas id="chunkTopPages" data-chart-height="420"></canvas>
              <div id="chunkTopPagesTooltip" class="chart-tooltip" hidden></div>
            </div>
          </div>
          <div class="section">
            <h2>Secoes mais frequentes</h2>
            <div class="muted" style="margin-bottom: 10px;">Aqui voce ve quais secoes aparecem mais no dataset. Isso ajuda a identificar vieses de cobertura e excesso de repeticao estrutural.</div>
            <div id="chunkSections" class="muted"></div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const chartSeries = [
      { label: "CPU", key: "cpu_percent", color: "#176b87", enabled: true },
      { label: "RAM", key: "mem_percent", color: "#5f6f52", enabled: true },
      { label: "GPU", key: "gpu_percent", color: "#a36b12", enabled: true },
      { label: "VRAM", key: "gpu_mem_percent", color: "#7f4f9f", enabled: true },
    ];
    const chartState = { hoverIndex: null, points: [] };
    const chunkChartState = {
      histogram: { hoverIndex: null, bars: [] },
      scatter: { hoverIndex: null, points: [] },
      topPages: { hoverIndex: null, bars: [] },
      boxplot: { hoverIndex: null, items: [] },
    };
    const state = {
      runs: [],
      selected: null,
      detail: null,
      models: [],
      view: "runs",
      chunkAnalysis: null,
      chunkSignature: null,
      chunkMetaSupported: true,
    };
    const $ = (id) => document.getElementById(id);

    function fmt(value, suffix = "") {
      return value === null || value === undefined ? "--" : `${Number(value).toFixed(1)}${suffix}`;
    }

    function fmtElapsed(seconds) {
      if (seconds === null || seconds === undefined) return "--";
      const total = Math.max(0, Math.round(Number(seconds)));
      const minutes = Math.floor(total / 60);
      const rest = total % 60;
      return minutes ? `${minutes}m ${String(rest).padStart(2, "0")}s` : `${rest}s`;
    }

    function statusClass(status) {
      if (status.startsWith("failed")) return "failed";
      if (status === "running") return "running";
      if (status === "completed") return "completed";
      return "";
    }

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function loadRuns() {
      state.runs = await api("/api/runs");
      $("runCount").textContent = state.runs.length;
      if (!state.selected && state.runs.length) state.selected = state.runs[0].id;
      renderRuns();
      if (state.selected) await loadDetail(state.selected);
      $("serverStatus").textContent = "online";
    }

    async function loadModels() {
      state.models = await api("/api/models");
      renderModels();
    }

    async function loadChunkAnalysis() {
      state.chunkAnalysis = await api("/api/chunk-analysis");
      state.chunkSignature = state.chunkAnalysis?.signature || null;
      renderChunkAnalysis();
    }

    async function loadChunkAnalysisMeta() {
      if (!state.chunkMetaSupported) return null;
      try {
        return await api("/api/chunk-analysis/meta");
      } catch (error) {
        if (String(error.message || "").includes("404")) {
          state.chunkMetaSupported = false;
          return null;
        }
        throw error;
      }
    }

    function sameChunkSignature(a, b) {
      if (!a && !b) return true;
      if (!a || !b) return false;
      return a.mtime_ns === b.mtime_ns && a.size === b.size;
    }

    async function refreshView(background = false) {
      if (state.view === "chunks") {
        if (!background) {
          await loadChunkAnalysis();
        } else {
          const meta = await loadChunkAnalysisMeta();
          if (!meta) return;
          const nextSignature = meta?.signature || null;
          if (!sameChunkSignature(state.chunkSignature, nextSignature)) {
            await loadChunkAnalysis();
          }
        }
      } else {
        await loadRuns();
      }
      $("serverStatus").textContent = `atualizado ${new Date().toLocaleTimeString("pt-BR")}`;
    }

    function renderModels() {
      const select = $("modelInput");
      select.innerHTML = "";
      if (!state.models.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "Nenhum modelo encontrado em models/";
        select.appendChild(option);
        select.disabled = true;
        return;
      }
      select.disabled = false;
      state.models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.path;
        option.textContent = model.has_config ? model.name : `${model.name} (sem config detectado)`;
        select.appendChild(option);
      });
      const defaultOption = [...select.options].find((option) => option.value === "models/qwen3-0.6b");
      if (defaultOption) select.value = defaultOption.value;
    }

    async function loadDetail(runId) {
      state.detail = await api(`/api/runs/${runId}`);
      state.selected = runId;
      renderRuns();
      renderDetail();
    }

    function renderRuns() {
      const root = $("runs");
      root.innerHTML = "";
      state.runs.forEach((run) => {
        const button = document.createElement("button");
        button.className = `run ${run.id === state.selected ? "active" : ""}`;
        button.type = "button";
        button.onclick = () => loadDetail(run.id);
        const model = run.env.MODEL || "";
        button.innerHTML = `
          <span class="row"><strong>${run.id}</strong><span class="status ${statusClass(run.status)}">${run.status}</span></span>
          <span class="muted">${model}</span>
        `;
        root.appendChild(button);
      });
    }

    function metricStat(name) {
      return state.detail?.summary?.metrics?.[name] || {};
    }

    function currentMetric(name) {
      const rows = state.detail?.metrics || [];
      return rows.length ? rows[rows.length - 1].metrics[name] : null;
    }

    function card(label, metricName, suffix = "") {
      const stat = metricStat(metricName);
      const current = currentMetric(metricName) ?? stat.current;
      return `
        <div class="card">
          <h2>${label}</h2>
          <div class="value">${fmt(current, suffix)}</div>
          <div class="stats">
            <span>min ${fmt(stat.min, suffix)}</span>
            <span>max ${fmt(stat.max, suffix)}</span>
            <span>med ${fmt(stat.avg, suffix)}</span>
          </div>
        </div>
      `;
    }

    function renderDetail() {
      const detail = state.detail;
      $("empty").hidden = true;
      $("detail").hidden = false;
      $("runTitle").textContent = detail.id;
      const metricsMode = detail.env.METRICS_ENABLED === "false"
        ? "metricas desligadas"
        : `metricas a cada ${detail.env.METRICS_INTERVAL || "?"}s`;
      $("runMeta").textContent = `${detail.env.MODEL || ""} | ${detail.status} | ${metricsMode}`;
      $("stopBtn").disabled = detail.status !== "running";
      $("cards").innerHTML = [
        card("CPU", "cpu_percent", "%"),
        card("RAM", "mem_percent", "%"),
        card("GPU", "gpu_percent", "%"),
        card("VRAM", "gpu_mem_percent", "%"),
        card("Processo CPU", "pid_cpu_percent", "%"),
        card("Processo RAM", "pid_mem_mb", " MB"),
        card("RAM usada", "mem_used_gb", " GB"),
        card("Temp CPU", "cpu_temp_c", " C"),
      ].join("");
      $("paths").innerHTML = Object.entries(detail.paths || {})
        .map(([key, value]) => `<div><strong>${key}</strong>: ${value}</div>`)
        .join("");
      drawChart(detail.metrics || []);
    }

    function setView(view) {
      state.view = view;
      $("runsTab").classList.toggle("active", view === "runs");
      $("chunksTab").classList.toggle("active", view === "chunks");
      $("runsView").hidden = view !== "runs";
      $("chunksView").hidden = view !== "chunks";
      if (view === "chunks") {
        if (state.chunkAnalysis) {
          renderChunkAnalysis();
          loadChunkAnalysisMeta()
            .then((meta) => {
              if (!meta) return null;
              const nextSignature = meta?.signature || null;
              if (!sameChunkSignature(state.chunkSignature, nextSignature)) {
                return loadChunkAnalysis();
              }
              return null;
            })
            .catch(() => null);
        } else {
          loadChunkAnalysis().catch((error) => {
            $("chunkEmpty").hidden = false;
            $("chunkDetail").hidden = true;
            $("chunkEmpty").textContent = error.message;
          });
        }
      }
    }

    function chunkCard(label, value, hint = "") {
      return `
        <div class="card chunk-card">
          <h2>${label}</h2>
          <div class="value">${value}</div>
          <div class="muted">${hint}</div>
        </div>
      `;
    }

    function renderChunkAnalysis() {
      const payload = state.chunkAnalysis;
      if (!payload || !payload.available) {
        $("chunkEmpty").hidden = false;
        $("chunkDetail").hidden = true;
        $("chunkEmpty").textContent = payload?.message || "Analise de chunks indisponivel.";
        return;
      }

      $("chunkEmpty").hidden = true;
      $("chunkDetail").hidden = false;
      const summary = payload.summary || {};
      $("chunkCards").innerHTML = [
        chunkCard("Chunks", Number(summary.total_chunks || 0).toLocaleString("pt-BR")),
        chunkCard("Paginas", Number(summary.total_pages || 0).toLocaleString("pt-BR")),
        chunkCard("Media de palavras", fmt(summary.avg_words || 0)),
        chunkCard("Mediana", fmt(summary.median_words || 0), "Ponto central real da distribuicao"),
        chunkCard("P90", fmt(summary.p90_words || 0), "90% dos chunks ficam abaixo deste valor"),
        chunkCard("Maximo", Number(summary.max_words || 0).toLocaleString("pt-BR")),
        chunkCard("Chunks curtos", Number(summary.below_100 || 0).toLocaleString("pt-BR"), `${fmt(summary.below_100_ratio || 0, "%")} abaixo de 100 palavras`),
        chunkCard("Chunks longos", Number(summary.above_1000 || 0).toLocaleString("pt-BR"), `${fmt(summary.above_1000_ratio || 0, "%")} acima de 1000 palavras`),
        chunkCard("Chunks por pagina", fmt(summary.avg_chunks_per_page || 0), "Media geral de fragmentacao"),
        chunkCard("Concentracao da maior pagina", fmt(summary.top_page_share || 0, "%"), "Parcela dos chunks que vem da pagina mais fragmentada"),
      ].join("");
      $("chunkSections").innerHTML = (payload.top_sections || [])
        .map((item) => `<div><strong>${item.section}</strong>: ${item.count} chunks</div>`)
        .join("");
      drawChunkHistogram(payload.histogram || []);
      drawChunkScatter(payload.scatter || []);
      drawChunkTopPages(payload.top_pages || []);
      drawChunkBoxplot(payload.boxplot || []);
    }

    function renderChartLegend() {
      $("chartLegend").innerHTML = chartSeries.map((serie, index) => `
        <button class="legend-item ${serie.enabled ? "" : "off"}" type="button" data-index="${index}">
          <span class="legend-swatch" style="background:${serie.color}"></span>
          ${serie.label}
        </button>
      `).join("");
      document.querySelectorAll(".legend-item").forEach((button) => {
        button.onclick = () => {
          const index = Number(button.dataset.index);
          chartSeries[index].enabled = !chartSeries[index].enabled;
          renderChartLegend();
          drawChart(state.detail?.metrics || []);
        };
      });
    }

    function drawChart(rows) {
      const canvas = $("chart");
      const { ctx, width: w, height: h } = clearCanvas(canvas);
      const left = 54;
      const right = 24;
      const top = 18;
      const bottom = 42;
      const plotW = w - left - right;
      const plotH = h - top - bottom;
      const enabledSeries = chartSeries.filter((serie) => serie.enabled);
      chartState.points = [];

      ctx.fillStyle = "#fbfcfd";
      ctx.fillRect(left, top, plotW, plotH);
      ctx.strokeStyle = "#d9dee7";
      ctx.lineWidth = 1;
      ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      for (let i = 0; i <= 4; i++) {
        const y = top + i * (plotH / 4);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(w - right, y);
        ctx.stroke();
        ctx.fillStyle = "#647184";
        ctx.textAlign = "right";
        ctx.fillText(`${100 - i * 25}%`, left - 10, y + 4);
      }
      ctx.strokeStyle = "#9aa4b2";
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, top + plotH);
      ctx.lineTo(left + plotW, top + plotH);
      ctx.stroke();

      if (!rows.length) {
        ctx.fillStyle = "#647184";
        ctx.textAlign = "left";
        ctx.fillText("Sem amostras para exibir.", left, h / 2);
        return;
      }

      const xLabelIndexes = [...new Set([0, Math.floor((rows.length - 1) / 2), rows.length - 1])];
      ctx.fillStyle = "#647184";
      ctx.textAlign = "center";
      xLabelIndexes.forEach((index) => {
        const row = rows[index];
        const x = left + (index / Math.max(rows.length - 1, 1)) * plotW;
        ctx.fillText(fmtElapsed(row.elapsed_seconds), x, h - 12);
      });

      enabledSeries.forEach((serie) => {
        ctx.strokeStyle = serie.color;
        ctx.lineWidth = 2.4;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        rows.forEach((row, i) => {
          const value = row.metrics[serie.key] ?? 0;
          const x = left + (i / Math.max(rows.length - 1, 1)) * plotW;
          const y = top + (1 - Math.max(0, Math.min(100, value)) / 100) * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        const pointStep = rows.length > 90 ? Math.ceil(rows.length / 90) : 1;
        rows.forEach((row, i) => {
          if (i % pointStep !== 0 && chartState.hoverIndex !== i) return;
          const value = row.metrics[serie.key] ?? 0;
          const x = left + (i / Math.max(rows.length - 1, 1)) * plotW;
          const y = top + (1 - Math.max(0, Math.min(100, value)) / 100) * plotH;
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = serie.color;
          ctx.lineWidth = chartState.hoverIndex === i ? 2.4 : 1.2;
          ctx.beginPath();
          ctx.arc(x, y, chartState.hoverIndex === i ? 5 : 2.4, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          chartState.points.push({ index: i, x, y, serie, row, value });
        });
      });

      if (chartState.hoverIndex !== null && rows[chartState.hoverIndex]) {
        const x = left + (chartState.hoverIndex / Math.max(rows.length - 1, 1)) * plotW;
        ctx.strokeStyle = "rgba(23, 32, 42, 0.45)";
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    function updateChartHover(event) {
      const rows = state.detail?.metrics || [];
      const canvas = $("chart");
      const tooltip = $("chartTooltip");
      if (!rows.length) {
        tooltip.hidden = true;
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const left = 54;
      const right = 24;
      const plotW = rect.width - left - right;
      const rawIndex = Math.round(((x - left) / plotW) * Math.max(rows.length - 1, 1));
      const index = Math.max(0, Math.min(rows.length - 1, rawIndex));
      chartState.hoverIndex = index;
      drawChart(rows);

      const row = rows[index];
      const timestamp = row.timestamp || "";
      const values = chartSeries
        .filter((serie) => serie.enabled)
        .map((serie) => `
          <div class="tooltip-row">
            <span><span class="legend-swatch" style="background:${serie.color}"></span> ${serie.label}</span>
            <strong>${fmt(row.metrics[serie.key], "%")}</strong>
          </div>
        `)
        .join("");
      tooltip.innerHTML = `<strong>Amostra ${index + 1}</strong><div class="muted">${fmtElapsed(row.elapsed_seconds)} | ${timestamp}</div>${values}`;
      tooltip.hidden = false;
      const localX = canvas.offsetLeft + event.clientX - rect.left;
      const localY = canvas.offsetTop + event.clientY - rect.top;
      const maxLeft = canvas.offsetLeft + rect.width - 318;
      tooltip.style.left = `${Math.max(8, Math.min(localX + 14, maxLeft))}px`;
      tooltip.style.top = `${Math.max(localY - 18, 8)}px`;
    }

    function clearChartHover() {
      chartState.hoverIndex = null;
      $("chartTooltip").hidden = true;
      drawChart(state.detail?.metrics || []);
    }

    function positionTooltip(canvas, tooltip, event) {
      const rect = canvas.getBoundingClientRect();
      const localX = canvas.offsetLeft + event.clientX - rect.left;
      const localY = canvas.offsetTop + event.clientY - rect.top;
      const maxLeft = canvas.offsetLeft + rect.width - 318;
      tooltip.style.left = `${Math.max(8, Math.min(localX + 14, maxLeft))}px`;
      tooltip.style.top = `${Math.max(localY - 18, 8)}px`;
    }

    function drawAxisFrame(ctx, left, top, plotW, plotH) {
      ctx.strokeStyle = "#9aa4b2";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, top + plotH);
      ctx.lineTo(left + plotW, top + plotH);
      ctx.stroke();
    }

    function clearCanvas(canvas) {
      const { ctx, width, height } = prepareCanvas(canvas);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      return { ctx, width, height };
    }

    function prepareCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const cssWidth = Math.max(320, Math.round(rect.width || canvas.parentElement?.clientWidth || 320));
      const cssHeight = Math.max(180, Number(canvas.dataset.chartHeight || 300));
      const dpr = window.devicePixelRatio || 1;
      const pixelWidth = Math.round(cssWidth * dpr);
      const pixelHeight = Math.round(cssHeight * dpr);
      canvas.style.height = `${cssHeight}px`;
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, width: cssWidth, height: cssHeight };
    }

    function redrawVisibleCharts() {
      if (!$("detail").hidden && state.detail) {
        drawChart(state.detail.metrics || []);
      }
      if (!$("chunkDetail").hidden && state.chunkAnalysis?.available) {
        drawChunkHistogram(state.chunkAnalysis.histogram || []);
        drawChunkScatter(state.chunkAnalysis.scatter || []);
        drawChunkTopPages(state.chunkAnalysis.top_pages || []);
        drawChunkBoxplot(state.chunkAnalysis.boxplot || []);
      }
    }

    function drawChunkHistogram(rows) {
      const canvas = $("chunkHistogram");
      const { ctx, width: canvasWidth, height: canvasHeight } = clearCanvas(canvas);
      chunkChartState.histogram.bars = [];
      const left = 52;
      const right = 18;
      const top = 18;
      const bottom = 40;
      const plotW = canvasWidth - left - right;
      const plotH = canvasHeight - top - bottom;
      const maxCount = Math.max(...rows.map((row) => row.count || 0), 1);

      ctx.fillStyle = "#fbfcfd";
      ctx.fillRect(left, top, plotW, plotH);
      ctx.strokeStyle = "#d9dee7";
      for (let i = 0; i <= 4; i++) {
        const y = top + i * (plotH / 4);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + plotW, y);
        ctx.stroke();
      }

      const barW = plotW / Math.max(rows.length, 1);
      ctx.textAlign = "center";
      ctx.font = "12px system-ui, sans-serif";
      rows.forEach((row, index) => {
        const height = ((row.count || 0) / maxCount) * (plotH - 10);
        const x = left + index * barW + 8;
        const y = top + plotH - height;
        const alert = row.max !== null && row.max <= 99;
        const high = row.min >= 1000;
        const width = Math.max(18, barW - 16);
        const hovered = chunkChartState.histogram.hoverIndex === index;
        ctx.fillStyle = alert ? "#a36b12" : high ? "#a43f3f" : "#176b87";
        ctx.globalAlpha = hovered ? 1 : 0.88;
        ctx.fillRect(x, y, width, height);
        ctx.globalAlpha = 1;
        if (hovered) {
          ctx.strokeStyle = "#17202a";
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, width, height);
        }
        ctx.fillStyle = "#647184";
        ctx.fillText(row.label, x + width / 2, canvasHeight - 14);
        chunkChartState.histogram.bars.push({ index, x, y, width, height, row });
      });
    }

    function drawChunkScatter(rows) {
      const canvas = $("chunkScatter");
      const { ctx, width: canvasWidth, height: canvasHeight } = clearCanvas(canvas);
      chunkChartState.scatter.points = [];
      const left = 52;
      const right = 18;
      const top = 18;
      const bottom = 40;
      const plotW = canvasWidth - left - right;
      const plotH = canvasHeight - top - bottom;
      const maxWord = Math.max(...rows.map((row) => row.word_count || 0), 1000);
      const maxIndex = Math.max(...rows.map((row) => row.chunk_index || 0), 1);

      ctx.fillStyle = "#fbfcfd";
      ctx.fillRect(left, top, plotW, plotH);
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillStyle = "#647184";
      ctx.textAlign = "right";
      for (let i = 0; i <= 4; i++) {
        const value = Math.round((maxWord * (4 - i)) / 4);
        const y = top + i * (plotH / 4);
        ctx.strokeStyle = "#d9dee7";
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + plotW, y);
        ctx.stroke();
        ctx.fillText(String(value), left - 8, y + 4);
      }
      [100, 1000].forEach((limit, idx) => {
        const y = top + (1 - Math.min(limit / maxWord, 1)) * plotH;
        ctx.strokeStyle = idx === 0 ? "#a36b12" : "#a43f3f";
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + plotW, y);
        ctx.stroke();
      });
      ctx.setLineDash([]);
      drawAxisFrame(ctx, left, top, plotW, plotH);

      const xTicks = [0, Math.round(maxIndex / 2), maxIndex];
      ctx.fillStyle = "#647184";
      ctx.textAlign = "center";
      xTicks.forEach((tick) => {
        const x = left + (tick / Math.max(maxIndex, 1)) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, top + plotH);
        ctx.lineTo(x, top + plotH + 5);
        ctx.strokeStyle = "#9aa4b2";
        ctx.stroke();
        ctx.fillText(String(tick), x, canvasHeight - 12);
      });

      rows.forEach((row) => {
        const x = left + ((row.chunk_index || 0) / maxIndex) * plotW;
        const y = top + (1 - ((row.word_count || 0) / maxWord)) * plotH;
        const hovered = chunkChartState.scatter.hoverIndex === chunkChartState.scatter.points.length;
        ctx.fillStyle = row.word_count > 1000 ? "#a43f3f" : row.word_count < 100 ? "#a36b12" : "#176b87";
        ctx.beginPath();
        ctx.arc(x, y, hovered ? 5 : 2.4, 0, Math.PI * 2);
        ctx.fill();
        if (hovered) {
          ctx.strokeStyle = "#17202a";
          ctx.lineWidth = 1.4;
          ctx.stroke();
        }
        chunkChartState.scatter.points.push({ x, y, row });
      });
    }

    function drawChunkTopPages(rows) {
      const canvas = $("chunkTopPages");
      const { ctx, width: canvasWidth, height: canvasHeight } = clearCanvas(canvas);
      chunkChartState.topPages.bars = [];
      const left = 220;
      const right = 24;
      const top = 18;
      const bottom = 18;
      const plotW = canvasWidth - left - right;
      const plotH = canvasHeight - top - bottom;
      const barH = plotH / Math.max(rows.length, 1);
      const maxCount = Math.max(...rows.map((row) => row.count || 0), 1);

      ctx.font = "12px system-ui, sans-serif";
      rows.forEach((row, index) => {
        const y = top + index * barH;
        const width = ((row.count || 0) / maxCount) * (plotW - 20);
        const height = Math.max(16, barH - 12);
        const hovered = chunkChartState.topPages.hoverIndex === index;
        ctx.fillStyle = "#176b87";
        ctx.globalAlpha = hovered ? 1 : 0.88;
        ctx.fillRect(left, y + 6, width, height);
        ctx.globalAlpha = 1;
        if (hovered) {
          ctx.strokeStyle = "#17202a";
          ctx.lineWidth = 2;
          ctx.strokeRect(left, y + 6, width, height);
        }
        ctx.fillStyle = "#17202a";
        ctx.textAlign = "right";
        ctx.fillText(row.title, left - 10, y + height);
        ctx.textAlign = "left";
        ctx.fillText(String(row.count), left + width + 6, y + height);
        chunkChartState.topPages.bars.push({ index, x: left, y: y + 6, width, height, row });
      });
    }

    function drawChunkBoxplot(rows) {
      const canvas = $("chunkBoxplot");
      const { ctx, width: canvasWidth, height: canvasHeight } = clearCanvas(canvas);
      chunkChartState.boxplot.items = [];
      const left = 54;
      const right = 18;
      const top = 18;
      const bottom = 92;
      const plotW = canvasWidth - left - right;
      const plotH = canvasHeight - top - bottom;
      const maxValue = Math.max(...rows.map((row) => row.max || 0), 1000);

      ctx.fillStyle = "#fbfcfd";
      ctx.fillRect(left, top, plotW, plotH);
      ctx.strokeStyle = "#d9dee7";
      ctx.font = "11px system-ui, sans-serif";
      for (let i = 0; i <= 4; i++) {
        const y = top + i * (plotH / 4);
        const value = Math.round((maxValue * (4 - i)) / 4);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + plotW, y);
        ctx.stroke();
        ctx.fillStyle = "#647184";
        ctx.textAlign = "right";
        ctx.fillText(String(value), left - 8, y + 4);
      }
      drawAxisFrame(ctx, left, top, plotW, plotH);

      const step = plotW / Math.max(rows.length, 1);
      const labelStep = Math.max(1, Math.ceil(rows.length / Math.max(4, Math.floor(plotW / 120))));
      rows.forEach((row, index) => {
        const x = left + index * step + step / 2;
        const toY = (value) => top + (1 - (value / maxValue)) * plotH;
        const minY = toY(row.min || 0);
        const q1Y = toY(row.q1 || 0);
        const medianY = toY(row.median || 0);
        const q3Y = toY(row.q3 || 0);
        const maxY = toY(row.max || 0);
        const hovered = chunkChartState.boxplot.hoverIndex === index;

        ctx.strokeStyle = hovered ? "#176b87" : "#17202a";
        ctx.lineWidth = hovered ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, minY);
        ctx.lineTo(x, maxY);
        ctx.stroke();

        ctx.fillStyle = hovered ? "#9fc3d3" : "#c8d9e2";
        ctx.fillRect(x - 12, q3Y, 24, Math.max(4, q1Y - q3Y));
        ctx.strokeStyle = "#176b87";
        ctx.strokeRect(x - 12, q3Y, 24, Math.max(4, q1Y - q3Y));

        ctx.strokeStyle = "#a43f3f";
        ctx.beginPath();
        ctx.moveTo(x - 12, medianY);
        ctx.lineTo(x + 12, medianY);
        ctx.stroke();

        if (index % labelStep === 0 || index === rows.length - 1) {
          ctx.beginPath();
          ctx.moveTo(x, top + plotH);
          ctx.lineTo(x, top + plotH + 5);
          ctx.strokeStyle = "#9aa4b2";
          ctx.stroke();
          ctx.save();
          ctx.translate(x, canvasHeight - 10);
          ctx.rotate(-Math.PI / 4);
          ctx.fillStyle = "#647184";
          ctx.textAlign = "right";
          const label = row.title.length > 24 ? `${row.title.slice(0, 24)}...` : row.title;
          ctx.fillText(label, 0, 0);
          ctx.restore();
        }
        chunkChartState.boxplot.items.push({
          index,
          x,
          minY,
          q1Y,
          medianY,
          q3Y,
          maxY,
          width: 24,
          row,
        });
      });
    }

    function updateChunkHistogramHover(event) {
      const rows = state.chunkAnalysis?.histogram || [];
      const canvas = $("chunkHistogram");
      const tooltip = $("chunkHistogramTooltip");
      if (!rows.length) {
        tooltip.hidden = true;
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const hit = chunkChartState.histogram.bars.find((bar) => x >= bar.x && x <= bar.x + bar.width);
      chunkChartState.histogram.hoverIndex = hit ? hit.index : null;
      drawChunkHistogram(rows);
      if (!hit) {
        tooltip.hidden = true;
        return;
      }
      tooltip.innerHTML = `
        <strong>Faixa ${hit.row.label}</strong>
        <div class="muted">${hit.row.min} a ${hit.row.max === null ? "acima" : hit.row.max} palavras</div>
        <div class="tooltip-row"><span>Chunks</span><strong>${hit.row.count}</strong></div>
      `;
      tooltip.hidden = false;
      positionTooltip(canvas, tooltip, event);
    }

    function clearChunkHistogramHover() {
      chunkChartState.histogram.hoverIndex = null;
      $("chunkHistogramTooltip").hidden = true;
      drawChunkHistogram(state.chunkAnalysis?.histogram || []);
    }

    function updateChunkScatterHover(event) {
      const rows = state.chunkAnalysis?.scatter || [];
      const canvas = $("chunkScatter");
      const tooltip = $("chunkScatterTooltip");
      if (!rows.length) {
        tooltip.hidden = true;
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let bestIndex = null;
      let bestDistance = 999999;
      chunkChartState.scatter.points.forEach((point, index) => {
        const distance = Math.hypot(point.x - x, point.y - y);
        if (distance < bestDistance && distance <= 12) {
          bestDistance = distance;
          bestIndex = index;
        }
      });
      chunkChartState.scatter.hoverIndex = bestIndex;
      drawChunkScatter(rows);
      if (bestIndex === null) {
        tooltip.hidden = true;
        return;
      }
      const point = chunkChartState.scatter.points[bestIndex];
      tooltip.innerHTML = `
        <strong>${point.row.title}</strong>
        <div class="muted">${point.row.section}</div>
        <div class="tooltip-row"><span>Chunk</span><strong>${point.row.chunk_index}</strong></div>
        <div class="tooltip-row"><span>Palavras</span><strong>${point.row.word_count}</strong></div>
      `;
      tooltip.hidden = false;
      positionTooltip(canvas, tooltip, event);
    }

    function clearChunkScatterHover() {
      chunkChartState.scatter.hoverIndex = null;
      $("chunkScatterTooltip").hidden = true;
      drawChunkScatter(state.chunkAnalysis?.scatter || []);
    }

    function updateChunkTopPagesHover(event) {
      const rows = state.chunkAnalysis?.top_pages || [];
      const canvas = $("chunkTopPages");
      const tooltip = $("chunkTopPagesTooltip");
      if (!rows.length) {
        tooltip.hidden = true;
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const y = event.clientY - rect.top;
      const hit = chunkChartState.topPages.bars.find((bar) => y >= bar.y && y <= bar.y + bar.height);
      chunkChartState.topPages.hoverIndex = hit ? hit.index : null;
      drawChunkTopPages(rows);
      if (!hit) {
        tooltip.hidden = true;
        return;
      }
      tooltip.innerHTML = `
        <strong>${hit.row.title}</strong>
        <div class="tooltip-row"><span>Chunks</span><strong>${hit.row.count}</strong></div>
      `;
      tooltip.hidden = false;
      positionTooltip(canvas, tooltip, event);
    }

    function clearChunkTopPagesHover() {
      chunkChartState.topPages.hoverIndex = null;
      $("chunkTopPagesTooltip").hidden = true;
      drawChunkTopPages(state.chunkAnalysis?.top_pages || []);
    }

    function updateChunkBoxplotHover(event) {
      const rows = state.chunkAnalysis?.boxplot || [];
      const canvas = $("chunkBoxplot");
      const tooltip = $("chunkBoxplotTooltip");
      if (!rows.length) {
        tooltip.hidden = true;
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      let hit = null;
      let bestDistance = 999999;
      chunkChartState.boxplot.items.forEach((item) => {
        const distance = Math.abs(item.x - x);
        if (distance < bestDistance && distance <= 18) {
          bestDistance = distance;
          hit = item;
        }
      });
      chunkChartState.boxplot.hoverIndex = hit ? hit.index : null;
      drawChunkBoxplot(rows);
      if (!hit) {
        tooltip.hidden = true;
        return;
      }
      tooltip.innerHTML = `
        <strong>${hit.row.title}</strong>
        <div class="tooltip-row"><span>Min</span><strong>${hit.row.min}</strong></div>
        <div class="tooltip-row"><span>Q1</span><strong>${hit.row.q1}</strong></div>
        <div class="tooltip-row"><span>Mediana</span><strong>${hit.row.median}</strong></div>
        <div class="tooltip-row"><span>Q3</span><strong>${hit.row.q3}</strong></div>
        <div class="tooltip-row"><span>Max</span><strong>${hit.row.max}</strong></div>
      `;
      tooltip.hidden = false;
      positionTooltip(canvas, tooltip, event);
    }

    function clearChunkBoxplotHover() {
      chunkChartState.boxplot.hoverIndex = null;
      $("chunkBoxplotTooltip").hidden = true;
      drawChunkBoxplot(state.chunkAnalysis?.boxplot || []);
    }

    $("runForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = {
        model: $("modelInput").value,
        prompt: $("promptInput").value,
        max_new_tokens: Number($("tokensInput").value || 300),
        metrics_enabled: $("metricsEnabledInput").checked,
        metrics_interval: Number($("metricsIntervalInput").value || 2),
      };
      if (!payload.model) {
        $("serverStatus").textContent = "nenhum modelo disponivel";
        return;
      }
      const run = await api("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.selected = run.id;
      await loadRuns();
    });

    $("stopBtn").addEventListener("click", async () => {
      if (!state.selected) return;
      await api(`/api/runs/${state.selected}/stop`, { method: "POST" });
      await loadRuns();
    });
    $("runsTab").addEventListener("click", () => setView("runs"));
    $("chunksTab").addEventListener("click", async () => {
      setView("chunks");
      if (!state.chunkAnalysis) await loadChunkAnalysis();
    });
    $("refreshBtn").addEventListener("click", refreshView);
    $("chart").addEventListener("mousemove", updateChartHover);
    $("chart").addEventListener("mouseleave", clearChartHover);
    $("chunkHistogram").addEventListener("mousemove", updateChunkHistogramHover);
    $("chunkHistogram").addEventListener("mouseleave", clearChunkHistogramHover);
    $("chunkScatter").addEventListener("mousemove", updateChunkScatterHover);
    $("chunkScatter").addEventListener("mouseleave", clearChunkScatterHover);
    $("chunkTopPages").addEventListener("mousemove", updateChunkTopPagesHover);
    $("chunkTopPages").addEventListener("mouseleave", clearChunkTopPagesHover);
    $("chunkBoxplot").addEventListener("mousemove", updateChunkBoxplotHover);
    $("chunkBoxplot").addEventListener("mouseleave", clearChunkBoxplotHover);
    window.addEventListener("resize", () => {
      clearTimeout(window.__nortaChartResizeTimer);
      window.__nortaChartResizeTimer = setTimeout(redrawVisibleCharts, 80);
    });

    setInterval(async () => {
      try {
        await refreshView(true);
      } catch (error) {
        $("serverStatus").textContent = "offline";
      }
    }, 1500);
    renderChartLegend();
    Promise.all([loadModels(), loadRuns()]).catch((error) => {
      $("serverStatus").textContent = error.message;
    });
  </script>
</body>
</html>
"""


manager = RunManager()


@app.get("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/api/runs")
def api_runs() -> Response:
    return jsonify(list_runs())


@app.get("/api/models")
def api_models() -> Response:
    return jsonify(list_models())


@app.get("/api/chunk-analysis")
def api_chunk_analysis() -> Response:
    return jsonify(chunk_analysis_payload())


@app.get("/api/chunk-analysis/meta")
def api_chunk_analysis_meta() -> Response:
    return jsonify(chunk_file_signature())


@app.get("/api/runs/<run_id>")
def api_run_detail(run_id: str) -> Response:
    metrics_limit = request.args.get("metrics_limit", default=180, type=int) or 180
    return jsonify(run_payload(run_id, metrics_limit))


@app.post("/api/runs")
def api_start_run() -> tuple[Response, int]:
    payload = request.get_json(silent=True) or {}
    model = str(payload.get("model") or DEFAULT_MODEL)
    prompt = str(payload.get("prompt") or DEFAULT_PROMPT)
    max_new_tokens = int(payload.get("max_new_tokens") or 300)
    metrics_enabled = bool(payload.get("metrics_enabled", True))
    metrics_interval = max(1.0, float(payload.get("metrics_interval") or 2.0))
    return (
        jsonify(
            manager.start(
                model,
                prompt,
                max_new_tokens,
                metrics_enabled,
                metrics_interval,
            )
        ),
        201,
    )


@app.post("/api/runs/<run_id>/stop")
def api_stop_run(run_id: str) -> Response:
    return jsonify({"stopped": manager.stop(run_id)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEB_PORT", "8000")))
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.environ.get("WEB_RELOAD", "").lower() in {"1", "true", "yes"},
        help="Reinicia o servidor automaticamente quando houver mudancas em src/.",
    )
    parser.add_argument(
        "--no-reload",
        action="store_false",
        dest="reload",
        help="Desabilita autoreload do Flask.",
    )
    args = parser.parse_args()
    if os.environ.get("WEB_RELOAD") == "0":
        args.reload = False
    return args


def main() -> None:
    args = parse_args()
    print(f"norta-llm-hoi4-commander web monitor: http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.reload, use_reloader=args.reload)


if __name__ == "__main__":
    main()
