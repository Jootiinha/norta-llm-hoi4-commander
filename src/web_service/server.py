import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from web_service.metrics import (
    RunStats,
    append_metrics_sample,
    collect_metrics,
    read_cpu_snapshot,
    write_summary,
)


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs" / "runs"
DEFAULT_MODEL = "models/qwen3-0.6b"
DEFAULT_PROMPT = (
    "Monte uma build inicial para o Brasil em Hearts of Iron IV "
    "focada em industria e exercito."
)


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
    .value {
      font-size: 27px;
      font-weight: 700;
      letter-spacing: 0;
      margin-bottom: 8px;
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
    canvas {
      width: 100%;
      height: 300px;
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
    }
    @media (max-width: 620px) {
      .grid { grid-template-columns: 1fr; }
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
            <canvas id="chart" width="1100" height="340"></canvas>
            <div id="chartTooltip" class="chart-tooltip" hidden></div>
          </div>
        </div>
        <div class="section">
          <h2>Arquivos</h2>
          <div id="paths" class="muted"></div>
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
    const state = { runs: [], selected: null, detail: null, models: [] };
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
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      const left = 54;
      const right = 24;
      const top = 18;
      const bottom = 42;
      const plotW = w - left - right;
      const plotH = h - top - bottom;
      const enabledSeries = chartSeries.filter((serie) => serie.enabled);
      chartState.points = [];
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

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
      const scaleX = canvas.width / rect.width;
      const x = (event.clientX - rect.left) * scaleX;
      const left = 54;
      const right = 24;
      const plotW = canvas.width - left - right;
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
    $("refreshBtn").addEventListener("click", loadRuns);
    $("chart").addEventListener("mousemove", updateChartHover);
    $("chart").addEventListener("mouseleave", clearChartHover);

    setInterval(async () => {
      try {
        await loadRuns();
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


class Handler(BaseHTTPRequestHandler):
    manager = RunManager()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/runs":
            self.send_json(list_runs())
            return
        if parsed.path == "/api/models":
            self.send_json(list_models())
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.removeprefix("/api/runs/").strip("/")
            if not run_id:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            params = parse_qs(parsed.query)
            metrics_limit = int(params.get("metrics_limit", ["180"])[0])
            self.send_json(run_payload(run_id, metrics_limit))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            payload = self.read_json_body()
            model = str(payload.get("model") or DEFAULT_MODEL)
            prompt = str(payload.get("prompt") or DEFAULT_PROMPT)
            max_new_tokens = int(payload.get("max_new_tokens") or 300)
            metrics_enabled = bool(payload.get("metrics_enabled", True))
            metrics_interval = max(1.0, float(payload.get("metrics_interval") or 2.0))
            self.send_json(
                self.manager.start(
                    model,
                    prompt,
                    max_new_tokens,
                    metrics_enabled,
                    metrics_interval,
                ),
                HTTPStatus.CREATED,
            )
            return
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
            run_id = parsed.path.removeprefix("/api/runs/").removesuffix("/stop").strip("/")
            stopped = self.manager.stop(run_id)
            self.send_json({"stopped": stopped})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"norta-llm-hoi4-commander web monitor: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
