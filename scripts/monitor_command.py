import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa um comando e registra metricas de CPU, memoria, disco e GPU.",
    )
    parser.add_argument("--name", default="command", help="Nome usado na pasta do relatorio.")
    parser.add_argument("--output-dir", type=Path, default=Path("metrics"), help="Diretorio base dos relatorios.")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo entre amostras, em segundos.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Comando a executar depois de --.")
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("informe o comando depois de --")
    if args.interval <= 0:
        parser.error("--interval deve ser maior que zero")
    return args


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower())
    return cleaned.strip("-") or "command"


def read_system_cpu() -> tuple[int, int] | None:
    try:
        values = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    except OSError:
        return None

    numbers = [int(value) for value in values]
    idle = numbers[3] + (numbers[4] if len(numbers) > 4 else 0)
    total = sum(numbers)
    return idle, total


def system_cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    previous_idle, previous_total = previous
    current_idle, current_total = current
    total_delta = current_total - previous_total
    idle_delta = current_idle - previous_idle
    if total_delta <= 0:
        return None
    return 100.0 * (1.0 - idle_delta / total_delta)


def list_process_tree(root_pid: int) -> list[int]:
    parent_by_pid: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue

        parts = stat.rsplit(")", 1)
        if len(parts) != 2:
            continue
        fields = parts[1].strip().split()
        if len(fields) < 2:
            continue

        pid = int(entry.name)
        ppid = int(fields[1])
        parent_by_pid[pid] = ppid

    descendants = [root_pid]
    seen = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_by_pid.items():
            if pid not in seen and ppid in seen:
                seen.add(pid)
                descendants.append(pid)
                changed = True
    return descendants


def read_process_stats(pid: int) -> dict[str, int] | None:
    proc_dir = Path("/proc") / str(pid)
    try:
        stat = (proc_dir / "stat").read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        fields = stat.rsplit(")", 1)[1].strip().split()
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        rss_pages = int(fields[21])
    except (IndexError, ValueError):
        return None

    read_bytes = 0
    write_bytes = 0
    try:
        for line in (proc_dir / "io").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key == "read_bytes":
                read_bytes = int(value.strip())
            elif key == "write_bytes":
                write_bytes = int(value.strip())
    except OSError:
        pass

    return {
        "cpu_seconds": (user_ticks + system_ticks) / CLOCK_TICKS,
        "rss_bytes": rss_pages * PAGE_SIZE,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
    }


def aggregate_process_stats(root_pid: int) -> dict[str, Any]:
    pids = list_process_tree(root_pid)
    stats = [read_process_stats(pid) for pid in pids]
    stats = [item for item in stats if item is not None]
    return {
        "process_count": len(stats),
        "cpu_seconds": sum(item["cpu_seconds"] for item in stats),
        "rss_bytes": sum(item["rss_bytes"] for item in stats),
        "read_bytes": sum(item["read_bytes"] for item in stats),
        "write_bytes": sum(item["write_bytes"] for item in stats),
    }


def read_gpu_stats() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return {"gpu_available": False}

    if result.returncode != 0:
        return {"gpu_available": False}

    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_percent": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                }
            )
        except ValueError:
            continue

    if not rows:
        return {"gpu_available": False}

    return {
        "gpu_available": True,
        "gpu_count": len(rows),
        "gpu_util_percent_max": max(row["util_percent"] for row in rows),
        "gpu_memory_used_mb_sum": sum(row["memory_used_mb"] for row in rows),
        "gpu_memory_total_mb_sum": sum(row["memory_total_mb"] for row in rows),
        "gpu_names": "; ".join(row["name"] for row in rows),
    }


def bytes_to_mb(value: float) -> float:
    return value / 1024 / 1024


def make_sample(
    process: subprocess.Popen[Any],
    start_time: float,
    previous_process_cpu: float | None,
    previous_sample_time: float | None,
    previous_system_cpu: tuple[int, int] | None,
) -> tuple[dict[str, Any], float, float, tuple[int, int] | None]:
    now = time.time()
    process_stats = aggregate_process_stats(process.pid)
    current_system_cpu = read_system_cpu()
    gpu_stats = read_gpu_stats()

    cpu_percent = None
    if previous_process_cpu is not None and previous_sample_time is not None:
        elapsed = now - previous_sample_time
        if elapsed > 0:
            cpu_percent = 100.0 * (process_stats["cpu_seconds"] - previous_process_cpu) / elapsed

    sample = {
        "timestamp": iso_now(),
        "elapsed_seconds": round(now - start_time, 3),
        "process_running": process.poll() is None,
        "process_count": process_stats["process_count"],
        "process_cpu_percent": cpu_percent,
        "system_cpu_percent": system_cpu_percent(previous_system_cpu, current_system_cpu),
        "rss_mb": bytes_to_mb(process_stats["rss_bytes"]),
        "disk_read_mb": bytes_to_mb(process_stats["read_bytes"]),
        "disk_write_mb": bytes_to_mb(process_stats["write_bytes"]),
        **gpu_stats,
    }
    return sample, process_stats["cpu_seconds"], now, current_system_cpu


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    if not samples:
        return

    columns: list[str] = []
    for sample in samples:
        for key in sample:
            if key not in columns:
                columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(samples)


def build_summary(
    command: list[str],
    returncode: int,
    samples: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    numeric_defaults = {
        "process_cpu_percent": 0.0,
        "system_cpu_percent": 0.0,
        "rss_mb": 0.0,
        "disk_read_mb": 0.0,
        "disk_write_mb": 0.0,
        "gpu_util_percent_max": 0.0,
        "gpu_memory_used_mb_sum": 0.0,
    }

    def max_value(key: str) -> float:
        values = [sample.get(key) for sample in samples if isinstance(sample.get(key), int | float)]
        return max(values) if values else numeric_defaults[key]

    elapsed = samples[-1]["elapsed_seconds"] if samples else 0.0
    return {
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "sample_count": len(samples),
        "peak_process_cpu_percent": max_value("process_cpu_percent"),
        "peak_system_cpu_percent": max_value("system_cpu_percent"),
        "peak_rss_mb": max_value("rss_mb"),
        "max_disk_read_mb": max_value("disk_read_mb"),
        "max_disk_write_mb": max_value("disk_write_mb"),
        "gpu_available": any(sample.get("gpu_available") for sample in samples),
        "peak_gpu_util_percent": max_value("gpu_util_percent_max"),
        "peak_gpu_memory_used_mb": max_value("gpu_memory_used_mb_sum"),
    }


def copy_stream_with_timestamps(stream: TextIO, output_file: TextIO, label: str) -> None:
    output_file.write(f"{iso_now()} [{label}] log_started\n")
    output_file.flush()
    for line in stream:
        output_file.write(f"{iso_now()} [{label}] {line}")
        output_file.flush()
    output_file.write(f"{iso_now()} [{label}] log_finished\n")
    output_file.flush()


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / f"{timestamp()}-{safe_name(args.name)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    samples_path = run_dir / "samples.csv"
    summary_path = run_dir / "summary.json"

    started_at = iso_now()
    start_time = time.time()
    samples: list[dict[str, Any]] = []
    previous_process_cpu: float | None = None
    previous_sample_time: float | None = None
    previous_system_cpu = read_system_cpu()

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        stdout_file.write(f"{started_at} [monitor] command_started command={json.dumps(args.command, ensure_ascii=False)}\n")
        stderr_file.write(f"{started_at} [monitor] command_started command={json.dumps(args.command, ensure_ascii=False)}\n")
        stdout_file.flush()
        stderr_file.flush()

        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_thread = threading.Thread(
            target=copy_stream_with_timestamps,
            args=(process.stdout, stdout_file, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=copy_stream_with_timestamps,
            args=(process.stderr, stderr_file, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        while True:
            sample, previous_process_cpu, previous_sample_time, previous_system_cpu = make_sample(
                process=process,
                start_time=start_time,
                previous_process_cpu=previous_process_cpu,
                previous_sample_time=previous_sample_time,
                previous_system_cpu=previous_system_cpu,
            )
            samples.append(sample)

            if process.poll() is not None:
                break
            time.sleep(args.interval)

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        finished_at = iso_now()
        stdout_file.write(f"{finished_at} [monitor] command_finished returncode={process.returncode}\n")
        stderr_file.write(f"{finished_at} [monitor] command_finished returncode={process.returncode}\n")

    write_csv(samples_path, samples)
    summary = build_summary(args.command, process.returncode, samples, started_at, finished_at)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Metricas salvas em: {run_dir}")
    print(f"Resumo: {summary_path}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
