#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run a python workflow and save its resource use as json
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# Cctober 2026
# ---------------------------------------------------------------------------

"""Run a command and save execution time, host ram, and gpu use as json.

Example
-------
python -u run_with_resource_monitor.py \
    --label mlp_SEE_w24_sd0042 \
    --output-dir ../scripts/gpu_slurm_jobs \
    -- python -u train_atmoswave_baselineMLP.py \
       --poi SEE --window 24 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import torch


def arguments() -> tuple[argparse.Namespace, list[str]]:
    """Read monitoring settings and the command after --."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../scripts/gpu_slurm_jobs"),
    )
    parser.add_argument("--sample-seconds", type=int, default=30)
    args, command = parser.parse_known_args()

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("provide a command after --")

    return args, command


def gpu_monitor(samples: list[dict], stop: threading.Event, interval: int) -> None:
    """Record allocated gpu status until the command finishes."""
    query = (
        "timestamp,index,name,memory.used,memory.total,"
        "utilization.gpu,utilization.memory,power.draw,temperature.gpu"
    )

    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            for line in result.stdout.strip().splitlines():
                values = [value.strip() for value in line.split(",")]

                if len(values) != 9:
                    continue

                samples.append(
                    {
                        "timestamp": values[0],
                        "index": int(values[1]),
                        "name": values[2],
                        "memory_used_mib": float(values[3]),
                        "memory_total_mib": float(values[4]),
                        "gpu_utilisation_percent": float(values[5]),
                        "memory_utilisation_percent": float(values[6]),
                        "power_w": float(values[7]),
                        "temperature_c": float(values[8]),
                    }
                )
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass

        stop.wait(interval)


def parse_time_log(path: Path) -> dict:
    """Read the useful fields written by /usr/bin/time -v."""
    if not path.is_file():
        return {}

    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "user_seconds": r"User time \(seconds\):\s+([\d.]+)",
        "system_seconds": r"System time \(seconds\):\s+([\d.]+)",
        "cpu_percent": r"Percent of CPU this job got:\s+(.+)",
        "wall_clock": r"Elapsed \(wall clock\) time.*:\s+(.+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s+(\d+)",
        "file_inputs": r"File system inputs:\s+(\d+)",
        "file_outputs": r"File system outputs:\s+(\d+)",
        "exit_status": r"Exit status:\s+(\d+)",
    }
    output = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text)

        if match is None:
            continue

        value = match.group(1).strip()

        if key in {"user_seconds", "system_seconds"}:
            value = float(value)
        elif key not in {"cpu_percent", "wall_clock"}:
            value = int(value)

        output[key] = value

    if "max_rss_kb" in output:
        output["max_rss_gib"] = output["max_rss_kb"] / 1024**2

    return output


def mean(samples: list[dict], key: str):
    values = [sample[key] for sample in samples]
    return sum(values) / len(values) if values else None


def maximum(samples: list[dict], key: str):
    values = [sample[key] for sample in samples]
    return max(values) if values else None


def main() -> None:
    """Run the requested command and write one json resource report."""
    args, command = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    job_id = os.getenv("SLURM_JOB_ID", "local")
    output_path = args.output_dir / f"{args.label}_{job_id}.json"

    start_time = datetime.now().astimezone()
    started = time.perf_counter()
    gpu_samples: list[dict] = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=gpu_monitor,
        args=(gpu_samples, stop_event, args.sample_seconds),
        daemon=True,
    )
    monitor.start()

    time_program = shutil.which("time") or "/usr/bin/time"

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as file:
        time_log = Path(file.name)

    try:
        process = subprocess.run(
            [time_program, "-v", "-o", str(time_log), *command],
            check=False,
        )
        exit_code = process.returncode
    finally:
        stop_event.set()
        monitor.join(timeout=args.sample_seconds + 5)

    end_time = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started
    host_resources = parse_time_log(time_log)
    time_log.unlink(missing_ok=True)

    summary = {
        "run": {
            "label": args.label,
            "command": shlex.join(command),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
        },
        "slurm": {
            "job_id": os.getenv("SLURM_JOB_ID"),
            "job_name": os.getenv("SLURM_JOB_NAME"),
            "partition": os.getenv("SLURM_JOB_PARTITION"),
            "node_list": os.getenv("SLURM_JOB_NODELIST"),
            "cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
            "allocated_gpus": os.getenv("SLURM_JOB_GPUS"),
            "requested_memory_mb": os.getenv("SLURM_MEM_PER_NODE"),
        },
        "system": {
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "host_resources": host_resources,
        "gpu_resources": {
            "sample_interval_seconds": args.sample_seconds,
            "n_samples": len(gpu_samples),
            "gpu_name": gpu_samples[0]["name"] if gpu_samples else None,
            "gpu_memory_total_mib": (
                gpu_samples[0]["memory_total_mib"] if gpu_samples else None
            ),
            "gpu_memory_peak_mib": maximum(gpu_samples, "memory_used_mib"),
            "gpu_utilisation_mean_percent": mean(
                gpu_samples, "gpu_utilisation_percent"
            ),
            "gpu_utilisation_peak_percent": maximum(
                gpu_samples, "gpu_utilisation_percent"
            ),
            "power_mean_w": mean(gpu_samples, "power_w"),
            "power_peak_w": maximum(gpu_samples, "power_w"),
            "temperature_peak_c": maximum(gpu_samples, "temperature_c"),
        },
    }

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"resource summary saved: {output_path}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
