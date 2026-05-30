"""Print a compact local system profile for CashSnap harness decisions."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]


def run_text(command: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def windows_cpu_name() -> str:
    if not platform.system().lower().startswith("windows"):
        return ""
    return run_text(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name",
        ]
    )


def windows_machine() -> dict[str, str]:
    if not platform.system().lower().startswith("windows"):
        return {}
    text = run_text(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model | ConvertTo-Json -Compress",
        ]
    )
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return {"manufacturer": str(data.get("Manufacturer", "")), "model": str(data.get("Model", ""))}


def gpu_profile() -> dict[str, object] | None:
    text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not text:
        return None
    first = text.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 6:
        return None
    try:
        return {
            "name": parts[0],
            "driver_version": parts[1],
            "memory_total_mb": float(parts[2]),
            "memory_used_mb": float(parts[3]),
            "utilization_percent": float(parts[4]),
            "temperature_c": float(parts[5]),
        }
    except ValueError:
        return {"raw": first}


def profile() -> dict[str, object]:
    ram = psutil.virtual_memory()
    machine = windows_machine()
    cpu_name = windows_cpu_name() or platform.processor()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": machine,
        "python": platform.python_version(),
        "cpu": {
            "name": cpu_name,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "current_percent": psutil.cpu_percent(interval=1.0),
        },
        "ram": {
            "total_gb": round(ram.total / (1024**3), 2),
            "available_gb": round(ram.available / (1024**3), 2),
            "used_percent": ram.percent,
        },
        "gpu": gpu_profile(),
        "headroom_policy": {
            "preferred_max_percent": 90,
            "hard_max_percent": 95,
            "resume_percent": 82,
            "training_default_workers": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    data = profile()
    text = json.dumps(data, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
