#!/usr/bin/env python
"""Benchmark short YOLO train loops across laptop-specific harness settings."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache
from hardware_profile import (
    HEADROOM_MAX_GPU_MEM_PERCENT,
    HEADROOM_MAX_PERCENT,
    HEADROOM_MAX_RAM_PERCENT,
    HEADROOM_MIN_FREE_RAM_GB,
    HEADROOM_RESUME_PERCENT,
    recommended_device,
)


ROOT = Path(__file__).resolve().parents[1]
configure_project_cache()
DEFAULT_PROJECT = ROOT / "runs" / "cashsnap"


def parse_csv_ints(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    parsed = [int(item) for item in items]
    if any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be >= 0")
    return parsed


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name-prefix", default="harness_bench")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_PROJECT / "yolo_harness_throughput_latest.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=96)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batches", type=parse_csv_ints, default=parse_csv_ints("8,16,24"))
    parser.add_argument("--workers", type=parse_csv_ints, default=parse_csv_ints("0,2"))
    parser.add_argument(
        "--cache-modes",
        type=parse_csv_strings,
        default=parse_csv_strings("false"),
        help="Comma-separated Ultralytics cache modes: false, disk, ram, true.",
    )
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", nargs="?", const="default", default=None)
    parser.add_argument("--min-free-ram-gb", type=float, default=HEADROOM_MIN_FREE_RAM_GB)
    parser.add_argument("--max-percent", type=float, default=HEADROOM_MAX_PERCENT)
    parser.add_argument("--resume-percent", type=float, default=HEADROOM_RESUME_PERCENT)
    parser.add_argument("--max-ram-percent", type=float, default=HEADROOM_MAX_RAM_PERCENT)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=HEADROOM_MAX_GPU_MEM_PERCENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return document


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_path(config_path: Path, config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return data_root(config_path, config) / path


def split_rows(config_path: Path, split_value: str | list[str]) -> list[str]:
    config = read_yaml(config_path)
    values = split_value if isinstance(split_value, list) else [split_value]
    rows: list[str] = []
    for value in values:
        path = split_path(config_path, config, str(value))
        if path.suffix.lower() == ".txt":
            rows.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif path.is_dir():
            rows.extend(str(item) for item in path.iterdir() if item.is_file())
    return rows


def train_row_count(config_path: Path) -> int | None:
    config = read_yaml(config_path)
    train = config.get("train")
    if not isinstance(train, (str, list)):
        return None
    return len(split_rows(config_path, train))


def command_for(args: argparse.Namespace, *, batch: int, workers: int, cache: str, run_name: str) -> list[str]:
    command = [
        sys.executable,
        "scripts/bench_train_with_headroom.py",
        "--model",
        args.model,
        "--data",
        repo_rel(args.data),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--name",
        run_name,
        "--project",
        repo_rel(args.project),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--device",
        args.device,
        "--optimizer",
        args.optimizer,
        "--lr0",
        str(args.lr0),
        "--lrf",
        str(args.lrf),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--warmup-bias-lr",
        str(args.warmup_bias_lr),
        "--warmup-momentum",
        str(args.warmup_momentum),
        "--seed",
        str(args.seed),
        "--cache",
        cache,
        "--max-train-batches",
        str(args.max_train_batches),
        "--no-val",
        "--quiet",
        "--exist-ok",
        "--min-free-ram-gb",
        str(args.min_free_ram_gb),
        "--max-percent",
        str(args.max_percent),
        "--resume-percent",
        str(args.resume_percent),
        "--max-ram-percent",
        str(args.max_ram_percent),
        "--max-gpu-mem-percent",
        str(args.max_gpu_mem_percent),
        "--floor-memory-action",
        "exit",
        "--adaptive-restarts",
        "0",
    ]
    if args.no_amp:
        command.append("--no-amp")
    if args.compile is not None:
        command.extend(["--compile", args.compile])
    return command


def main() -> int:
    args = parse_args()
    args.data = resolve(args.data)
    args.project = resolve(args.project)
    args.json_out = resolve(args.json_out)
    args.device = recommended_device(args.device)
    train_rows = train_row_count(args.data)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "data": repo_rel(args.data),
        "model": args.model,
        "run_id": run_id,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "max_train_batches": args.max_train_batches,
        "train_rows": train_rows,
        "device": args.device,
        "amp": not args.no_amp,
        "compile": args.compile,
        "results": results,
    }

    for batch in args.batches:
        if batch < 1:
            raise SystemExit("--batches values must be >= 1")
        for workers in args.workers:
            for cache in args.cache_modes:
                if cache not in {"false", "disk", "ram", "true"}:
                    raise SystemExit(f"unsupported cache mode: {cache}")
                run_name = (
                    f"{args.name_prefix}_{run_id}_i{args.imgsz}_"
                    f"b{batch}_w{workers}_cache{cache}_steps{args.max_train_batches}"
                )
                command = command_for(args, batch=batch, workers=workers, cache=cache, run_name=run_name)
                print("[harness-bench] command:", " ".join(command), flush=True)
                started = time.monotonic()
                if args.dry_run:
                    code = 0
                    elapsed = None
                else:
                    code = subprocess.call(command, cwd=ROOT)
                    elapsed = time.monotonic() - started
                row = {
                    "run_name": run_name,
                    "batch": batch,
                    "workers": workers,
                    "cache": cache,
                    "command": command,
                    "dry_run": args.dry_run,
                    "elapsed_seconds": elapsed,
                    "batches_per_second": (
                        (args.max_train_batches / elapsed) if elapsed is not None and elapsed > 0 and code == 0 else None
                    ),
                    "nominal_images_per_second": (
                        (batch * args.max_train_batches / elapsed)
                        if elapsed is not None and elapsed > 0 and code == 0
                        else None
                    ),
                    "reference_batches_per_epoch": math.ceil(train_rows / batch) if train_rows else None,
                    "estimated_epoch_seconds": (
                        (math.ceil(train_rows / batch) * elapsed / args.max_train_batches)
                        if train_rows and elapsed is not None and elapsed > 0 and code == 0
                        else None
                    ),
                    "exit_code": code,
                }
                results.append(row)
                write_json(args.json_out, payload)
                if code != 0:
                    print(f"[harness-bench] failed exit_code={code}: {run_name}", flush=True)
                elif args.dry_run:
                    print(f"[harness-bench] dry_run={run_name}", flush=True)
                else:
                    print(
                        "[harness-bench] ok "
                        f"run={run_name} elapsed={elapsed:.1f}s "
                        f"batches_per_second={row['batches_per_second']:.3f}",
                        flush=True,
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
