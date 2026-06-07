#!/usr/bin/env python
"""Benchmark short WebGL generation runs across renderer harness settings."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hardware_profile import (
    HEADROOM_MAX_GPU_MEM_PERCENT,
    HEADROOM_MAX_PERCENT,
    HEADROOM_MAX_RAM_PERCENT,
    HEADROOM_MIN_FREE_RAM_GB,
    HEADROOM_RESUME_PERCENT,
    WEBGL_CHECK_JOBS,
    WEBGL_RENDERER_BATCH_SIZE,
    WEBGL_RENDER_JOBS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PARENT = ROOT / "data" / "synthetic" / "webgl_generation_benchmarks"
DEFAULT_JSON_OUT = ROOT / "runs" / "cashsnap" / "webgl_generation_throughput_latest.json"


def parse_csv_ints(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    parsed = [int(item) for item in items]
    if any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("values must be >= 1")
    return parsed


def parse_csv_bools(value: str) -> list[bool]:
    aliases = {
        "0": False,
        "false": False,
        "no": False,
        "off": False,
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
    }
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    parsed: list[bool] = []
    for item in items:
        if item not in aliases:
            raise argparse.ArgumentTypeError(f"unsupported boolean value: {item}")
        parsed.append(aliases[item])
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-id", default="webgl_clean_topdown_readable_v1")
    parser.add_argument("--out-parent", type=Path, default=DEFAULT_OUT_PARENT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--start-variant", type=int, default=9000)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--visual-scale", default="1")
    parser.add_argument("--render-jobs", type=parse_csv_ints, default=parse_csv_ints(str(WEBGL_RENDER_JOBS)))
    parser.add_argument("--renderer-batch-sizes", type=parse_csv_ints, default=parse_csv_ints(f"1,8,{WEBGL_RENDERER_BATCH_SIZE}"))
    parser.add_argument("--check-jobs", type=parse_csv_ints, default=parse_csv_ints(str(WEBGL_CHECK_JOBS)))
    parser.add_argument("--check-mode", choices=["in-process", "subprocess"], default="subprocess")
    parser.add_argument("--shared-browser-modes", type=parse_csv_bools, default=parse_csv_bools("false,true"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--headroom-max-percent", default=str(int(HEADROOM_MAX_PERCENT)))
    parser.add_argument("--headroom-resume-percent", default=str(int(HEADROOM_RESUME_PERCENT)))
    parser.add_argument("--headroom-max-ram-percent", default=str(int(HEADROOM_MAX_RAM_PERCENT)))
    parser.add_argument("--headroom-max-gpu-mem-percent", default=str(int(HEADROOM_MAX_GPU_MEM_PERCENT)))
    parser.add_argument("--min-free-ram-gb", default=str(int(HEADROOM_MIN_FREE_RAM_GB)))
    parser.add_argument("--preflight-timeout", default="120")
    parser.add_argument("--browser-executable", type=Path, default=None)
    parser.add_argument("--include-yolo-check", action="store_true")
    parser.add_argument("--include-label-view-check", action="store_true")
    parser.add_argument("--use-catalog-balanced-subset", action="store_true")
    parser.add_argument("--measure-package-only", action="store_true", default=True)
    parser.add_argument("--no-measure-package-only", dest="measure_package_only", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_for(
    args: argparse.Namespace,
    *,
    out_root: Path,
    start_variant: int,
    render_jobs: int,
    renderer_batch_size: int,
    check_jobs: int,
    shared_browser: bool,
    skip_render: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_webgl_recipe.py",
        "--recipe-id",
        args.recipe_id,
        "--out-root",
        repo_rel(out_root),
        "--start-variant",
        str(start_variant),
        "--count",
        str(args.count),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--visual-scale",
        str(args.visual_scale),
        "--headroom-max-percent",
        args.headroom_max_percent,
        "--headroom-resume-percent",
        args.headroom_resume_percent,
        "--headroom-max-ram-percent",
        args.headroom_max_ram_percent,
        "--headroom-max-gpu-mem-percent",
        args.headroom_max_gpu_mem_percent,
        "--min-free-ram-gb",
        args.min_free_ram_gb,
        "--preflight-timeout",
        args.preflight_timeout,
        "--render-jobs",
        str(render_jobs),
        "--renderer-batch-size",
        str(renderer_batch_size),
        "--check-jobs",
        str(check_jobs),
        "--check-mode",
        args.check_mode,
        "--skip-smoke-gate",
        "--skip-trainable-gate",
    ]
    if not args.use_catalog_balanced_subset:
        command.extend(["--balanced-subset-count", "0"])
    if not args.include_yolo_check:
        command.append("--skip-yolo-check")
    if not args.include_label_view_check:
        command.append("--skip-label-view-check")
    if args.browser_executable is not None:
        command.extend(["--browser-executable", str(args.browser_executable)])
    if shared_browser:
        command.append("--shared-browser")
    if skip_render:
        command.append("--skip-render")
    return command


def timed_call(command: list[str], dry_run: bool) -> tuple[int, float | None]:
    print("[webgl-bench] command:", " ".join(command), flush=True)
    started = time.monotonic()
    if dry_run:
        return 0, None
    code = subprocess.call(command, cwd=ROOT)
    return code, time.monotonic() - started


def best_success(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    successes = [
        row for row in results
        if row.get("exit_code") == 0 and isinstance(row.get("elapsed_seconds"), (int, float))
    ]
    if not successes:
        return None
    return min(successes, key=lambda row: float(row["elapsed_seconds"]))


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    args.out_parent = resolve(args.out_parent)
    args.json_out = resolve(args.json_out)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "recipe_id": args.recipe_id,
        "run_id": run_id,
        "count": args.count,
        "width": args.width,
        "height": args.height,
        "visual_scale": str(args.visual_scale),
        "include_yolo_check": bool(args.include_yolo_check),
        "include_label_view_check": bool(args.include_label_view_check),
        "use_catalog_balanced_subset": bool(args.use_catalog_balanced_subset),
        "measure_package_only": bool(args.measure_package_only),
        "check_mode": args.check_mode,
        "results": results,
        "best": None,
    }

    for repeat in range(args.repeats):
        for render_jobs in args.render_jobs:
            for renderer_batch_size in args.renderer_batch_sizes:
                for check_jobs in args.check_jobs:
                    for shared_browser in args.shared_browser_modes:
                        label = (
                            f"{slug(args.recipe_id)}_{run_id}_r{repeat + 1}_"
                            f"j{render_jobs}_b{renderer_batch_size}_c{check_jobs}_sb{int(shared_browser)}"
                        )
                        out_root = args.out_parent / label
                        start_variant = args.start_variant + repeat * 100_000
                        command = command_for(
                            args,
                            out_root=out_root,
                            start_variant=start_variant,
                            render_jobs=render_jobs,
                            renderer_batch_size=renderer_batch_size,
                            check_jobs=check_jobs,
                            shared_browser=shared_browser,
                        )
                        code, elapsed = timed_call(command, args.dry_run)
                        row: dict[str, Any] = {
                            "label": label,
                            "repeat": repeat + 1,
                            "out_root": repo_rel(out_root),
                            "start_variant": start_variant,
                            "render_jobs": render_jobs,
                            "renderer_batch_size": renderer_batch_size,
                            "check_jobs": check_jobs,
                            "shared_browser": shared_browser,
                            "command": command,
                            "dry_run": bool(args.dry_run),
                            "elapsed_seconds": elapsed,
                            "images_per_second": (
                                (args.count / elapsed) if elapsed is not None and elapsed > 0 and code == 0 else None
                            ),
                            "exit_code": code,
                        }
                        if code == 0 and args.measure_package_only:
                            package_command = command_for(
                                args,
                                out_root=out_root,
                                start_variant=start_variant,
                                render_jobs=render_jobs,
                                renderer_batch_size=renderer_batch_size,
                                check_jobs=check_jobs,
                                shared_browser=shared_browser,
                                skip_render=True,
                            )
                            package_code, package_elapsed = timed_call(package_command, args.dry_run)
                            row.update(
                                {
                                    "package_only_command": package_command,
                                    "package_only_elapsed_seconds": package_elapsed,
                                    "package_only_images_per_second": (
                                        (args.count / package_elapsed)
                                        if package_elapsed is not None and package_elapsed > 0 and package_code == 0
                                        else None
                                    ),
                                    "package_only_exit_code": package_code,
                                    "render_plus_browser_estimated_seconds": (
                                        max(0.0, elapsed - package_elapsed)
                                        if elapsed is not None
                                        and package_elapsed is not None
                                        and code == 0
                                        and package_code == 0
                                        else None
                                    ),
                                }
                            )
                        results.append(row)
                        payload["best"] = best_success(results)
                        write_json(args.json_out, payload)
                        if code != 0:
                            print(f"[webgl-bench] failed exit_code={code}: {label}", flush=True)
                        elif args.dry_run:
                            print(f"[webgl-bench] dry_run={label}", flush=True)
                        else:
                            print(
                                "[webgl-bench] ok "
                                f"run={label} elapsed={elapsed:.1f}s images_per_second={row['images_per_second']:.3f}",
                                flush=True,
                            )

    payload["best"] = best_success(results)
    write_json(args.json_out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
