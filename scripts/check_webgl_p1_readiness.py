#!/usr/bin/env python
"""Run lightweight readiness checks for a WebGL P1 transfer proof."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_MIX = ROOT / "configs" / "cashsnap_webgl_smoke_suite_mix.yaml"
DEFAULT_SOURCES = ROOT / "manifests" / "real_fan_benchmark_sources.csv"
DEFAULT_QUALITY = ROOT / "manifests" / "real_fan_benchmark_label_quality.csv"
DEFAULT_BROWSER_CASES = ROOT / "manifests" / "browser_smoke_cases.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-mix", type=Path, default=DEFAULT_SMOKE_MIX)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--browser-cases", type=Path, default=DEFAULT_BROWSER_CASES)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run_check(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(f"check failed: {' '.join(command)}")
    return result.stdout


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "score", "keep"}


def main() -> int:
    args = parse_args()
    yolo_stdout = run_check([sys.executable, "scripts/check_yolo_dataset.py", "--data", str(args.smoke_mix)])
    real_stdout = run_check([sys.executable, "scripts/check_real_fan_benchmark.py"])
    browser_stdout = run_check(
        [
            sys.executable,
            "scripts/run_browser_smoke_cases.py",
            "--cases",
            str(args.browser_cases),
            "--validate-only",
            "--no-artifacts",
        ]
    )

    source_rows = read_csv(args.sources)
    quality_rows = read_csv(args.quality)
    labeled_images = [
        row
        for row in source_rows
        if row.get("benchmark_status", "").strip() == "labeled" or row.get("label_status", "").strip() == "labeled"
    ]
    draft_scoreable_rows = [
        row
        for row in quality_rows
        if row.get("quality", "").strip() in {"clear", "partial_clear"} and truthy(row.get("count_for_score", ""))
    ]
    browser_cases = read_csv(args.browser_cases)
    blockers: list[str] = []
    if not labeled_images:
        blockers.append("no promoted real benchmark labels; only draft diagnostics are available")
    if not draft_scoreable_rows:
        blockers.append("no scoreable draft real labels")
    if not browser_cases:
        blockers.append("browser smoke manifest has no cases")

    summary = {
        "smoke_mix": resolve(args.smoke_mix).relative_to(ROOT).as_posix(),
        "smoke_mix_check": "passed",
        "real_benchmark_check": "passed",
        "browser_manifest_check": "passed",
        "promoted_labeled_images": len(labeled_images),
        "draft_scoreable_boxes": len(draft_scoreable_rows),
        "browser_cases": len(browser_cases),
        "ready_for_full_p1_transfer": not blockers,
        "blockers": blockers,
    }
    if args.json_out is not None:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("yolo_check:", yolo_stdout.splitlines()[0] if yolo_stdout.splitlines() else "passed")
    print("real_check:", real_stdout.splitlines()[-1] if real_stdout.splitlines() else "passed")
    print("browser_check:", browser_stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
