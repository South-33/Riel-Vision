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
DEFAULT_DRAFT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "drafts"
P1_STRESS_ROLES = {"fan_stress", "dense_overlap_stress", "hand_occlusion_stress"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-mix", type=Path, default=DEFAULT_SMOKE_MIX)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--browser-cases", type=Path, default=DEFAULT_BROWSER_CASES)
    parser.add_argument("--draft-label-dir", type=Path, default=DEFAULT_DRAFT_LABEL_DIR)
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


def benchmark_role(row: dict[str, str]) -> str:
    return row.get("benchmark_role", "").strip()


def is_p1_stress(row: dict[str, str]) -> bool:
    return benchmark_role(row) in P1_STRESS_ROLES


def repo_path(path: Path) -> str:
    return resolve(path).relative_to(ROOT).as_posix()


def count_yolo_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def review_ready_drafts(source_rows: list[dict[str, str]], quality_rows: list[dict[str, str]], draft_label_dir: Path) -> list[str]:
    scoreable_indices: dict[tuple[str, str], set[int]] = {}
    for row in quality_rows:
        if row.get("quality", "").strip() not in {"clear", "partial_clear"}:
            continue
        if not truthy(row.get("count_for_score", "")):
            continue
        try:
            label_index = int(row.get("label_index", ""))
        except ValueError:
            continue
        key = (row.get("image_id", ""), row.get("label_path", "").replace("\\", "/"))
        scoreable_indices.setdefault(key, set()).add(label_index)

    ready: list[str] = []
    draft_root = resolve(draft_label_dir)
    for row in source_rows:
        image_id = row.get("image_id", "")
        if row.get("label_status", "").strip() == "labeled" or row.get("benchmark_status", "").strip() == "labeled":
            continue
        draft_path = draft_root / f"{image_id}.txt"
        label_count = count_yolo_rows(draft_path)
        if label_count <= 0:
            continue
        key = (image_id, repo_path(draft_path))
        if scoreable_indices.get(key) == set(range(label_count)):
            ready.append(image_id)
    return ready


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
    labeled_stress_images = [row for row in labeled_images if is_p1_stress(row)]
    draft_scoreable_rows = [
        row
        for row in quality_rows
        if row.get("quality", "").strip() in {"clear", "partial_clear"} and truthy(row.get("count_for_score", ""))
    ]
    review_ready_draft_ids = review_ready_drafts(source_rows, quality_rows, args.draft_label_dir)
    source_by_id = {row.get("image_id", ""): row for row in source_rows}
    review_ready_stress_draft_ids = [
        image_id for image_id in review_ready_draft_ids if is_p1_stress(source_by_id.get(image_id, {}))
    ]
    browser_cases = read_csv(args.browser_cases)
    blockers: list[str] = []
    if not labeled_images:
        blockers.append("no promoted real benchmark labels; only draft diagnostics are available")
    elif not labeled_stress_images:
        blockers.append("no promoted real fan/overlap stress labels; visible-denomination labels alone are not a full P1 proof")
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
        "promoted_stress_labeled_images": len(labeled_stress_images),
        "p1_stress_roles": sorted(P1_STRESS_ROLES),
        "draft_scoreable_boxes": len(draft_scoreable_rows),
        "review_ready_draft_images": len(review_ready_draft_ids),
        "review_ready_draft_image_ids": review_ready_draft_ids,
        "review_ready_stress_draft_images": len(review_ready_stress_draft_ids),
        "review_ready_stress_draft_image_ids": review_ready_stress_draft_ids,
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
