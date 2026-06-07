#!/usr/bin/env python
"""Summarize YOLO transfer guardrails for synthetic curriculum probes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLIT_RE = re.compile(r"#([^:#]+):")
CLEAN_TRANSFER_COMPARISONS = (
    "full_val",
    "full_test",
    "clean_visible_val",
    "clean_visible_test",
    "labeled_all_test",
    "geometry_stress_test",
    "protected_riel_val",
    "protected_riel_test",
)
CLEAN_TRANSFER_BACKGROUND_SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Comparison JSON from compare_yolo_metrics.py. Repeat for full_val, full_test, etc.",
    )
    parser.add_argument("--background-fp-json", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--preset",
        choices=["clean-transfer"],
        default=None,
        help="Add required checks for a known promotion gate.",
    )
    parser.add_argument(
        "--require-compare",
        action="append",
        default=[],
        metavar="NAME",
        help="Require a named --compare entry to be present. Repeatable.",
    )
    parser.add_argument(
        "--require-background-split",
        action="append",
        default=[],
        metavar="SPLIT",
        help="Require background FP coverage for a split name such as val or test. Repeatable.",
    )
    parser.add_argument(
        "--max-fp-detection-increase",
        type=int,
        default=0,
        help="Maximum allowed candidate-baseline increase in empty-frame detections per split.",
    )
    parser.add_argument(
        "--max-fp-image-increase",
        type=int,
        default=0,
        help="Maximum allowed candidate-baseline increase in empty-frame images_with_fp per split.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected JSON object")
    return document


def unique_preserve_order(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        rows.append(item)
    return rows


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"--compare must be NAME=JSON, got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"--compare has empty name: {raw!r}")
    return name, Path(path.strip())


def compare_summary(name: str, path: Path) -> dict[str, Any]:
    document = read_json(path)
    failures = document.get("per_class_failures", [])
    if not isinstance(failures, list):
        failures = []
    return {
        "name": name,
        "path": repo_rel(resolve(path)),
        "passed": bool(document.get("passed", False)),
        "metric": document.get("metric"),
        "baseline": document.get("baseline"),
        "candidate": document.get("candidate"),
        "delta": document.get("delta"),
        "per_class_failed": [row.get("class_name") for row in failures if isinstance(row, dict)],
    }


def split_name(image_root: str) -> str:
    match = SPLIT_RE.search(image_root)
    return match.group(1) if match else image_root


def fp_rows_by_label_and_split(document: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = document.get("rows", [])
    if not isinstance(rows, list):
        raise SystemExit("background FP JSON missing rows list")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("model_label", ""))
        split = split_name(str(row.get("image_root", "")))
        indexed[(label, split)] = row
    return indexed


def class_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_counts = baseline.get("by_class", {})
    candidate_counts = candidate.get("by_class", {})
    if not isinstance(baseline_counts, dict):
        baseline_counts = {}
    if not isinstance(candidate_counts, dict):
        candidate_counts = {}
    rows = []
    for class_name in sorted(set(baseline_counts) | set(candidate_counts)):
        base = int(baseline_counts.get(class_name, 0))
        cand = int(candidate_counts.get(class_name, 0))
        rows.append({"class_name": class_name, "baseline": base, "candidate": cand, "delta": cand - base})
    return sorted(rows, key=lambda row: int(row["delta"]), reverse=True)


def background_fp_summary(
    *,
    path: Path,
    baseline_label: str,
    candidate_label: str,
    max_detection_increase: int,
    max_image_increase: int,
) -> list[dict[str, Any]]:
    document = read_json(path)
    indexed = fp_rows_by_label_and_split(document)
    splits = sorted({split for label, split in indexed if label in {baseline_label, candidate_label}})
    summaries = []
    for split in splits:
        baseline = indexed.get((baseline_label, split))
        candidate = indexed.get((candidate_label, split))
        if baseline is None or candidate is None:
            summaries.append(
                {
                    "split": split,
                    "passed": False,
                    "missing": [
                        label
                        for label, row in ((baseline_label, baseline), (candidate_label, candidate))
                        if row is None
                    ],
                }
            )
            continue
        detection_delta = int(candidate.get("detections", 0)) - int(baseline.get("detections", 0))
        image_delta = int(candidate.get("images_with_fp", 0)) - int(baseline.get("images_with_fp", 0))
        summaries.append(
            {
                "split": split,
                "passed": detection_delta <= max_detection_increase and image_delta <= max_image_increase,
                "images": int(candidate.get("images", baseline.get("images", 0))),
                "baseline_detections": int(baseline.get("detections", 0)),
                "candidate_detections": int(candidate.get("detections", 0)),
                "detection_delta": detection_delta,
                "baseline_images_with_fp": int(baseline.get("images_with_fp", 0)),
                "candidate_images_with_fp": int(candidate.get("images_with_fp", 0)),
                "images_with_fp_delta": image_delta,
                "top_class_deltas": class_deltas(baseline, candidate)[:5],
            }
        )
    return summaries


def blocker_summary(
    comparisons: list[dict[str, Any]],
    fp_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    class_failures: Counter[str] = Counter()
    failed_comparisons = []
    for row in comparisons:
        if not row.get("passed", False):
            failed_comparisons.append(str(row.get("name", "")))
        class_failures.update(
            str(class_name)
            for class_name in row.get("per_class_failed", [])
            if class_name
        )

    background_class_deltas: Counter[str] = Counter()
    failed_background_splits = []
    for row in fp_summaries:
        if not row.get("passed", False):
            failed_background_splits.append(str(row.get("split", "")))
        for delta_row in row.get("top_class_deltas", []):
            if not isinstance(delta_row, dict):
                continue
            delta = int(delta_row.get("delta", 0))
            if delta > 0:
                background_class_deltas[str(delta_row.get("class_name", "unknown"))] += delta

    return {
        "failed_comparisons": failed_comparisons,
        "failed_background_splits": failed_background_splits,
        "per_class_failure_counts": [
            {"class_name": class_name, "failed_comparisons": int(count)}
            for class_name, count in class_failures.most_common()
        ],
        "background_fp_positive_class_deltas": [
            {"class_name": class_name, "positive_detection_delta": int(delta)}
            for class_name, delta in background_class_deltas.most_common()
        ],
    }


def requirement_summary(
    *,
    comparisons: list[dict[str, Any]],
    fp_summaries: list[dict[str, Any]],
    required_comparisons: list[str],
    required_background_splits: list[str],
) -> dict[str, Any]:
    comparison_names = {str(row.get("name", "")) for row in comparisons}
    background_splits = {str(row.get("split", "")) for row in fp_summaries}
    missing_comparisons = [name for name in required_comparisons if name not in comparison_names]
    missing_background_splits = [
        split for split in required_background_splits if split not in background_splits
    ]
    return {
        "required_comparisons": required_comparisons,
        "missing_comparisons": missing_comparisons,
        "required_background_splits": required_background_splits,
        "missing_background_splits": missing_background_splits,
        "passed": not missing_comparisons and not missing_background_splits,
    }


def main() -> int:
    args = parse_args()
    required_comparisons = list(args.require_compare)
    required_background_splits = list(args.require_background_split)
    if args.preset == "clean-transfer":
        required_comparisons.extend(CLEAN_TRANSFER_COMPARISONS)
        required_background_splits.extend(CLEAN_TRANSFER_BACKGROUND_SPLITS)
    required_comparisons = unique_preserve_order(required_comparisons)
    required_background_splits = unique_preserve_order(required_background_splits)
    comparisons = [compare_summary(name, path) for name, path in map(parse_named_path, args.compare)]
    fp_summaries = []
    if args.background_fp_json:
        fp_summaries = background_fp_summary(
            path=args.background_fp_json,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            max_detection_increase=args.max_fp_detection_increase,
            max_image_increase=args.max_fp_image_increase,
        )
    requirements = requirement_summary(
        comparisons=comparisons,
        fp_summaries=fp_summaries,
        required_comparisons=required_comparisons,
        required_background_splits=required_background_splits,
    )
    passed = (
        all(row["passed"] for row in comparisons)
        and all(row["passed"] for row in fp_summaries)
        and bool(requirements["passed"])
    )
    blockers = blocker_summary(comparisons, fp_summaries)
    payload = {
        "schema": "cashsnap_yolo_transfer_guardrails_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "preset": args.preset,
        "passed": passed,
        "requirements": requirements,
        "comparisons": comparisons,
        "background_fp": fp_summaries,
        "blockers": blockers,
    }
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_rel(out)}")
    status = "PASS" if passed else "FAIL"
    print(f"{status}: comparisons={len(comparisons)} background_fp_splits={len(fp_summaries)}")
    if requirements["missing_comparisons"]:
        print(f"missing_comparisons: {', '.join(requirements['missing_comparisons'])}")
    if requirements["missing_background_splits"]:
        print(f"missing_background_splits: {', '.join(requirements['missing_background_splits'])}")
    for row in comparisons:
        print(
            f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} "
            f"baseline={row['baseline']:.6f} candidate={row['candidate']:.6f} delta={row['delta']:+.6f}"
        )
    for row in fp_summaries:
        if row.get("missing"):
            print(f"background_fp {row['split']}: FAIL missing={','.join(row['missing'])}")
            continue
        print(
            f"background_fp {row['split']}: {'PASS' if row['passed'] else 'FAIL'} "
            f"detections={row['baseline_detections']}->{row['candidate_detections']} "
            f"images_with_fp={row['baseline_images_with_fp']}->{row['candidate_images_with_fp']}"
        )
    if blockers["per_class_failure_counts"]:
        classes = ", ".join(
            f"{row['class_name']}x{row['failed_comparisons']}"
            for row in blockers["per_class_failure_counts"][:8]
        )
        print(f"class_blockers: {classes}")
    if blockers["background_fp_positive_class_deltas"]:
        classes = ", ".join(
            f"{row['class_name']}+{row['positive_detection_delta']}"
            for row in blockers["background_fp_positive_class_deltas"][:8]
        )
        print(f"background_fp_positive_deltas: {classes}")
    return 0 if passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
