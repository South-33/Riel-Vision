#!/usr/bin/env python
"""Build a YOLO transfer scorecard from named metric pairs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_yolo_transfer_guardrails as guardrails
from scripts import compare_yolo_metrics as compare_metrics

CLEAN_TRANSFER_CLASS_FILTERS = {
    "protected_riel_val": {"KHR_20000", "KHR_50000"},
    "protected_riel_test": {"KHR_20000", "KHR_50000"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="NAME=BASELINE_JSON,CANDIDATE_JSON",
        help="Named metric pair to compare. Repeat for full_val, full_test, etc.",
    )
    parser.add_argument("--background-fp-json", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--preset", choices=["clean-transfer"], default=None)
    parser.add_argument("--metric", default="box.map50_95")
    parser.add_argument("--per-class-metric", default="map50_95")
    parser.add_argument(
        "--only-classes-for",
        action="append",
        default=[],
        metavar="NAME=CLASS,CLASS",
        help="Restrict per-class guard for one comparison name. Repeatable.",
    )
    parser.add_argument("--max-drop", type=float, default=0.0)
    parser.add_argument("--max-per-class-drop", type=float, default=0.05)
    parser.add_argument("--max-fp-detection-increase", type=int, default=0)
    parser.add_argument("--max-fp-image-increase", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Directory for generated comparison JSONs. Defaults beside --json-out.",
    )
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_pair(raw: str) -> tuple[str, Path, Path]:
    if "=" not in raw:
        raise SystemExit(f"--pair must be NAME=BASELINE_JSON,CANDIDATE_JSON, got {raw!r}")
    name, rest = raw.split("=", 1)
    paths = [value.strip() for value in rest.split(",", 1)]
    if not name.strip() or len(paths) != 2 or not paths[0] or not paths[1]:
        raise SystemExit(f"--pair must be NAME=BASELINE_JSON,CANDIDATE_JSON, got {raw!r}")
    return name.strip(), Path(paths[0]), Path(paths[1])


def parse_only_classes_for(rows: list[str]) -> dict[str, set[str]]:
    filters: dict[str, set[str]] = {}
    for raw in rows:
        if "=" not in raw:
            raise SystemExit(f"--only-classes-for must be NAME=CLASS,CLASS, got {raw!r}")
        name, classes = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--only-classes-for has empty name: {raw!r}")
        parsed = compare_metrics.parse_class_filter(classes)
        if not parsed:
            raise SystemExit(f"--only-classes-for has no classes: {raw!r}")
        filters[name] = parsed
    return filters


def compare_payload(
    *,
    baseline_path: Path,
    candidate_path: Path,
    metric: str,
    per_class_metric: str,
    max_drop: float,
    max_per_class_drop: float,
    only_classes: set[str] | None,
) -> dict[str, Any]:
    baseline = compare_metrics.read_json(baseline_path)
    candidate = compare_metrics.read_json(candidate_path)
    baseline_value = compare_metrics.result_metric(baseline, metric)
    candidate_value = compare_metrics.result_metric(candidate, metric)
    delta = candidate_value - baseline_value
    per_class_rows = compare_metrics.compare_per_class(
        baseline,
        candidate,
        per_class_metric,
        only_classes=only_classes,
    )
    checks: list[dict[str, Any]] = [
        {
            "name": "max_drop",
            "passed": delta >= -max_drop,
            "threshold": -max_drop,
            "actual_delta": delta,
        }
    ]
    check, per_class_failures = compare_metrics.per_class_guard(
        per_class_rows,
        max_per_class_drop,
        ignore_missing_classes=False,
    )
    checks.append(check)
    return {
        "passed": all(row["passed"] for row in checks),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metric": metric,
        "baseline_path": repo_rel(baseline_path),
        "baseline_sha256": compare_metrics.file_sha256(baseline_path),
        "candidate_path": repo_rel(candidate_path),
        "candidate_sha256": compare_metrics.file_sha256(candidate_path),
        "classes_from_summary_path": "",
        "classes_from_summary_sha256": "",
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "checks": checks,
        "per_class_metric": per_class_metric,
        "max_per_class_drop": max_per_class_drop,
        "only_classes": sorted(only_classes or []),
        "per_class_failures": per_class_failures,
        "per_class_map50_95": per_class_rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    json_out = resolve(args.json_out)
    compare_dir = resolve(args.compare_dir) if args.compare_dir else json_out.parent / f"{json_out.stem}_comparisons"
    only_classes_by_name = parse_only_classes_for(args.only_classes_for)
    if args.preset == "clean-transfer":
        for name, classes in CLEAN_TRANSFER_CLASS_FILTERS.items():
            only_classes_by_name.setdefault(name, classes)
    generated_comparisons = []
    for name, raw_baseline_path, raw_candidate_path in map(parse_pair, args.pair):
        baseline_path = resolve(raw_baseline_path)
        candidate_path = resolve(raw_candidate_path)
        payload = compare_payload(
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            metric=args.metric,
            per_class_metric=args.per_class_metric,
            max_drop=args.max_drop,
            max_per_class_drop=args.max_per_class_drop,
            only_classes=only_classes_by_name.get(name),
        )
        compare_path = compare_dir / f"{name}.json"
        write_json(compare_path, payload)
        generated_comparisons.append(guardrails.compare_summary(name, compare_path))

    fp_summaries = []
    if args.background_fp_json:
        fp_summaries = guardrails.background_fp_summary(
            path=args.background_fp_json,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            max_detection_increase=args.max_fp_detection_increase,
            max_image_increase=args.max_fp_image_increase,
        )

    required_comparisons = []
    required_background_splits = []
    if args.preset == "clean-transfer":
        required_comparisons.extend(guardrails.CLEAN_TRANSFER_COMPARISONS)
        required_background_splits.extend(guardrails.CLEAN_TRANSFER_BACKGROUND_SPLITS)
    required_comparisons = guardrails.unique_preserve_order(required_comparisons)
    required_background_splits = guardrails.unique_preserve_order(required_background_splits)
    requirements = guardrails.requirement_summary(
        comparisons=generated_comparisons,
        fp_summaries=fp_summaries,
        required_comparisons=required_comparisons,
        required_background_splits=required_background_splits,
    )
    blockers = guardrails.blocker_summary(generated_comparisons, fp_summaries)
    passed = (
        all(row["passed"] for row in generated_comparisons)
        and all(row["passed"] for row in fp_summaries)
        and bool(requirements["passed"])
    )
    scorecard = {
        "schema": "cashsnap_yolo_transfer_scorecard_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "preset": args.preset,
        "passed": passed,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "metric": args.metric,
        "per_class_metric": args.per_class_metric,
        "max_drop": args.max_drop,
        "max_per_class_drop": args.max_per_class_drop,
        "requirements": requirements,
        "comparisons": generated_comparisons,
        "background_fp": fp_summaries,
        "blockers": blockers,
    }
    write_json(json_out, scorecard)

    status = "PASS" if passed else "FAIL"
    print(f"wrote_json={repo_rel(json_out)}")
    print(f"{status}: comparisons={len(generated_comparisons)} background_fp_splits={len(fp_summaries)}")
    if requirements["missing_comparisons"]:
        print(f"missing_comparisons: {', '.join(requirements['missing_comparisons'])}")
    if requirements["missing_background_splits"]:
        print(f"missing_background_splits: {', '.join(requirements['missing_background_splits'])}")
    for row in generated_comparisons:
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
    return 0 if passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
