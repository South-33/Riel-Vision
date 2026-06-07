from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = [
    "foreground_visible_fraction",
    "inpaint_source_box_fraction",
    "source_box_covered_by_foreground_fraction",
    "source_box_outside_foreground_fraction",
    "foreground_inside_source_box_fraction",
    "inpaint_mask_fraction",
    "inpaint_mask_to_foreground_area_ratio",
    "inpaint_mask_outside_foreground_fraction",
    "inpaint_mask_outside_foreground_to_foreground_ratio",
]

SOURCE_CONTEXT_DEFAULTS = {
    "max_row_inpaint_source_box_fraction": 0.90,
    "max_p95_inpaint_source_box_fraction": 0.80,
    "max_row_inpaint_mask_fraction": 0.95,
    "max_p95_inpaint_mask_fraction": 0.85,
    "max_row_inpaint_mask_to_foreground_area_ratio": 10.0,
    "max_p95_inpaint_mask_to_foreground_area_ratio": 6.5,
    "max_row_inpaint_mask_outside_foreground_to_foreground_ratio": 9.0,
    "max_p95_inpaint_mask_outside_foreground_to_foreground_ratio": 5.5,
    "max_p95_source_box_outside_foreground_fraction": 0.86,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check target-anchor transplant inpaint metadata. This catches broad "
            "source-context erase masks and source boxes that mostly sit outside "
            "the rendered foreground before expensive detector/visual audits."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path, help="Path to metadata/train.jsonl.")
    source.add_argument("--synthetic-root", type=Path, help="Synthetic root containing metadata/train.jsonl.")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("source_context", "report"),
        default="source_context",
        help="source_context applies broad-erase default limits; report only summarizes unless explicit limits are set.",
    )
    parser.add_argument("--max-row-inpaint-source-box-fraction", type=float)
    parser.add_argument("--max-p95-inpaint-source-box-fraction", type=float)
    parser.add_argument("--max-row-inpaint-mask-fraction", type=float)
    parser.add_argument("--max-p95-inpaint-mask-fraction", type=float)
    parser.add_argument("--max-row-inpaint-mask-to-foreground-area-ratio", type=float)
    parser.add_argument("--max-p95-inpaint-mask-to-foreground-area-ratio", type=float)
    parser.add_argument("--max-row-inpaint-mask-outside-foreground-to-foreground-ratio", type=float)
    parser.add_argument("--max-p95-inpaint-mask-outside-foreground-to-foreground-ratio", type=float)
    parser.add_argument("--max-p95-source-box-outside-foreground-fraction", type=float)
    parser.add_argument("--allow-missing-fields", action="store_true")
    parser.add_argument("--max-row-violations", type=int, default=100)
    parser.add_argument("--fail-on-violations", action="store_true")
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def metadata_path_from_args(args: argparse.Namespace) -> Path:
    if args.metadata is not None:
        return resolve_repo_path(args.metadata)
    assert args.synthetic_root is not None
    return resolve_repo_path(args.synthetic_root) / "metadata" / "train.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing metadata: {repo_rel(path)}")
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{repo_rel(path)}:{line_no}: invalid JSON: {exc}") from exc
        row["_line"] = line_no
        rows.append(row)
    if not rows:
        raise SystemExit(f"empty metadata: {repo_rel(path)}")
    return rows


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "p05": round(float(np.percentile(arr, 5)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def numeric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
    return None


def arg_limit(args: argparse.Namespace, name: str) -> float | None:
    value = getattr(args, name)
    if value is not None:
        return float(value)
    if args.profile == "source_context":
        return SOURCE_CONTEXT_DEFAULTS.get(name)
    return None


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    return {
        key: quantiles([value for row in rows if (value := numeric_value(row, key)) is not None])
        for key in REQUIRED_FIELDS
    }


def add_row_violations(
    rows: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    key: str,
    limit: float | None,
    label: str,
    max_rows: int,
) -> int:
    if limit is None:
        return 0
    count = 0
    for row in rows:
        value = numeric_value(row, key)
        if value is None or value <= limit:
            continue
        count += 1
        if len(violations) < max_rows:
            violations.append(
                {
                    "line": row.get("_line"),
                    "image": row.get("image"),
                    "class_name": row.get("class_name"),
                    "metric": key,
                    "value": round(value, 6),
                    "limit": limit,
                    "violation": label,
                }
            )
    return count


def add_p95_violation(
    summary: dict[str, dict[str, float | int | None]],
    violations: list[dict[str, Any]],
    key: str,
    limit: float | None,
    label: str,
) -> None:
    if limit is None:
        return
    p95 = summary.get(key, {}).get("p95")
    if not isinstance(p95, (int, float)) or p95 <= limit:
        return
    violations.append(
        {
            "metric": key,
            "p95": round(float(p95), 6),
            "limit": limit,
            "violation": label,
        }
    )


def main() -> None:
    args = parse_args()
    metadata_path = metadata_path_from_args(args)
    rows = read_jsonl(metadata_path)
    missing_fields = [
        {"line": row.get("_line"), "image": row.get("image"), "field": key}
        for row in rows
        for key in REQUIRED_FIELDS
        if key not in row
    ]

    summary = metric_summary(rows)
    row_violations: list[dict[str, Any]] = []
    row_violation_count = 0
    aggregate_violations: list[dict[str, Any]] = []

    row_violation_count += add_row_violations(
        rows,
        row_violations,
        "inpaint_source_box_fraction",
        arg_limit(args, "max_row_inpaint_source_box_fraction"),
        "row_source_box_too_large",
        args.max_row_violations,
    )
    row_violation_count += add_row_violations(
        rows,
        row_violations,
        "inpaint_mask_fraction",
        arg_limit(args, "max_row_inpaint_mask_fraction"),
        "row_inpaint_mask_too_large",
        args.max_row_violations,
    )
    row_violation_count += add_row_violations(
        rows,
        row_violations,
        "inpaint_mask_to_foreground_area_ratio",
        arg_limit(args, "max_row_inpaint_mask_to_foreground_area_ratio"),
        "row_inpaint_mask_too_large_vs_foreground",
        args.max_row_violations,
    )
    row_violation_count += add_row_violations(
        rows,
        row_violations,
        "inpaint_mask_outside_foreground_to_foreground_ratio",
        arg_limit(args, "max_row_inpaint_mask_outside_foreground_to_foreground_ratio"),
        "row_inpaint_context_too_large_vs_foreground",
        args.max_row_violations,
    )

    add_p95_violation(
        summary,
        aggregate_violations,
        "inpaint_source_box_fraction",
        arg_limit(args, "max_p95_inpaint_source_box_fraction"),
        "p95_source_box_too_large",
    )
    add_p95_violation(
        summary,
        aggregate_violations,
        "inpaint_mask_fraction",
        arg_limit(args, "max_p95_inpaint_mask_fraction"),
        "p95_inpaint_mask_too_large",
    )
    add_p95_violation(
        summary,
        aggregate_violations,
        "inpaint_mask_to_foreground_area_ratio",
        arg_limit(args, "max_p95_inpaint_mask_to_foreground_area_ratio"),
        "p95_inpaint_mask_too_large_vs_foreground",
    )
    add_p95_violation(
        summary,
        aggregate_violations,
        "inpaint_mask_outside_foreground_to_foreground_ratio",
        arg_limit(args, "max_p95_inpaint_mask_outside_foreground_to_foreground_ratio"),
        "p95_inpaint_context_too_large_vs_foreground",
    )
    add_p95_violation(
        summary,
        aggregate_violations,
        "source_box_outside_foreground_fraction",
        arg_limit(args, "max_p95_source_box_outside_foreground_fraction"),
        "p95_source_box_poorly_covered_by_foreground",
    )

    missing_violation = [] if args.allow_missing_fields else missing_fields
    status = "pass" if not (missing_violation or row_violations or aggregate_violations) else "fail"
    payload = {
        "schema": "target_anchor_inpaint_metadata_check_v1",
        "status": status,
        "profile": args.profile,
        "metadata": repo_rel(metadata_path),
        "rows": len(rows),
        "missing_fields": missing_fields[: args.max_row_violations],
        "missing_field_count": len(missing_fields),
        "metrics": summary,
        "row_violations": row_violations,
        "row_violation_count": row_violation_count,
        "aggregate_violations": aggregate_violations,
        "aggregate_violation_count": len(aggregate_violations),
    }
    out_path = resolve_repo_path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "rows": len(rows),
                "row_violations": row_violation_count,
                "aggregate_violations": len(aggregate_violations),
                "missing_fields": len(missing_fields),
                "json_out": repo_rel(out_path),
            },
            sort_keys=True,
        )
    )
    if args.fail_on_violations and status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
