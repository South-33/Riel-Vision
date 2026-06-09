#!/usr/bin/env python
"""Annotate overlap review rows with YOLO positive-error review signals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = Path("runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1.csv")

ADDED_FIELDS = [
    "model_error_total",
    "model_error_priority",
    "model_error_by_model",
    "model_error_types",
    "model_error_pairs",
    "model_error_top_overlays",
    "model_error_sources",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--error-csv", action="append", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--top-overlays", type=int, default=4)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_path(value: str) -> str:
    return value.strip().replace("\\", "/")


def row_images(row: dict[str, str]) -> list[str]:
    values = [normalized_path(str(row.get("image", "")))]
    values.extend(normalized_path(value) for value in str(row.get("variant_images", "")).split("|"))
    return [value for value in dict.fromkeys(values) if value]


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compact_counter(counter: Counter[str]) -> str:
    return ";".join(f"{key}={counter[key]}" for key in sorted(counter))


def error_pair(row: dict[str, str]) -> str:
    error_type = str(row.get("error_type", "")).strip()
    gt_class = str(row.get("gt_class", "")).strip()
    pred_class = str(row.get("pred_class", "")).strip()
    nearest_gt_class = str(row.get("nearest_gt_class", "")).strip()
    if error_type == "wrong_class":
        return f"{gt_class or '?'}->{pred_class or '?'}"
    if error_type == "unmatched_fp":
        return f"{nearest_gt_class or 'no_gt'}=>fp:{pred_class or '?'}"
    if error_type == "missed_gt":
        return f"{gt_class or nearest_gt_class or '?'}=>miss"
    return f"{error_type or 'error'}:{gt_class or nearest_gt_class or '?'}->{pred_class or '?'}"


def annotate_row(
    row: dict[str, str],
    errors_by_image: dict[str, list[dict[str, str]]],
    *,
    top_overlays: int,
) -> dict[str, str]:
    matched_errors: list[dict[str, str]] = []
    for image in row_images(row):
        matched_errors.extend(errors_by_image.get(image, []))

    out = dict(row)
    if not matched_errors:
        out.update({field: "" for field in ADDED_FIELDS})
        out["model_error_total"] = "0"
        return out

    by_model_type: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    sources: set[str] = set()
    for error in matched_errors:
        model = str(error.get("model", "")).strip() or "model"
        error_type = str(error.get("error_type", "")).strip() or "error"
        by_model_type[f"{model}:{error_type}"] += 1
        by_type[error_type] += 1
        pairs[error_pair(error)] += 1
        source = str(error.get("_source_csv", "")).strip()
        if source:
            sources.add(source)

    ranked_overlays = sorted(
        (error for error in matched_errors if str(error.get("overlay", "")).strip()),
        key=lambda item: as_float(str(item.get("review_score", ""))),
        reverse=True,
    )
    overlays = [normalized_path(str(error["overlay"])) for error in ranked_overlays[: max(0, top_overlays)]]

    out.update(
        {
            "model_error_total": str(len(matched_errors)),
            "model_error_priority": f"{max(as_float(str(error.get('review_score', ''))) for error in matched_errors):.6f}",
            "model_error_by_model": compact_counter(by_model_type),
            "model_error_types": compact_counter(by_type),
            "model_error_pairs": compact_counter(pairs),
            "model_error_top_overlays": "|".join(overlays),
            "model_error_sources": "|".join(sorted(sources)),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    review_csv = resolve(args.review_csv)
    out_csv = resolve(args.out_csv)
    review_rows = read_csv(review_csv)

    errors_by_image: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    error_sources: list[str] = []
    total_errors = 0
    for raw_error_csv in args.error_csv:
        error_csv = resolve(raw_error_csv)
        error_sources.append(repo_rel(error_csv))
        for error in read_csv(error_csv):
            image = normalized_path(str(error.get("image", "")))
            if not image:
                continue
            errors_by_image[image].append({**error, "_source_csv": repo_rel(error_csv)})
            total_errors += 1

    annotated = [
        annotate_row(row, errors_by_image, top_overlays=args.top_overlays)
        for row in review_rows
    ]
    fieldnames = list(review_rows[0].keys()) if review_rows else []
    for field in ADDED_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(out_csv, annotated, fieldnames)

    rows_with_errors = [row for row in annotated if int(row.get("model_error_total") or 0) > 0]
    summary = {
        "schema": "cashsnap_overlap_review_model_error_merge_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review_csv": repo_rel(review_csv),
        "error_csvs": error_sources,
        "out_csv": repo_rel(out_csv),
        "review_rows": len(review_rows),
        "input_error_rows": total_errors,
        "matched_review_rows": len(rows_with_errors),
        "unmatched_error_images": len(
            set(errors_by_image)
            - {image for row in review_rows for image in row_images(row)}
        ),
        "top_packet_buckets_with_errors": dict(
            Counter(str(row.get("packet_bucket", "")) for row in rows_with_errors).most_common()
        ),
        "top_sources_with_errors": dict(
            Counter(str(row.get("source_group", "")) for row in rows_with_errors).most_common()
        ),
        "not_training_data": True,
    }
    summary_json = resolve(args.summary_json) if args.summary_json else out_csv.with_suffix(".summary.json")
    write_summary(summary_json, summary)
    print(
        f"merged_overlap_errors={repo_rel(out_csv)} rows={len(review_rows)} "
        f"matched_rows={len(rows_with_errors)} errors={total_errors}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
