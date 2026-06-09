#!/usr/bin/env python
"""Build diagnostic YOLO views from a real-overlap review CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATA = Path("data/cashsnap_v1/data.yaml")
DEFAULT_REVIEW_CSV = Path(
    "runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1_model_error_triage.csv"
)
DEFAULT_OUT_DIR = Path("runs/cashsnap/real_overlap_review_diagnostic_views_v1")
DEFAULT_HELDOUT_SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--heldout-split",
        action="append",
        default=[],
        help="Split to include in the heldout diagnostic view. Repeatable. Defaults to val and test.",
    )
    parser.add_argument(
        "--include-bucket",
        action="append",
        default=[],
        help="Optional packet_bucket filter. Repeatable. Defaults to all buckets.",
    )
    parser.add_argument(
        "--min-model-error-total",
        type=int,
        default=0,
        help="Optional minimum model_error_total filter for the all/error views.",
    )
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Write empty list/YAML files for empty filtered views.",
    )
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


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "unknown"


def image_split(image: str) -> str:
    parts = image.replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        if part == "images" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def int_value(value: str) -> int:
    try:
        return int(float(str(value).strip() or "0"))
    except ValueError:
        return 0


def float_value(value: str) -> float:
    try:
        return float(str(value).strip() or "0")
    except ValueError:
        return 0.0


def split_values(values: list[str], default: tuple[str, ...]) -> set[str]:
    selected: list[str] = []
    for value in values:
        selected.extend(part.strip() for part in value.split(",") if part.strip())
    if not selected:
        selected = list(default)
    return {value.lower() for value in selected}


def label_path_for(image: str) -> Path:
    raw = Path(image.replace("\\", "/"))
    parts = list(raw.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return ROOT / raw.with_suffix(".txt")
    parts[index] = "labels"
    return ROOT / Path(*parts).with_suffix(".txt")


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("split", ""),
            row.get("packet_bucket", ""),
            row.get("source_group", ""),
            -float_value(row.get("priority", "")),
            row.get("image", ""),
        ),
    )


def unique_images(rows: list[dict[str, str]]) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    for row in sort_rows(rows):
        image = row.get("image", "").strip().replace("\\", "/")
        if not image or image in seen:
            continue
        seen.add(image)
        images.append(image)
    return images


def view_summary(name: str, rows: list[dict[str, str]], images: list[str]) -> dict[str, Any]:
    missing_images = [image for image in images if not (ROOT / image).exists()]
    missing_labels = [image for image in images if not label_path_for(image).exists()]
    return {
        "name": name,
        "rows": len(rows),
        "images": len(images),
        "split_counts": dict(Counter(row.get("split", "") or image_split(row.get("image", "")) for row in rows)),
        "packet_bucket_counts": dict(Counter(row.get("packet_bucket", "") or "unbucketed" for row in rows)),
        "source_counts": dict(Counter(row.get("source_group", "") or "unknown" for row in rows)),
        "model_error_total": sum(int_value(row.get("model_error_total", "0")) for row in rows),
        "missing_images": missing_images[:20],
        "missing_labels": missing_labels[:20],
        "missing_image_count": len(missing_images),
        "missing_label_count": len(missing_labels),
    }


def write_view(
    *,
    out_dir: Path,
    base_names: Any,
    review_csv: Path,
    name: str,
    rows: list[dict[str, str]],
    write_empty: bool,
    reason: str,
) -> dict[str, Any] | None:
    images = unique_images(rows)
    if not images and not write_empty:
        return None

    list_path = out_dir / f"{slug(name)}_images.txt"
    data_path = out_dir / f"{slug(name)}_data.yaml"
    list_path.write_text("".join(f"{image}\n" for image in images), encoding="utf-8")

    data = {
        "path": rel_between(out_dir, ROOT),
        "train": repo_rel(list_path),
        "val": repo_rel(list_path),
        "test": repo_rel(list_path),
        "names": base_names,
        "cashsnap_diagnostic": {
            "purpose": "Diagnostic YOLO view for real overlap review packet analysis.",
            "review_csv": repo_rel(review_csv),
            "view": name,
            "filter": reason,
            "not_a_promotion_config": True,
            "uses_representative_images_only": True,
        },
    }
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    summary = view_summary(name, rows, images)
    summary.update(
        {
            "images_txt": repo_rel(list_path),
            "data_yaml": repo_rel(data_path),
            "filter": reason,
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    review_csv = resolve(args.review_csv)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_data = read_yaml(args.base_data)
    base_names = base_data.get("names")
    if base_names is None:
        raise SystemExit(f"{repo_rel(resolve(args.base_data))} is missing names")

    rows = read_csv(review_csv)
    for row in rows:
        if not row.get("split", "").strip():
            row["split"] = image_split(row.get("image", ""))

    include_buckets = {slug(value) for value in args.include_bucket}
    if include_buckets:
        rows = [row for row in rows if slug(row.get("packet_bucket", "")) in include_buckets]
    if args.min_model_error_total > 0:
        rows = [row for row in rows if int_value(row.get("model_error_total", "0")) >= args.min_model_error_total]

    heldout_splits = split_values(args.heldout_split, DEFAULT_HELDOUT_SPLITS)
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bucket[row.get("packet_bucket", "") or "unbucketed"].append(row)

    views: list[tuple[str, list[dict[str, str]], str]] = [
        ("all_representatives", rows, "all representative rows"),
        (
            "heldout_representatives",
            [row for row in rows if row.get("split", "").lower() in heldout_splits],
            f"split in {','.join(sorted(heldout_splits))}",
        ),
        (
            "train_representatives",
            [row for row in rows if row.get("split", "").lower() == "train"],
            "split=train",
        ),
    ]

    error_rows = [row for row in rows if int_value(row.get("model_error_total", "0")) > 0]
    views.append(("model_error_representatives", error_rows, "model_error_total>0"))
    views.append(
        (
            "heldout_model_error_representatives",
            [row for row in error_rows if row.get("split", "").lower() in heldout_splits],
            f"model_error_total>0 and split in {','.join(sorted(heldout_splits))}",
        )
    )

    for bucket, bucket_rows in sorted(by_bucket.items()):
        views.append((f"bucket_{bucket}", bucket_rows, f"packet_bucket={bucket}"))
        heldout_bucket_rows = [
            row for row in bucket_rows if row.get("split", "").lower() in heldout_splits
        ]
        views.append(
            (
                f"heldout_bucket_{bucket}",
                heldout_bucket_rows,
                f"packet_bucket={bucket} and split in {','.join(sorted(heldout_splits))}",
            )
        )

    written = []
    for name, view_rows, reason in views:
        summary = write_view(
            out_dir=out_dir,
            base_names=base_names,
            review_csv=review_csv,
            name=name,
            rows=view_rows,
            write_empty=args.write_empty,
            reason=reason,
        )
        if summary is not None:
            written.append(summary)

    summary = {
        "schema": "cashsnap_real_overlap_review_diagnostic_views_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_data": repo_rel(resolve(args.base_data)),
        "review_csv": repo_rel(review_csv),
        "out_dir": repo_rel(out_dir),
        "heldout_splits": sorted(heldout_splits),
        "include_buckets": sorted(include_buckets),
        "min_model_error_total": args.min_model_error_total,
        "input_rows_after_filters": len(rows),
        "not_training_data": True,
        "not_a_promotion_config": True,
        "views_written": len(written),
        "views": written,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
