#!/usr/bin/env python
"""Filter a JSONL image manifest by source YOLO label geometry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--cashsnap-root", type=Path, default=Path("data/cashsnap_v1"))
    parser.add_argument("--max-boxes", type=int, default=1, help="Maximum source YOLO boxes per image; <0 disables.")
    parser.add_argument(
        "--max-box-area-fraction",
        type=float,
        default=0.80,
        help="Maximum area fraction for any single source YOLO box; <=0 disables.",
    )
    parser.add_argument(
        "--max-total-box-area-fraction",
        type=float,
        default=0.95,
        help="Maximum summed source YOLO box area fraction per image; <=0 disables.",
    )
    parser.add_argument(
        "--min-box-area-fraction",
        type=float,
        default=0.0,
        help="Minimum area fraction for every source YOLO box; <=0 disables.",
    )
    parser.add_argument("--keep-empty-labels", action="store_true")
    return parser.parse_args()


def label_path_for_image(image_path: Path, cashsnap_root: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return cashsnap_root / "labels" / "train" / f"{image_path.stem}.txt"
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if "image" not in payload:
            raise SystemExit(f"{repo_rel(path)}:{line_no} missing image field")
        payload["_line"] = line_no
        rows.append(payload)
    if not rows:
        raise SystemExit(f"Manifest is empty: {repo_rel(path)}")
    return rows


def read_label_boxes(label_path: Path) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    boxes: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields, got {len(parts)}")
        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} unknown class id {class_id}")
        width = float(parts[3])
        height = float(parts[4])
        boxes.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "width": width,
                "height": height,
                "area_fraction": width * height,
            }
        )
    return boxes


def rejection_reasons(boxes: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    if not boxes:
        return [] if args.keep_empty_labels else ["empty_labels"]
    reasons: list[str] = []
    if args.max_boxes >= 0 and len(boxes) > args.max_boxes:
        reasons.append("too_many_boxes")
    max_area = max(float(box["area_fraction"]) for box in boxes)
    total_area = sum(float(box["area_fraction"]) for box in boxes)
    min_area = min(float(box["area_fraction"]) for box in boxes)
    if args.max_box_area_fraction > 0 and max_area > args.max_box_area_fraction:
        reasons.append("box_too_large")
    if args.max_total_box_area_fraction > 0 and total_area > args.max_total_box_area_fraction:
        reasons.append("total_box_area_too_large")
    if args.min_box_area_fraction > 0 and min_area < args.min_box_area_fraction:
        reasons.append("box_too_small")
    return reasons


def compact_metrics(boxes: list[dict[str, Any]]) -> dict[str, Any]:
    if not boxes:
        return {"boxes": 0, "classes": [], "max_box_area_fraction": 0.0, "total_box_area_fraction": 0.0}
    return {
        "boxes": len(boxes),
        "classes": sorted({str(box["class_name"]) for box in boxes}),
        "max_box_area_fraction": round(max(float(box["area_fraction"]) for box in boxes), 6),
        "total_box_area_fraction": round(sum(float(box["area_fraction"]) for box in boxes), 6),
    }


def main() -> None:
    args = parse_args()
    if args.max_boxes < -1:
        raise SystemExit("--max-boxes must be >= -1")
    if args.max_box_area_fraction < 0 or args.max_total_box_area_fraction < 0 or args.min_box_area_fraction < 0:
        raise SystemExit("Area fraction thresholds must be >= 0")

    manifest_path = resolve(args.manifest)
    out_path = resolve(args.out)
    cashsnap_root = resolve(args.cashsnap_root)
    rows = read_jsonl(manifest_path)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    kept_classes: Counter[str] = Counter()
    removed_classes: Counter[str] = Counter()

    for row in rows:
        image_path = resolve(row["image"])
        label_path = label_path_for_image(image_path, cashsnap_root)
        boxes = read_label_boxes(label_path)
        reasons = rejection_reasons(boxes, args)
        clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
        metrics = compact_metrics(boxes)
        for class_name in metrics["classes"]:
            if reasons:
                removed_classes[class_name] += 1
            else:
                kept_classes[class_name] += 1
        if reasons:
            removed.append(
                {
                    "line": row["_line"],
                    "image": repo_rel(image_path),
                    "label": repo_rel(label_path),
                    "reasons": reasons,
                    **metrics,
                }
            )
        else:
            kept.append(clean_row)

    if not kept:
        raise SystemExit("Filtering removed every manifest row")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept), encoding="utf-8")
    summary = {
        "schema": "cashsnap_jsonl_manifest_yolo_geometry_filter_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": repo_rel(manifest_path),
        "out": repo_rel(out_path),
        "cashsnap_root": repo_rel(cashsnap_root),
        "thresholds": {
            "max_boxes": args.max_boxes,
            "max_box_area_fraction": args.max_box_area_fraction,
            "max_total_box_area_fraction": args.max_total_box_area_fraction,
            "min_box_area_fraction": args.min_box_area_fraction,
            "keep_empty_labels": args.keep_empty_labels,
        },
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "kept_classes": dict(sorted(kept_classes.items())),
        "removed_classes": dict(sorted(removed_classes.items())),
        "removed": removed,
    }
    summary_path = resolve(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"filtered_manifest={repo_rel(out_path)} input={len(rows)} "
        f"kept={len(kept)} removed={len(removed)} summary={repo_rel(summary_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
