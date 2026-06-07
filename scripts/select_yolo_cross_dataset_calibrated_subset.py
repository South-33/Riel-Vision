#!/usr/bin/env python
"""Select synthetic YOLO images that best match a reference YOLO bridge."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

import audit_yolo_cross_dataset_visual_gap as cross_gap
import audit_yolo_domain_gap as domain_gap


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "box_area",
    "box_aspect",
    "box_width",
    "box_height",
    "luma_mean",
    "luma_std",
    "luma_p05",
    "luma_p95",
    "saturation_mean",
    "saturation_std",
    "sharpness_grad_var",
]
FEATURE_WEIGHTS = {
    "box_area": 1.35,
    "box_aspect": 1.35,
    "box_width": 0.65,
    "box_height": 0.65,
    "luma_mean": 0.55,
    "luma_std": 1.25,
    "luma_p05": 1.05,
    "luma_p95": 0.85,
    "saturation_mean": 0.55,
    "saturation_std": 0.90,
    "sharpness_grad_var": 0.45,
}
MIN_SCALE = {
    "box_area": 0.08,
    "box_aspect": 0.20,
    "box_width": 0.06,
    "box_height": 0.06,
    "luma_mean": 0.05,
    "luma_std": 0.035,
    "luma_p05": 0.05,
    "luma_p95": 0.05,
    "saturation_mean": 0.05,
    "saturation_std": 0.035,
    "sharpness_grad_var": 0.006,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-data", required=True, type=Path)
    parser.add_argument("--reference-split", action="append", required=True)
    parser.add_argument("--candidate-data", action="append", required=True, type=Path)
    parser.add_argument("--candidate-split", default="train")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out-list", required=True, type=Path)
    parser.add_argument("--out-data", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--pad-frac", type=float, default=0.02)
    parser.add_argument("--min-crop-pixels", type=int, default=16)
    parser.add_argument("--max-images-per-family", type=int, default=None)
    parser.add_argument("--min-per-class", type=int, default=1)
    parser.add_argument("--val", default="data/cashsnap_v1/images/val")
    parser.add_argument("--test", default="data/cashsnap_v1/images/test")
    parser.add_argument("--metadata-note", default="")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SystemExit(f"expected YAML mapping: {repo_rel(path)}")
    return doc


def class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names")
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise SystemExit("YOLO config must include names")


def collect_rows(data_path: Path, splits: list[str], family: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    image_rows, box_rows, crop_rows = cross_gap.collect_family_rows(
        family=family,
        config_path=data_path,
        splits=splits,
        class_filter=set(),
        args=args,
    )
    crop_by_key = {
        (row["image"], int(row["label_index"])): row
        for row in crop_rows
    }
    rows: list[dict[str, Any]] = []
    for box in box_rows:
        key = (box["image"], int(box["label_index"]))
        crop = crop_by_key.get(key)
        if crop is None:
            continue
        row = {
            "family": family,
            "source_data": repo_rel(data_path),
            "split": box["split"],
            "image": box["image"],
            "class_id": int(box["class_id"]),
            "class_name": str(box["class_name"]),
            "label_index": int(box["label_index"]),
        }
        for feature in FEATURES:
            if feature in box:
                row[feature] = float(box[feature])
            elif feature in crop:
                row[feature] = float(crop[feature])
        rows.append(row)
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    assert avg is not None
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def target_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    by_class: dict[str, dict[str, dict[str, float]]] = {}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["class_name"])].append(row)
    for class_name, class_rows in grouped.items():
        by_class[class_name] = {}
        for feature in FEATURES:
            values = [float(row[feature]) for row in class_rows if row.get(feature) is not None]
            avg = mean(values)
            if avg is None:
                continue
            spread = stdev(values)
            by_class[class_name][feature] = {
                "mean": avg,
                "stdev": spread if spread is not None else 0.0,
            }
    return by_class


def row_score(row: dict[str, Any], targets: dict[str, dict[str, dict[str, float]]]) -> tuple[float, dict[str, float]]:
    class_targets = targets.get(str(row["class_name"]), {})
    total = 0.0
    parts: dict[str, float] = {}
    for feature in FEATURES:
        if feature not in row or feature not in class_targets:
            continue
        target = class_targets[feature]["mean"]
        scale = max(MIN_SCALE[feature], class_targets[feature].get("stdev", 0.0))
        contribution = FEATURE_WEIGHTS[feature] * abs(float(row[feature]) - target) / scale
        parts[feature] = contribution
        total += contribution
    return total, parts


def desired_counts(class_order: list[str], reference_counts: Counter[str], count: int, min_per_class: int) -> dict[str, int]:
    if count < len(class_order) * min_per_class:
        raise SystemExit(f"--count {count} cannot satisfy --min-per-class {min_per_class} for {len(class_order)} classes")
    desired = {class_name: min_per_class for class_name in class_order}
    remaining = count - len(class_order) * min_per_class
    weights = {class_name: max(1, int(reference_counts[class_name])) for class_name in class_order}
    while remaining > 0:
        class_name = min(
            class_order,
            key=lambda name: (desired[name] / weights[name], desired[name], name),
        )
        desired[class_name] += 1
        remaining -= 1
    return desired


def select_rows(candidates: list[dict[str, Any]], reference: list[dict[str, Any]], count: int, min_per_class: int) -> list[dict[str, Any]]:
    targets = target_stats(reference)
    reference_counts: Counter[str] = Counter(str(row["class_name"]) for row in reference)
    class_order = sorted(targets)
    desired = desired_counts(class_order, reference_counts, count, min_per_class)
    scored: list[dict[str, Any]] = []
    for row in candidates:
        if str(row["class_name"]) not in desired:
            continue
        score, parts = row_score(row, targets)
        scored.append({**row, "score": score, "score_parts": parts})

    by_class: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_class[str(row["class_name"])].append(row)
    for rows in by_class.values():
        rows.sort(key=lambda row: (float(row["score"]), row["image"]))

    selected: list[dict[str, Any]] = []
    selected_images: set[str] = set()
    for class_name in class_order:
        available = [row for row in by_class.get(class_name, []) if row["image"] not in selected_images]
        take = min(desired[class_name], len(available))
        if take < min_per_class:
            raise SystemExit(f"class {class_name} has only {take} selectable rows, below --min-per-class {min_per_class}")
        for row in available[:take]:
            selected.append(row)
            selected_images.add(str(row["image"]))

    if len(selected) < count:
        leftovers = sorted(
            (row for row in scored if row["image"] not in selected_images),
            key=lambda row: (float(row["score"]), row["image"]),
        )
        for row in leftovers:
            selected.append(row)
            selected_images.add(str(row["image"]))
            if len(selected) >= count:
                break
    if len(selected) != count:
        raise SystemExit(f"selected {len(selected)} rows, expected {count}")
    selected.sort(key=lambda row: (str(row["class_name"]), float(row["score"]), str(row["image"])))
    return selected


def write_data_yaml(path: Path, train_list: Path, reference_data: Path, args: argparse.Namespace) -> None:
    config = read_yaml(reference_data)
    names = class_names(config)
    payload = {
        "path": "../..",
        "train": repo_rel(train_list),
        "val": args.val,
        "test": args.test,
        "names": {int(index): name for index, name in sorted(names.items())},
        "cashsnap_calibration": {
            "schema": "cashsnap_yolo_cross_dataset_calibrated_subset_v1",
            "reference_data": repo_rel(reference_data),
            "reference_splits": args.reference_split,
            "candidate_data": [repo_rel(resolve(path)) for path in args.candidate_data],
            "note": args.metadata_note,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.min_per_class < 0:
        raise SystemExit("--min-per-class must be >= 0")
    reference_data = resolve(args.reference_data)
    reference_rows = collect_rows(reference_data, args.reference_split, "reference", args)
    candidate_rows: list[dict[str, Any]] = []
    for data_path in args.candidate_data:
        candidate_rows.extend(collect_rows(resolve(data_path), [args.candidate_split], "candidate", args))
    selected = select_rows(candidate_rows, reference_rows, args.count, args.min_per_class)

    out_list = resolve(args.out_list)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(row["image"] for row in selected) + "\n", encoding="utf-8")
    out_data = resolve(args.out_data)
    write_data_yaml(out_data, out_list, reference_data, args)

    selected_counts = Counter(str(row["class_name"]) for row in selected)
    source_counts = Counter(str(row["source_data"]) for row in selected)
    payload = {
        "schema": "cashsnap_yolo_cross_dataset_calibrated_subset_v1",
        "reference_data": repo_rel(reference_data),
        "reference_splits": args.reference_split,
        "candidate_data": [repo_rel(resolve(path)) for path in args.candidate_data],
        "count": len(selected),
        "selected_class_counts": dict(sorted(selected_counts.items())),
        "selected_source_counts": dict(sorted(source_counts.items())),
        "mean_score": sum(float(row["score"]) for row in selected) / len(selected),
        "outputs": {
            "train_list": repo_rel(out_list),
            "data_yaml": repo_rel(out_data),
        },
        "selected": [
            {
                "image": row["image"],
                "class_name": row["class_name"],
                "source_data": row["source_data"],
                "score": round(float(row["score"]), 6),
                "score_parts": {key: round(float(value), 6) for key, value in sorted(row["score_parts"].items())},
            }
            for row in selected
        ],
    }
    summary_path = resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"selected={len(selected)} classes={dict(sorted(selected_counts.items()))} "
        f"sources={dict(sorted(source_counts.items()))} mean_score={payload['mean_score']:.3f}"
    )
    print(f"wrote_list={repo_rel(out_list)}")
    print(f"wrote_data={repo_rel(out_data)}")
    print(f"wrote_summary={repo_rel(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
