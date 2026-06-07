#!/usr/bin/env python
"""Compare visual/geometry stats between two arbitrary YOLO datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

import audit_yolo_crop_visual_domain_gap as crop_gap
import audit_yolo_domain_gap as domain_gap


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-data", required=True, type=Path, help="Reference YOLO YAML, e.g. real bridge.")
    parser.add_argument("--reference-split", action="append", required=True, help="Reference split key. Repeatable.")
    parser.add_argument("--candidate-data", required=True, type=Path, help="Candidate YOLO YAML, e.g. synthetic train config.")
    parser.add_argument("--candidate-split", action="append", required=True, help="Candidate split key. Repeatable.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--image-csv-out", type=Path, default=None)
    parser.add_argument("--box-csv-out", type=Path, default=None)
    parser.add_argument("--crop-csv-out", type=Path, default=None)
    parser.add_argument("--class-name", action="append", default=[], help="Optional class filter. Repeatable/comma-separated.")
    parser.add_argument("--pad-frac", type=float, default=0.0)
    parser.add_argument("--min-crop-pixels", type=int, default=16)
    parser.add_argument("--top-class-deltas", type=int, default=12)
    parser.add_argument("--max-images-per-family", type=int, default=None)
    parser.add_argument(
        "--min-labels",
        type=int,
        default=0,
        help="Skip images with fewer labels after optional class filtering.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Shuffle each family before applying --max-images-per-family.",
    )
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
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"expected YAML mapping: {repo_rel(path)}")
    return document


def parse_class_names(values: list[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        for item in re.split(r"[,\s]+", value):
            class_name = item.strip()
            if class_name:
                names.add(class_name)
    return names


def class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names")
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise SystemExit("YOLO config must include names")


def split_images(config_path: Path, config: dict[str, Any], splits: list[str]) -> list[tuple[str, Path]]:
    root = domain_gap.data_root(config_path, config)
    rows: list[tuple[str, Path]] = []
    for split in splits:
        if split not in config:
            raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
        rows.extend((split, image) for image in domain_gap.iter_split_images(root, config[split]))
    return rows


def yolo_crop_box(row: dict[str, Any], image_size: tuple[int, int], pad_frac: float) -> tuple[int, int, int, int] | None:
    width, height = image_size
    cx = float(row["x_center"]) * width
    cy = float(row["y_center"]) * height
    box_w = float(row["box_width"]) * width
    box_h = float(row["box_height"]) * height
    left = cx - box_w / 2.0
    top = cy - box_h / 2.0
    right = cx + box_w / 2.0
    bottom = cy + box_h / 2.0
    pad_x = box_w * max(0.0, pad_frac)
    pad_y = box_h * max(0.0, pad_frac)
    x1 = max(0, int(round(left - pad_x)))
    y1 = max(0, int(round(top - pad_y)))
    x2 = min(width, int(round(right + pad_x)))
    y2 = min(height, int(round(bottom + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def collect_family_rows(
    *,
    family: str,
    config_path: Path,
    splits: list[str],
    class_filter: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = read_yaml(config_path)
    names = class_names(config)
    images = split_images(config_path, config, splits)
    if args.sample_seed is not None:
        seed_offset = 0 if family == "reference" else 1_000_003
        rng = random.Random(args.sample_seed + seed_offset)
        rng.shuffle(images)
    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    for split, image_path in images:
        labels = [
            label_row
            for label_row in domain_gap.label_rows(image_path, names)
            if not class_filter or str(label_row["class_name"]) in class_filter
        ]
        if len(labels) < args.min_labels:
            continue
        if args.max_images_per_family is not None and len(image_rows) >= args.max_images_per_family:
            break
        stats = domain_gap.image_stats(image_path)
        base = {
            "family": family,
            "split": split,
            "image": repo_rel(image_path),
            "source_group": domain_gap.source_group(image_path),
        }
        image_rows.append({**base, **stats})
        with Image.open(image_path) as image:
            image_size = image.size
            for label_index, label_row in enumerate(labels):
                class_name = str(label_row["class_name"])
                box_row = {**base, **label_row, "label_index": label_index}
                box_rows.append(box_row)
                crop_box = yolo_crop_box(label_row, image_size, args.pad_frac)
                if crop_box is None:
                    continue
                left, top, right, bottom = crop_box
                if (right - left) * (bottom - top) < args.min_crop_pixels:
                    continue
                crop_stats = crop_gap.crop_stats(image.crop(crop_box))
                crop_rows.append(
                    {
                        **base,
                        "label_index": label_index,
                        "class_id": int(label_row["class_id"]),
                        "class_name": class_name,
                        "crop_left": left,
                        "crop_top": top,
                        "crop_right": right,
                        "crop_bottom": bottom,
                        **crop_stats,
                    }
                )
    return image_rows, box_rows, crop_rows


def summary_by_family(
    image_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    crop_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    families = sorted({row["family"] for row in image_rows})
    payload: dict[str, Any] = {}
    for family in families:
        family_images = [row for row in image_rows if row["family"] == family]
        family_boxes = [row for row in box_rows if row["family"] == family]
        family_crops = [row for row in crop_rows if row["family"] == family]
        payload[family] = {
            "images": len(family_images),
            "boxes": len(family_boxes),
            "crops": len(family_crops),
            "class_counts": dict(sorted((str(k), int(v)) for k, v in class_counts(family_boxes).items())),
            "image_stats": domain_gap.summarize_numeric(family_images, domain_gap.IMAGE_STAT_KEYS),
            "box_stats": domain_gap.summarize_numeric(family_boxes, domain_gap.BOX_STAT_KEYS),
            "crop_stats": domain_gap.summarize_numeric(family_crops, crop_gap.CROP_STAT_KEYS),
            "class_box_stats": domain_gap.summarize_class_box_stats(family_boxes),
            "class_crop_stats": crop_gap.summarize_class_stats(family_crops),
        }
    return payload


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["class_name"])] += 1
    return counts


def mean_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for section in ["image_stats", "box_stats", "crop_stats"]:
        deltas[section] = {}
        for metric, reference_stats in reference.get(section, {}).items():
            reference_mean = reference_stats.get("mean")
            candidate_mean = candidate.get(section, {}).get(metric, {}).get("mean")
            deltas[section][metric] = (
                None if reference_mean is None or candidate_mean is None else candidate_mean - reference_mean
            )
    deltas["class_box_stats"] = class_metric_deltas(
        candidate.get("class_box_stats", {}),
        reference.get("class_box_stats", {}),
        domain_gap.BOX_STAT_KEYS,
    )
    deltas["class_crop_stats"] = class_metric_deltas(
        candidate.get("class_crop_stats", {}),
        reference.get("class_crop_stats", {}),
        crop_gap.CROP_STAT_KEYS,
    )
    return deltas


def class_metric_deltas(candidate: dict[str, Any], reference: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for class_name in sorted(set(candidate) | set(reference)):
        out[class_name] = {}
        for metric in metrics:
            reference_mean = reference.get(class_name, {}).get(metric, {}).get("mean")
            candidate_mean = candidate.get(class_name, {}).get(metric, {}).get("mean")
            out[class_name][metric] = (
                None if reference_mean is None or candidate_mean is None else candidate_mean - reference_mean
            )
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_class_delta_text(deltas: dict[str, Any], metric: str, limit: int) -> str:
    rows: list[tuple[float, str]] = []
    for class_name, metrics in deltas.items():
        value = metrics.get(metric)
        if value is not None:
            rows.append((float(value), str(class_name)))
    if not rows:
        return "none"
    rows.sort(key=lambda item: abs(item[0]), reverse=True)
    return ", ".join(f"{name} {value:+.3f}" for value, name in rows[:limit])


def main() -> int:
    args = parse_args()
    class_filter = parse_class_names(args.class_name)
    reference_data = resolve(args.reference_data)
    candidate_data = resolve(args.candidate_data)
    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    for family, config_path, splits in [
        ("reference", reference_data, args.reference_split),
        ("candidate", candidate_data, args.candidate_split),
    ]:
        family_images, family_boxes, family_crops = collect_family_rows(
            family=family,
            config_path=config_path,
            splits=splits,
            class_filter=class_filter,
            args=args,
        )
        image_rows.extend(family_images)
        box_rows.extend(family_boxes)
        crop_rows.extend(family_crops)

    by_family = summary_by_family(image_rows, box_rows, crop_rows)
    reference = by_family.get("reference", {})
    candidate = by_family.get("candidate", {})
    deltas = {"candidate_minus_reference": mean_delta(candidate, reference)}
    payload = {
        "schema": "cashsnap_yolo_cross_dataset_visual_gap_v1",
        "reference": {
            "data": repo_rel(reference_data),
            "splits": args.reference_split,
        },
        "candidate": {
            "data": repo_rel(candidate_data),
            "splits": args.candidate_split,
        },
        "class_filter": sorted(class_filter),
        "min_labels": args.min_labels,
        "pad_frac": args.pad_frac,
        "sample_seed": args.sample_seed,
        "by_family": by_family,
        "deltas": deltas,
    }

    if args.image_csv_out:
        write_csv(image_rows, resolve(args.image_csv_out))
    if args.box_csv_out:
        write_csv(box_rows, resolve(args.box_csv_out))
    if args.crop_csv_out:
        write_csv(crop_rows, resolve(args.crop_csv_out))
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_rel(out)}")

    print(
        "families="
        f"reference images={reference.get('images', 0)} boxes={reference.get('boxes', 0)} crops={reference.get('crops', 0)}; "
        f"candidate images={candidate.get('images', 0)} boxes={candidate.get('boxes', 0)} crops={candidate.get('crops', 0)}"
    )
    crop_deltas = deltas["candidate_minus_reference"].get("crop_stats", {})
    if crop_deltas:
        key_text = ", ".join(
            f"{metric} {float(crop_deltas[metric]):+.3f}"
            for metric in ["luma_mean", "luma_std", "luma_p05", "luma_p95", "saturation_mean", "saturation_std"]
            if crop_deltas.get(metric) is not None
        )
        print(f"crop_deltas candidate-reference: {key_text}")
    class_crop = deltas["candidate_minus_reference"].get("class_crop_stats", {})
    print("top_class_luma_std_delta: " + top_class_delta_text(class_crop, "luma_std", args.top_class_deltas))
    print("top_class_saturation_std_delta: " + top_class_delta_text(class_crop, "saturation_std", args.top_class_deltas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
