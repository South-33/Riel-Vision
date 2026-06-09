#!/usr/bin/env python
"""Build a review queue for real overlap, fan, partial, and multi-note candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PROTECTED_RIEL = {"KHR_20000", "KHR_50000"}
SOURCE_PREFIXES = {
    "asian_currency_": "asian_currency",
    "billsbank_": "billsbank",
    "cambodia_currency_project_": "cambodia_currency_project",
    "cashcountingxl_": "cashcountingxl",
    "khmer_us_currency_": "khmer_us_currency",
    "usd_total_": "usd_total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cashsnap_v1/data.yaml"))
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("runs/cashsnap/real_overlap_review_queue_v1"))
    parser.add_argument("--edge-margin", type=float, default=0.03)
    parser.add_argument("--tight-gap", type=float, default=0.035)
    parser.add_argument("--overlap-small-ratio", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=400)
    parser.add_argument("--sheet-items", type=int, default=120)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--cols", type=int, default=5)
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


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = Path(str(config.get("path", "."))).expanduser()
    return raw_root if raw_root.is_absolute() else (resolve(config_path).parent / raw_root).resolve()


def split_root(root: Path, split_path: str) -> Path:
    raw_path = Path(split_path).expanduser()
    return raw_path if raw_path.is_absolute() else root / raw_path


def read_split_list(root: Path, split_path: str) -> list[Path]:
    list_path = split_root(root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line).expanduser()
        images.append(image if image.is_absolute() else root / image)
    return images


def split_images(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(resolve(config_path))} missing split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for value in values:
        split_path = split_root(root, str(value))
        if split_path.suffix.lower() == ".txt":
            images.extend(read_split_list(root, str(value)))
            continue
        if split_path.is_dir():
            images.extend(sorted(path for path in split_path.iterdir() if path.suffix.lower() in IMAGE_EXTS))
            continue
        raise SystemExit(f"unsupported split path: {repo_rel(split_path)}")
    return images


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names", {})
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(class_id): str(name) for class_id, name in raw.items()}
    raise SystemExit("data YAML names must be a list or mapping")


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_group_for_image(image_path: Path) -> str:
    name = image_path.name.lower()
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return name.split("_", 1)[0] if "_" in name else "unknown"


def canonical_image_key(image_path: Path) -> str:
    stem = image_path.stem
    stem = re.sub(r"\.rf\.[0-9a-fA-F]+$", "", stem)
    return f"{source_group_for_image(image_path)}|{stem}"


def read_labels(image_path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        class_id = int(float(parts[0]))
        if class_id not in names:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} class id {class_id} outside names")
        cx, cy, width, height = [float(value) for value in parts[1:]]
        x1 = cx - width / 2.0
        y1 = cy - height / 2.0
        x2 = cx + width / 2.0
        y2 = cy + height / 2.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": width * height,
            }
        )
    return rows


def box_intersection(a: dict[str, Any], b: dict[str, Any]) -> float:
    width = max(0.0, min(float(a["x2"]), float(b["x2"])) - max(float(a["x1"]), float(b["x1"])))
    height = max(0.0, min(float(a["y2"]), float(b["y2"])) - max(float(a["y1"]), float(b["y1"])))
    return width * height


def pair_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    horizontal = max(0.0, max(float(a["x1"]), float(b["x1"])) - min(float(a["x2"]), float(b["x2"])))
    vertical = max(0.0, max(float(a["y1"]), float(b["y1"])) - min(float(a["y2"]), float(b["y2"])))
    return math.hypot(horizontal, vertical)


def pair_metrics(labels: list[dict[str, Any]]) -> dict[str, Any]:
    max_iou = 0.0
    max_intersection_small_ratio = 0.0
    min_gap = None
    pairs = 0
    best_pair = ""
    for left_index, left in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right = labels[right_index]
            pairs += 1
            intersection = box_intersection(left, right)
            union = float(left["area"]) + float(right["area"]) - intersection
            iou = intersection / union if union > 0 else 0.0
            small_area = max(1e-9, min(float(left["area"]), float(right["area"])))
            small_ratio = intersection / small_area
            gap = pair_gap(left, right)
            if iou > max_iou:
                max_iou = iou
            if small_ratio > max_intersection_small_ratio:
                max_intersection_small_ratio = small_ratio
            if min_gap is None or gap < min_gap:
                min_gap = gap
                best_pair = f"{left['class_name']}|{right['class_name']}"
    return {
        "pairs": pairs,
        "max_iou": max_iou,
        "max_intersection_small_ratio": max_intersection_small_ratio,
        "min_gap": min_gap,
        "closest_pair": best_pair,
    }


def touches_edge(label: dict[str, Any], edge_margin: float) -> bool:
    return (
        float(label["x1"]) <= edge_margin
        or float(label["y1"]) <= edge_margin
        or float(label["x2"]) >= 1.0 - edge_margin
        or float(label["y2"]) >= 1.0 - edge_margin
    )


def label_currency(class_name: str) -> str:
    if class_name.startswith("USD_"):
        return "USD"
    if class_name.startswith("KHR_"):
        return "KHR"
    return "OTHER"


def candidate_row(
    *,
    split: str,
    image_path: Path,
    labels: list[dict[str, Any]],
    edge_margin: float,
    tight_gap: float,
    overlap_small_ratio: float,
) -> dict[str, Any]:
    metrics = pair_metrics(labels)
    class_names = [str(label["class_name"]) for label in labels]
    currencies = {label_currency(name) for name in class_names}
    edge_count = sum(1 for label in labels if touches_edge(label, edge_margin))
    protected_count = sum(1 for name in class_names if name in PROTECTED_RIEL)
    tags: list[str] = []
    if len(labels) >= 2:
        tags.append("multi_note")
    if float(metrics["max_intersection_small_ratio"]) >= overlap_small_ratio:
        tags.append("bbox_overlap")
    if metrics["min_gap"] is not None and float(metrics["min_gap"]) <= tight_gap and len(labels) >= 2:
        tags.append("tight_pair")
    if edge_count:
        tags.append("partial_edge")
    if protected_count:
        tags.append("protected_riel")
    if {"USD", "KHR"}.issubset(currencies):
        tags.append("mixed_usd_khr")
    if len(set(class_names)) < len(class_names) and len(labels) >= 2:
        tags.append("same_class_repeat")

    priority = 0.0
    priority += 6.0 if "bbox_overlap" in tags else 0.0
    priority += 4.0 if "tight_pair" in tags else 0.0
    priority += 3.0 if "multi_note" in tags else 0.0
    priority += 2.0 if "protected_riel" in tags else 0.0
    priority += 1.5 if "mixed_usd_khr" in tags else 0.0
    priority += 1.0 if "partial_edge" in tags else 0.0
    priority += min(4.0, 0.5 * len(labels))
    priority += 0.5 if split in {"val", "test"} else 0.0

    return {
        "image": repo_rel(image_path),
        "canonical_key": canonical_image_key(image_path),
        "split": split,
        "source_group": source_group_for_image(image_path),
        "boxes": len(labels),
        "classes": ",".join(class_names),
        "unique_classes": len(set(class_names)),
        "tags": ",".join(tags),
        "priority": round(priority, 4),
        "pairs": int(metrics["pairs"]),
        "max_iou": round(float(metrics["max_iou"]), 6),
        "max_intersection_small_ratio": round(float(metrics["max_intersection_small_ratio"]), 6),
        "min_gap": "" if metrics["min_gap"] is None else round(float(metrics["min_gap"]), 6),
        "closest_pair": metrics["closest_pair"],
        "edge_touch_boxes": edge_count,
        "protected_riel_boxes": protected_count,
        "labels": labels,
        "usable_as": "",
        "review_decision": "",
        "final_route": "",
        "review_notes": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "priority",
        "split",
        "source_group",
        "canonical_key",
        "variant_count",
        "variant_splits",
        "image",
        "boxes",
        "unique_classes",
        "classes",
        "tags",
        "pairs",
        "max_iou",
        "max_intersection_small_ratio",
        "min_gap",
        "closest_pair",
        "edge_touch_boxes",
        "protected_riel_boxes",
        "usable_as",
        "review_decision",
        "final_route",
        "review_notes",
        "variant_images",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_list(path: Path, images: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(images))
    path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")


def build_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_key"])].append(row)
    clusters: list[dict[str, Any]] = []
    for key, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda row: (
                float(row["priority"]),
                int(row["boxes"]),
                float(row["max_intersection_small_ratio"]),
                0.0 if row["min_gap"] == "" else -float(row["min_gap"]),
            ),
            reverse=True,
        )
        representative = dict(ordered[0])
        representative["variant_count"] = len(group)
        representative["variant_splits"] = ",".join(sorted({str(row["split"]) for row in group}))
        representative["variant_images"] = "|".join(str(row["image"]) for row in ordered[:12])
        representative["cluster_tags"] = ",".join(
            sorted({tag for row in group for tag in str(row["tags"]).split(",") if tag})
        )
        representative["tags"] = representative["cluster_tags"]
        clusters.append(representative)
    clusters.sort(
        key=lambda row: (
            float(row["priority"]),
            int(row["variant_count"]),
            int(row["boxes"]),
            float(row["max_intersection_small_ratio"]),
        ),
        reverse=True,
    )
    return clusters


def scale_box(label: dict[str, Any], source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    x1 = float(label["x1"]) * source_width
    y1 = float(label["y1"]) * source_height
    x2 = float(label["x2"]) * source_width
    y2 = float(label["y2"]) * source_height
    return (
        int(round(x1 * target_width / max(1, source_width))),
        int(round(y1 * target_height / max(1, source_height))),
        int(round(x2 * target_width / max(1, source_width))),
        int(round(y2 * target_height / max(1, source_height))),
    )


def draw_sheet(path: Path, rows: list[dict[str, Any]], *, items: int, thumb_width: int, cols: int) -> None:
    selected = rows[:items]
    if not selected:
        return
    thumb_height = thumb_width
    caption_height = 76
    row_count = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, row_count * (thumb_height + caption_height)), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        tile_x = (index % cols) * thumb_width
        tile_y = (index // cols) * (thumb_height + caption_height)
        image_path = resolve(str(row["image"]))
        try:
            with Image.open(image_path) as loaded:
                original = ImageOps.exif_transpose(loaded.convert("RGB"))
            source_size = original.size
            preview = original.copy()
            preview.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_width, thumb_height), "white")
            paste_xy = ((thumb_width - preview.width) // 2, (thumb_height - preview.height) // 2)
            tile.paste(preview, paste_xy)
            tile_draw = ImageDraw.Draw(tile)
            for label in row.get("labels", []):
                x1, y1, x2, y2 = scale_box(label, source_size, preview.size)
                box = (x1 + paste_xy[0], y1 + paste_xy[1], x2 + paste_xy[0], y2 + paste_xy[1])
                outline = (35, 145, 60) if str(label["class_name"]).startswith("KHR_") else (35, 95, 210)
                tile_draw.rectangle(box, outline=outline, width=3)
                tile_draw.text((box[0] + 2, max(2, box[1] - 12)), str(label["class_name"]), fill=outline, font=font)
            sheet.paste(tile, (tile_x, tile_y))
        except Exception as exc:  # pragma: no cover - visual fallback
            draw.rectangle((tile_x, tile_y, tile_x + thumb_width, tile_y + thumb_height), fill=(250, 220, 220))
            draw.text((tile_x + 4, tile_y + 4), f"open failed: {exc}", fill=(120, 0, 0), font=font)
        caption = [
            f"{row['split']} {row['source_group']} p={row['priority']}",
            f"boxes={row['boxes']} vars={row.get('variant_count', 1)} gap={row['min_gap']}",
            f"ov={row['max_intersection_small_ratio']} {str(row['canonical_key'])[:28]}",
            str(row["tags"])[:52],
        ]
        for line_no, text in enumerate(caption):
            draw.text((tile_x + 4, tile_y + thumb_height + 3 + line_no * 17), text, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_yaml(data_path)
    names = names_by_id(config)
    splits = args.split or ["train", "val", "test"]
    out_dir = resolve(args.out_dir)

    rows: list[dict[str, Any]] = []
    split_counts: dict[str, Any] = {}
    tag_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for split in splits:
        images = split_images(data_path, config, split)
        split_rows = 0
        labeled_images = 0
        multi_note_images = 0
        for image_path in images:
            labels = read_labels(image_path, names)
            if labels:
                labeled_images += 1
            if len(labels) >= 2:
                multi_note_images += 1
            row = candidate_row(
                split=split,
                image_path=image_path,
                labels=labels,
                edge_margin=args.edge_margin,
                tight_gap=args.tight_gap,
                overlap_small_ratio=args.overlap_small_ratio,
            )
            if row["tags"]:
                rows.append(row)
                split_rows += 1
                source_counts[str(row["source_group"])] += 1
                for tag in str(row["tags"]).split(","):
                    if tag:
                        tag_counts[tag] += 1
        split_counts[split] = {
            "images": len(images),
            "labeled_images": labeled_images,
            "multi_note_images": multi_note_images,
            "queued_images": split_rows,
        }

    rows.sort(
        key=lambda row: (
            float(row["priority"]),
            int(row["boxes"]),
            float(row["max_intersection_small_ratio"]),
            0.0 if row["min_gap"] == "" else -float(row["min_gap"]),
        ),
        reverse=True,
    )
    review_rows = rows[: args.top_k]
    cluster_rows = build_clusters(rows)
    review_clusters = cluster_rows[: args.top_k]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "review_queue.csv"
    write_csv(csv_path, review_rows)
    sheet_path = out_dir / "review_queue_sheet.jpg"
    draw_sheet(sheet_path, review_rows, items=args.sheet_items, thumb_width=args.thumb_width, cols=args.cols)
    cluster_csv_path = out_dir / "review_clusters.csv"
    write_csv(cluster_csv_path, review_clusters)
    cluster_sheet_path = out_dir / "review_clusters_sheet.jpg"
    draw_sheet(
        cluster_sheet_path,
        review_clusters,
        items=args.sheet_items,
        thumb_width=args.thumb_width,
        cols=args.cols,
    )

    list_paths: dict[str, str] = {}
    for tag in sorted(tag_counts):
        images = [str(row["image"]) for row in rows if tag in str(row["tags"]).split(",")]
        path = out_dir / f"{tag}_images.txt"
        write_list(path, images)
        list_paths[tag] = repo_rel(path)

    summary = {
        "schema": "cashsnap_real_overlap_review_queue_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": repo_rel(data_path),
        "out_dir": repo_rel(out_dir),
        "review_queue_csv": repo_rel(csv_path),
        "review_queue_sheet": repo_rel(sheet_path),
        "review_clusters_csv": repo_rel(cluster_csv_path),
        "review_clusters_sheet": repo_rel(cluster_sheet_path),
        "list_paths": list_paths,
        "splits": split_counts,
        "queued_images": len(rows),
        "queued_clusters": len(cluster_rows),
        "review_rows_written": len(review_rows),
        "review_clusters_written": len(review_clusters),
        "tag_counts": dict(tag_counts.most_common()),
        "cluster_tag_counts": dict(
            Counter(
                tag
                for row in cluster_rows
                for tag in str(row["tags"]).split(",")
                if tag
            ).most_common()
        ),
        "source_counts": dict(source_counts.most_common()),
        "cluster_source_counts": dict(Counter(str(row["source_group"]) for row in cluster_rows).most_common()),
        "top_rows_preview": [
            {
                key: row[key]
                for key in [
                    "priority",
                    "split",
                    "source_group",
                    "canonical_key",
                    "image",
                    "boxes",
                    "classes",
                    "tags",
                    "min_gap",
                ]
            }
            for row in review_rows[:20]
        ],
        "top_clusters_preview": [
            {
                key: row[key]
                for key in [
                    "priority",
                    "split",
                    "source_group",
                    "canonical_key",
                    "variant_count",
                    "image",
                    "boxes",
                    "classes",
                    "tags",
                    "min_gap",
                ]
            }
            for row in review_clusters[:20]
        ],
        "not_training_data": True,
        "purpose": (
            "Review and validation bridge for real overlap/fan/partial/multi-note candidates. "
            "Rows must be visually reviewed before becoming training or promotion evidence."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"review_queue={repo_rel(csv_path)} rows={len(review_rows)} "
        f"clusters={len(review_clusters)} queued={len(rows)} "
        f"queued_clusters={len(cluster_rows)} multi_note={tag_counts.get('multi_note', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
