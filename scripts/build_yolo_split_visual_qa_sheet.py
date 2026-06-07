#!/usr/bin/env python
"""Build a YOLO visual QA contact sheet from the exact images in a split."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


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
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--items", type=int, default=52)
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def load_data_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return config


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_root(root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else root / path


def read_split_list(root: Path, split_path: str) -> list[Path]:
    list_path = split_root(root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line)
        images.append(image if image.is_absolute() else root / image)
    return images


def split_images(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for value in values:
        resolved = split_root(root, str(value))
        if resolved.suffix.lower() == ".txt":
            images.extend(read_split_list(root, str(value)))
        else:
            images.extend(sorted(path for path in resolved.glob("*") if path.suffix.lower() in IMAGE_EXTS))
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(label_path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    labels: list[dict[str, Any]] = []
    if not label_path.exists():
        return labels
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        class_id = int(parts[0])
        cx, cy, bw, bh = [float(value) for value in parts[1:]]
        labels.append(
            {
                "class_id": class_id,
                "xyxy": [
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                ],
            }
        )
    return labels


def choose_images(images: list[Path], names: dict[Any, Any], per_class: int, items: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if per_class > 0:
        by_class: dict[int, list[Path]] = defaultdict(list)
        for image in images:
            with Image.open(image) as opened:
                labels = read_labels(label_path_for_image(image), opened.size)
            for class_id in {int(label["class_id"]) for label in labels}:
                by_class[class_id].append(image)
        chosen: list[Path] = []
        for raw_class_id in sorted(names, key=lambda item: int(item)):
            class_id = int(raw_class_id)
            pool = list(by_class[class_id])
            rng.shuffle(pool)
            chosen.extend(pool[:per_class])
        return chosen
    pool = list(images)
    rng.shuffle(pool)
    return pool[:items]


def draw_sheet(records: list[dict[str, Any]], names: dict[Any, Any], out_path: Path, thumb_width: int, cols: int) -> None:
    if not records:
        raise SystemExit("No records selected")
    thumb_h = int(thumb_width * 0.82)
    caption_h = 34
    cols = max(1, min(cols, len(records)))
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, rows * (thumb_h + caption_h)), (238, 238, 238))
    for idx, record in enumerate(records):
        with Image.open(resolve(record["image"])).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for label in record["labels"]:
                draw.rectangle(label["xyxy"], outline=(40, 190, 70), width=3)
            thumb = ImageOps.contain(image, (thumb_width, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_width
        y = (idx // cols) * (thumb_h + caption_h)
        sheet.paste(thumb, (x + (thumb_width - thumb.width) // 2, y))
        labels = record["labels"]
        class_text = ",".join(str(names.get(label["class_id"], label["class_id"])) for label in labels) or "background"
        caption = f"{class_text} | {Path(record['image']).name}"
        ImageDraw.Draw(sheet).text((x + 5, y + thumb_h + 5), caption[:38], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_data_config(data_path)
    images = split_images(data_path, config, args.split)
    if not images:
        raise SystemExit("No images selected")
    names = config.get("names") or {}
    selected = choose_images(images, names, args.per_class, args.items, args.seed)
    records: list[dict[str, Any]] = []
    for image_path in selected:
        with Image.open(image_path) as image:
            labels = read_labels(label_path_for_image(image_path), image.size)
        records.append({"image": repo_rel(image_path), "label": repo_rel(label_path_for_image(image_path)), "labels": labels})

    out_path = resolve(args.out)
    draw_sheet(records, names, out_path, args.thumb_width, args.cols)
    json_out = resolve(args.json_out) if args.json_out else out_path.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "schema": "cashsnap_yolo_split_visual_qa_sheet_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "data": repo_rel(data_path),
                "split": args.split,
                "images": len(images),
                "selected": len(records),
                "per_class": args.per_class,
                "seed": args.seed,
                "sheet": repo_rel(out_path),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"visual_qa_sheet={repo_rel(out_path)} selected={len(records)} json={repo_rel(json_out)}")


if __name__ == "__main__":
    main()
