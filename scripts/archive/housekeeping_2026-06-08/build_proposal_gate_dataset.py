from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from local_runtime import configure_project_cache

configure_project_cache()

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = ["banknote", "background"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a binary ImageFolder dataset from detector proposal crops."
    )
    parser.add_argument("--detector", required=True, help="YOLO detector .pt path.")
    parser.add_argument("--out", required=True, help="Output ImageFolder dataset path.")
    parser.add_argument(
        "--source",
        nargs=5,
        action="append",
        metavar=("ROOT", "SOURCE_SPLIT", "OUT_SPLIT", "MAX_IMAGES", "GLOB"),
        required=True,
        help="Dataset source. Example: data/cashsnap_v1 train train 240 *.jpg",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--positive-iou", type=float, default=0.45)
    parser.add_argument("--negative-iou", type=float, default=0.05)
    parser.add_argument("--crop-padding", type=float, default=0.06)
    parser.add_argument("--random-backgrounds-per-image", type=int, default=1)
    parser.add_argument("--max-per-split-class", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def yolo_label_boxes(path: Path, image_size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []
    width, height = image_size
    boxes: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, box_w, box_h = (float(part) for part in parts)
        x1 = (cx - box_w / 2) * width
        y1 = (cy - box_h / 2) * height
        x2 = (cx + box_w / 2) * width
        y2 = (cy + box_h / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def max_iou(box: tuple[float, float, float, float], gt_boxes: list[tuple[float, float, float, float]]) -> float:
    return max((box_iou(box, gt_box) for gt_box in gt_boxes), default=0.0)


def padded_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )


def random_background_box(
    image_size: tuple[int, int],
    gt_boxes: list[tuple[float, float, float, float]],
    max_overlap: float,
    rng: random.Random,
) -> tuple[float, float, float, float] | None:
    width, height = image_size
    for _ in range(80):
        crop_w = rng.uniform(0.12, 0.42) * width
        crop_h = rng.uniform(0.12, 0.42) * height
        if crop_w < 24 or crop_h < 24:
            continue
        x1 = rng.uniform(0, max(1, width - crop_w))
        y1 = rng.uniform(0, max(1, height - crop_h))
        box = (x1, y1, x1 + crop_w, y1 + crop_h)
        if max_iou(box, gt_boxes) <= max_overlap:
            return box
    return None


def discover_images(root: Path, split: str, pattern: str, limit: int, rng: random.Random) -> list[Path]:
    image_dir = root / "images" / split
    images = [
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    images = sorted(images)
    rng.shuffle(images)
    return images if limit <= 0 else images[:limit]


def label_for_iou(value: float, positive_iou: float, negative_iou: float) -> str:
    if value >= positive_iou:
        return "banknote"
    if value <= negative_iou:
        return "background"
    return "ignore"


def write_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
    label: str,
    out_dir: Path,
    split: str,
    stem: str,
    index: int,
    padding: float,
) -> Path:
    target_dir = out_dir / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    crop = image.crop(padded_box(box, image.size, padding))
    target = target_dir / f"{stem}_{index:04d}.jpg"
    crop.save(target, quality=92)
    return target


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    detector = YOLO(str(resolve(args.detector)))
    counters: Counter[tuple[str, str]] = Counter()
    source_counters: Counter[tuple[str, str, str]] = Counter()
    rows: list[dict[str, str]] = []

    for source_index, source in enumerate(args.source):
        root_text, source_split, out_split, max_images_text, pattern = source
        source_root = resolve(root_text)
        max_images = int(max_images_text)
        images = discover_images(source_root, source_split, pattern, max_images, rng)
        for image_path in images:
            label_path = source_root / "labels" / source_split / f"{image_path.stem}.txt"
            with Image.open(image_path).convert("RGB") as image:
                gt_boxes = yolo_label_boxes(label_path, image.size)
                result = detector.predict(
                    source=str(image_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    agnostic_nms=args.agnostic_nms,
                    device=args.device,
                    verbose=False,
                )[0]
                proposal_boxes: list[tuple[float, float, float, float]] = []
                proposal_scores: list[float] = []
                if result.boxes is not None and len(result.boxes):
                    xyxy = result.boxes.xyxy.cpu().numpy().astype(float)
                    scores = result.boxes.conf.cpu().numpy().astype(float)
                    proposal_boxes = [tuple(float(v) for v in box) for box in xyxy]
                    proposal_scores = [float(score) for score in scores]

                crop_index = 0
                for proposal_index, box in enumerate(proposal_boxes):
                    overlap = max_iou(box, gt_boxes)
                    label = label_for_iou(overlap, args.positive_iou, args.negative_iou)
                    if label == "ignore":
                        continue
                    if counters[(out_split, label)] >= args.max_per_split_class:
                        continue
                    crop_path = write_crop(
                        image,
                        box,
                        label,
                        out_dir,
                        out_split,
                        f"s{source_index}_{image_path.stem}_p{proposal_index}",
                        crop_index,
                        args.crop_padding,
                    )
                    crop_index += 1
                    counters[(out_split, label)] += 1
                    source_counters[(root_text, out_split, label)] += 1
                    rows.append(
                        {
                            "split": out_split,
                            "label": label,
                            "crop_path": str(crop_path.relative_to(ROOT)),
                            "image_path": str(image_path.relative_to(ROOT)),
                            "label_path": str(label_path.relative_to(ROOT)),
                            "source_root": root_text,
                            "source_split": source_split,
                            "proposal_index": str(proposal_index),
                            "detector_conf": f"{proposal_scores[proposal_index]:.5f}",
                            "max_iou": f"{overlap:.5f}",
                            "kind": "proposal",
                        }
                    )

                for random_index in range(args.random_backgrounds_per_image):
                    if counters[(out_split, "background")] >= args.max_per_split_class:
                        break
                    box = random_background_box(image.size, gt_boxes, args.negative_iou, rng)
                    if box is None:
                        continue
                    crop_path = write_crop(
                        image,
                        box,
                        "background",
                        out_dir,
                        out_split,
                        f"s{source_index}_{image_path.stem}_r{random_index}",
                        crop_index,
                        args.crop_padding,
                    )
                    crop_index += 1
                    counters[(out_split, "background")] += 1
                    source_counters[(root_text, out_split, "background")] += 1
                    rows.append(
                        {
                            "split": out_split,
                            "label": "background",
                            "crop_path": str(crop_path.relative_to(ROOT)),
                            "image_path": str(image_path.relative_to(ROOT)),
                            "label_path": str(label_path.relative_to(ROOT)),
                            "source_root": root_text,
                            "source_split": source_split,
                            "proposal_index": "",
                            "detector_conf": "",
                            "max_iou": "0.00000",
                            "kind": "random_background",
                        }
                    )

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "label",
            "crop_path",
            "image_path",
            "label_path",
            "source_root",
            "source_split",
            "proposal_index",
            "detector_conf",
            "max_iou",
            "kind",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counters.items())},
        "source_counts": {
            f"{root}|{split}|{label}": count
            for (root, split, label), count in sorted(source_counters.items())
        },
        "settings": {
            "detector": args.detector,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "positive_iou": args.positive_iou,
            "negative_iou": args.negative_iou,
            "random_backgrounds_per_image": args.random_backgrounds_per_image,
            "max_per_split_class": args.max_per_split_class,
            "seed": args.seed,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} proposal-gate crops to {out_dir.relative_to(ROOT)}")
    for key, count in sorted(counters.items()):
        print(f"{key[0]} {key[1]}: {count}")


if __name__ == "__main__":
    main()
