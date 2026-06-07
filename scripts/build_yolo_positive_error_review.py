#!/usr/bin/env python
"""Build a visual error review pack for YOLO on labeled positive splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from local_runtime import configure_project_cache


configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO dataset YAML.")
    parser.add_argument("--split", action="append", required=True, help="Dataset split key. Repeatable.")
    parser.add_argument("--model", action="append", required=True, help="Model path or label=path. Repeatable.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output review directory.")
    parser.add_argument("--conf", type=float, default=0.05, help="Prediction confidence threshold.")
    parser.add_argument("--iou-match", type=float, default=0.50, help="IoU threshold for GT/prediction matches.")
    parser.add_argument("--pred-iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--focus-classes",
        default="",
        help="Comma/space separated class-name filter for visual review selection only.",
    )
    parser.add_argument("--max-review", type=int, default=180, help="Maximum visual rows to export.")
    parser.add_argument("--max-per-group", type=int, default=24, help="Cap per model/split/error/class group.")
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--thumb-size", type=int, default=190)
    parser.add_argument("--sheet-columns", type=int, default=5)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def slug(value: str, *, max_length: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        cleaned = "item"
    return cleaned[:max_length].rstrip("._") or "item"


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_roots = [(ROOT / "runs").resolve(), (ROOT / "data" / "review").resolve()]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        allowed_text = ", ".join(str(root) for root in allowed_roots)
        raise SystemExit(f"Refusing to clean outside allowed roots ({allowed_text}): {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_model(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(value.strip())
        label = path.parent.parent.name if path.name == "best.pt" else path.stem
    if not label:
        raise SystemExit(f"empty model label: {value!r}")
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing model: {resolved}")
    return label, resolved


def parse_focus_classes(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[,\s]+", value) if item.strip()}


def read_names(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names")
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    raise SystemExit("dataset YAML must include names as a list or mapping")


def dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = Path(str(config.get("path", ".")))
    return raw_root if raw_root.is_absolute() else (config_path.parent / raw_root).resolve()


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
        path = Path(line)
        images.append(path if path.is_absolute() else root / path)
    return images


def split_images(root: Path, split_value: str | list[str]) -> list[Path]:
    rows: list[Path] = []
    for raw_split in split_value if isinstance(split_value, list) else [split_value]:
        resolved = split_root(root, str(raw_split))
        if resolved.suffix.lower() == ".txt":
            rows.extend(read_split_list(root, str(raw_split)))
            continue
        rows.extend(path for path in sorted(resolved.glob("*")) if path.suffix.lower() in IMAGE_EXTS)
    return rows


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    cx, cy, box_w, box_h = values
    x1 = (cx - box_w / 2.0) * width
    y1 = (cy - box_h / 2.0) * height
    x2 = (cx + box_w / 2.0) * width
    y2 = (cy + box_h / 2.0) * height
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def read_gt_boxes(image: Path, names: dict[int, str], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image)
    if not label_path.exists():
        return []
    width, height = image_size
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{label_path}:{index + 1} expected 5 YOLO fields, found {len(parts)}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        rows.append(
            {
                "gt_index": index,
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "bbox_xyxy": yolo_to_xyxy(values, width, height),
            }
        )
    return rows


def box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(a: list[float], b: list[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def match_image(
    *,
    model_label: str,
    split: str,
    image: Path,
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    errors: list[dict[str, Any]] = []
    tp_by_class: Counter[str] = Counter()

    same_class_pairs: list[tuple[float, float, int, int]] = []
    for gt_index, gt in enumerate(gt_rows):
        for pred_index, pred in enumerate(pred_rows):
            if int(gt["class_id"]) != int(pred["class_id"]):
                continue
            iou = box_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou >= iou_threshold:
                same_class_pairs.append((iou, float(pred["confidence"]), gt_index, pred_index))
    same_class_pairs.sort(reverse=True)
    for _iou, _score, gt_index, pred_index in same_class_pairs:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        tp_by_class[str(gt_rows[gt_index]["class_name"])] += 1

    wrong_pairs: list[tuple[float, float, int, int]] = []
    for gt_index, gt in enumerate(gt_rows):
        if gt_index in matched_gt:
            continue
        for pred_index, pred in enumerate(pred_rows):
            if pred_index in matched_pred:
                continue
            if int(gt["class_id"]) == int(pred["class_id"]):
                continue
            iou = box_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou >= iou_threshold:
                wrong_pairs.append((iou, float(pred["confidence"]), gt_index, pred_index))
    wrong_pairs.sort(reverse=True)
    for iou, _score, gt_index, pred_index in wrong_pairs:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        gt = gt_rows[gt_index]
        pred = pred_rows[pred_index]
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        errors.append(
            base_error_row(
                model_label=model_label,
                split=split,
                image=image,
                error_type="wrong_class",
                gt=gt,
                pred=pred,
                iou=iou,
                nearest_gt=None,
            )
        )

    for gt_index, gt in enumerate(gt_rows):
        if gt_index in matched_gt:
            continue
        nearest_pred = nearest_box(gt["bbox_xyxy"], pred_rows, exclude=matched_pred)
        errors.append(
            base_error_row(
                model_label=model_label,
                split=split,
                image=image,
                error_type="missed_gt",
                gt=gt,
                pred=nearest_pred[1] if nearest_pred else None,
                iou=nearest_pred[0] if nearest_pred else 0.0,
                nearest_gt=None,
            )
        )

    for pred_index, pred in enumerate(pred_rows):
        if pred_index in matched_pred:
            continue
        nearest_gt = nearest_box(pred["bbox_xyxy"], gt_rows, exclude=set())
        errors.append(
            base_error_row(
                model_label=model_label,
                split=split,
                image=image,
                error_type="unmatched_fp",
                gt=None,
                pred=pred,
                iou=nearest_gt[0] if nearest_gt else 0.0,
                nearest_gt=nearest_gt[1] if nearest_gt else None,
            )
        )

    summary = {
        "gt": Counter(str(row["class_name"]) for row in gt_rows),
        "predictions": Counter(str(row["class_name"]) for row in pred_rows),
        "tp": tp_by_class,
    }
    return errors, summary


def nearest_box(
    box: list[float],
    rows: list[dict[str, Any]],
    *,
    exclude: set[int],
) -> tuple[float, dict[str, Any]] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for index, row in enumerate(rows):
        if index in exclude:
            continue
        iou = box_iou(box, row["bbox_xyxy"])
        if best is None or iou > best[0]:
            best = (iou, row)
    return best


def base_error_row(
    *,
    model_label: str,
    split: str,
    image: Path,
    error_type: str,
    gt: dict[str, Any] | None,
    pred: dict[str, Any] | None,
    iou: float,
    nearest_gt: dict[str, Any] | None,
) -> dict[str, Any]:
    gt_class = str(gt["class_name"]) if gt else ""
    pred_class = str(pred["class_name"]) if pred else ""
    nearest_gt_class = str(nearest_gt["class_name"]) if nearest_gt else ""
    confidence = float(pred["confidence"]) if pred else 0.0
    area = box_area((gt or pred or {"bbox_xyxy": [0, 0, 0, 0]})["bbox_xyxy"])
    if error_type == "wrong_class":
        review_score = 3.0 + confidence + iou
    elif error_type == "unmatched_fp":
        review_score = 2.0 + confidence + max(0.0, iou)
    else:
        review_score = 1.0 + math.log1p(area) / 20.0 + max(0.0, iou)
    return {
        "model": model_label,
        "split": split,
        "image": repo_rel(image),
        "error_type": error_type,
        "gt_class": gt_class,
        "pred_class": pred_class,
        "nearest_gt_class": nearest_gt_class,
        "confidence": round(confidence, 6),
        "iou": round(float(iou), 6),
        "gt_bbox_xyxy": gt["bbox_xyxy"] if gt else None,
        "pred_bbox_xyxy": pred["bbox_xyxy"] if pred else None,
        "nearest_gt_bbox_xyxy": nearest_gt["bbox_xyxy"] if nearest_gt else None,
        "review_score": round(review_score, 6),
    }


def class_names_from_model(model: YOLO, fallback: dict[int, str]) -> dict[int, str]:
    names = getattr(model, "names", None) or getattr(model.model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return fallback


def predictions_from_result(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return []
    classes = boxes.cls.cpu().numpy().astype(int).tolist()
    scores = boxes.conf.cpu().numpy().astype(float).tolist()
    xyxy = boxes.xyxy.cpu().numpy().astype(float).tolist()
    rows: list[dict[str, Any]] = []
    for index, (class_id, score, box) in enumerate(zip(classes, scores, xyxy, strict=True)):
        rows.append(
            {
                "pred_index": index,
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), f"class_{class_id}"),
                "confidence": float(score),
                "bbox_xyxy": [float(value) for value in box],
            }
        )
    return rows


def run_model_split(
    *,
    model_label: str,
    model_path: Path,
    split: str,
    images: list[Path],
    dataset_names: dict[int, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = YOLO(str(model_path))
    pred_names = class_names_from_model(model, dataset_names)
    errors: list[dict[str, Any]] = []
    totals = {
        "images": len(images),
        "gt": Counter(),
        "predictions": Counter(),
        "tp": Counter(),
        "error_types": Counter(),
        "wrong_class_pairs": Counter(),
    }
    with tempfile.TemporaryDirectory(prefix="cashsnap_positive_error_") as tmp_dir:
        source_file = Path(tmp_dir) / "images.txt"
        source_file.write_text("\n".join(path.resolve().as_posix() for path in images) + "\n", encoding="utf-8")
        results = model.predict(
            source=str(source_file),
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.pred_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            save=False,
            stream=True,
        )
        for image_path, result in zip(images, results, strict=True):
            with Image.open(image_path) as image:
                image_size = image.size
            gt_rows = read_gt_boxes(image_path, dataset_names, image_size)
            pred_rows = predictions_from_result(result, pred_names)
            image_errors, image_summary = match_image(
                model_label=model_label,
                split=split,
                image=image_path,
                gt_rows=gt_rows,
                pred_rows=pred_rows,
                iou_threshold=args.iou_match,
            )
            errors.extend(image_errors)
            totals["gt"].update(image_summary["gt"])
            totals["predictions"].update(image_summary["predictions"])
            totals["tp"].update(image_summary["tp"])
            totals["error_types"].update(row["error_type"] for row in image_errors)
            totals["wrong_class_pairs"].update(
                f"{row['gt_class']}->{row['pred_class']}"
                for row in image_errors
                if row["error_type"] == "wrong_class"
            )
    return errors, summarize_totals(totals, dataset_names)


def summarize_totals(totals: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    error_rows = totals["error_types"]
    for class_name in [names[index] for index in sorted(names)]:
        gt_count = int(totals["gt"][class_name])
        tp_count = int(totals["tp"][class_name])
        by_class[class_name] = {
            "gt": gt_count,
            "tp": tp_count,
            "recall_at_iou": round(tp_count / gt_count, 6) if gt_count else None,
            "predictions": int(totals["predictions"][class_name]),
        }
    return {
        "images": int(totals["images"]),
        "gt": int(sum(totals["gt"].values())),
        "predictions": int(sum(totals["predictions"].values())),
        "tp": int(sum(totals["tp"].values())),
        "error_types": dict(sorted((str(key), int(value)) for key, value in error_rows.items())),
        "wrong_class_pairs": dict(totals["wrong_class_pairs"].most_common(30)),
        "by_class": by_class,
    }


def wants_review(row: dict[str, Any], focus: set[str]) -> bool:
    if not focus:
        return True
    candidates = {
        str(row.get("gt_class", "")),
        str(row.get("pred_class", "")),
        str(row.get("nearest_gt_class", "")),
    }
    return bool(candidates & focus)


def select_review_rows(rows: list[dict[str, Any]], focus: set[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = [row for row in rows if wants_review(row, focus)]
    candidates.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["split"]),
            -float(row["review_score"]),
            str(row["image"]),
        )
    )
    selected: list[dict[str, Any]] = []
    group_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in sorted(candidates, key=lambda item: -float(item["review_score"])):
        group_class = str(row.get("gt_class") or row.get("pred_class") or row.get("nearest_gt_class") or "unknown")
        group = (str(row["model"]), str(row["split"]), str(row["error_type"]), group_class)
        if group_counts[group] >= args.max_per_group:
            continue
        group_counts[group] += 1
        selected.append(row)
        if len(selected) >= args.max_review:
            break
    selected.sort(key=lambda row: (str(row["model"]), str(row["split"]), str(row["error_type"]), str(row["image"])))
    for index, row in enumerate(selected, start=1):
        row["review_rank"] = index
    return selected


def parse_box(value: Any) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [float(item) for item in json.loads(value)]
    return [float(item) for item in value]


def union_box(boxes: list[list[float]], width: int, height: int, pad_frac: float) -> tuple[int, int, int, int]:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    pad = max(0.0, pad_frac) * max(1.0, right - left, bottom - top)
    return (
        max(0, int(round(left - pad))),
        max(0, int(round(top - pad))),
        min(width, int(round(right + pad))),
        min(height, int(round(bottom + pad))),
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int], width: int) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)


def draw_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rectangle((left - 3, top - 3, right + 3, bottom + 3), fill=(20, 20, 20))
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def write_artifacts(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    crop_dir = out_dir / "crops"
    overlay_dir = out_dir / "overlays"
    crop_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_path = resolve(Path(str(row["image"])))
        gt_box = parse_box(row.get("gt_bbox_xyxy"))
        pred_box = parse_box(row.get("pred_bbox_xyxy"))
        nearest_gt_box = parse_box(row.get("nearest_gt_bbox_xyxy"))
        boxes = [box for box in [gt_box, pred_box, nearest_gt_box] if box is not None and box_area(box) > 0]
        if not boxes:
            continue
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            line_width = max(2, min(width, height) // 220)
            font = load_font(max(13, min(width, height) // 55))
            if nearest_gt_box:
                draw_box(draw, nearest_gt_box, (255, 210, 40), max(1, line_width - 1))
            if gt_box:
                draw_box(draw, gt_box, (40, 210, 90), line_width)
            if pred_box:
                draw_box(draw, pred_box, (255, 50, 50), line_width)
            label = (
                f"{row['error_type']} gt={row.get('gt_class') or '-'} "
                f"pred={row.get('pred_class') or '-'} conf={float(row.get('confidence') or 0):.2f}"
            )
            draw_text(draw, label, (8, 8), font)
            if max(overlay.size) > 1500:
                overlay.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
            crop = image.crop(union_box(boxes, width, height, args.crop_pad))

        base = (
            f"{int(row['review_rank']):03d}_{slug(str(row['model']), max_length=28)}_"
            f"{slug(str(row['split']), max_length=16)}_{slug(str(row['error_type']), max_length=20)}_"
            f"{slug(str(row.get('gt_class') or row.get('pred_class') or 'unknown'), max_length=24)}"
        )
        crop_path = crop_dir / f"{base}_crop.jpg"
        overlay_path = overlay_dir / f"{base}_overlay.jpg"
        crop.save(crop_path, quality=92)
        overlay.save(overlay_path, quality=92)
        row["crop"] = repo_rel(crop_path)
        row["overlay"] = repo_rel(overlay_path)


def fit_tile(image: Image.Image, size: int) -> Image.Image:
    tile = Image.new("RGB", (size, size), (245, 245, 245))
    thumb = ImageOps.contain(image.convert("RGB"), (size, size))
    tile.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    return tile


def write_contact_sheet(rows: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    selected = [row for row in rows if row.get("crop")][:96]
    if not selected:
        return
    columns = max(1, args.sheet_columns)
    thumb = max(64, args.thumb_size)
    label_h = 64
    sheet = Image.new("RGB", (columns * thumb, math.ceil(len(selected) / columns) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(13)
    for index, row in enumerate(selected):
        crop_path = resolve(Path(str(row["crop"])))
        with Image.open(crop_path) as crop:
            tile = fit_tile(crop, thumb)
        col = index % columns
        grid_row = index // columns
        x = col * thumb
        y = grid_row * (thumb + label_h)
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb - 1, y + thumb + label_h - 1), outline=(180, 180, 180))
        draw.text((x + 4, y + thumb + 4), f"{row['review_rank']} {row['model']} {row['split']}"[:34], fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + thumb + 22), str(row["error_type"])[:34], fill=(0, 0, 0), font=font)
        pair = f"{row.get('gt_class') or '-'}->{row.get('pred_class') or '-'}"
        draw.text((x + 4, y + thumb + 40), pair[:34], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "review_rank",
        "model",
        "split",
        "image",
        "error_type",
        "gt_class",
        "pred_class",
        "nearest_gt_class",
        "confidence",
        "iou",
        "review_score",
        "gt_bbox_xyxy",
        "pred_bbox_xyxy",
        "nearest_gt_bbox_xyxy",
        "crop",
        "overlay",
        "review_include",
        "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            for key in ["gt_bbox_xyxy", "pred_bbox_xyxy", "nearest_gt_bbox_xyxy"]:
                if out_row.get(key) is not None and not isinstance(out_row[key], str):
                    out_row[key] = json.dumps([round(float(value), 2) for value in out_row[key]])
            out_row.setdefault("review_include", "")
            out_row.setdefault("review_notes", "")
            writer.writerow(out_row)


def print_summary(payload: dict[str, Any]) -> None:
    for key, summary in sorted(payload["summaries"].items()):
        error_types = summary.get("error_types", {})
        print(
            f"{key}: images={summary['images']} gt={summary['gt']} pred={summary['predictions']} "
            f"tp={summary['tp']} errors={error_types}"
        )
        by_class = summary.get("by_class", {})
        weak = [
            (class_name, values)
            for class_name, values in by_class.items()
            if values.get("gt") and (values.get("recall_at_iou") or 0.0) < 0.50
        ]
        if weak:
            text = ", ".join(
                f"{name} recall={values['recall_at_iou']:.3f} gt={values['gt']}"
                for name, values in weak[:8]
            )
            print(f"  weak_classes: {text}")


def main() -> int:
    args = parse_args()
    if args.max_review < 0:
        raise SystemExit("--max-review must be >= 0")
    if args.max_per_group < 1:
        raise SystemExit("--max-per-group must be >= 1")
    data_path = resolve(args.data)
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"dataset YAML must be a mapping: {data_path}")
    names = read_names(config)
    root = dataset_root(data_path, config)
    split_map: dict[str, list[Path]] = {}
    for split in args.split:
        if split not in config:
            raise SystemExit(f"dataset YAML has no split {split!r}: {data_path}")
        images = split_images(root, config[split])
        if not images:
            raise SystemExit(f"split {split!r} has no images")
        split_map[split] = images

    out_dir = resolve(args.out_dir)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [parse_model(value) for value in args.model]
    all_errors: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for model_label, model_path in models:
        for split, images in split_map.items():
            errors, summary = run_model_split(
                model_label=model_label,
                model_path=model_path,
                split=split,
                images=images,
                dataset_names=names,
                args=args,
            )
            all_errors.extend(errors)
            summaries[f"{model_label}/{split}"] = summary

    focus = parse_focus_classes(args.focus_classes)
    review_rows = select_review_rows(all_errors, focus, args)
    if review_rows:
        write_artifacts(review_rows, out_dir, args)
    write_csv(all_errors, out_dir / "errors.csv")
    write_csv(review_rows, out_dir / "review.csv")
    write_contact_sheet(review_rows, out_dir / "contact_sheet_mixed.jpg", args)

    by_sheet: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        by_sheet[(str(row["model"]), str(row["split"]), str(row["error_type"]))].append(row)
    for (model_label, split, error_type), rows in sorted(by_sheet.items()):
        write_contact_sheet(
            rows,
            out_dir / "sheets" / f"{slug(model_label)}_{slug(split)}_{slug(error_type)}.jpg",
            args,
        )

    payload = {
        "schema": "cashsnap_yolo_positive_error_review_v1",
        "data": repo_rel(data_path),
        "dataset_root": repo_rel(root),
        "splits": {split: len(images) for split, images in split_map.items()},
        "models": [{"label": label, "path": repo_rel(path)} for label, path in models],
        "conf": args.conf,
        "iou_match": args.iou_match,
        "imgsz": args.imgsz,
        "focus_classes": sorted(focus),
        "errors": len(all_errors),
        "review_rows": len(review_rows),
        "outputs": {
            "errors_csv": repo_rel(out_dir / "errors.csv"),
            "review_csv": repo_rel(out_dir / "review.csv"),
            "contact_sheet": repo_rel(out_dir / "contact_sheet_mixed.jpg"),
        },
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_summary(payload)
    print(f"wrote_review={repo_rel(out_dir)} rows={len(review_rows)} errors={len(all_errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
