from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from local_runtime import configure_project_cache

configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_csv_numbers(value: str, caster):
    return [caster(item.strip()) for item in value.split(",") if item.strip()]


def read_yolo_detect_labels(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    boxes: list[list[float]] = []
    classes: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}: line {line_number} has {len(parts)} fields; expected 5")
        class_id = int(parts[0])
        cx, cy, box_width, box_height = [float(value) for value in parts[1:]]
        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height
        boxes.append([x1, y1, x2, y2])
        classes.append(class_id)
    return np.asarray(boxes, dtype=float), np.asarray(classes, dtype=int)


def result_boxes(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        boxes = getattr(result, "obb", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=float)
    return (
        boxes.xyxy.cpu().numpy().astype(float),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy().astype(float),
    )


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
    require_same_class: bool,
) -> int:
    order = np.argsort(-pred_scores)
    ious = box_iou(pred_boxes, gt_boxes)
    used_gt: set[int] = set()
    matches = 0
    for pred_idx in order:
        best_gt = -1
        best_iou = 0.0
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in used_gt:
                continue
            if require_same_class and pred_classes[pred_idx] != gt_classes[gt_idx]:
                continue
            iou = float(ious[pred_idx, gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt)
            matches += 1
    return matches


def class_counts(classes: np.ndarray, names: dict[int, str]) -> str:
    counts: dict[str, int] = {}
    for class_id in classes:
        name = names.get(int(class_id), str(int(class_id)))
        counts[name] = counts.get(name, 0) + 1
    return ";".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model count/recall against private real-image draft labels.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--imgsz", default="640,960")
    parser.add_argument("--conf", default="0.03,0.05,0.25")
    parser.add_argument("--nms-iou", default="0.70")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    label_path = resolve_path(args.labels)
    model_paths = [resolve_path(Path(value)) for value in args.model]
    image_sizes = parse_csv_numbers(args.imgsz, int)
    confs = parse_csv_numbers(args.conf, float)
    nms_ious = parse_csv_numbers(args.nms_iou, float)
    with Image.open(image_path) as image:
        image_size = image.size
    gt_boxes, gt_classes = read_yolo_detect_labels(label_path, image_size)

    rows: list[dict[str, str | int | float]] = []
    for model_path in model_paths:
        model = YOLO(str(model_path))
        names = {int(key): value for key, value in model.names.items()}
        gt_counts = class_counts(gt_classes, names)
        for image_size_arg in image_sizes:
            for conf in confs:
                for nms_iou in nms_ious:
                    result = model.predict(
                        source=str(image_path),
                        imgsz=image_size_arg,
                        conf=conf,
                        iou=nms_iou,
                        verbose=False,
                    )[0]
                    pred_boxes, pred_classes, pred_scores = result_boxes(result)
                    same_class = greedy_match(
                        pred_boxes,
                        pred_classes,
                        pred_scores,
                        gt_boxes,
                        gt_classes,
                        args.match_iou,
                        require_same_class=True,
                    )
                    any_class = greedy_match(
                        pred_boxes,
                        pred_classes,
                        pred_scores,
                        gt_boxes,
                        gt_classes,
                        args.match_iou,
                        require_same_class=False,
                    )
                    rows.append(
                        {
                            "image": image_path.relative_to(ROOT).as_posix(),
                            "labels": label_path.relative_to(ROOT).as_posix(),
                            "model": model_path.relative_to(ROOT).as_posix(),
                            "imgsz": image_size_arg,
                            "conf": conf,
                            "nms_iou": nms_iou,
                            "match_iou": args.match_iou,
                            "gt_count": int(len(gt_boxes)),
                            "pred_count": int(len(pred_boxes)),
                            "count_error": int(len(pred_boxes) - len(gt_boxes)),
                            "matched_same_class": same_class,
                            "matched_any_class": any_class,
                            "recall_same_class": f"{same_class / len(gt_boxes):.3f}" if len(gt_boxes) else "",
                            "recall_any_class": f"{any_class / len(gt_boxes):.3f}" if len(gt_boxes) else "",
                            "gt_classes": gt_counts,
                            "pred_classes": class_counts(pred_classes, names),
                            "top_conf": f"{float(pred_scores.max()):.3f}" if len(pred_scores) else "",
                        }
                    )

    fieldnames = list(rows[0].keys()) if rows else []
    if args.out is not None:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
