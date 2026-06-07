from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cashsnap_classes import CLASS_TO_ID, ID_TO_CLASS
from evaluate_real_draft_labels import box_iou, read_yolo_detect_labels


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect two-stage proposal rows against YOLO draft labels.")
    parser.add_argument("--csv", required=True, help="Raw or fused two-stage proposal CSV.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--top-probs", type=int, default=3)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def parse_probs(row: dict[str, str], top: int) -> str:
    raw = row.get("fragment_probs", "").strip()
    if not raw:
        return ""
    try:
        probs = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return ";".join(f"{key}:{value:.4f}" for key, value in sorted(probs.items(), key=lambda item: item[1], reverse=True)[:top])


def best_label_for_rows(rows: list[dict[str, str]], gt_boxes: np.ndarray, gt_classes: np.ndarray) -> list[tuple[int, float, str]]:
    if not rows:
        return []
    boxes = np.asarray([[float(row[key]) for key in ["x1", "y1", "x2", "y2"]] for row in rows], dtype=float)
    ious = box_iou(boxes, gt_boxes)
    matches: list[tuple[int, float, str]] = []
    for row_index in range(len(rows)):
        if ious.shape[1] == 0:
            matches.append((-1, 0.0, ""))
            continue
        label_index = int(np.argmax(ious[row_index]))
        iou = float(ious[row_index, label_index])
        class_name = ID_TO_CLASS.get(int(gt_classes[label_index]), str(int(gt_classes[label_index])))
        matches.append((label_index, iou, class_name))
    return matches


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(resolve(args.csv).open("r", newline="", encoding="utf-8")))
    with Image.open(resolve(args.image)) as image:
        gt_boxes, gt_classes = read_yolo_detect_labels(resolve(args.labels), image.size)
    matches = best_label_for_rows(rows, gt_boxes, gt_classes)
    fieldnames = [
        "index",
        "gt_index",
        "gt_class",
        "gt_iou",
        "detector_class",
        "detector_conf",
        "fragment_class",
        "fragment_conf",
        "detector_correct",
        "fragment_correct",
        "matched",
        "top_fragment_probs",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row, (label_index, iou, gt_class) in zip(rows, matches):
        detector_class = row.get("detector_class", "")
        fragment_class = row.get("fragment_class", "")
        writer.writerow(
            {
                "index": row.get("index", ""),
                "gt_index": label_index,
                "gt_class": gt_class,
                "gt_iou": f"{iou:.4f}",
                "detector_class": detector_class,
                "detector_conf": row.get("detector_conf", ""),
                "fragment_class": fragment_class,
                "fragment_conf": row.get("fragment_conf", ""),
                "detector_correct": int(iou >= args.match_iou and detector_class == gt_class),
                "fragment_correct": int(iou >= args.match_iou and fragment_class == gt_class),
                "matched": int(iou >= args.match_iou),
                "top_fragment_probs": parse_probs(row, args.top_probs),
            }
        )


if __name__ == "__main__":
    main()
