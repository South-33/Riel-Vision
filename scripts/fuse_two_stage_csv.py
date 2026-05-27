from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse detector and fragment-classifier labels in a proposal CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--det-threshold", type=float, default=0.15)
    parser.add_argument("--det-class-column", default="detector_class")
    parser.add_argument("--det-score-column", default="detector_conf")
    parser.add_argument("--out-class-column", default="fragment_class")
    parser.add_argument("--out-score-column", default="fragment_conf")
    parser.add_argument("--nms-iou", type=float, default=None, help="Optional class-agnostic NMS IoU threshold.")
    parser.add_argument("--nms-score-column", default="", help="Score column for NMS ranking. Defaults to out score.")
    parser.add_argument("--image", default="", help="Optional source image for rendering a fused preview.")
    parser.add_argument("--out-preview", default="", help="Optional preview image path.")
    return parser.parse_args()


def iou(a: dict[str, str], b: dict[str, str]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[key]) for key in ["x1", "y1", "x2", "y2"])
    bx1, by1, bx2, by2 = (float(b[key]) for key in ["x1", "y1", "x2", "y2"])
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def nms(rows: list[dict[str, str]], score_column: str, threshold: float) -> list[dict[str, str]]:
    pending = sorted(rows, key=lambda row: float(row.get(score_column, "0") or 0), reverse=True)
    kept: list[dict[str, str]] = []
    for row in pending:
        if all(iou(row, kept_row) < threshold for kept_row in kept):
            kept.append(row)
    return sorted(kept, key=lambda row: int(row.get("index", "0") or 0))


def main() -> None:
    args = parse_args()
    with open(args.csv, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No rows found")
    fieldnames = list(rows[0].keys())
    fused = 0
    for row in rows:
        score = float(row.get(args.det_score_column, "0") or 0)
        if score >= args.det_threshold:
            row[args.out_class_column] = row[args.det_class_column]
            row[args.out_score_column] = row[args.det_score_column]
            fused += 1
    before_nms = len(rows)
    if args.nms_iou is not None:
        rows = nms(rows, args.nms_score_column or args.out_score_column, args.nms_iou)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.image and args.out_preview:
        render_preview(args.image, args.out_preview, rows, args.out_class_column, args.out_score_column)
    print(f"wrote {len(rows)} rows to {args.out}; detector_overrides={fused}; nms_removed={before_nms - len(rows)}")


def render_preview(
    image_path: str,
    out_path: str,
    rows: list[dict[str, str]],
    class_column: str,
    score_column: str,
) -> None:
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "dodgerblue", "white"]
    with Image.open(image_path).convert("RGB") as image:
        preview_width = min(image.width, 1920)
        scale = preview_width / image.width
        preview_height = round(image.height * scale)
        output = image.resize((preview_width, preview_height))
        draw = ImageDraw.Draw(output)
        for index, row in enumerate(rows):
            x1, y1, x2, y2 = [float(row[key]) for key in ["x1", "y1", "x2", "y2"]]
            x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
            color = colors[index % len(colors)]
            label = f"{row[class_column]} {float(row.get(score_column, '0') or 0):.2f}"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, preview_width // 700))
            text_box = draw.textbbox((x1, y1), label)
            draw.rectangle((text_box[0], text_box[1], text_box[2] + 4, text_box[3] + 4), fill=color)
            draw.text((x1 + 2, y1 + 2), label, fill="black")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        output.save(out_path, quality=92)


if __name__ == "__main__":
    main()
