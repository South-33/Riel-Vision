from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BoxStats:
    cls: int
    xyxy: tuple[float, float, float, float]
    short_px: float
    long_px: float
    area_frac: float
    train_short_px: float
    train_area_px: float
    sharpness: float
    luma: float


@dataclass(frozen=True)
class ImageStats:
    image_path: Path
    label_path: Path
    width: int
    height: int
    note_count: int
    min_short_px: float
    median_short_px: float
    min_train_short_px: float
    median_train_short_px: float
    min_train_area_px: float
    median_area_frac: float
    min_crop_sharpness: float
    median_crop_luma: float
    boxes: tuple[BoxStats, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build small visual QA sheets and note-pixel stats for a synthetic YOLO root."
    )
    parser.add_argument("--root", type=Path, required=True, help="Synthetic dataset root containing images/train.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416, help="Model input size used to estimate train-time note pixels.")
    parser.add_argument("--items-per-sheet", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=520)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tiny-short-px", type=float, default=48.0)
    parser.add_argument("--small-short-px", type=float, default=80.0)
    parser.add_argument("--fail-on-quality", action="store_true")
    parser.add_argument("--max-tiny-fraction", type=float, default=0.03)
    parser.add_argument("--max-small-fraction", type=float, default=0.45)
    parser.add_argument("--max-soft-fraction", type=float, default=0.15)
    parser.add_argument("--min-p05-short-px", type=float, default=55.0)
    parser.add_argument("--min-p50-short-px", type=float, default=88.0)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_labels(path: Path, width: int, height: int, train_scale: float) -> list[BoxStats]:
    if not path.exists():
        return []

    boxes: list[BoxStats] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        x1 = max(0.0, (cx - bw / 2.0) * width)
        y1 = max(0.0, (cy - bh / 2.0) * height)
        x2 = min(float(width), (cx + bw / 2.0) * width)
        y2 = min(float(height), (cy + bh / 2.0) * height)
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w <= 1 or box_h <= 1:
            continue
        area_frac = (box_w * box_h) / float(width * height)
        boxes.append(
            BoxStats(
                cls=cls,
                xyxy=(x1, y1, x2, y2),
                short_px=min(box_w, box_h),
                long_px=max(box_w, box_h),
                area_frac=area_frac,
                train_short_px=min(box_w, box_h) * train_scale,
                train_area_px=(box_w * train_scale) * (box_h * train_scale),
                sharpness=0.0,
                luma=0.0,
            )
        )
    return boxes


def crop_quality(image: Image.Image, box: BoxStats) -> tuple[float, float]:
    x1, y1, x2, y2 = box.xyxy
    crop = image.crop((int(x1), int(y1), int(x2), int(y2))).convert("L")
    if crop.width < 3 or crop.height < 3:
        return 0.0, float(ImageStat.Stat(crop).mean[0]) if crop.width and crop.height else 0.0
    arr = np.asarray(crop, dtype=np.float32)
    dx = np.abs(np.diff(arr, axis=1)).mean()
    dy = np.abs(np.diff(arr, axis=0)).mean()
    return float(dx + dy), float(arr.mean())


def collect(root: Path, imgsz: int) -> list[ImageStats]:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    rows: list[ImageStats] = []

    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size
            train_scale = min(imgsz / width, imgsz / height)
            raw_boxes = read_labels(label_path, width, height, train_scale)
            boxes = []
            for box in raw_boxes:
                sharpness, luma = crop_quality(image, box)
                boxes.append(
                    BoxStats(
                        cls=box.cls,
                        xyxy=box.xyxy,
                        short_px=box.short_px,
                        long_px=box.long_px,
                        area_frac=box.area_frac,
                        train_short_px=box.train_short_px,
                        train_area_px=box.train_area_px,
                        sharpness=sharpness,
                        luma=luma,
                    )
                )
        if not boxes:
            rows.append(
                ImageStats(
                    image_path=image_path,
                    label_path=label_path,
                    width=width,
                    height=height,
                    note_count=0,
                    min_short_px=0.0,
                    median_short_px=0.0,
                    min_train_short_px=0.0,
                    median_train_short_px=0.0,
                    min_train_area_px=0.0,
                    median_area_frac=0.0,
                    min_crop_sharpness=0.0,
                    median_crop_luma=0.0,
                    boxes=(),
                )
            )
            continue

        rows.append(
            ImageStats(
                image_path=image_path,
                label_path=label_path,
                width=width,
                height=height,
                note_count=len(boxes),
                min_short_px=min(box.short_px for box in boxes),
                median_short_px=float(median(box.short_px for box in boxes)),
                min_train_short_px=min(box.train_short_px for box in boxes),
                median_train_short_px=float(median(box.train_short_px for box in boxes)),
                min_train_area_px=min(box.train_area_px for box in boxes),
                median_area_frac=float(median(box.area_frac for box in boxes)),
                min_crop_sharpness=min(box.sharpness for box in boxes),
                median_crop_luma=float(median(box.luma for box in boxes)),
                boxes=tuple(boxes),
            )
        )
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    if low == high:
        return values[low]
    frac = pos - low
    return values[low] * (1.0 - frac) + values[high] * frac


def flags(row: ImageStats, tiny: float, small: float, soft_cutoff: float) -> list[str]:
    out: list[str] = []
    if row.note_count == 0:
        out.append("no_labels")
    if row.min_train_short_px < tiny:
        out.append("tiny_at_416")
    elif row.min_train_short_px < small:
        out.append("small_at_416")
    if row.min_crop_sharpness < soft_cutoff:
        out.append("soft_crop")
    return out


def draw_sheet(rows: list[ImageStats], out_path: Path, title: str, thumb_width: int) -> None:
    if not rows:
        return
    font = ImageFont.load_default()
    cols = 2
    rows_per_sheet = (len(rows) + cols - 1) // cols
    text_h = 64
    padding = 12
    cell_w = thumb_width + padding * 2
    cell_h = int(thumb_width * 0.75) + text_h + padding * 2
    title_h = 34
    sheet = Image.new("RGB", (cols * cell_w, title_h + rows_per_sheet * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black", font=font)

    for idx, row in enumerate(rows):
        col = idx % cols
        r = idx // cols
        x0 = col * cell_w + padding
        y0 = title_h + r * cell_h + padding
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
            scale = min(thumb_width / image.width, (cell_h - text_h - padding) / image.height)
            resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_width, cell_h - text_h - padding), (235, 235, 235))
            ox = (tile.width - resized.width) // 2
            oy = (tile.height - resized.height) // 2
            tile.paste(resized, (ox, oy))
            tile_draw = ImageDraw.Draw(tile)
            for box in row.boxes:
                x1, y1, x2, y2 = box.xyxy
                tile_draw.rectangle(
                    (ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale),
                    outline=(255, 40, 40),
                    width=3,
                )
            sheet.paste(tile, (x0, y0))
        label = (
            f"{row.image_path.stem}  notes={row.note_count}  "
            f"minShort@416={row.min_train_short_px:.0f}px  med={row.median_train_short_px:.0f}px"
        )
        label2 = (
            f"sharpMin={row.min_crop_sharpness:.1f}  areaMed={row.median_area_frac:.3f}  "
            f"{row.width}x{row.height}"
        )
        draw.text((x0, y0 + tile.height + 6), label[:92], fill="black", font=font)
        draw.text((x0, y0 + tile.height + 22), label2[:92], fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_outputs(args: argparse.Namespace, rows: list[ImageStats]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sharp_values = [row.min_crop_sharpness for row in rows if row.note_count]
    soft_cutoff = percentile(sharp_values, 10)

    csv_path = args.out_dir / "image_quality.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "label",
                "width",
                "height",
                "note_count",
                "min_short_px",
                "median_short_px",
                "min_train_short_px",
                "median_train_short_px",
                "min_train_area_px",
                "median_area_frac",
                "min_crop_sharpness",
                "median_crop_luma",
                "flags",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": rel(row.image_path),
                    "label": rel(row.label_path),
                    "width": row.width,
                    "height": row.height,
                    "note_count": row.note_count,
                    "min_short_px": f"{row.min_short_px:.3f}",
                    "median_short_px": f"{row.median_short_px:.3f}",
                    "min_train_short_px": f"{row.min_train_short_px:.3f}",
                    "median_train_short_px": f"{row.median_train_short_px:.3f}",
                    "min_train_area_px": f"{row.min_train_area_px:.3f}",
                    "median_area_frac": f"{row.median_area_frac:.6f}",
                    "min_crop_sharpness": f"{row.min_crop_sharpness:.3f}",
                    "median_crop_luma": f"{row.median_crop_luma:.3f}",
                    "flags": ";".join(flags(row, args.tiny_short_px, args.small_short_px, soft_cutoff)),
                }
            )

    train_shorts = [row.min_train_short_px for row in rows if row.note_count]
    flag_counts: dict[str, int] = {}
    for row in rows:
        for flag in flags(row, args.tiny_short_px, args.small_short_px, soft_cutoff):
            flag_counts[flag] = int(flag_counts.get(flag, 0)) + 1

    note_rows = max(1, len([row for row in rows if row.note_count]))
    p05_short = percentile(train_shorts, 5)
    p50_short = percentile(train_shorts, 50)
    summary = {
        "root": rel(args.root),
        "images": len(rows),
        "imgsz": args.imgsz,
        "tiny_short_px": args.tiny_short_px,
        "small_short_px": args.small_short_px,
        "soft_cutoff_p10": soft_cutoff,
        "min_train_short_px": {
            "p00": percentile(train_shorts, 0),
            "p05": p05_short,
            "p10": percentile(train_shorts, 10),
            "p25": percentile(train_shorts, 25),
            "p50": p50_short,
            "p75": percentile(train_shorts, 75),
            "p90": percentile(train_shorts, 90),
            "p95": percentile(train_shorts, 95),
            "p100": percentile(train_shorts, 100),
        },
        "flag_counts": flag_counts,
        "quality_gate": {
            "status": "pending",
            "limits": {
                "max_tiny_fraction": args.max_tiny_fraction,
                "max_small_fraction": args.max_small_fraction,
                "max_soft_fraction": args.max_soft_fraction,
                "min_p05_short_px": args.min_p05_short_px,
                "min_p50_short_px": args.min_p50_short_px,
            },
            "observed": {
                "tiny_fraction": flag_counts.get("tiny_at_416", 0) / note_rows,
                "small_fraction": flag_counts.get("small_at_416", 0) / note_rows,
                "soft_fraction": flag_counts.get("soft_crop", 0) / note_rows,
                "p05_short_px": p05_short,
                "p50_short_px": p50_short,
            },
            "failures": [],
        },
        "csv": rel(csv_path),
    }
    gate = summary["quality_gate"]
    assert isinstance(gate, dict)
    observed = gate["observed"]
    failures: list[str] = []
    if observed["tiny_fraction"] > args.max_tiny_fraction:
        failures.append(f"tiny_fraction {observed['tiny_fraction']:.3f} > {args.max_tiny_fraction:.3f}")
    if observed["small_fraction"] > args.max_small_fraction:
        failures.append(f"small_fraction {observed['small_fraction']:.3f} > {args.max_small_fraction:.3f}")
    if observed["soft_fraction"] > args.max_soft_fraction:
        failures.append(f"soft_fraction {observed['soft_fraction']:.3f} > {args.max_soft_fraction:.3f}")
    if observed["p05_short_px"] < args.min_p05_short_px:
        failures.append(f"p05_short_px {observed['p05_short_px']:.1f} < {args.min_p05_short_px:.1f}")
    if observed["p50_short_px"] < args.min_p50_short_px:
        failures.append(f"p50_short_px {observed['p50_short_px']:.1f} < {args.min_p50_short_px:.1f}")
    gate["failures"] = failures
    gate["status"] = "failed" if failures else "passed"

    sheet_rows = {
        "smallest_notes": sorted(rows, key=lambda row: row.min_train_short_px)[: args.items_per_sheet],
        "largest_notes": sorted(rows, key=lambda row: row.min_train_short_px, reverse=True)[: args.items_per_sheet],
        "softest_crops": sorted(rows, key=lambda row: row.min_crop_sharpness)[: args.items_per_sheet],
    }
    rng = random.Random(args.seed)
    random_rows = list(rows)
    rng.shuffle(random_rows)
    sheet_rows["random_blend"] = random_rows[: args.items_per_sheet]

    sheets = {}
    for name, selected in sheet_rows.items():
        path = args.out_dir / f"{name}.png"
        draw_sheet(selected, path, name.replace("_", " "), args.thumb_width)
        sheets[name] = rel(path)
    summary["sheets"] = sheets

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_on_quality and failures:
        raise SystemExit("visual QA gate failed: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    rows = collect(root, args.imgsz)
    if not rows:
        raise SystemExit(f"no train images found under {root / 'images' / 'train'}")
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
