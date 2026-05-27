from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build human-review crop sheets from two-stage proposal CSVs without creating training labels."
    )
    parser.add_argument(
        "--item",
        nargs=2,
        action="append",
        metavar=("IMAGE", "CSV"),
        required=True,
        help="Source image and proposal CSV pair. Repeat for multiple images.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--class-column", default="fragment_class")
    parser.add_argument("--score-column", default="fragment_conf")
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--sheet-columns", type=int, default=5)
    parser.add_argument("--padding", type=float, default=0.06)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def crop_box(
    image_size: tuple[int, int],
    row: dict[str, str],
    padding: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = (float(row[key]) for key in ["x1", "y1", "x2", "y2"])
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )


def fit_thumbnail(image: Image.Image, size: int) -> Image.Image:
    output = Image.new("RGB", (size, size), "white")
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    output.paste(thumb, (x, y))
    return output


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace(".", "_")


def write_contact_sheet(
    thumbs: list[tuple[Image.Image, str]],
    out_path: Path,
    columns: int,
    thumb_size: int,
) -> None:
    if not thumbs:
        return
    label_height = 42
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(14)
    for idx, (thumb, label) in enumerate(thumbs):
        col = idx % columns
        row = idx // columns
        x = col * thumb_size
        y = row * (thumb_size + label_height)
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size + label_height - 1), outline=(180, 180, 180))
        draw.text((x + 6, y + thumb_size + 6), label[:42], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    crop_dir = out_dir / "crops"
    sheet_dir = out_dir / "sheets"
    crop_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, str]] = []
    for image_text, csv_text in args.item:
        image_path = resolve(image_text)
        csv_path = resolve(csv_text)
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
        thumbs: list[tuple[Image.Image, str]] = []
        with Image.open(image_path).convert("RGB") as image:
            image_key = safe_name(image_path)
            for row in rows:
                proposal_index = row.get("index", str(len(review_rows)))
                box = crop_box(image.size, row, args.padding)
                crop = image.crop(box)
                crop_name = f"{image_key}_{int(proposal_index):03d}.jpg"
                crop_path = crop_dir / crop_name
                crop.save(crop_path, quality=92)
                pred_class = row.get(args.class_column, "")
                pred_score = row.get(args.score_column, "")
                thumbs.append((fit_thumbnail(crop, args.thumb_size), f"{proposal_index} {pred_class} {pred_score}"))
                review_rows.append(
                    {
                        "image": str(image_path.relative_to(ROOT) if image_path.is_relative_to(ROOT) else image_path),
                        "proposal_csv": str(csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path),
                        "proposal_index": proposal_index,
                        "crop_path": str(crop_path.relative_to(ROOT) if crop_path.is_relative_to(ROOT) else crop_path),
                        "x1": row.get("x1", ""),
                        "y1": row.get("y1", ""),
                        "x2": row.get("x2", ""),
                        "y2": row.get("y2", ""),
                        "detector_class": row.get("detector_class", ""),
                        "detector_conf": row.get("detector_conf", ""),
                        "fragment_class": row.get("fragment_class", ""),
                        "fragment_conf": row.get("fragment_conf", ""),
                        "review_include": "",
                        "review_class": "",
                        "review_notes": "",
                    }
                )
        write_contact_sheet(thumbs, sheet_dir / f"{safe_name(image_path)}.jpg", args.sheet_columns, args.thumb_size)

    fieldnames = [
        "image",
        "proposal_csv",
        "proposal_index",
        "crop_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "detector_class",
        "detector_conf",
        "fragment_class",
        "fragment_conf",
        "review_include",
        "review_class",
        "review_notes",
    ]
    review_csv = out_dir / "review.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"wrote {len(review_rows)} proposal crops to {review_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
