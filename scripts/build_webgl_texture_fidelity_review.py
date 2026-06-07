#!/usr/bin/env python
"""Build simple raw-scan vs WebGL-render crop review sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Rendered WebGL dataset root with manifest.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for review sheet and CSV.")
    parser.add_argument("--max-items", type=int, default=80, help="Maximum variants to include in one sheet.")
    parser.add_argument("--crop-pad", type=int, default=24, help="Pixel padding around the visible-box crop.")
    parser.add_argument("--pair-dir-name", default="pairs", help="Subdirectory for one large raw/render card per texture.")
    return parser.parse_args()


def resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def rgba_on_neutral(path: Path) -> Image.Image:
    with Image.open(path).convert("RGBA") as image:
        bg = Image.new("RGB", image.size, (246, 246, 246))
        bg.paste(image, mask=image.getchannel("A"))
        return bg


def crop_rendered_note(dataset_root: Path, manifest_row: dict[str, Any], pad: int) -> Image.Image:
    image_path = resolve(manifest_row["image"], dataset_root)
    boxes = read_json(resolve(manifest_row["visible_boxes"], dataset_root)).get("boxes", [])
    if not boxes:
        with Image.open(image_path).convert("RGB") as image:
            return image.copy()
    box = boxes[0]
    with Image.open(image_path).convert("RGB") as image:
        x1 = max(0, int(box["minX"]) - pad)
        y1 = max(0, int(box["minY"]) - pad)
        x2 = min(image.width, int(box["maxX"]) + pad)
        y2 = min(image.height, int(box["maxY"]) + pad)
        return image.crop((x1, y1, x2, y2))


def condition_label(asset: dict[str, Any]) -> str:
    condition = asset.get("condition", {}) or {}
    return (
        f"dirt={condition.get('dirtiness')} "
        f"crinkle={condition.get('crinkle')} "
        f"edge={condition.get('edgeWear')} "
        f"crease={condition.get('creaseCount')} "
        f"profile={condition.get('profile')}"
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pair_card(
    raw: Image.Image,
    rendered: Image.Image,
    asset: dict[str, Any],
    manifest_row: dict[str, Any],
    out_path: Path,
) -> None:
    font = safe_font(24)
    small = safe_font(17)
    panel_w, panel_h = 860, 520
    label_h = 72
    card = Image.new("RGB", (panel_w * 2, panel_h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    raw_fit = ImageOps.contain(raw, (panel_w - 48, panel_h - 32), Image.Resampling.LANCZOS)
    rendered_fit = ImageOps.contain(rendered, (panel_w - 48, panel_h - 32), Image.Resampling.LANCZOS)
    card.paste(raw_fit, (24 + (panel_w - 48 - raw_fit.width) // 2, label_h + (panel_h - 32 - raw_fit.height) // 2))
    card.paste(
        rendered_fit,
        (panel_w + 24 + (panel_w - 48 - rendered_fit.width) // 2, label_h + (panel_h - 32 - rendered_fit.height) // 2),
    )
    draw.line((panel_w, 0, panel_w, panel_h + label_h), fill=(216, 216, 216), width=1)
    draw.text(
        (24, 14),
        f"RAW {asset.get('className', '')} {asset.get('side', '')}",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((24, 42), Path(asset["path"]).name, fill=(80, 80, 80), font=small)
    draw.text(
        (panel_w + 24, 14),
        f"RENDER variant_{manifest_row.get('variant')}",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((panel_w + 24, 42), condition_label(asset), fill=(80, 80, 80), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(out_path, quality=94)


def build_sheet(dataset_root: Path, out_dir: Path, max_items: int, crop_pad: int, pair_dir_name: str) -> None:
    manifest = read_json(dataset_root / "manifest.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = out_dir / pair_dir_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    font = safe_font(18)
    tile_w, tile_h = 1040, 286
    raw_box = (450, 216)
    render_box = (450, 216)
    tiles: list[Image.Image] = []
    rows: list[dict[str, str]] = []

    for manifest_row in manifest[:max_items]:
        metadata = read_json(resolve(manifest_row["source_metadata"], dataset_root))
        assets = metadata.get("assets", [])
        if not assets:
            continue
        asset = assets[0]
        raw_path = Path(asset["path"])
        raw = ImageOps.contain(rgba_on_neutral(raw_path), raw_box, Image.Resampling.LANCZOS)
        raw_full = rgba_on_neutral(raw_path)
        rendered_full = crop_rendered_note(dataset_root, manifest_row, crop_pad)
        rendered = ImageOps.contain(
            rendered_full,
            render_box,
            Image.Resampling.LANCZOS,
        )

        tile = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
        draw = ImageDraw.Draw(tile)
        tile.paste(raw, (20, 54 + (raw_box[1] - raw.height) // 2))
        tile.paste(rendered, (560, 54 + (render_box[1] - rendered.height) // 2))
        draw.line((520, 0, 520, tile_h), fill=(210, 210, 210), width=1)
        draw.text(
            (20, 14),
            f"RAW {asset.get('className', '')} {asset.get('side', '')} {raw_path.name}",
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (560, 14),
            f"RENDER variant_{manifest_row.get('variant')} {condition_label(asset)}",
            fill=(0, 0, 0),
            font=font,
        )
        pair_name = f"variant_{int(manifest_row.get('variant')):04d}_{asset.get('className', '')}_{asset.get('side', '')}.jpg"
        pair_path = pair_dir / pair_name
        write_pair_card(raw_full, rendered_full, asset, manifest_row, pair_path)
        tiles.append(tile)
        rows.append(
            {
                "variant": str(manifest_row.get("variant")),
                "class_name": str(asset.get("className", "")),
                "side": str(asset.get("side", "")),
                "source_path": str(raw_path),
                "condition": condition_label(asset),
                "render_image": str(resolve(manifest_row["image"], dataset_root)),
                "pair_card": str(pair_path),
            }
        )

    if not tiles:
        raise SystemExit(f"no reviewable assets found in {dataset_root}")
    sheet = Image.new("RGB", (tile_w, tile_h * len(tiles)), (236, 236, 236))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, index * tile_h))
    sheet_path = out_dir / "scan_vs_render_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    write_csv(out_dir / "scan_vs_render_rows.csv", rows)
    print(f"sheet: {sheet_path}")
    print(f"rows: {out_dir / 'scan_vs_render_rows.csv'}")
    print(f"pairs: {pair_dir}")


def main() -> int:
    args = parse_args()
    dataset_root = args.root if args.root.is_absolute() else ROOT / args.root
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    build_sheet(dataset_root.resolve(), out_dir.resolve(), args.max_items, args.crop_pad, args.pair_dir_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
