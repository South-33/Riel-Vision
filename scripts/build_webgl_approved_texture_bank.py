#!/usr/bin/env python
"""Select and document approved WebGL banknote textures from the cutout manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from cashsnap_currency_taxonomy import class_names_for_scope

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "data" / "asset_candidates" / "numista_current_cutout_bank_v1"
DEFAULT_OUT = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_approved_texture_bank_v1.json"
DEFAULT_CONTACT = ROOT / "runs" / "cashsnap" / "texture_bank_review" / "approved_texture_bank_v1.jpg"
SIDES = ["front", "back"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="Cutout bank root containing manifest.csv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Approved texture-bank JSON to write.")
    parser.add_argument("--contact-sheet", type=Path, default=DEFAULT_CONTACT, help="Visual sheet of selected textures.")
    parser.add_argument("--class-scope", choices=["operational", "official"], default="operational")
    parser.add_argument(
        "--name",
        default=None,
        help="Override output JSON name. Defaults to active operational bank name.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a partial bank and record missing class/sides instead of failing.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_manifest(bank: Path) -> list[dict[str, str]]:
    manifest = bank / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_existing_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT).as_posix())


def start_year(value: str) -> int:
    match = re.search(r"\d{4}", value or "")
    return int(match.group(0)) if match else 0


def numeric(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def score(row: dict[str, str]) -> tuple[float, ...]:
    return (
        1.0 if row.get("status") == "in_circulation" else 0.0,
        numeric(row, "max_year"),
        float(start_year(row.get("years", ""))),
        numeric(row, "width") * numeric(row, "height"),
        numeric(row, "alpha_area"),
    )


def select_rows(rows: list[dict[str, str]], class_names: list[str], *, allow_missing: bool) -> tuple[list[dict[str, str]], list[str]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        class_name = row.get("class_name", "")
        side = row.get("side", "")
        if class_name in class_names and side in SIDES:
            by_key[(class_name, side)].append(row)

    selected: list[dict[str, str]] = []
    missing: list[str] = []
    for class_name in class_names:
        for side in SIDES:
            candidates = by_key.get((class_name, side), [])
            if not candidates:
                missing.append(f"{class_name}/{side}")
                continue
            selected.append(max(candidates, key=score))
    if missing and not allow_missing:
        raise SystemExit(f"missing class sides: {', '.join(missing)}")
    return selected, missing


def safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_texture(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path).convert("RGBA") as image:
        bg = Image.new("RGB", image.size, (246, 246, 246))
        bg.paste(image, mask=image.getchannel("A"))
        return ImageOps.contain(bg, size, Image.Resampling.LANCZOS)


def write_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = safe_font(15)
    cols = 2
    thumb_w, thumb_h = 520, 236
    label_h = 56
    sheet = Image.new("RGB", (cols * thumb_w, ((len(rows) + cols - 1) // cols) * (thumb_h + label_h)), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + label_h)
        tile = Image.new("RGB", (thumb_w, thumb_h + label_h), (255, 255, 255))
        texture = render_texture(ROOT / row["asset_path"], (thumb_w - 24, thumb_h - 20))
        tile.paste(texture, ((thumb_w - texture.width) // 2, 10 + (thumb_h - 20 - texture.height) // 2))
        sheet.paste(tile, (x, y))
        label = f"{row['class_name']} {row['side']} {row['years']} {row['width']}x{row['height']}"
        draw.text((x + 8, y + thumb_h + 8), label, fill=(0, 0, 0), font=font)
        draw.text((x + 8, y + thumb_h + 30), Path(row["asset_path"]).name, fill=(76, 76, 76), font=font)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    bank = resolve(args.bank).resolve()
    out = resolve(args.out).resolve()
    contact = resolve(args.contact_sheet).resolve()
    class_names = class_names_for_scope(args.class_scope)
    existing_payload = read_existing_payload(out)
    existing_reviews = {
        (row.get("class_name"), row.get("side"), row.get("asset_path")): row
        for row in existing_payload.get("rows", [])
        if isinstance(row, dict)
    }
    selected, missing = select_rows(read_manifest(bank), class_names, allow_missing=args.allow_missing)
    rows: list[dict[str, Any]] = []
    for row in selected:
        asset_path = repo_path(ROOT / row["asset_path"])
        output_row = {
            "class_name": row["class_name"],
            "side": row["side"],
            "asset_path": asset_path,
            "source_path": row.get("source_path", ""),
            "status": row.get("status", ""),
            "years": row.get("years", ""),
            "max_year": row.get("max_year", ""),
            "width": int(float(row.get("width", 0) or 0)),
            "height": int(float(row.get("height", 0) or 0)),
            "selection_basis": "latest_design_by_status_max_year_start_year_resolution",
            "visual_review_status": "pending_manual_review",
        }
        preserved = existing_reviews.get((row["class_name"], row["side"], asset_path))
        if preserved:
            for key, value in preserved.items():
                if key.startswith("visual_review_"):
                    output_row[key] = value
        rows.append(output_row)
    payload = {
        "schema_version": 1,
        "name": args.name or "cashsnap_webgl_approved_texture_bank_v1",
        "source_bank": repo_path(bank),
        "class_scope": args.class_scope,
        "target_class_names": class_names,
        "selection_policy": "latest_design",
        "class_side_count": len(rows),
        "missing_class_sides": missing,
        "rows": rows,
        "outputs": {"contact_sheet": repo_path(contact)},
    }
    if isinstance(existing_payload.get("visual_review"), dict):
        payload["visual_review"] = existing_payload["visual_review"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_contact_sheet(rows, contact)
    print(f"approved texture bank: {out}")
    print(f"contact sheet: {contact}")
    print(f"textures: {len(rows)}")
    if missing:
        print(f"missing: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
