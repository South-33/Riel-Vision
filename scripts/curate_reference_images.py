from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TARGET_KHR = {"500", "1000", "2000", "5000", "10000", "20000", "50000"}
DEFERRED_KHR = {"100", "100000"}


def parse_nbc_filename(path: Path) -> tuple[str, str]:
    name = path.stem.lower()
    match = re.search(r"_(\d{2,6})_(front|back|reverse)", name)
    if not match:
        return "junk", ""
    denomination = match.group(1)
    if denomination in TARGET_KHR:
        return "target", denomination
    if denomination in DEFERRED_KHR:
        return "deferred", denomination
    return "other_banknote", denomination


def image_ok(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        return False, f"unreadable: {exc}"
    if width < 80 or height < 30:
        return False, f"too small {width}x{height}"
    return True, f"{width}x{height}"


def curate_nbc() -> list[dict[str, str]]:
    source_dir = ROOT / "data" / "reference" / "khr_nbc"
    curated_dir = ROOT / "data" / "curated" / "reference" / "khr_nbc"
    rows: list[dict[str, str]] = []
    for path in sorted(source_dir.glob("*")):
        if not path.is_file():
            continue
        ok, note = image_ok(path)
        bucket, denomination = parse_nbc_filename(path) if ok else ("junk", "")
        target_dir = curated_dir / bucket / (f"KHR_{denomination}" if denomination else "site_assets")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        rows.append(
            {
                "source": str(path.relative_to(ROOT)),
                "curated_path": str(target.relative_to(ROOT)),
                "bucket": bucket,
                "denomination": denomination,
                "note": note,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(image_paths: list[Path], output: Path, title: str, thumb_w: int = 180) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        with Image.open(path).convert("RGB") as img:
            ratio = thumb_w / img.width
            thumb_h = max(1, int(img.height * ratio))
            img = img.resize((thumb_w, thumb_h))
            thumbs.append((path, img))
    cols = 4
    label_h = 36
    row_h = max(img.height for _, img in thumbs) + label_h
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * row_h + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for index, (path, img) in enumerate(thumbs):
        x = (index % cols) * thumb_w
        y = 40 + (index // cols) * row_h
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + img.height + 4), path.name[:28], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    rows = curate_nbc()
    write_csv(ROOT / "manifests" / "khr_nbc_curated_manifest.csv", rows)
    for bucket in ["target", "deferred", "other_banknote", "junk"]:
        paths = [
            ROOT / row["curated_path"]
            for row in rows
            if row["bucket"] == bucket and Path(row["curated_path"]).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        make_contact_sheet(paths, ROOT / "data" / "curated" / "reference" / f"khr_nbc_{bucket}_contact.jpg", f"KHR NBC {bucket}")
    print("Curated counts:")
    for bucket in ["target", "deferred", "other_banknote", "junk"]:
        print(f"{bucket}: {sum(1 for row in rows if row['bucket'] == bucket)}")


if __name__ == "__main__":
    main()
