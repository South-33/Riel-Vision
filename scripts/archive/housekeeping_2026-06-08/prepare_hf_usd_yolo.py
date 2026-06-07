from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

CANONICAL = {
    "USD_1": 0,
    "USD_5": 1,
    "USD_10": 2,
    "USD_20": 3,
    "USD_50": 4,
    "USD_100": 5,
}


def canonical_usd(raw_name: str) -> str | None:
    match = re.search(r"(100|50|20|10|5|1)\s*USD|(100|50|20|10|5|1)USD", raw_name)
    if not match:
        return None
    value = next(group for group in match.groups() if group)
    return f"USD_{value}"


def convert_split(source_root: Path, output_root: Path, split: str) -> dict[str, int]:
    annotation_path = source_root / split / "_annotations.coco.json"
    if not annotation_path.exists():
        return {"images": 0, "boxes": 0, "missing_images": 0}

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    images = {image["id"]: image for image in coco["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    image_out = output_root / "images" / split
    label_out = output_root / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    boxes = 0
    missing = 0
    for image_id, image in images.items():
        filename = image["file_name"]
        source_image = source_root / split / filename
        if not source_image.exists():
            missing += 1
            continue
        shutil.copy2(source_image, image_out / filename)
        width = image["width"]
        height = image["height"]
        lines = []
        for annotation in by_image.get(image_id, []):
            raw_name = categories[annotation["category_id"]]
            canonical = canonical_usd(raw_name)
            if canonical is None:
                continue
            x, y, w, h = annotation["bbox"]
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            lines.append(f"{CANONICAL[canonical]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (label_out / f"{Path(filename).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        copied += 1
        boxes += len(lines)

    return {"images": copied, "boxes": boxes, "missing_images": missing}


def write_yaml(output_root: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for name, index in CANONICAL.items())
    text = f"""path: {output_root.as_posix()}
train: images/train
val: images/valid
test: images/test

names:
{names}
"""
    (output_root / "data.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = root / "data" / "raw_datasets" / "hf_usd_side_coco_annotations"
    output_root = root / "data" / "processed" / "hf_usd_side_yolo_canonical"
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {split: convert_split(source_root, output_root, split) for split in ["train", "valid", "test"]}
    write_yaml(output_root)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
