#!/usr/bin/env python
"""Build a tiny YOLO calibration dataset from saved browser demo scene renders."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "runs" / "cashsnap" / "demo_scene_debug_v1"
OUT_ROOT = ROOT / "data" / "synthetic" / "cashsnap_browser_scene_calibration_v1"

CLASS_NAMES = [
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]
CLASS_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Box:
    class_name: str
    xyxy: tuple[float, float, float, float]


SCENES: dict[str, list[Box]] = {
    "clean": [
        Box("KHR_50000", (344, 326, 592, 408)),
        Box("KHR_20000", (540, 272, 785, 350)),
        Box("KHR_10000", (746, 319, 1002, 425)),
        Box("KHR_5000", (932, 392, 1218, 526)),
        Box("KHR_1000", (370, 451, 671, 571)),
        Box("USD_100", (633, 458, 1015, 654)),
    ],
    "fan": [
        Box("KHR_50000", (371, 371, 606, 489)),
        Box("KHR_20000", (480, 314, 706, 425)),
        Box("KHR_10000", (583, 295, 816, 373)),
        Box("KHR_5000", (714, 294, 916, 378)),
        Box("KHR_1000", (804, 325, 1014, 430)),
        Box("USD_100", (902, 318, 1158, 499)),
    ],
}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def yolo_line(box: Box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box.xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{CLASS_ID[box.class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def augment(image: Image.Image, rng: random.Random, variant: int) -> Image.Image:
    out = image.convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.86, 1.14))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.88, 1.18))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.82, 1.18))
    out = ImageEnhance.Sharpness(out).enhance(rng.uniform(0.88, 1.12))
    if variant % 7 == 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.10, 0.28)))
    return out


def write_split(split: str, rows: list[tuple[str, int]]) -> dict[str, int]:
    image_dir = OUT_ROOT / "images" / split
    label_dir = OUT_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    class_counts = {name: 0 for name in CLASS_NAMES}
    for scene_name, variant in rows:
        src = SRC_DIR / f"cashsnap_slide_scene_{scene_name}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        with Image.open(src) as raw:
            image = raw.convert("RGB")
        rng = random.Random(91017 + variant * 97 + len(scene_name))
        out_image = augment(image, rng, variant)
        out_name = f"{scene_name}_{variant:03d}.jpg"
        out_image.save(image_dir / out_name, quality=92, subsampling=1)

        width, height = image.size
        lines = [yolo_line(box, width, height) for box in SCENES[scene_name]]
        for box in SCENES[scene_name]:
            class_counts[box.class_name] += 1
        (label_dir / out_name).with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {name: count for name, count in class_counts.items() if count}


def main() -> None:
    reset_dir(OUT_ROOT)
    train_rows = [(scene, variant) for scene in SCENES for variant in range(36)]
    val_rows = [(scene, variant) for scene in SCENES for variant in range(36, 42)]
    test_rows = [(scene, variant) for scene in SCENES for variant in range(42, 48)]

    summary = {
        "schema": "cashsnap_browser_scene_calibration_v1",
        "source_dir": SRC_DIR.relative_to(ROOT).as_posix(),
        "note": "Demo-only synthetic calibration data. Do not use as evidence of real-world robustness.",
        "splits": {
            "train": {"images": len(train_rows), "class_counts": write_split("train", train_rows)},
            "val": {"images": len(val_rows), "class_counts": write_split("val", val_rows)},
            "test": {"images": len(test_rows), "class_counts": write_split("test", test_rows)},
        },
        "manual_boxes_xyxy": {
            scene: [{"class_name": box.class_name, "xyxy": list(box.xyxy)} for box in boxes]
            for scene, boxes in SCENES.items()
        },
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (OUT_ROOT / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"wrote={OUT_ROOT.relative_to(ROOT).as_posix()}")
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
