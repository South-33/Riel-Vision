#!/usr/bin/env python
"""Compile full real + 2x scaled WebGL synthetic dataset + targeted underperf mix + oblique fan mix.

This version adds the oblique fan dataset (phone_hard_eval_mix camera) on top of the underperf mix,
closing the camera distribution gap that caused regression in the previous fine-tune.
"""

import os
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

V16_SCALED2X_TXT = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_train.txt"
UNDERPERF_DIR = ROOT / "data" / "synthetic" / "cashsnap_webgl_underperf_target_fan_candidate_v1" / "images" / "train"
OBLIQUE_FAN_DIR = ROOT / "data" / "synthetic" / "cashsnap_webgl_oblique_fan_full_candidate_v1" / "images" / "train"
OUT_TXT = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_train.txt"

CLASS_NAMES = [
    "USD_1", "USD_5", "USD_10", "USD_20", "USD_50", "USD_100",
    "KHR_500", "KHR_1000", "KHR_2000", "KHR_5000", "KHR_10000", "KHR_20000", "KHR_50000"
]

def gather_images(directory: Path) -> list[str]:
    """Gather all image files from a directory, sorted, as repo-relative posix paths."""
    images = []
    for file in sorted(directory.glob("*")):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            images.append(file.relative_to(ROOT).as_posix())
    return images

def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")

def label_class_ids(image: str) -> list[int]:
    label_path = ROOT / label_path_for_image(image)
    if not label_path.exists():
        return []
    class_ids = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 1:
            try:
                class_ids.append(int(float(parts[0])))
            except ValueError:
                pass
    return class_ids

def main():
    if not V16_SCALED2X_TXT.exists():
        print(f"Error: Base V16 Scaled2x list not found at {V16_SCALED2X_TXT}")
        return

    # 1. Load the base list
    base_lines = [l.strip() for l in V16_SCALED2X_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(base_lines)} base training images.")

    # 2. Load the underperf targeted synthetic images (USD_100 / KHR_10000, phone_auto)
    underperf_images = []
    if UNDERPERF_DIR.exists():
        underperf_images = gather_images(UNDERPERF_DIR)
        print(f"Found {len(underperf_images)} underperf targeted images (phone_auto fan).")
    else:
        print(f"Warning: Underperf dir not found: {UNDERPERF_DIR}")

    # 3. Load the oblique fan images (all classes, phone_hard_eval_mix camera)
    oblique_images = []
    if OBLIQUE_FAN_DIR.exists():
        oblique_images = gather_images(OBLIQUE_FAN_DIR)
        print(f"Found {len(oblique_images)} oblique fan images (phone_hard_eval_mix).")
    else:
        print(f"Error: Oblique fan dir not found: {OBLIQUE_FAN_DIR}")
        return

    # Repeat underperf 2x (match 2x scaling of other WebGL sets), oblique 2x (same treatment)
    repeated_underperf = underperf_images * 2
    repeated_oblique = oblique_images * 2
    print(f"Underperf exposures: {len(repeated_underperf)}")
    print(f"Oblique fan exposures: {len(repeated_oblique)}")

    # 4. Merge
    final_list = base_lines + repeated_underperf + repeated_oblique
    print(f"Total final images in mix: {len(final_list)}")

    # Write text list file
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(final_list) + "\n", encoding="utf-8")
    print(f"Wrote compiled training list to: {OUT_TXT.relative_to(ROOT)}")

    # 5. Class distribution report (sample from oblique fan)
    print("\n--- Oblique fan class distribution sample ---")
    counter: Counter = Counter()
    for img in oblique_images[:64]:  # sample first 64
        for cid in label_class_ids(img):
            counter[cid] += 1
    for cid, cnt in sorted(counter.items()):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls_{cid}"
        print(f"  {name}: {cnt}")

if __name__ == "__main__":
    main()
