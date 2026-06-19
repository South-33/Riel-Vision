#!/usr/bin/env python
"""Compile full real + 2x scaled WebGL synthetic dataset + targeted underperf mix."""

import os
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parents[1]

V16_SCALED2X_TXT = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_train.txt"
TARGET_DIR = ROOT / "data" / "synthetic" / "cashsnap_webgl_underperf_target_fan_candidate_v1" / "images" / "train"
OUT_TXT = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_target_underperf_train.txt"

CLASS_NAMES = [
    "USD_1", "USD_5", "USD_10", "USD_20", "USD_50", "USD_100",
    "KHR_500", "KHR_1000", "KHR_2000", "KHR_5000", "KHR_10000", "KHR_20000", "KHR_50000"
]

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

    # 2. Load the targeted synthetic images
    if not TARGET_DIR.exists():
        print(f"Error: Target directory does not exist yet: {TARGET_DIR}")
        return

    target_images = []
    for file in sorted(TARGET_DIR.glob("*")):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            # Convert to relative path from ROOT
            rel_path = file.relative_to(ROOT).as_posix()
            target_images.append(rel_path)

    print(f"Found {len(target_images)} unique targeted synthetic images.")

    # Repeat them 2x to match the 2x scaling factor of other WebGL synthetic images
    repeated_targets = target_images * 2
    print(f"Total targeted synthetic exposures to add: {len(repeated_targets)}")

    # 3. Merge
    final_list = base_lines + repeated_targets
    print(f"Total final images in mix: {len(final_list)}")

    # Write text list file
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(final_list) + "\n", encoding="utf-8")
    print(f"Wrote compiled training list to: {OUT_TXT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
