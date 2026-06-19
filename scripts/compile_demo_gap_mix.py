#!/usr/bin/env python
"""Compile the oblique-fan champion mix plus the targeted browser/demo KHR fan dose."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

BASE_TXT = (
    ROOT
    / "configs"
    / "generated_lists"
    / "webgl_ablation"
    / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_train.txt"
)
BASE_YAML = ROOT / "configs" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_oblique_fan.yaml"
DEMO_GAP_DIR = ROOT / "data" / "synthetic" / "cashsnap_webgl_demo_gap_khr_fan_candidate_v1" / "images" / "train"

OUT_TXT = (
    ROOT
    / "configs"
    / "generated_lists"
    / "webgl_ablation"
    / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap_train.txt"
)
OUT_YAML = ROOT / "configs" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap.yaml"

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


def gather_images(directory: Path) -> list[str]:
    return [
        path.relative_to(ROOT).as_posix()
        for path in sorted(directory.glob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
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
    class_ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_ids.append(int(float(parts[0])))
        except ValueError:
            continue
    return class_ids


def summarize(lines: list[str]) -> dict:
    class_counter: Counter[int] = Counter()
    empty_rows = 0
    for image in lines:
        class_ids = label_class_ids(image)
        if class_ids:
            class_counter.update(class_ids)
        else:
            empty_rows += 1
    return {
        "rows": len(lines),
        "unique_rows": len(set(lines)),
        "duplicate_exposures": len(lines) - len(set(lines)),
        "empty_rows": empty_rows,
        "class_counts": {
            CLASS_NAMES[class_id]: count
            for class_id, count in sorted(class_counter.items())
            if 0 <= class_id < len(CLASS_NAMES)
        },
    }


def main() -> None:
    if not BASE_TXT.exists():
        raise FileNotFoundError(f"base list not found: {BASE_TXT}")
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"base YAML not found: {BASE_YAML}")
    if not DEMO_GAP_DIR.exists():
        raise FileNotFoundError(f"demo-gap image dir not found: {DEMO_GAP_DIR}")

    base_lines = [line.strip() for line in BASE_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]
    demo_gap_images = gather_images(DEMO_GAP_DIR)
    if not demo_gap_images:
        raise RuntimeError(f"no demo-gap images found under: {DEMO_GAP_DIR}")

    demo_gap_exposures = demo_gap_images * 2
    final_lines = base_lines + demo_gap_exposures

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    yaml_data = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_data["train"] = OUT_TXT.relative_to(ROOT).as_posix()

    pilot = dict(yaml_data.get("cashsnap_production_pilot", {}))
    component_sources = dict(pilot.get("component_sources", {}))
    component_sources["targeted_demo_gap_oblique_fan"] = [
        "data/synthetic/cashsnap_webgl_demo_gap_khr_fan_candidate_v1/images/train"
    ]
    pilot.update(
        {
            "tag": "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap",
            "base_reference_config": "configs/webgl_ablation/cashsnap_production_pilot_v16_scaled2x_oblique_fan.yaml",
            "recommended_init_checkpoint": "runs/cashsnap/cashsnap_v16_oblique_fan_finetune/weights/best.pt",
            "component_sources": component_sources,
            "summary": {
                **summarize(final_lines),
                "base_rows": len(base_lines),
                "demo_gap_unique": len(demo_gap_images),
                "demo_gap_repeated_exposures": len(demo_gap_exposures),
            },
        }
    )
    yaml_data["cashsnap_production_pilot"] = pilot
    yaml_data["cashsnap_policy"] = {
        "intended_use": (
            "Final demo-gap fine-tune: current oblique-fan champion mix plus a 2x targeted "
            "front-only no-hand KHR_5000/KHR_10000 wide-fan dose for browser demo misses."
        ),
        "promotion_rule": "Promote only if browser fan demo and hard oblique fan eval improve without clean real regression.",
    }

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    print(f"base_rows={len(base_lines)}")
    print(f"demo_gap_unique={len(demo_gap_images)}")
    print(f"demo_gap_exposures={len(demo_gap_exposures)}")
    print(f"total_rows={len(final_lines)}")
    print(f"wrote_list={OUT_TXT.relative_to(ROOT).as_posix()}")
    print(f"wrote_yaml={OUT_YAML.relative_to(ROOT).as_posix()}")
    print("class_counts:")
    for name, count in summarize(final_lines)["class_counts"].items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
