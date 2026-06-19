#!/usr/bin/env python
"""Compile the current demo-gap mix plus a tight KHR_10000/KHR_20000 contrast dose."""

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
    / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap_train.txt"
)
BASE_YAML = ROOT / "configs" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap.yaml"

CONTRAST_SOURCES = [
    (
        "targeted_khr10k20k_clean_contrast",
        ROOT / "data" / "synthetic" / "cashsnap_webgl_demo_gap_khr10k20k_clean_candidate_v1" / "images" / "train",
        3,
    ),
    (
        "targeted_khr10k20k_fan_contrast",
        ROOT / "data" / "synthetic" / "cashsnap_webgl_demo_gap_khr10k20k_fan_candidate_v1" / "images" / "train",
        2,
    ),
]

OUT_TXT = (
    ROOT
    / "configs"
    / "generated_lists"
    / "webgl_ablation"
    / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap_khr10k20k_train.txt"
)
OUT_YAML = (
    ROOT
    / "configs"
    / "webgl_ablation"
    / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap_khr10k20k.yaml"
)

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

    base_lines = [line.strip() for line in BASE_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]
    final_lines = list(base_lines)
    source_counts: dict[str, dict[str, int]] = {}
    component_sources: dict[str, list[str]] = {}

    for source_name, image_dir, repeat in CONTRAST_SOURCES:
        if not image_dir.exists():
            raise FileNotFoundError(f"{source_name} image dir not found: {image_dir}")
        images = gather_images(image_dir)
        if not images:
            raise RuntimeError(f"no images found for {source_name}: {image_dir}")
        final_lines.extend(images * repeat)
        source_counts[source_name] = {
            "unique": len(images),
            "repeat": repeat,
            "repeated_exposures": len(images) * repeat,
        }
        component_sources[source_name] = [image_dir.relative_to(ROOT).as_posix()]

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    yaml_data = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_data["train"] = OUT_TXT.relative_to(ROOT).as_posix()

    pilot = dict(yaml_data.get("cashsnap_production_pilot", {}))
    existing_sources = dict(pilot.get("component_sources", {}))
    existing_sources.update(component_sources)
    pilot.update(
        {
            "tag": "cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap_khr10k20k",
            "base_reference_config": "configs/webgl_ablation/cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap.yaml",
            "recommended_init_checkpoint": "runs/cashsnap/cashsnap_v16_oblique_fan_demogap_fast_b32/weights/best.pt",
            "component_sources": existing_sources,
            "summary": {
                **summarize(final_lines),
                "base_rows": len(base_lines),
                **source_counts,
            },
        }
    )
    yaml_data["cashsnap_production_pilot"] = pilot
    yaml_data["cashsnap_policy"] = {
        "intended_use": (
            "Short final contrast fine-tune: current demo-gap mix plus tight clean/fan "
            "KHR_10000-vs-KHR_20000 synthetic dose for browser demo class confusion."
        ),
        "promotion_rule": (
            "Promote only if exact browser clean/fan smoke improves and hard oblique plus clean real "
            "guardrails do not materially regress."
        ),
    }

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    print(f"base_rows={len(base_lines)}")
    for name, stats in source_counts.items():
        print(
            f"{name}_unique={stats['unique']} repeat={stats['repeat']} "
            f"exposures={stats['repeated_exposures']}"
        )
    print(f"total_rows={len(final_lines)}")
    print(f"wrote_list={OUT_TXT.relative_to(ROOT).as_posix()}")
    print(f"wrote_yaml={OUT_YAML.relative_to(ROOT).as_posix()}")
    print("class_counts:")
    for name, count in summarize(final_lines)["class_counts"].items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
