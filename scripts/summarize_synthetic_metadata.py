"""Summarize synthetic scene metadata emitted by generate_synthetic_fan_dataset.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(frac: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * frac)))
        return ordered[index]

    return {
        "p10": at(0.10),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
    }


def iter_metadata(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield path, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def metadata_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    metadata_dir = root / "metadata"
    if metadata_dir.exists():
        return sorted(metadata_dir.glob("*.jsonl"))
    return sorted(root.glob("*.jsonl"))


def crop_counts(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    crop_root = root / "crops"
    if not crop_root.exists():
        return counts
    for path in crop_root.rglob("*.jpg"):
        counts[path.parent.name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Synthetic dataset root, metadata directory, or JSONL file.")
    args = parser.parse_args()

    paths = metadata_paths(args.path)
    if not paths:
        raise SystemExit(f"No metadata JSONL files found under {args.path}")

    scenes = 0
    instances = 0
    exported = 0
    tier_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    exported_class_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    hand_scene_counts: Counter[str] = Counter()
    hand_occluder_counts: list[float] = []
    hand_occluded_note_pixels: list[float] = []
    visible_area_by_tier: dict[str, list[float]] = defaultdict(list)
    visibility_ratio_by_tier: dict[str, list[float]] = defaultdict(list)

    for _, scene in iter_metadata(paths):
        scenes += 1
        split_counts[str(scene.get("split", "unknown"))] += 1
        background_counts[str(scene.get("background", "unknown"))] += 1
        hand_applied = bool(scene.get("hand_occluder_applied", False))
        hand_scene_counts["applied" if hand_applied else "none"] += 1
        if hand_applied:
            hand_occluder_counts.append(float(scene.get("hand_occluder_count", 0)))
            hand_occluded_note_pixels.append(float(scene.get("hand_occluded_note_pixels", 0)))
        for instance in scene.get("instances", []):
            instances += 1
            class_name = str(instance.get("class_name", "unknown"))
            tier = str(instance.get("evidence_tier", "unknown"))
            layout = str(instance.get("layout_mode", "unknown"))
            class_counts[class_name] += 1
            tier_counts[tier] += 1
            layout_counts[layout] += 1
            if instance.get("drop_reason"):
                drop_counts[str(instance["drop_reason"])] += 1
            if instance.get("exported_label"):
                exported += 1
                exported_class_counts[class_name] += 1
            visible_area_by_tier[tier].append(float(instance.get("visible_area_frac", 0.0)))
            visibility_ratio_by_tier[tier].append(float(instance.get("visibility_ratio", 0.0)))

    print(f"metadata_files: {len(paths)}")
    print(f"scenes: {scenes}")
    print(f"instances: {instances}")
    print(f"exported_labels: {exported}")
    print(f"splits: {dict(sorted(split_counts.items()))}")
    print(f"evidence_tiers: {dict(sorted(tier_counts.items()))}")
    print(f"drop_reasons: {dict(sorted(drop_counts.items()))}")
    print(f"layouts: {dict(sorted(layout_counts.items()))}")
    print(f"hand_occluders: {dict(sorted(hand_scene_counts.items()))}")
    if hand_occluder_counts:
        print(f"hand_occluder_count_quantiles: {quantiles(hand_occluder_counts)}")
        print(f"hand_occluded_note_pixels_quantiles: {quantiles(hand_occluded_note_pixels)}")
    if background_counts:
        print(f"backgrounds: {dict(background_counts.most_common(12))}")
    print(f"classes_all: {dict(sorted(class_counts.items()))}")
    print(f"classes_exported: {dict(sorted(exported_class_counts.items()))}")
    crops = crop_counts(args.path)
    if crops:
        print(f"crops: {dict(sorted(crops.items()))}")
    print("visible_area_frac_quantiles:")
    for tier, values in sorted(visible_area_by_tier.items()):
        print(f"  {tier}: {quantiles(values)}")
    print("visibility_ratio_quantiles:")
    for tier, values in sorted(visibility_ratio_by_tier.items()):
        print(f"  {tier}: {quantiles(values)}")


if __name__ == "__main__":
    main()
