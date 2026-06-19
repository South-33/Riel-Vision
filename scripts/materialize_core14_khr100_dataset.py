#!/usr/bin/env python
"""Materialize a 13-class CashSnap mix plus KHR_100 into one 14-class YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

TARGET_NAMES = {
    0: "USD_1",
    1: "USD_5",
    2: "USD_10",
    3: "USD_20",
    4: "USD_50",
    5: "USD_100",
    6: "KHR_100",
    7: "KHR_500",
    8: "KHR_1000",
    9: "KHR_2000",
    10: "KHR_5000",
    11: "KHR_10000",
    12: "KHR_20000",
    13: "KHR_50000",
}

CORE13_TO_CORE14 = {old: (old if old <= 5 else old + 1) for old in range(13)}
OFFICIAL21_TO_CORE14 = {
    0: 0,   # USD_1
    2: 1,   # USD_5
    3: 2,   # USD_10
    4: 3,   # USD_20
    5: 4,   # USD_50
    6: 5,   # USD_100
    8: 6,   # KHR_100
    10: 7,  # KHR_500
    11: 8,  # KHR_1000
    12: 9,  # KHR_2000
    13: 10, # KHR_5000
    14: 11, # KHR_10000
    16: 12, # KHR_20000
    18: 13, # KHR_50000
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core13-config",
        default="configs/webgl_ablation/cashsnap_production_pilot_v16_scaled2x_oblique_fan_demogap_browsercalib3x.yaml",
        type=Path,
    )
    parser.add_argument(
        "--official21-config",
        default="configs/official21/cashsnap_official21_roboflow_currentcap180_empty360_khr100repeat3_current24_v1.yaml",
        type=Path,
    )
    parser.add_argument("--out-root", default="data/processed/cashsnap_core14_khr100_one_model_v1", type=Path)
    parser.add_argument("--out-config", default="configs/generated/cashsnap_core14_khr100_one_model_v1.yaml", type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return payload


def split_root(config_path: Path, config: dict[str, Any], split_value: str) -> Path:
    value = Path(split_value)
    if value.is_absolute():
        return value
    dataset_root = (config_path.parent / str(config.get("path", "."))).resolve()
    return dataset_root / value


def image_rows(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    split_value = config.get(split)
    if not split_value:
        return []
    resolved = split_root(config_path, config, str(split_value))
    if resolved.suffix.lower() == ".txt":
        rows: list[Path] = []
        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            rows.append(path if path.is_absolute() else ROOT / path)
        return rows
    return sorted(
        path for path in resolved.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_remapped_labels(image: Path, class_map: dict[int, int], require_class: int | None = None) -> list[str]:
    label = label_path_for_image(image)
    if not label.exists():
        raise FileNotFoundError(f"Missing label for {image}: {label}")
    remapped: list[str] = []
    seen_required = require_class is None
    for line_no, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label)}:{line_no}: expected 5 YOLO fields, got {len(parts)}")
        old_class = int(parts[0])
        if old_class == require_class:
            seen_required = True
        new_class = class_map.get(old_class)
        if new_class is None:
            continue
        remapped.append(" ".join([str(new_class), *parts[1:]]))
    return remapped if seen_required else []


def stable_stem(image: Path, row_index: int) -> str:
    digest = hashlib.sha1(f"{repo_rel(image)}\n{row_index}".encode("utf-8")).hexdigest()[:12]
    return f"{row_index:06d}_{image.stem}_{digest}"


def hardlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_rows(
    *,
    rows: list[tuple[Path, dict[int, int], int | None, str]],
    out_root: Path,
    split: str,
) -> dict[str, Any]:
    image_dir = out_root / "images" / split
    label_dir = out_root / "labels" / split
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    written = 0
    skipped_empty = 0
    for row_index, (image, class_map, require_class, source_tag) in enumerate(rows):
        remapped = read_remapped_labels(image, class_map, require_class=require_class)
        if not remapped:
            skipped_empty += 1
            continue
        stem = stable_stem(image, row_index)
        out_image = image_dir / f"{stem}{image.suffix.lower()}"
        out_label = label_dir / f"{stem}.txt"
        hardlink_or_copy(image, out_image)
        out_label.write_text("\n".join(remapped) + "\n", encoding="utf-8")
        for line in remapped:
            class_counts[TARGET_NAMES[int(line.split()[0])]] += 1
        source_counts[source_tag] += 1
        written += 1

    return {
        "images": written,
        "skipped_empty": skipped_empty,
        "boxes_by_class": dict(sorted(class_counts.items())),
        "rows_by_source": dict(sorted(source_counts.items())),
    }


def write_config(out_config: Path, out_root: Path) -> None:
    payload = {
        "path": "../..",
        "train": f"{repo_rel(out_root)}/images/train",
        "val": f"{repo_rel(out_root)}/images/val",
        "test": f"{repo_rel(out_root)}/images/test",
        "names": TARGET_NAMES,
        "cashsnap_policy": {
            "intended_use": "one promoted detector schema: current 13 CashSnap classes plus labeled KHR_100",
            "note": "Classes without training labels are intentionally excluded instead of advertised.",
        },
    }
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    core_path = resolve(args.core13_config)
    official_path = resolve(args.official21_config)
    out_root = resolve(args.out_root)
    out_config = resolve(args.out_config)
    core_config = load_yaml(core_path)
    official_config = load_yaml(official_path)

    train_rows: list[tuple[Path, dict[int, int], int | None, str]] = []
    train_rows.extend((image, CORE13_TO_CORE14, None, "core13_current_demo_mix") for image in image_rows(core_path, core_config, "train"))
    train_rows.extend((image, OFFICIAL21_TO_CORE14, 8, "khr100_official21_replay") for image in image_rows(official_path, official_config, "train"))

    eval_rows: dict[str, list[tuple[Path, dict[int, int], int | None, str]]] = {}
    for split in ("val", "test"):
        rows: list[tuple[Path, dict[int, int], int | None, str]] = []
        rows.extend((image, CORE13_TO_CORE14, None, f"core13_{split}") for image in image_rows(core_path, core_config, split))
        rows.extend((image, OFFICIAL21_TO_CORE14, 8, f"khr100_official21_{split}") for image in image_rows(official_path, official_config, split))
        eval_rows[split] = rows

    summary = {
        "schema": "cashsnap_core14_khr100_one_model_dataset_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "core13_config": repo_rel(core_path),
        "official21_config": repo_rel(official_path),
        "out_root": repo_rel(out_root),
        "out_config": repo_rel(out_config),
        "target_names": TARGET_NAMES,
        "splits": {
            "train": materialize_rows(rows=train_rows, out_root=out_root, split="train"),
            "val": materialize_rows(rows=eval_rows["val"], out_root=out_root, split="val"),
            "test": materialize_rows(rows=eval_rows["test"], out_root=out_root, split="test"),
        },
    }
    write_config(out_config, out_root)
    summary_path = resolve(args.summary_json) if args.summary_json else out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_config={repo_rel(out_config)} summary={repo_rel(summary_path)}")
    for split, split_summary in summary["splits"].items():
        print(
            f"{split}: images={split_summary['images']} skipped_empty={split_summary['skipped_empty']} "
            f"classes={split_summary['boxes_by_class']}"
        )


if __name__ == "__main__":
    main()
