#!/usr/bin/env python
"""Remap a core13 YOLO eval root into the core14 + KHR_100 schema."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CORE13_TO_CORE14 = {old: (old if old <= 5 else old + 1) for old in range(13)}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--out-config", required=True, type=Path)
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
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return payload


def split_root(config_path: Path, config: dict[str, Any], split: str) -> Path:
    value = Path(str(config[split]))
    if value.is_absolute():
        return value
    root = Path(str(config.get("path", ".")))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root / value


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_split(images_root: Path, out_root: Path, split: str) -> dict[str, Any]:
    image_dir = out_root / "images" / split
    label_dir = out_root / "labels" / split
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in images_root.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    counts: Counter[str] = Counter()
    for image in images:
        source_label = label_path_for_image(image)
        if not source_label.exists():
            raise FileNotFoundError(f"Missing label for {repo_rel(image)}")
        out_image = image_dir / image.name
        out_label = label_dir / f"{image.stem}.txt"
        hardlink_or_copy(image, out_image)
        out_lines: list[str] = []
        for line_no, raw_line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            old_class = int(parts[0])
            new_class = CORE13_TO_CORE14.get(old_class)
            if new_class is None:
                raise SystemExit(f"{repo_rel(source_label)}:{line_no}: class {old_class} cannot map to core14")
            out_lines.append(" ".join([str(new_class), *parts[1:]]))
            counts[TARGET_NAMES[new_class]] += 1
        out_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return {"images": len(images), "boxes_by_class": dict(sorted(counts.items()))}


def main() -> None:
    args = parse_args()
    source_data = resolve(args.source_data)
    out_root = resolve(args.out_root)
    out_config = resolve(args.out_config)
    source_config = load_yaml(source_data)

    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        splits[split] = materialize_split(split_root(source_data, source_config, split), out_root, split)

    config = {
        "path": "../..",
        "train": f"{repo_rel(out_root)}/images/train",
        "val": f"{repo_rel(out_root)}/images/val",
        "test": f"{repo_rel(out_root)}/images/test",
        "names": TARGET_NAMES,
        "cashsnap_policy": {
            "intended_use": "core13 eval remapped for the promoted core14 + KHR_100 detector",
            "note": "KHR_100 is absent from this source eval; old KHR ids are shifted by one.",
        },
    }
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")

    summary = {
        "schema": "cashsnap_core13_eval_as_core14_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data": repo_rel(source_data),
        "out_root": repo_rel(out_root),
        "out_config": repo_rel(out_config),
        "splits": splits,
    }
    summary_path = resolve(args.summary_json) if args.summary_json else out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_config={repo_rel(out_config)} summary={repo_rel(summary_path)}")


if __name__ == "__main__":
    main()
