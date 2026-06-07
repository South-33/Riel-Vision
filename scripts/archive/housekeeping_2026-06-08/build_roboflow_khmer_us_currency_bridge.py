#!/usr/bin/env python
"""Materialize a schema-aware Roboflow Khmer/US currency YOLO bridge dataset.

The Roboflow project has multiple useful exports with different class coverage.
This script keeps the raw exports as intake, converts labels into a CashSnap
taxonomy scope, and writes an ignored processed dataset for model probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from cashsnap_currency_taxonomy import ROOT, class_names_for_scope, repo_path, resolve_repo_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
INPUT_SPLITS = ("train", "valid", "test")
OUTPUT_SPLIT = {"train": "train", "valid": "val", "test": "test"}
RAW_CLASS_ALIASES = {
    "1-us": "USD_1",
    "5-us": "USD_5",
    "10-us": "USD_10",
    "20-us": "USD_20",
    "50-us": "USD_50",
    "100-us": "USD_100",
    "50-riel": "KHR_50",
    "100-riel": "KHR_100",
    "200-riel": "KHR_200",
    "500-riel": "KHR_500",
    "1000-riel": "KHR_1000",
    "2000-riel": "KHR_2000",
    "5000-riel": "KHR_5000",
    "10000-riel": "KHR_10000",
    "15000-riel": "KHR_15000",
    "20000-riel": "KHR_20000",
    "30000-riel": "KHR_30000",
    "50000-riel": "KHR_50000",
    "100000-riel": "KHR_100000",
    "200000-riel": "KHR_200000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        action="append",
        type=Path,
        default=[],
        help="Raw Roboflow YOLO export root. Repeat to build a deduped union.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "processed" / "roboflow_khmer_us_currency_core13_bridge_v1",
    )
    parser.add_argument("--scope", choices=["operational", "official"], default="operational")
    parser.add_argument(
        "--unsupported-policy",
        choices=["exclude_image", "drop_object"],
        default="exclude_image",
        help="How to handle labels outside the selected scope, e.g. KHR_100 for operational.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete an existing out-root before writing.")
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep exact duplicate image bytes across source exports.",
    )
    parser.add_argument(
        "--require-all-target-classes",
        action="store_true",
        help="Exit non-zero when any class in the selected scope has no output objects.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "runs" / "cashsnap" / "roboflow_khmer_us_currency_core13_bridge_v1_summary.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return resolve_repo_path(path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"malformed YAML: {repo_path(path)}")
    return data


def normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise SystemExit("data.yaml names must be a list or mapping")


def raw_class_to_canonical(raw_class: str) -> str | None:
    normalized = raw_class.strip().lower().replace("_", "-")
    return RAW_CLASS_ALIASES.get(normalized)


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_clean_out_root(out_root: Path) -> None:
    data_processed = (ROOT / "data" / "processed").resolve()
    resolved = out_root.resolve()
    try:
        resolved.relative_to(data_processed)
    except ValueError as exc:
        raise SystemExit(f"refusing to clean outside data/processed: {repo_path(resolved)}") from exc
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_label(
    label_path: Path,
    names: dict[int, str],
    class_to_id: dict[str, int],
    *,
    unsupported_policy: str,
) -> tuple[list[str], Counter[str], Counter[str], Counter[str], list[str]]:
    converted: list[str] = []
    raw_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    format_counter: Counter[str] = Counter()
    unsupported: list[str] = []

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split()
        try:
            raw_class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise SystemExit(f"{repo_path(label_path)}:{line_number}: non-numeric label") from exc
        if raw_class_id not in names:
            raise SystemExit(f"{repo_path(label_path)}:{line_number}: class id {raw_class_id} missing from names")

        if len(coords) == 4:
            x, y, w, h = coords
            format_counter["bbox"] += 1
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]
            ys = coords[1::2]
            if any(value < 0.0 or value > 1.0 for value in coords):
                raise SystemExit(f"{repo_path(label_path)}:{line_number}: polygon outside normalized YOLO range")
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            format_counter["polygon_to_bbox"] += 1
        else:
            raise SystemExit(f"{repo_path(label_path)}:{line_number}: unsupported YOLO label field count")

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise SystemExit(f"{repo_path(label_path)}:{line_number}: bbox outside normalized YOLO range")

        raw_name = names[raw_class_id]
        raw_counter[raw_name] += 1
        canonical = raw_class_to_canonical(raw_name)
        if canonical is None or canonical not in class_to_id:
            unsupported.append(canonical or raw_name)
            if unsupported_policy == "drop_object":
                continue
            continue
        target_id = class_to_id[canonical]
        converted.append(f"{target_id} {x:.8g} {y:.8g} {w:.8g} {h:.8g}")
        target_counter[canonical] += 1

    return converted, raw_counter, target_counter, format_counter, unsupported


def split_dirs(raw_root: Path, split: str) -> tuple[Path, Path]:
    image_dir = raw_root / split / "images"
    label_dir = raw_root / split / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise SystemExit(f"missing {split} images/labels under {repo_path(raw_root)}")
    return image_dir, label_dir


def copy_bridge_image(
    image_path: Path,
    label_lines: list[str],
    *,
    raw_root: Path,
    split: str,
    out_root: Path,
) -> tuple[Path, Path]:
    out_split = OUTPUT_SPLIT[split]
    source_id = raw_root.name
    out_image_dir = out_root / "images" / out_split
    out_label_dir = out_root / "labels" / out_split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{source_id}__{image_path.name}"
    out_image = out_image_dir / out_name
    out_label = out_label_dir / f"{Path(out_name).stem}.txt"
    shutil.copy2(image_path, out_image)
    out_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return out_image, out_label


def source_metadata(raw_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    roboflow = config.get("roboflow", {})
    if not isinstance(roboflow, dict):
        roboflow = {}
    return {
        "root": repo_path(raw_root),
        "project": roboflow.get("project"),
        "workspace": roboflow.get("workspace"),
        "version": roboflow.get("version"),
        "url": roboflow.get("url"),
        "license": roboflow.get("license"),
        "names": list(normalize_names(config.get("names", {})).values()),
    }


def write_data_yaml(out_root: Path, target_names: list[str], summary: dict[str, Any]) -> None:
    names = {index: name for index, name in enumerate(target_names)}
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "cashsnap_bridge": {
            "schema": "roboflow_khmer_us_currency_bridge_v1",
            "class_scope": summary["class_scope"],
            "source_roots": summary["source_roots"],
            "unsupported_policy": summary["unsupported_policy"],
            "dedupe": summary["dedupe"],
        },
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_bridge(args: argparse.Namespace) -> dict[str, Any]:
    raw_roots = [resolve(path) for path in args.raw_root]
    if not raw_roots:
        raise SystemExit("--raw-root is required")
    out_root = resolve(args.out_root)
    if args.clean:
        safe_clean_out_root(out_root)
    elif out_root.exists():
        raise SystemExit(f"output root exists; pass --clean to replace: {repo_path(out_root)}")

    target_names = class_names_for_scope(args.scope)
    class_to_id = {name: index for index, name in enumerate(target_names)}
    seen_hashes: set[str] = set()
    skipped = Counter()
    raw_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    label_formats: Counter[str] = Counter()
    target_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    images_by_split: Counter[str] = Counter()
    source_rows: list[dict[str, Any]] = []
    source_roots: list[str] = []

    for raw_root in raw_roots:
        data_yaml = raw_root / "data.yaml"
        if not data_yaml.exists():
            raise SystemExit(f"missing data.yaml: {repo_path(raw_root)}")
        config = read_yaml(data_yaml)
        names = normalize_names(config.get("names", {}))
        source_rows.append(source_metadata(raw_root, config))
        source_roots.append(repo_path(raw_root))

        for split in INPUT_SPLITS:
            image_dir, label_dir = split_dirs(raw_root, split)
            for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    skipped["missing_label"] += 1
                    continue
                label_lines, image_raw_counts, image_target_counts, image_label_formats, unsupported = parse_label(
                    label_path,
                    names,
                    class_to_id,
                    unsupported_policy=args.unsupported_policy,
                )
                raw_counts.update(image_raw_counts)
                label_formats.update(image_label_formats)
                if unsupported and args.unsupported_policy == "exclude_image":
                    skipped["unsupported_image"] += 1
                    continue
                if not label_lines:
                    skipped["empty_after_mapping"] += 1
                    continue
                if not args.no_dedupe:
                    digest = image_sha256(image_path)
                    if digest in seen_hashes:
                        skipped["exact_duplicate_image"] += 1
                        continue
                    seen_hashes.add(digest)

                copy_bridge_image(image_path, label_lines, raw_root=raw_root, split=split, out_root=out_root)
                out_split = OUTPUT_SPLIT[split]
                images_by_split[out_split] += 1
                target_counts.update(image_target_counts)
                target_counts_by_split[out_split].update(image_target_counts)

    missing_target = [name for name in target_names if target_counts.get(name, 0) == 0]
    summary = {
        "schema": "roboflow_khmer_us_currency_bridge_v1",
        "out_root": repo_path(out_root),
        "class_scope": args.scope,
        "target_class_names": target_names,
        "source_roots": source_roots,
        "sources": source_rows,
        "unsupported_policy": args.unsupported_policy,
        "dedupe": not args.no_dedupe,
        "images_by_split": dict(sorted(images_by_split.items())),
        "objects_by_raw_class": dict(sorted(raw_counts.items())),
        "objects_by_target_class": dict(sorted(target_counts.items())),
        "objects_by_target_class_by_split": {
            split: dict(sorted(counter.items())) for split, counter in sorted(target_counts_by_split.items())
        },
        "label_formats": dict(sorted(label_formats.items())),
        "missing_target_class_names": missing_target,
        "skipped": dict(sorted(skipped.items())),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_root, target_names, summary)
    (out_root / "bridge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json = resolve(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.require_all_target_classes and missing_target:
        missing = ", ".join(missing_target)
        raise SystemExit(f"missing target classes after bridge materialization: {missing}")
    return summary


def main() -> int:
    args = parse_args()
    summary = build_bridge(args)
    print(
        "roboflow_bridge=ok "
        f"scope={summary['class_scope']} "
        f"images={sum(summary['images_by_split'].values())} "
        f"missing={len(summary['missing_target_class_names'])} "
        f"skipped={summary['skipped']}"
    )
    if summary["missing_target_class_names"]:
        print("missing_target_class_names: " + ", ".join(summary["missing_target_class_names"]))
    print(f"out_root: {summary['out_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
