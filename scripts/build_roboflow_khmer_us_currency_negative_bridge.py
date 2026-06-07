#!/usr/bin/env python
"""Materialize zero-label Roboflow bridge-negative diagnostics.

The positive bridge intentionally excludes unsupported denominations such as
KHR_100 from the core-13 eval. This script keeps those out-of-scope-only frames
as zero-label diagnostics so detectors can be probed for target-class
hallucinations without contaminating the positive bridge.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from cashsnap_currency_taxonomy import ROOT, class_names_for_scope, repo_path, resolve_repo_path
from build_roboflow_khmer_us_currency_bridge import (
    IMAGE_SUFFIXES,
    INPUT_SPLITS,
    OUTPUT_SPLIT,
    image_sha256,
    normalize_names,
    parse_label,
    read_yaml,
    safe_clean_out_root,
    source_metadata,
    split_dirs,
)


DEFAULT_OUT = ROOT / "data" / "processed" / "roboflow_khmer_us_currency_core13_bridge_negative_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", action="append", type=Path, default=[], help="Raw Roboflow YOLO export root.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scope", choices=["operational", "official"], default="operational")
    parser.add_argument("--clean", action="store_true", help="Delete an existing out-root before writing.")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep exact duplicate image bytes.")
    parser.add_argument(
        "--include-category",
        action="append",
        choices=["raw_empty", "unsupported_only"],
        default=[],
        help="Negative category to include. Defaults to both.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "runs" / "cashsnap" / "roboflow_khmer_us_currency_core13_bridge_negative_v1_summary.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return resolve_repo_path(path).resolve()


def write_data_yaml(out_root: Path, target_names: list[str], summary: dict[str, Any]) -> None:
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(target_names)},
        "cashsnap_bridge_negative": {
            "schema": "roboflow_khmer_us_currency_bridge_negative_v1",
            "class_scope": summary["class_scope"],
            "source_roots": summary["source_roots"],
            "categories": summary["included_categories"],
            "dedupe": summary["dedupe"],
            "label_policy": "zero-label diagnostics; out-of-scope-only frames are not target positives for this schema",
        },
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def copy_negative_image(image_path: Path, *, raw_root: Path, split: str, out_root: Path) -> None:
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
    out_label.write_text("", encoding="utf-8")


def category_for(label_lines: list[str], raw_counts: Counter[str], unsupported: list[str]) -> str | None:
    if label_lines:
        return None
    if not raw_counts:
        return "raw_empty"
    if unsupported:
        return "unsupported_only"
    return None


def build_negative_bridge(args: argparse.Namespace) -> dict[str, Any]:
    raw_roots = [resolve(path) for path in args.raw_root]
    if not raw_roots:
        raise SystemExit("--raw-root is required")
    out_root = resolve(args.out_root)
    if args.clean:
        safe_clean_out_root(out_root)
    elif out_root.exists():
        raise SystemExit(f"output root exists; pass --clean to replace: {repo_path(out_root)}")

    included = set(args.include_category or ["raw_empty", "unsupported_only"])
    target_names = class_names_for_scope(args.scope)
    class_to_id = {name: index for index, name in enumerate(target_names)}
    seen_hashes: set[str] = set()
    skipped = Counter()
    images_by_split = Counter()
    categories_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    raw_classes_by_category: dict[str, Counter[str]] = defaultdict(Counter)
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
                label_lines, raw_counts, _target_counts, _formats, unsupported = parse_label(
                    label_path,
                    names,
                    class_to_id,
                    unsupported_policy="drop_object",
                )
                category = category_for(label_lines, raw_counts, unsupported)
                if category is None:
                    skipped["has_target_label"] += 1
                    continue
                if category not in included:
                    skipped[f"excluded_{category}"] += 1
                    continue
                if not args.no_dedupe:
                    digest = image_sha256(image_path)
                    if digest in seen_hashes:
                        skipped["exact_duplicate_image"] += 1
                        continue
                    seen_hashes.add(digest)

                copy_negative_image(image_path, raw_root=raw_root, split=split, out_root=out_root)
                out_split = OUTPUT_SPLIT[split]
                images_by_split[out_split] += 1
                categories_by_split[out_split][category] += 1
                raw_classes_by_category[category].update(raw_counts)

    summary = {
        "schema": "roboflow_khmer_us_currency_bridge_negative_v1",
        "out_root": repo_path(out_root),
        "class_scope": args.scope,
        "target_class_names": target_names,
        "source_roots": source_roots,
        "sources": source_rows,
        "included_categories": sorted(included),
        "dedupe": not args.no_dedupe,
        "images_by_split": dict(sorted(images_by_split.items())),
        "categories_by_split": {
            split: dict(sorted(counter.items())) for split, counter in sorted(categories_by_split.items())
        },
        "raw_classes_by_category": {
            category: dict(sorted(counter.items())) for category, counter in sorted(raw_classes_by_category.items())
        },
        "skipped": dict(sorted(skipped.items())),
    }
    if not images_by_split:
        raise SystemExit("no bridge-negative images selected")
    out_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_root, target_names, summary)
    (out_root / "bridge_negative_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = build_negative_bridge(args)
    print(
        "roboflow_bridge_negative=ok "
        f"scope={summary['class_scope']} "
        f"images={sum(summary['images_by_split'].values())} "
        f"categories={summary['categories_by_split']} "
        f"skipped={summary['skipped']}"
    )
    print(f"out_root: {summary['out_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
