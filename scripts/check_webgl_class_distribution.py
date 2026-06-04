#!/usr/bin/env python
"""Gate packaged WebGL class distributions for targeted dose audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COUNT_BLOCKS = (
    "physical_visible_instances",
    "parent_fused_all_fragments",
    "parent_fused_kept_fragments",
    "kept_fragments",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Packaged WebGL root, or a counts/summary.json file.",
    )
    parser.add_argument(
        "--expected-classes",
        required=True,
        help="Comma-separated class names expected in the selected count block.",
    )
    parser.add_argument(
        "--count-block",
        choices=COUNT_BLOCKS,
        default="physical_visible_instances",
        help="Count block to gate. Physical instances are the count truth.",
    )
    parser.add_argument("--min-images", type=int, default=1, help="Minimum packaged image count.")
    parser.add_argument("--min-total", type=int, default=1, help="Minimum selected count-block total.")
    parser.add_argument("--min-per-class", type=int, default=1, help="Minimum count for every expected class.")
    parser.add_argument(
        "--max-class-spread",
        type=int,
        help="Maximum allowed max-min spread across expected class counts.",
    )
    parser.add_argument(
        "--max-class-ratio",
        type=float,
        help="Maximum allowed max/min ratio across expected class counts.",
    )
    parser.add_argument(
        "--allow-extra-classes",
        action="store_true",
        help="Allow positive counts for classes outside --expected-classes.",
    )
    parser.add_argument(
        "--require-parent-fusion-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require parent_fused_all_fragments to match physical_visible_instances.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def summary_path_for(root_arg: Path) -> Path:
    root = resolve_path(root_arg)
    if root.is_file():
        return root
    return root / "counts" / "summary.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing count summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def parse_expected_classes(raw_value: str) -> list[str]:
    classes = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not classes:
        raise SystemExit("--expected-classes must contain at least one class")
    duplicates = sorted({name for name in classes if classes.count(name) > 1})
    if duplicates:
        raise SystemExit(f"--expected-classes contains duplicates: {', '.join(duplicates)}")
    return classes


def count_block(summary: dict[str, Any], block_name: str) -> tuple[int, dict[str, int]]:
    block = summary.get(block_name)
    if not isinstance(block, dict):
        raise SystemExit(f"counts summary missing {block_name!r} block")
    raw_by_class = block.get("by_class")
    if not isinstance(raw_by_class, dict):
        raise SystemExit(f"{block_name}.by_class must be an object")
    by_class: dict[str, int] = {}
    for class_name, raw_count in raw_by_class.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{block_name}.by_class[{class_name!r}] must be an integer") from exc
        if count < 0:
            raise SystemExit(f"{block_name}.by_class[{class_name!r}] must not be negative")
        by_class[str(class_name)] = count
    total = int(block.get("total", -1))
    if total != sum(by_class.values()):
        raise SystemExit(f"{block_name}.total={total} does not match by_class sum={sum(by_class.values())}")
    return total, by_class


def fail_if_errors(errors: list[str]) -> None:
    if errors:
        detail = "\n  - ".join(errors)
        raise SystemExit(f"class distribution audit failed:\n  - {detail}")


def main() -> int:
    args = parse_args()
    summary_path = summary_path_for(args.root)
    summary = read_json(summary_path)
    expected_classes = parse_expected_classes(args.expected_classes)

    errors: list[str] = []
    images = int(summary.get("images", 0))
    if images < args.min_images:
        errors.append(f"expected at least {args.min_images} images, got {images}")

    if args.require_parent_fusion_match and not summary.get("parent_fused_all_matches_physical", False):
        errors.append("parent_fused_all_matches_physical must be true")

    total, by_class = count_block(summary, args.count_block)
    if total < args.min_total:
        errors.append(f"expected at least {args.min_total} {args.count_block} targets, got {total}")

    positive_extra = sorted(
        class_name for class_name, count in by_class.items() if count > 0 and class_name not in expected_classes
    )
    if positive_extra and not args.allow_extra_classes:
        errors.append(f"unexpected positive classes: {', '.join(positive_extra)}")

    expected_counts = [by_class.get(class_name, 0) for class_name in expected_classes]
    below_minimum = [
        f"{class_name}={count}"
        for class_name, count in zip(expected_classes, expected_counts, strict=True)
        if count < args.min_per_class
    ]
    if below_minimum:
        errors.append(f"expected every class >= {args.min_per_class}: {', '.join(below_minimum)}")

    spread = max(expected_counts) - min(expected_counts)
    if args.max_class_spread is not None and spread > args.max_class_spread:
        errors.append(f"class spread {spread} exceeds max {args.max_class_spread}")

    min_count = min(expected_counts)
    max_count = max(expected_counts)
    ratio = float("inf") if min_count == 0 else max_count / min_count
    if args.max_class_ratio is not None and ratio > args.max_class_ratio:
        errors.append(f"class ratio {ratio:.3f} exceeds max {args.max_class_ratio:.3f}")

    fail_if_errors(errors)

    class_report = ", ".join(f"{name}:{by_class.get(name, 0)}" for name in expected_classes)
    print(
        "ok: "
        f"{display_path(summary_path)} images={images}, "
        f"{args.count_block}={total}, classes={class_report}, "
        f"spread={spread}, ratio={ratio:.3f}, "
        f"parent_fused_all_matches_physical={bool(summary.get('parent_fused_all_matches_physical', False))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
