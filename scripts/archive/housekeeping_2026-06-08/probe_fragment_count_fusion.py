#!/usr/bin/env python
"""Probe synthetic fragment counts against physical-note count targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON summary output.")
    parser.add_argument("--per-image", action="store_true", help="Print per-image count rows.")
    parser.add_argument(
        "--proposal-min-fragment-overlap",
        type=float,
        default=0.50,
        help="Minimum fraction of a fragment box that must lie inside a proposal box for proposal fusion.",
    )
    parser.add_argument(
        "--require-parent-fused-all-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require oracle parent fusion over kept+ignored fragments to match physical counts.",
    )
    parser.add_argument(
        "--require-parent-fused-kept-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require oracle parent fusion over kept fragments to match physical counts.",
    )
    parser.add_argument(
        "--require-proposal-fused-kept-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require same-class proposal fusion over kept fragments to match physical counts.",
    )
    parser.add_argument(
        "--require-proposal-fused-all-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require same-class proposal fusion over kept+ignored fragments to match physical counts.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SystemExit(f"{label} must be a list")
    return value


def class_name(class_index: int) -> str:
    if 0 <= class_index < len(CLASS_NAMES):
        return CLASS_NAMES[class_index]
    return f"class_{class_index}"


def counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def count_block(counter: Counter[str]) -> dict[str, Any]:
    return {"total": int(sum(counter.values())), "by_class": counter_payload(counter)}


def counter_text(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter)) or "none"


def count_by_class(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("className", "unknown")) for row in rows)


def parent_fused_counts(fragment_rows: list[dict[str, Any]]) -> Counter[str]:
    parent_keys = {
        (int(row["parentVisibleIndex"]), str(row.get("className", "unknown")))
        for row in fragment_rows
    }
    return Counter(class_name for _parent_index, class_name in parent_keys)


def split_parent_count(fragment_rows: list[dict[str, Any]]) -> int:
    parent_fragment_counts: Counter[tuple[int, str]] = Counter(
        (int(row["parentVisibleIndex"]), str(row.get("className", "unknown")))
        for row in fragment_rows
    )
    return sum(1 for count in parent_fragment_counts.values() if count > 1)


def fragment_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in row["bbox_xywh_px"]]
    return x, y, x + width, y + height


def visible_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(row["minX"]), float(row["minY"]), float(row["maxX"]), float(row["maxY"])


def intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def proposal_fused_counts(
    visible_boxes: list[dict[str, Any]],
    fragment_rows: list[dict[str, Any]],
    min_fragment_overlap: float,
) -> tuple[Counter[str], int]:
    proposal_boxes = [visible_box(row) for row in visible_boxes]
    assigned_proposals: set[int] = set()
    unassigned = 0
    for fragment in fragment_rows:
        fragment_class = str(fragment.get("className", "unknown"))
        frag_box = fragment_box(fragment)
        frag_area = max(1.0, box_area(frag_box))
        fragment_assignments = 0
        for index, proposal in enumerate(visible_boxes):
            if str(proposal.get("className", "unknown")) != fragment_class:
                continue
            overlap = intersection_area(frag_box, proposal_boxes[index]) / frag_area
            if overlap >= min_fragment_overlap:
                assigned_proposals.add(index)
                fragment_assignments += 1
        if fragment_assignments == 0:
            unassigned += 1
    return Counter(str(visible_boxes[index].get("className", "unknown")) for index in assigned_proposals), unassigned


def load_fragment_rows(dataset_root: Path, row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = row.get(key, "")
    if not value:
        return []
    document = read_json(dataset_root / str(value))
    rows = require_list(document, str(value))
    return [require_mapping(item, f"{value} row") for item in rows]


def analyze_image(dataset_root: Path, row: dict[str, Any], min_fragment_overlap: float) -> dict[str, Any]:
    boxes_doc = require_mapping(read_json(dataset_root / str(row["visible_boxes"])), str(row["visible_boxes"]))
    visible_boxes = [require_mapping(item, "visible box") for item in require_list(boxes_doc.get("boxes", []), "visible boxes")]
    kept_fragments = load_fragment_rows(dataset_root, row, "fragment_metadata")
    ignored_fragments = load_fragment_rows(dataset_root, row, "fragment_ignored_metadata")
    all_fragments = [*kept_fragments, *ignored_fragments]

    physical = count_by_class(visible_boxes)
    kept = count_by_class(kept_fragments)
    all_fragment_counts = count_by_class(all_fragments)
    parent_kept = parent_fused_counts(kept_fragments)
    parent_all = parent_fused_counts(all_fragments)
    proposal_kept, unassigned_kept = proposal_fused_counts(visible_boxes, kept_fragments, min_fragment_overlap)
    proposal_all, unassigned_all = proposal_fused_counts(visible_boxes, all_fragments, min_fragment_overlap)
    return {
        "variant": row.get("variant"),
        "image": row.get("image"),
        "physical": physical,
        "kept_fragments": kept,
        "all_fragments": all_fragment_counts,
        "parent_fused_kept": parent_kept,
        "parent_fused_all": parent_all,
        "proposal_fused_kept": proposal_kept,
        "proposal_fused_all": proposal_all,
        "unassigned_kept_fragments": unassigned_kept,
        "unassigned_all_fragments": unassigned_all,
        "kept_split_parent_count": split_parent_count(kept_fragments),
        "all_split_parent_count": split_parent_count(all_fragments),
    }


def merge_counter(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update(row[key])
    return total


def summarize(rows: list[dict[str, Any]], root: Path, min_fragment_overlap: float) -> dict[str, Any]:
    physical = merge_counter(rows, "physical")
    kept = merge_counter(rows, "kept_fragments")
    all_fragments = merge_counter(rows, "all_fragments")
    parent_kept = merge_counter(rows, "parent_fused_kept")
    parent_all = merge_counter(rows, "parent_fused_all")
    proposal_kept = merge_counter(rows, "proposal_fused_kept")
    proposal_all = merge_counter(rows, "proposal_fused_all")
    physical_total = int(sum(physical.values()))
    return {
        "root": repo_path(root),
        "images": len(rows),
        "proposal_min_fragment_overlap": min_fragment_overlap,
        "counts": {
            "physical_visible_instances": count_block(physical),
            "kept_fragments": count_block(kept),
            "all_fragments": count_block(all_fragments),
            "parent_fused_kept": count_block(parent_kept),
            "parent_fused_all": count_block(parent_all),
            "proposal_fused_kept": count_block(proposal_kept),
            "proposal_fused_all": count_block(proposal_all),
        },
        "deltas": {
            "naive_kept_fragment_overcount": int(sum(kept.values()) - physical_total),
            "naive_all_fragment_overcount": int(sum(all_fragments.values()) - physical_total),
            "parent_fused_kept_delta": int(sum(parent_kept.values()) - physical_total),
            "parent_fused_all_delta": int(sum(parent_all.values()) - physical_total),
            "proposal_fused_kept_delta": int(sum(proposal_kept.values()) - physical_total),
            "proposal_fused_all_delta": int(sum(proposal_all.values()) - physical_total),
            "unassigned_kept_fragments": int(sum(row["unassigned_kept_fragments"] for row in rows)),
            "unassigned_all_fragments": int(sum(row["unassigned_all_fragments"] for row in rows)),
            "kept_split_parent_count": int(sum(row["kept_split_parent_count"] for row in rows)),
            "all_split_parent_count": int(sum(row["all_split_parent_count"] for row in rows)),
        },
        "matches": {
            "parent_fused_kept_matches_physical": parent_kept == physical,
            "parent_fused_all_matches_physical": parent_all == physical,
            "proposal_fused_kept_matches_physical": proposal_kept == physical,
            "proposal_fused_all_matches_physical": proposal_all == physical,
        },
    }


def json_ready_image(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": row["variant"],
        "image": row["image"],
        "physical": count_block(row["physical"]),
        "kept_fragments": count_block(row["kept_fragments"]),
        "all_fragments": count_block(row["all_fragments"]),
        "parent_fused_kept": count_block(row["parent_fused_kept"]),
        "parent_fused_all": count_block(row["parent_fused_all"]),
        "proposal_fused_kept": count_block(row["proposal_fused_kept"]),
        "proposal_fused_all": count_block(row["proposal_fused_all"]),
        "unassigned_kept_fragments": row["unassigned_kept_fragments"],
        "unassigned_all_fragments": row["unassigned_all_fragments"],
        "kept_split_parent_count": row["kept_split_parent_count"],
        "all_split_parent_count": row["all_split_parent_count"],
    }


def require_match(enabled: bool, matches: dict[str, bool], key: str) -> None:
    if enabled and not matches[key]:
        raise SystemExit(f"{key} is false")


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    manifest_rows = [require_mapping(row, "manifest row") for row in require_list(manifest, "manifest.json")]
    if not manifest_rows:
        raise SystemExit("manifest.json must be a non-empty list")
    image_rows = [
        analyze_image(dataset_root, row, args.proposal_min_fragment_overlap)
        for row in manifest_rows
    ]
    summary = summarize(image_rows, dataset_root, args.proposal_min_fragment_overlap)
    require_match(args.require_parent_fused_kept_match, summary["matches"], "parent_fused_kept_matches_physical")
    require_match(args.require_parent_fused_all_match, summary["matches"], "parent_fused_all_matches_physical")
    require_match(args.require_proposal_fused_kept_match, summary["matches"], "proposal_fused_kept_matches_physical")
    require_match(args.require_proposal_fused_all_match, summary["matches"], "proposal_fused_all_matches_physical")

    if args.per_image:
        for row in image_rows:
            print(
                f"{row['image']}: physical={sum(row['physical'].values())} "
                f"kept_fragments={sum(row['kept_fragments'].values())} "
                f"all_fragments={sum(row['all_fragments'].values())} "
                f"proposal_kept={sum(row['proposal_fused_kept'].values())} "
                f"proposal_all={sum(row['proposal_fused_all'].values())} "
                f"physical_by_class=({counter_text(row['physical'])})"
            )

    counts = summary["counts"]
    deltas = summary["deltas"]
    matches = summary["matches"]
    print(f"images: {summary['images']}")
    print(
        "physical visible instances: "
        f"{counts['physical_visible_instances']['total']} "
        f"({counter_text(Counter(counts['physical_visible_instances']['by_class']))})"
    )
    print(f"kept fragments: {counts['kept_fragments']['total']} ({counter_text(Counter(counts['kept_fragments']['by_class']))})")
    print(f"all fragments: {counts['all_fragments']['total']} ({counter_text(Counter(counts['all_fragments']['by_class']))})")
    print(f"naive kept fragment overcount: {deltas['naive_kept_fragment_overcount']}")
    print(f"naive all fragment overcount: {deltas['naive_all_fragment_overcount']}")
    print(
        "parent fused: "
        f"kept={counts['parent_fused_kept']['total']} match={matches['parent_fused_kept_matches_physical']} "
        f"all={counts['parent_fused_all']['total']} match={matches['parent_fused_all_matches_physical']}"
    )
    print(
        "proposal fused: "
        f"kept={counts['proposal_fused_kept']['total']} match={matches['proposal_fused_kept_matches_physical']} "
        f"all={counts['proposal_fused_all']['total']} match={matches['proposal_fused_all_matches_physical']} "
        f"unassigned_kept={deltas['unassigned_kept_fragments']} unassigned_all={deltas['unassigned_all_fragments']}"
    )

    if args.json_out is not None:
        out_path = resolve_path(args.json_out)
        payload = dict(summary)
        payload["per_image"] = [json_ready_image(row) for row in image_rows]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
