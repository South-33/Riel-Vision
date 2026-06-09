#!/usr/bin/env python
"""Materialize reviewed real overlap/fan cluster decisions into image lists."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATA = Path("data/cashsnap_v1/data.yaml")
DEFAULT_REVIEW_CSV = Path(
    "runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1.csv"
)
DEFAULT_OUT_DIR = Path("runs/cashsnap/real_overlap_review_materialized_v1")

ALLOWED_USABLE_AS = {
    "trusted_overlap_eval",
    "train_anchor_candidate",
    "hard_negative_context",
    "partial_policy_unclear",
    "unknown_or_foreign",
    "exclude_duplicate_or_flat",
    "exclude",
}
ACCEPTED_REVIEW_DECISIONS = {"accept", "accepted", "approved", "reviewed", "use", "usable"}
TRAIN_BUCKETS = {"train_anchor_candidate", "hard_negative_context"}
HELDOUT_EVAL_BUCKETS = {"trusted_overlap_eval"}
HELDOUT_SPLITS = {"val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--variant-policy",
        choices=["route_default", "representative", "all_variants"],
        default="route_default",
        help=(
            "route_default uses all train variants for train buckets and representative images "
            "for all other buckets."
        ),
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty manifests/lists instead of failing when no reviewed rows are materialized.",
    )
    parser.add_argument(
        "--require-train-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require train_anchor_candidate and hard_negative_context images to come from images/train.",
    )
    parser.add_argument(
        "--require-heldout-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require trusted_overlap_eval representative images to come from images/val or images/test.",
    )
    parser.add_argument(
        "--write-data-yamls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write diagnostic YOLO data YAML views for active eval and train-anchor lists.",
    )
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


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def image_split(image: str) -> str:
    parts = image.replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        if part == "images" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def split_variant_images(raw: str, fallback: str) -> list[str]:
    values = [value.strip() for value in raw.split("|") if value.strip()]
    if not values and fallback.strip():
        values = [fallback.strip()]
    return list(dict.fromkeys(values))


def cluster_key(row: dict[str, str]) -> str:
    existing = str(row.get("canonical_key", "")).strip()
    if existing:
        return existing
    source = str(row.get("source_group", "")).strip()
    image = Path(str(row.get("image", "")).strip()).stem
    return f"{source}|{image}"


def bucket_for_row(row: dict[str, str]) -> tuple[str | None, dict[str, Any] | None]:
    usable_as = normalized(str(row.get("usable_as", "")))
    final_route = normalized(str(row.get("final_route", "")))
    review_decision = normalized(str(row.get("review_decision", "")))
    key = cluster_key(row)

    if not usable_as and not final_route and not review_decision:
        return None, None
    if usable_as and usable_as not in ALLOWED_USABLE_AS:
        return None, {
            "cluster_key": key,
            "reason": "invalid_usable_as",
            "usable_as": row.get("usable_as", ""),
            "review_decision": row.get("review_decision", ""),
            "final_route": row.get("final_route", ""),
        }
    if final_route in ALLOWED_USABLE_AS and usable_as and final_route != usable_as:
        return None, {
            "cluster_key": key,
            "reason": "conflicting_usable_as_and_final_route",
            "usable_as": row.get("usable_as", ""),
            "final_route": row.get("final_route", ""),
            "review_decision": row.get("review_decision", ""),
        }
    bucket = usable_as or (final_route if final_route in ALLOWED_USABLE_AS else "")
    if not bucket:
        return None, {
            "cluster_key": key,
            "reason": "missing_usable_as",
            "usable_as": row.get("usable_as", ""),
            "review_decision": row.get("review_decision", ""),
            "final_route": row.get("final_route", ""),
        }
    if review_decision not in ACCEPTED_REVIEW_DECISIONS:
        return None, {
            "cluster_key": key,
            "reason": "review_decision_not_accepted",
            "usable_as": row.get("usable_as", ""),
            "review_decision": row.get("review_decision", ""),
            "final_route": row.get("final_route", ""),
        }
    return bucket, None


def selected_images_for_bucket(row: dict[str, str], bucket: str, variant_policy: str) -> list[tuple[str, str]]:
    representative = str(row.get("image", "")).strip()
    variants = split_variant_images(str(row.get("variant_images", "")), representative)
    if variant_policy == "representative":
        return [(representative, "representative")] if representative else []
    if variant_policy == "all_variants":
        return [(image, "representative" if image == representative else "variant") for image in variants]
    if bucket in TRAIN_BUCKETS:
        return [
            (image, "representative" if image == representative else "train_variant")
            for image in variants
            if image_split(image) == "train"
        ]
    return [(representative, "representative")] if representative else []


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_list(path: Path, images: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(images))
    path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")


def write_view_data_yaml(
    path: Path,
    *,
    base_data_path: Path,
    base_config: dict[str, Any],
    split_list: Path,
    split_name: str,
    purpose: str,
) -> None:
    payload = {
        "path": rel_between(path.parent, ROOT),
        "train": repo_rel(split_list),
        "val": repo_rel(split_list),
        "test": repo_rel(split_list),
        "names": base_config.get("names", {}),
        "cashsnap_diagnostic": {
            "purpose": purpose,
            "source_data": repo_rel(base_data_path),
            "source_split_list": repo_rel(split_list),
            "active_split": split_name,
            "not_a_promotion_config": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_data_path = resolve(args.base_data)
    base_config = read_yaml(base_data_path)
    review_csv = resolve(args.review_csv)
    out_dir = resolve(args.out_dir)
    rows = read_csv(review_csv)

    decisions: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    skipped_decisions: list[dict[str, Any]] = []
    skipped_materialization: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for row in rows:
        key = cluster_key(row)
        bucket, skip = bucket_for_row(row)
        if skip:
            skipped_decisions.append(skip)
            continue
        if not bucket:
            continue
        if key in seen_keys:
            skipped_decisions.append({"cluster_key": key, "reason": "duplicate_reviewed_cluster"})
            continue
        seen_keys.add(key)

        decision = {
            "cluster_key": key,
            "usable_as": bucket,
            "review_decision": normalized(str(row.get("review_decision", ""))),
            "final_route": str(row.get("final_route", "")).strip(),
            "review_notes": str(row.get("review_notes", "")).strip(),
            "packet_bucket": str(row.get("packet_bucket", "")).strip(),
            "split": str(row.get("split", "")).strip(),
            "source_group": str(row.get("source_group", "")).strip(),
            "representative_image": str(row.get("image", "")).strip(),
            "variant_count": str(row.get("variant_count", "")).strip(),
            "tags": str(row.get("tags", "")).strip(),
            "classes": str(row.get("classes", "")).strip(),
            "boxes": str(row.get("boxes", "")).strip(),
        }
        decisions.append(decision)

        if args.require_heldout_eval and bucket in HELDOUT_EVAL_BUCKETS:
            rep_split = image_split(decision["representative_image"]) or decision["split"]
            if rep_split not in HELDOUT_SPLITS:
                skipped_materialization.append(
                    {
                        "cluster_key": key,
                        "usable_as": bucket,
                        "image": decision["representative_image"],
                        "reason": "trusted_eval_not_heldout",
                    }
                )
                continue

        selected = selected_images_for_bucket(row, bucket, args.variant_policy)
        if args.require_train_anchors and bucket in TRAIN_BUCKETS:
            selected = [(image, role) for image, role in selected if image_split(image) == "train"]
            if not selected:
                skipped_materialization.append(
                    {"cluster_key": key, "usable_as": bucket, "reason": "no_train_images_for_train_bucket"}
                )
                continue

        for image, role in selected:
            materialized.append(
                {
                    **decision,
                    "materialized_image": image,
                    "materialized_split": image_split(image),
                    "image_role": role,
                }
            )

    if not materialized and not args.allow_empty:
        raise SystemExit(
            "no reviewed overlap rows materialized; fill review_decision with one of "
            f"{sorted(ACCEPTED_REVIEW_DECISIONS)} and usable_as/final_route with one of "
            f"{sorted(ALLOWED_USABLE_AS)}, or pass --allow-empty for a dry summary"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    reviewed_clusters_path = out_dir / "reviewed_clusters.csv"
    fields = [
        "cluster_key",
        "usable_as",
        "review_decision",
        "final_route",
        "review_notes",
        "packet_bucket",
        "split",
        "source_group",
        "representative_image",
        "materialized_image",
        "materialized_split",
        "image_role",
        "variant_count",
        "tags",
        "classes",
        "boxes",
    ]
    write_csv(manifest_path, materialized, fields)
    reviewed_cluster_fields = [
        field for field in fields if field not in {"materialized_image", "materialized_split", "image_role"}
    ]
    write_csv(reviewed_clusters_path, decisions, reviewed_cluster_fields)

    list_paths: dict[str, str] = {}
    for bucket in sorted(ALLOWED_USABLE_AS):
        path = out_dir / f"{bucket}_images.txt"
        write_list(path, [row["materialized_image"] for row in materialized if row["usable_as"] == bucket])
        list_paths[bucket] = repo_rel(path)

    active_eval_path = out_dir / "active_eval_images.txt"
    write_list(
        active_eval_path,
        [row["materialized_image"] for row in materialized if row["usable_as"] in HELDOUT_EVAL_BUCKETS],
    )
    active_train_path = out_dir / "active_train_anchor_images.txt"
    write_list(
        active_train_path,
        [row["materialized_image"] for row in materialized if row["usable_as"] in TRAIN_BUCKETS],
    )
    data_yaml_paths: dict[str, str] = {}
    if args.write_data_yamls:
        eval_data_yaml = out_dir / "active_eval_data.yaml"
        write_view_data_yaml(
            eval_data_yaml,
            base_data_path=base_data_path,
            base_config=base_config,
            split_list=active_eval_path,
            split_name="test",
            purpose="Reviewed held-out overlap/fan evaluation view; use with --split test.",
        )
        train_anchor_data_yaml = out_dir / "active_train_anchor_view_data.yaml"
        write_view_data_yaml(
            train_anchor_data_yaml,
            base_data_path=base_data_path,
            base_config=base_config,
            split_list=active_train_path,
            split_name="train",
            purpose="Reviewed train-anchor overlap/fan view for inspection or future controlled mixing.",
        )
        data_yaml_paths = {
            "active_eval_data": repo_rel(eval_data_yaml),
            "active_train_anchor_view_data": repo_rel(train_anchor_data_yaml),
        }

    summary = {
        "schema": "cashsnap_real_overlap_review_materialization_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_data": repo_rel(base_data_path),
        "review_csv": repo_rel(review_csv),
        "out_dir": repo_rel(out_dir),
        "manifest_csv": repo_rel(manifest_path),
        "reviewed_clusters_csv": repo_rel(reviewed_clusters_path),
        "list_paths": list_paths,
        "active_eval_images": repo_rel(active_eval_path),
        "active_train_anchor_images": repo_rel(active_train_path),
        "data_yamls": data_yaml_paths,
        "allowed_usable_as": sorted(ALLOWED_USABLE_AS),
        "accepted_review_decisions": sorted(ACCEPTED_REVIEW_DECISIONS),
        "variant_policy": args.variant_policy,
        "require_train_anchors": bool(args.require_train_anchors),
        "require_heldout_eval": bool(args.require_heldout_eval),
        "input_rows": len(rows),
        "reviewed_clusters": len(decisions),
        "materialized_rows": len(materialized),
        "materialized_unique_images": len({row["materialized_image"] for row in materialized}),
        "by_usable_as": dict(Counter(row["usable_as"] for row in materialized).most_common()),
        "by_source": dict(Counter(row["source_group"] for row in materialized).most_common()),
        "by_split": dict(Counter(row["materialized_split"] for row in materialized).most_common()),
        "skipped_decisions": skipped_decisions,
        "skipped_materialization": skipped_materialization,
        "not_a_yolo_config": True,
        "note": (
            "This is a strict bridge from reviewed real overlap clusters to explicit image lists. "
            "Blank review rows are ignored, and train anchors stay separate from held-out overlap evaluation candidates."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"materialized_overlap_review={repo_rel(out_dir)} rows={len(materialized)} "
        f"clusters={len(decisions)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
