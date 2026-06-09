#!/usr/bin/env python
"""Materialize reviewed synth+real calibration decisions into train-only lists."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_USABLE_AS = {
    "trusted_positive",
    "hard_negative",
    "unknown_out_of_scope",
    "exclude",
}
ACCEPTED_REVIEW_DECISIONS = {"accept", "accepted", "approved", "reviewed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-csv", required=True, type=Path)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty summary instead of failing when no reviewed decisions are materialized.",
    )
    parser.add_argument(
        "--require-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every materialized image path to come from data/cashsnap_v1/images/train.",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_image_key(image: str) -> str:
    stem = Path(image).stem
    stem = re.sub(r"\.rf\.[0-9a-fA-F]+$", "", stem)
    return stem


def cluster_key_for_row(row: dict[str, str]) -> str:
    existing = str(row.get("cluster_key", "")).strip()
    if existing:
        return existing
    source = str(row.get("source_group", "")).strip()
    image = str(row.get("image") or row.get("sample_image") or "").strip()
    return f"{source}|{canonical_image_key(image)}"


def normalized(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def reviewed_cluster_decisions(clusters: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    decisions: dict[str, dict[str, str]] = {}
    skipped: list[dict[str, Any]] = []
    for row in clusters:
        key = cluster_key_for_row(row)
        usable_as = normalized(str(row.get("usable_as", "")))
        review_decision = normalized(str(row.get("review_decision", "")))
        if not usable_as and not review_decision:
            continue
        if usable_as not in ALLOWED_USABLE_AS:
            skipped.append(
                {
                    "cluster_key": key,
                    "reason": "invalid_usable_as",
                    "usable_as": row.get("usable_as", ""),
                    "review_decision": row.get("review_decision", ""),
                }
            )
            continue
        if review_decision not in ACCEPTED_REVIEW_DECISIONS:
            skipped.append(
                {
                    "cluster_key": key,
                    "reason": "review_decision_not_accepted",
                    "usable_as": row.get("usable_as", ""),
                    "review_decision": row.get("review_decision", ""),
                }
            )
            continue
        if key in decisions:
            skipped.append({"cluster_key": key, "reason": "duplicate_cluster_decision"})
            continue
        decisions[key] = {
            "usable_as": usable_as,
            "review_decision": review_decision,
            "final_class_or_route": str(row.get("final_class_or_route", "")).strip(),
            "review_notes": str(row.get("review_notes", "")).strip(),
        }
    return decisions, skipped


def is_train_image(image: str) -> bool:
    normalized_path = image.replace("\\", "/")
    return "/images/train/" in f"/{normalized_path}"


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


def main() -> int:
    args = parse_args()
    clusters_csv = resolve(args.clusters_csv)
    rows_csv = resolve(args.rows_csv)
    out_dir = resolve(args.out_dir)

    clusters = read_csv(clusters_csv)
    rows = read_csv(rows_csv)
    decisions, skipped_decisions = reviewed_cluster_decisions(clusters)

    rows_by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_cluster[cluster_key_for_row(row)].append(row)

    materialized: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for key, decision in sorted(decisions.items()):
        matched_rows = rows_by_cluster.get(key, [])
        if not matched_rows:
            skipped_rows.append({"cluster_key": key, "reason": "no_matching_rows"})
            continue
        for row in matched_rows:
            image = str(row.get("image", "")).strip()
            if args.require_train and not is_train_image(image):
                skipped_rows.append({"cluster_key": key, "image": image, "reason": "not_train_split"})
                continue
            materialized.append(
                {
                    "cluster_key": key,
                    "image": image,
                    "label": str(row.get("label", "")).strip(),
                    "source_group": str(row.get("source_group", "")).strip(),
                    "usable_as": decision["usable_as"],
                    "review_decision": decision["review_decision"],
                    "final_class_or_route": decision["final_class_or_route"],
                    "review_notes": decision["review_notes"],
                    "row_review_rank": row.get("review_rank", ""),
                    "row_score": row.get("score", ""),
                    "row_suggested_action": row.get("suggested_action", ""),
                }
            )

    if not materialized and not args.allow_empty:
        raise SystemExit(
            "no reviewed train rows materialized; fill review_decision with one of "
            f"{sorted(ACCEPTED_REVIEW_DECISIONS)} and usable_as with one of {sorted(ALLOWED_USABLE_AS)}, "
            "or pass --allow-empty for a dry summary"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    fields = [
        "cluster_key",
        "image",
        "label",
        "source_group",
        "usable_as",
        "review_decision",
        "final_class_or_route",
        "review_notes",
        "row_review_rank",
        "row_score",
        "row_suggested_action",
    ]
    write_csv(manifest_path, materialized, fields)

    list_paths: dict[str, str] = {}
    for bucket in sorted(ALLOWED_USABLE_AS):
        images = [row["image"] for row in materialized if row["usable_as"] == bucket]
        path = out_dir / f"{bucket}_images.txt"
        write_list(path, images)
        list_paths[bucket] = repo_rel(path)

    summary = {
        "schema": "cashsnap_synth_real_calibration_review_materialization_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clusters_csv": repo_rel(clusters_csv),
        "rows_csv": repo_rel(rows_csv),
        "manifest_csv": repo_rel(manifest_path),
        "list_paths": list_paths,
        "reviewed_clusters": len(decisions),
        "materialized_rows": len(materialized),
        "materialized_unique_images": len({row["image"] for row in materialized}),
        "by_usable_as": dict(Counter(row["usable_as"] for row in materialized).most_common()),
        "by_source": dict(Counter(row["source_group"] for row in materialized).most_common()),
        "skipped_decisions": skipped_decisions,
        "skipped_rows": skipped_rows,
        "require_train": bool(args.require_train),
        "not_a_yolo_config": True,
        "note": (
            "This materializes reviewed train rows into explicit buckets only. "
            "unknown_out_of_scope is intentionally separate from hard_negative because schema scope is a product decision."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"materialized_review={repo_rel(out_dir)} rows={len(materialized)} "
        f"clusters={len(decisions)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
