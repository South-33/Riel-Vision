#!/usr/bin/env python
"""Validate the external negative-image review registry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_external_negative_banks_v1.json"
VALID_REVIEW_STATUSES = {"planned", "pending_review", "proof_only", "accepted", "rejected"}
VALID_ARTIFACT_STATUSES = {"smoke", "diagnostic", "trainable-candidate", "promoted"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_CURRENCY_STATUSES = {"unreviewed", "ambiguous_review_required", "verified_absent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--require-path", type=Path, default=None, help="Require this external-negative directory to be allowed.")
    parser.add_argument(
        "--artifact-status",
        choices=sorted(VALID_ARTIFACT_STATUSES),
        default="",
        help="Artifact status that wants to use --require-path.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = resolve(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"path must stay inside repository: {resolved}") from exc


def read_json(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing external negative bank config: {resolved}")
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{resolved}: expected JSON object")
    return document


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def string_list(value: object, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    return [str(item).strip() for item in value]


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def relative_asset_path(bank_id: str, bank_path: Path, file_text: str) -> Path:
    file_path = Path(file_text)
    require(not file_path.is_absolute(), f"{bank_id}: asset file must be relative: {file_text}")
    local_path = (bank_path / file_path).resolve()
    try:
        local_path.relative_to(bank_path)
    except ValueError as exc:
        raise SystemExit(f"{bank_id}: asset file escapes bank path: {file_text}") from exc
    return local_path


def validate_category_list(value: object, label: str, valid_categories: set[str]) -> list[str]:
    categories = string_list(value, label)
    require(categories, f"{label} must be non-empty")
    unknown = sorted(set(categories) - valid_categories)
    require(not unknown, f"{label} has unknown categories {unknown}")
    return categories


def validate_assets(
    bank_id: str,
    bank_path: Path,
    assets: object,
    *,
    require_files: bool,
    accepted_trainable_licenses: set[str],
    valid_categories: set[str],
    require_trainable_review: bool,
) -> int:
    require(isinstance(assets, list), f"{bank_id}: assets must be a list")
    if require_files:
        require(assets, f"{bank_id}: reviewed external-negative bank must list assets with source/license metadata")

    seen_files: set[str] = set()
    for asset in assets:
        require(isinstance(asset, dict), f"{bank_id}: asset rows must be objects")
        file_text = str(asset.get("file", "")).strip()
        require(file_text, f"{bank_id}: asset missing file")
        require(file_text not in seen_files, f"{bank_id}: duplicate asset file {file_text}")
        seen_files.add(file_text)

        local_path = relative_asset_path(bank_id, bank_path, file_text)
        if require_files:
            require(local_path.exists() and local_path.is_file(), f"{bank_id}: missing asset file {repo_rel(local_path)}")
            require(local_path.suffix.lower() in IMAGE_SUFFIXES, f"{bank_id}: unsupported image suffix {local_path.suffix}")

        license_name = str(asset.get("license", "")).strip()
        require(license_name, f"{bank_id}: asset {file_text} missing license")
        require(str(asset.get("source_url", "")).strip(), f"{bank_id}: asset {file_text} missing source_url")
        require(str(asset.get("license_url", "")).strip(), f"{bank_id}: asset {file_text} missing license_url")
        require(str(asset.get("credit", "")).strip(), f"{bank_id}: asset {file_text} missing credit")
        validate_category_list(asset.get("categories", []), f"{bank_id}: asset {file_text} categories", valid_categories)

        target_status = str(asset.get("target_currency_status", "")).strip()
        require(
            target_status in TARGET_CURRENCY_STATUSES,
            f"{bank_id}: asset {file_text} invalid target_currency_status {target_status!r}",
        )

        if require_trainable_review:
            require(
                license_name in accepted_trainable_licenses,
                f"{bank_id}: asset {file_text} has non-trainable license {license_name!r}",
            )
            require(
                target_status == "verified_absent",
                f"{bank_id}: asset {file_text} must be verified_absent of target USD/KHR notes for trainable use",
            )
            require(str(asset.get("review_basis", "")).strip(), f"{bank_id}: asset {file_text} missing review_basis")

    return len(assets)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    min_trainable_images = int(config.get("min_trainable_images", 1))
    require(min_trainable_images > 0, "min_trainable_images must be positive")

    accepted_trainable_licenses = set(string_list(config.get("accepted_trainable_licenses", []), "accepted_trainable_licenses"))
    require(accepted_trainable_licenses, "accepted_trainable_licenses must be non-empty")
    valid_categories = set(string_list(config.get("valid_categories", []), "valid_categories"))
    require(valid_categories, "valid_categories must be non-empty")

    banks = config.get("banks", [])
    require(isinstance(banks, list) and banks, "external negative config must contain banks")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    allowed_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for row in banks:
        require(isinstance(row, dict), "external negative bank rows must be objects")
        bank_id = str(row.get("id", "")).strip()
        require(bank_id, "external negative bank id must be non-empty")
        require(bank_id not in seen_ids, f"duplicate external negative bank id: {bank_id}")
        seen_ids.add(bank_id)

        path_text = str(row.get("path", "")).strip()
        require(path_text, f"{bank_id}: missing path")
        path = resolve(Path(path_text)).resolve()
        try:
            path.relative_to((ROOT / "data" / "external_negatives").resolve())
        except ValueError as exc:
            raise SystemExit(f"{bank_id}: external negative path must stay under data/external_negatives") from exc

        path_key = path.relative_to(ROOT).as_posix()
        require(path_key not in seen_paths, f"duplicate external negative bank path: {path_key}")
        seen_paths.add(path_key)

        review_status = str(row.get("review_status", "")).strip()
        require(review_status in VALID_REVIEW_STATUSES, f"{bank_id}: invalid review_status {review_status!r}")
        allowed = string_list(row.get("allowed_artifact_statuses", []), f"{bank_id}: allowed_artifact_statuses")
        unknown_allowed = sorted(set(allowed) - VALID_ARTIFACT_STATUSES)
        require(not unknown_allowed, f"{bank_id}: unknown allowed artifact statuses {unknown_allowed}")

        intended_categories = validate_category_list(
            row.get("intended_categories", []),
            f"{bank_id}: intended_categories",
            valid_categories,
        )
        category_counts.update(intended_categories)

        if review_status == "planned":
            require(not allowed, f"{bank_id}: planned banks must not allow artifact use")
            count = 0
            asset_count = validate_assets(
                bank_id,
                path,
                row.get("assets", []),
                require_files=False,
                accepted_trainable_licenses=accepted_trainable_licenses,
                valid_categories=valid_categories,
                require_trainable_review=False,
            )
        else:
            require(path.exists() and path.is_dir(), f"{bank_id}: missing external negative directory {path}")
            count = image_count(path)
            require(count > 0, f"{bank_id}: no external negative images found")
            asset_count = validate_assets(
                bank_id,
                path,
                row.get("assets", []),
                require_files=True,
                accepted_trainable_licenses=accepted_trainable_licenses,
                valid_categories=valid_categories,
                require_trainable_review=review_status == "accepted",
            )
            require(asset_count == count, f"{bank_id}: asset metadata count {asset_count} != file count {count}")

        if review_status == "accepted":
            require(
                count >= min_trainable_images,
                f"{bank_id}: accepted trainable bank needs at least {min_trainable_images} images",
            )
            require("trainable-candidate" in allowed, f"{bank_id}: accepted bank must allow trainable-candidate")
            require(str(row.get("review_basis", "")).strip(), f"{bank_id}: accepted bank must include review_basis")
        else:
            require("trainable-candidate" not in allowed, f"{bank_id}: only accepted banks may allow trainable-candidate")
            require("promoted" not in allowed, f"{bank_id}: only accepted banks may allow promoted")

        status_counts[review_status] += 1
        allowed_counts.update(allowed)
        rows.append(
            {
                "id": bank_id,
                "path": path_key,
                "review_status": review_status,
                "allowed_artifact_statuses": allowed,
                "images": count,
                "assets": asset_count,
                "intended_categories": intended_categories,
                "blocker": str(row.get("blocker", "")),
            }
        )

    if args.require_path is not None:
        require(args.artifact_status, "--artifact-status is required with --require-path")
        requested_path = repo_rel(args.require_path)
        matches = [row for row in rows if row["path"] == requested_path]
        require(matches, f"external negative directory is not registered: {requested_path}")
        bank = matches[0]
        allowed = set(bank["allowed_artifact_statuses"])
        require(
            args.artifact_status in allowed,
            (
                f"{bank['id']}: review_status={bank['review_status']} does not allow "
                f"artifact_status={args.artifact_status}; blocker={bank['blocker']}"
            ),
        )

    print(
        f"ok: {config.get('name')} has {len(rows)} bank(s), "
        f"statuses={dict(sorted(status_counts.items()))}, "
        f"allowed={dict(sorted(allowed_counts.items()))}, "
        f"categories={dict(sorted(category_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
