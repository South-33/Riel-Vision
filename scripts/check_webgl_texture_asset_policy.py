#!/usr/bin/env python
"""Validate that WebGL renders use approved current banknote textures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_approved_texture_bank_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Rendered WebGL dataset root with manifest.json.")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="Approved texture-bank JSON.")
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--require-asset-quality-policy", default="latest_design")
    parser.add_argument("--require-source-status", default="in_circulation")
    parser.add_argument("--require-reviewed-status", default="manual_pass_texture_qa_v1")
    parser.add_argument(
        "--require-all-approved-class-sides",
        action="store_true",
        help="Require every approved class/side texture to appear at least once.",
    )
    return parser.parse_args()


def resolve(path: str | Path, root: Path = ROOT) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def norm_repo_path(value: str) -> str:
    path = Path(value)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")
    return path.as_posix().replace("\\", "/")


def norm_source_path(value: str) -> str:
    return value.replace("\\", "/")


def approved_rows(bank_path: Path, required_review_status: str) -> dict[tuple[str, str], dict[str, Any]]:
    payload = read_json(bank_path)
    require(isinstance(payload, dict), "approved texture bank must be an object")
    review = payload.get("visual_review", {})
    if required_review_status:
        require(isinstance(review, dict), "approved texture bank must include visual_review")
        require(
            review.get("status") == required_review_status,
            f"approved texture bank visual_review.status must be {required_review_status!r}",
        )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("rows", []):
        require(isinstance(row, dict), "approved texture bank rows must be objects")
        key = (str(row.get("class_name", "")), str(row.get("side", "")))
        require(all(key), f"approved texture bank has incomplete row key: {row}")
        if required_review_status:
            require(
                row.get("visual_review_status") == required_review_status,
                f"{'/'.join(key)} review status is not {required_review_status!r}",
            )
        require(key not in rows, f"duplicate approved texture row: {'/'.join(key)}")
        rows[key] = row
    require(rows, "approved texture bank has no rows")
    return rows


def image_assets(dataset_root: Path) -> tuple[list[dict[str, Any]], int]:
    manifest = read_json(dataset_root / "manifest.json")
    require(isinstance(manifest, list), "manifest.json must be a list")
    assets: list[dict[str, Any]] = []
    for manifest_row in manifest:
        require(isinstance(manifest_row, dict), "manifest rows must be objects")
        metadata_path = resolve(str(manifest_row.get("source_metadata", "")), dataset_root)
        metadata = read_json(metadata_path)
        require(isinstance(metadata, dict), f"{metadata_path} must be an object")
        row_assets = metadata.get("assets", [])
        require(isinstance(row_assets, list), f"{metadata_path} assets must be a list")
        for asset in row_assets:
            require(isinstance(asset, dict), f"{metadata_path} asset must be an object")
            assets.append(asset)
    return assets, len(manifest)


def compare_asset(
    asset: dict[str, Any],
    approved: dict[tuple[str, str], dict[str, Any]],
    required_quality_policy: str,
    required_source_status: str,
) -> tuple[str, str]:
    class_name = str(asset.get("className", ""))
    side = str(asset.get("side", ""))
    key = (class_name, side)
    require(all(key), f"asset is missing className/side: {asset}")
    approved_row = approved.get(key)
    require(approved_row is not None, f"{class_name}/{side} is not in approved texture bank")
    if required_quality_policy:
        require(
            asset.get("assetQualityPolicy") == required_quality_policy,
            f"{class_name}/{side} assetQualityPolicy={asset.get('assetQualityPolicy')!r}, expected {required_quality_policy!r}",
        )
    if required_source_status:
        require(
            asset.get("sourceStatus") == required_source_status,
            f"{class_name}/{side} sourceStatus={asset.get('sourceStatus')!r}, expected {required_source_status!r}",
        )

    asset_path = norm_repo_path(str(asset.get("path", "")))
    expected_path = norm_repo_path(str(approved_row.get("asset_path", "")))
    require(asset_path == expected_path, f"{class_name}/{side} asset path drifted: {asset_path} != {expected_path}")
    for asset_key, row_key in [
        ("sourceYears", "years"),
        ("sourceMaxYear", "max_year"),
        ("textureWidth", "width"),
        ("textureHeight", "height"),
    ]:
        require(
            str(asset.get(asset_key, "")) == str(approved_row.get(row_key, "")),
            f"{class_name}/{side} {asset_key} drifted: {asset.get(asset_key)!r} != {approved_row.get(row_key)!r}",
        )
    source_path = norm_source_path(str(asset.get("sourcePath", "")))
    expected_source_path = norm_source_path(str(approved_row.get("source_path", "")))
    require(
        source_path == expected_source_path,
        f"{class_name}/{side} sourcePath drifted: {source_path} != {expected_source_path}",
    )
    return key


def main() -> int:
    args = parse_args()
    dataset_root = resolve(args.root)
    bank_path = resolve(args.bank)
    approved = approved_rows(bank_path, args.require_reviewed_status)
    assets, images = image_assets(dataset_root)
    require(images >= args.min_images, f"expected at least {args.min_images} images, got {images}")
    require(assets, f"no WebGL assets found in {dataset_root}")

    counts: Counter[tuple[str, str]] = Counter()
    for asset in assets:
        counts[compare_asset(asset, approved, args.require_asset_quality_policy, args.require_source_status)] += 1

    if args.require_all_approved_class_sides:
        missing = sorted(set(approved) - set(counts))
        require(not missing, f"missing approved class/side textures: {', '.join('/'.join(key) for key in missing)}")

    print(
        f"ok: texture asset policy passed "
        f"({images} images, {len(assets)} assets, {len(counts)} approved class/sides)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
