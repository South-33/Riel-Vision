#!/usr/bin/env python
"""Validate that a WebGL environment-map review render covered a bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Rendered WebGL batch root.")
    parser.add_argument("--config", type=Path, required=True, help="Environment bank config used for the render.")
    parser.add_argument("--bank-id", default="", help="Bank id to check. Defaults to the only/first bank.")
    parser.add_argument("--min-images-per-asset", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    return resolve(path).resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> Any:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing JSON file: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def selected_bank(config: dict[str, Any], bank_id: str) -> dict[str, Any]:
    banks = config.get("banks", [])
    require(isinstance(banks, list) and banks, "environment config must contain banks")
    if bank_id:
        matches = [bank for bank in banks if isinstance(bank, dict) and bank.get("id") == bank_id]
        require(matches, f"bank id not found: {bank_id}")
        return matches[0]
    require(isinstance(banks[0], dict), "bank rows must be objects")
    return banks[0]


def asset_files(bank: dict[str, Any]) -> set[str]:
    assets = bank.get("assets", [])
    require(isinstance(assets, list) and assets, f"{bank.get('id', '<bank>')}: assets must be a non-empty list")
    files: set[str] = set()
    for asset in assets:
        require(isinstance(asset, dict), f"{bank.get('id', '<bank>')}: asset rows must be objects")
        file_name = str(asset.get("file", "")).strip()
        require(file_name, f"{bank.get('id', '<bank>')}: asset missing file")
        require(file_name not in files, f"{bank.get('id', '<bank>')}: duplicate asset file {file_name}")
        files.add(file_name)
    return files


def variant_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("variant_*") if path.is_dir())


def main() -> int:
    args = parse_args()
    require(args.min_images_per_asset > 0, "--min-images-per-asset must be positive")
    root = resolve(args.root).resolve()
    require(root.exists() and root.is_dir(), f"missing render root: {root}")

    config = read_json(args.config)
    require(isinstance(config, dict), "environment config must be a JSON object")
    bank = selected_bank(config, args.bank_id)
    bank_id = str(bank.get("id", "")).strip() or "<bank>"
    bank_path = resolve(Path(str(bank.get("path", "")))).resolve()
    expected_files = asset_files(bank)

    directories = variant_dirs(root)
    require(directories, f"{repo_rel(root)}: no variant_* directories found")

    seen: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for directory in directories:
        metadata_path = directory / "metadata.json"
        visual_path = directory / "visual.png"
        id_path = directory / "id.png"
        require(metadata_path.exists(), f"{repo_rel(directory)}: missing metadata.json")
        require(visual_path.exists(), f"{repo_rel(directory)}: missing visual.png")
        require(id_path.exists(), f"{repo_rel(directory)}: missing id.png")
        metadata = read_json(metadata_path)
        environment = metadata.get("sceneConfig", {}).get("environment") if isinstance(metadata, dict) else None
        require(isinstance(environment, dict), f"{repo_rel(directory)}: metadata has no sceneConfig.environment")
        env_path = Path(str(environment.get("path", ""))).resolve()
        require(env_path.is_file(), f"{repo_rel(directory)}: environment path missing on disk: {env_path}")
        try:
            env_path.relative_to(bank_path)
        except ValueError as exc:
            raise SystemExit(f"{repo_rel(directory)}: environment path is outside bank path: {env_path}") from exc
        file_name = env_path.name
        require(file_name in expected_files, f"{repo_rel(directory)}: environment file not listed in bank config: {file_name}")
        seen[file_name] += 1
        rows.append(
            {
                "variant": directory.name,
                "environment_file": file_name,
                "format": str(environment.get("format", "")),
            }
        )

    missing = sorted(file_name for file_name in expected_files if seen[file_name] < args.min_images_per_asset)
    require(
        not missing,
        f"{bank_id}: review render did not cover all assets at least {args.min_images_per_asset} time(s): {missing}",
    )

    summary_path = root / "counts" / "summary.json"
    require(summary_path.exists(), f"{repo_rel(root)}: missing counts/summary.json")
    counts = read_json(summary_path)
    image_count = int(counts.get("images", 0)) if isinstance(counts, dict) else 0
    physical_total = int(counts.get("physical_visible_instances", {}).get("total", 0)) if isinstance(counts, dict) else 0
    require(image_count == len(directories), f"{repo_rel(root)}: summary images {image_count} != variants {len(directories)}")
    require(physical_total > 0, f"{repo_rel(root)}: review render has no physical visible targets")

    report = {
        "root": repo_rel(root),
        "bank_id": bank_id,
        "bank_path": repo_rel(bank_path),
        "assets": len(expected_files),
        "images": len(directories),
        "physical_visible_targets": physical_total,
        "environment_hits": dict(sorted(seen.items())),
        "rows": rows,
    }
    if args.json_out:
        json_out = resolve(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_rel(json_out)}")
    print(
        f"ok: {bank_id} review covered {len(expected_files)} environment asset(s) "
        f"with {len(directories)} image(s), physical_targets={physical_total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
