#!/usr/bin/env python
"""Validate CashSnap data lifecycle registry and critical working asset rules."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "synthetic_recipes" / "cashsnap_data_lifecycle_registry_v1.json"
APPROVED_TEXTURE_BANK = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_approved_texture_bank_v1.json"
ALLOWED_STATES = {"working", "diagnostic", "planned", "intake", "container", "archive", "rejected"}
NON_TRAINABLE_STATES = {"archive", "rejected", "intake", "container", "planned"}
REQUIRED_TEXTURE_REVIEW = "manual_pass_texture_qa_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--allow-missing-data", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing JSON: {repo_rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing YAML: {repo_rel(path)}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check_exists(path: Path, *, allow_missing_data: bool) -> None:
    if path.exists():
        return
    if allow_missing_data and repo_rel(path).startswith("data/"):
        print(f"warn: missing ignored data path allowed: {repo_rel(path)}")
        return
    raise SystemExit(f"missing lifecycle path: {repo_rel(path)}")


def norm(value: str) -> str:
    return value.replace("\\", "/")


def check_approved_texture_bank(path: Path) -> None:
    payload = read_json(path)
    require(isinstance(payload, dict), "approved texture bank must be a JSON object")
    rows = payload.get("rows")
    require(isinstance(rows, list) and rows, "approved texture bank rows must be a non-empty list")
    require(int(payload.get("class_side_count", 0)) == 26, "approved texture bank must cover 26 class/side rows")

    seen: set[tuple[str, str]] = set()
    usd20_rows: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict), "approved texture bank row must be an object")
        class_name = str(row.get("class_name", ""))
        side = str(row.get("side", ""))
        key = (class_name, side)
        require(all(key), f"texture row missing class/side: {row}")
        require(key not in seen, f"duplicate texture row: {class_name}/{side}")
        seen.add(key)
        require(row.get("status") == "in_circulation", f"{class_name}/{side} is not in circulation")
        require(
            row.get("visual_review_status") == REQUIRED_TEXTURE_REVIEW,
            f"{class_name}/{side} lacks required visual review status {REQUIRED_TEXTURE_REVIEW}",
        )
        asset_path = resolve(str(row.get("asset_path", "")))
        source_path = resolve(str(row.get("source_path", "")))
        require(asset_path.exists(), f"{class_name}/{side} missing asset_path: {repo_rel(asset_path)}")
        require(source_path.exists(), f"{class_name}/{side} missing source_path: {repo_rel(source_path)}")
        source_text = norm(str(row.get("source_path", ""))).lower()
        require("/out_of_circulation/" not in source_text, f"{class_name}/{side} points to out_of_circulation source")
        if class_name == "USD_20":
            usd20_rows.append(row)

    require(len(seen) == 26, f"approved texture bank must have 26 unique class/side rows, got {len(seen)}")
    require(len(usd20_rows) == 2, f"USD_20 must have exactly two approved sides, got {len(usd20_rows)}")
    for row in usd20_rows:
        side = str(row.get("side", ""))
        years = str(row.get("years", ""))
        max_year = str(row.get("max_year", ""))
        asset_path = norm(str(row.get("asset_path", "")))
        source_path = norm(str(row.get("source_path", "")))
        require(years == "2004-2021", f"USD_20/{side} must use 2004-2021 current design, got {years!r}")
        require(max_year == "2021", f"USD_20/{side} max_year must be 2021, got {max_year!r}")
        require("2004-2021" in asset_path, f"USD_20/{side} asset path does not show 2004-2021: {asset_path}")
        require("2004-2021" in source_path, f"USD_20/{side} source path does not show 2004-2021: {source_path}")


def resolve_dataset_split(data_yaml: Path, config: dict[str, Any], split: str) -> Path:
    value = config.get(split)
    if value is None and split == "val":
        value = config.get("valid")
    require(isinstance(value, str) and value.strip(), f"{repo_rel(data_yaml)} missing {split} split")
    split_path = Path(value).expanduser()
    if split_path.is_absolute():
        return split_path
    root_value = config.get("path")
    root_path = data_yaml.parent
    if root_value is not None:
        root_path = Path(str(root_value)).expanduser()
        if not root_path.is_absolute():
            root_path = (data_yaml.parent / root_path).resolve()
    return (root_path / split_path).resolve()


def check_yolo_dataset(path: Path) -> None:
    data_yaml = path / "data.yaml"
    payload = read_yaml(data_yaml)
    require(isinstance(payload, dict), f"{repo_rel(data_yaml)} must be a mapping")
    raw_names = payload.get("names")
    if isinstance(raw_names, dict):
        names = [str(value) for _, value in sorted((int(key), value) for key, value in raw_names.items())]
    elif isinstance(raw_names, list):
        names = [str(value) for value in raw_names]
    else:
        raise SystemExit(f"{repo_rel(data_yaml)} must include names list or mapping")
    require(bool(names), f"{repo_rel(data_yaml)} has empty names")
    for split in ("train", "val", "test"):
        split_path = resolve_dataset_split(data_yaml, payload, split)
        require(split_path.exists(), f"{repo_rel(data_yaml)} {split} path missing: {repo_rel(split_path)}")


def check_registry(registry_path: Path, *, allow_missing_data: bool) -> None:
    payload = read_json(registry_path)
    require(isinstance(payload, dict), "data lifecycle registry must be an object")
    require(payload.get("schema_version") == 1, "data lifecycle registry schema_version must be 1")
    entries = payload.get("entries")
    require(isinstance(entries, list) and entries, "data lifecycle registry must include entries")

    ids: set[str] = set()
    path_states: dict[str, str] = {}
    states: Counter[str] = Counter()
    for entry in entries:
        require(isinstance(entry, dict), "lifecycle entry must be an object")
        entry_id = str(entry.get("id", "")).strip()
        path_text = str(entry.get("path", "")).strip()
        state = str(entry.get("state", "")).strip()
        require(entry_id, f"lifecycle entry missing id: {entry}")
        require(entry_id not in ids, f"duplicate lifecycle id: {entry_id}")
        ids.add(entry_id)
        require(path_text, f"{entry_id} missing path")
        require(state in ALLOWED_STATES, f"{entry_id} has unsupported state {state!r}")
        require(str(entry.get("purpose", "")).strip(), f"{entry_id} missing purpose")
        states[state] += 1
        normalized_path = norm(path_text)
        prior_state = path_states.get(normalized_path)
        require(prior_state is None, f"duplicate lifecycle path: {normalized_path}")
        path_states[normalized_path] = state

        path = resolve(path_text)
        checks = entry.get("checks", [])
        require(isinstance(checks, list), f"{entry_id} checks must be a list")
        for check in checks:
            check_name = str(check)
            if check_name == "exists":
                check_exists(path, allow_missing_data=allow_missing_data)
            elif check_name == "json":
                require(path.suffix.lower() == ".json", f"{entry_id} json check on non-json path: {path_text}")
                read_json(path)
            elif check_name == "approved_texture_bank_v1":
                require(path.resolve() == APPROVED_TEXTURE_BANK.resolve(), f"{entry_id} checks wrong texture bank")
                check_approved_texture_bank(path)
            elif check_name == "yolo_dataset":
                check_yolo_dataset(path)
            else:
                raise SystemExit(f"{entry_id} has unknown lifecycle check: {check_name}")

    policy = payload.get("policy", {})
    require(isinstance(policy, dict), "registry policy must be an object")
    do_not_train = set(policy.get("do_not_train_directly_from_states", []))
    require(NON_TRAINABLE_STATES <= do_not_train, "registry policy must mark all non-trainable states")
    print(
        "ok: data lifecycle registry passed "
        f"entries={len(entries)} states={dict(sorted(states.items()))}"
    )


def main() -> int:
    args = parse_args()
    check_registry(resolve(args.registry), allow_missing_data=args.allow_missing_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
