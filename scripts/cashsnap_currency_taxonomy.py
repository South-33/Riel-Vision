"""CashSnap currency taxonomy and asset-coverage helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OPERATIONAL_CLASS_NAMES = [
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

OFFICIAL_CURRENT_CLASS_NAMES = [
    "USD_1",
    "USD_2",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_50",
    "KHR_100",
    "KHR_200",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_15000",
    "KHR_20000",
    "KHR_30000",
    "KHR_50000",
    "KHR_100000",
    "KHR_200000",
]

OFFICIAL_TAXONOMY_SOURCES = {
    "USD": "https://www.uscurrency.gov/denominations",
    "KHR": "https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php",
}

TARGET_KHR_OFFICIAL = {
    "50": "KHR_50",
    "100": "KHR_100",
    "200": "KHR_200",
    "500": "KHR_500",
    "1000": "KHR_1000",
    "2000": "KHR_2000",
    "5000": "KHR_5000",
    "10000": "KHR_10000",
    "15000": "KHR_15000",
    "20000": "KHR_20000",
    "30000": "KHR_30000",
    "50000": "KHR_50000",
    "100000": "KHR_100000",
    "200000": "KHR_200000",
}

USD_VALUES_OFFICIAL = {"1", "2", "5", "10", "20", "50", "100"}


def class_names_for_scope(scope: str) -> list[str]:
    if scope == "operational":
        return OPERATIONAL_CLASS_NAMES
    if scope == "official":
        return OFFICIAL_CURRENT_CLASS_NAMES
    raise ValueError(f"unknown class scope: {scope}")


def class_name_for_metadata(row: dict[str, Any], *, scope: str = "official") -> str | None:
    allowed = set(class_names_for_scope(scope))
    country = str(row.get("country", ""))
    denomination = str(row.get("denomination", "")).strip()
    if country == "Cambodia":
        class_name = TARGET_KHR_OFFICIAL.get(denomination)
    elif country == "United States" and denomination in USD_VALUES_OFFICIAL:
        class_name = f"USD_{denomination}"
    else:
        class_name = None
    return class_name if class_name in allowed else None


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def empty_side_counts() -> dict[str, int]:
    return {"front": 0, "back": 0}


def side_ready(side_counts: dict[str, int]) -> bool:
    return int(side_counts.get("front", 0)) > 0 and int(side_counts.get("back", 0)) > 0


def normalize_source_path(metadata_path: Path, relative_path: str) -> Path:
    return metadata_path.parent / Path(relative_path.replace("\\", "/"))


def numista_raw_coverage(metadata_path: Path, *, scope: str = "official") -> dict[str, Any]:
    resolved = resolve_repo_path(metadata_path).resolve()
    class_names = class_names_for_scope(scope)
    coverage: dict[str, dict[str, Any]] = {
        class_name: {
            "current_note_count": 0,
            "any_note_count": 0,
            "current_side_counts": empty_side_counts(),
            "any_side_counts": empty_side_counts(),
            "current_examples": [],
            "any_examples": [],
        }
        for class_name in class_names
    }
    if not resolved.exists():
        return {
            "metadata": repo_path(resolved),
            "exists": False,
            "class_scope": scope,
            "class_names": class_names,
            "coverage": coverage,
            "current_front_back_ready_class_names": [],
            "any_front_back_ready_class_names": [],
            "missing_current_front_back_class_names": class_names,
            "missing_any_front_back_class_names": class_names,
            "errors": [f"metadata missing: {repo_path(resolved)}"],
        }

    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{repo_path(resolved)}: expected metadata JSON object")

    note_seen_current: dict[str, set[str]] = {class_name: set() for class_name in class_names}
    note_seen_any: dict[str, set[str]] = {class_name: set() for class_name in class_names}
    for note_id, note in sorted(data.items()):
        if not isinstance(note, dict):
            continue
        class_name = class_name_for_metadata(note, scope=scope)
        if not class_name:
            continue
        status = str(note.get("circulation_status", ""))
        files = note.get("files", {})
        if not isinstance(files, dict):
            continue
        current = status == "in_circulation"
        for side in ["front", "back"]:
            rel_source = files.get(side)
            if not rel_source:
                continue
            source = normalize_source_path(resolved, str(rel_source))
            if not source.exists():
                continue
            coverage[class_name]["any_side_counts"][side] += 1
            note_seen_any[class_name].add(str(note_id))
            if current:
                coverage[class_name]["current_side_counts"][side] += 1
                note_seen_current[class_name].add(str(note_id))

        title = str(note.get("title", "")).strip()
        example = {
            "note_id": str(note_id),
            "title": title,
            "status": status,
            "years": str(note.get("years", "")),
            "directory": str(note.get("directory", "")),
        }
        if len(coverage[class_name]["any_examples"]) < 3 and note_seen_any[class_name]:
            coverage[class_name]["any_examples"].append(example)
        if current and len(coverage[class_name]["current_examples"]) < 3 and note_seen_current[class_name]:
            coverage[class_name]["current_examples"].append(example)

    for class_name in class_names:
        row = coverage[class_name]
        row["current_note_count"] = len(note_seen_current[class_name])
        row["any_note_count"] = len(note_seen_any[class_name])
        row["current_front_back_ready"] = side_ready(row["current_side_counts"])
        row["any_front_back_ready"] = side_ready(row["any_side_counts"])

    current_ready = [class_name for class_name in class_names if coverage[class_name]["current_front_back_ready"]]
    any_ready = [class_name for class_name in class_names if coverage[class_name]["any_front_back_ready"]]
    return {
        "metadata": repo_path(resolved),
        "exists": True,
        "class_scope": scope,
        "class_names": class_names,
        "coverage": coverage,
        "current_front_back_ready_class_names": current_ready,
        "any_front_back_ready_class_names": any_ready,
        "missing_current_front_back_class_names": [class_name for class_name in class_names if class_name not in current_ready],
        "missing_any_front_back_class_names": [class_name for class_name in class_names if class_name not in any_ready],
        "errors": [],
    }


def resolve_manifest_asset(row: dict[str, str], bank: Path) -> Path:
    raw = row.get("asset_path") or row.get("path") or row.get("output") or ""
    path = Path(raw.replace("\\", "/"))
    if path.is_absolute():
        return path
    repo_resolved = ROOT / path
    if repo_resolved.exists():
        return repo_resolved
    return bank / path


def cutout_bank_coverage(bank: Path, *, scope: str = "official") -> dict[str, Any]:
    resolved = resolve_repo_path(bank).resolve()
    manifest = resolved / "manifest.csv"
    class_names = class_names_for_scope(scope)
    coverage: dict[str, dict[str, Any]] = {
        class_name: {
            "asset_count": 0,
            "side_counts": empty_side_counts(),
            "existing_asset_count": 0,
            "existing_side_counts": empty_side_counts(),
            "examples": [],
        }
        for class_name in class_names
    }
    if not manifest.exists():
        return {
            "bank": repo_path(resolved),
            "manifest": repo_path(manifest),
            "exists": resolved.exists(),
            "manifest_exists": False,
            "class_scope": scope,
            "class_names": class_names,
            "coverage": coverage,
            "front_back_ready_class_names": [],
            "missing_front_back_class_names": class_names,
            "errors": [f"manifest missing: {repo_path(manifest)}"],
        }

    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unknown_classes = sorted({row.get("class_name", "") for row in rows if row.get("class_name", "") not in set(class_names)})
    for row in rows:
        class_name = row.get("class_name", "")
        if class_name not in coverage:
            continue
        side = row.get("side", "")
        asset = resolve_manifest_asset(row, resolved)
        if side not in {"front", "back"}:
            continue
        coverage[class_name]["asset_count"] += 1
        coverage[class_name]["side_counts"][side] += 1
        if asset.exists():
            coverage[class_name]["existing_asset_count"] += 1
            coverage[class_name]["existing_side_counts"][side] += 1
        if len(coverage[class_name]["examples"]) < 3:
            coverage[class_name]["examples"].append(
                {
                    "asset_path": row.get("asset_path", ""),
                    "source_path": row.get("source_path", ""),
                    "side": side,
                    "exists": asset.exists(),
                }
            )

    ready = [class_name for class_name in class_names if side_ready(coverage[class_name]["existing_side_counts"])]
    return {
        "bank": repo_path(resolved),
        "manifest": repo_path(manifest),
        "exists": resolved.exists(),
        "manifest_exists": True,
        "class_scope": scope,
        "class_names": class_names,
        "coverage": coverage,
        "front_back_ready_class_names": ready,
        "missing_front_back_class_names": [class_name for class_name in class_names if class_name not in ready],
        "unknown_class_names": unknown_classes,
        "errors": [],
    }
