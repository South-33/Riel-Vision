#!/usr/bin/env python
"""Check CashSnap currency coverage without conflating raw assets, active bank, and model schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from cashsnap_currency_taxonomy import (
    OFFICIAL_CURRENT_CLASS_NAMES,
    OFFICIAL_TAXONOMY_SOURCES,
    ROOT,
    class_names_for_scope,
    cutout_bank_coverage,
    numista_raw_coverage,
    repo_path,
    resolve_repo_path,
)


DEFAULT_METADATA = ROOT / "data" / "numista_raw" / "metadata.json"
DEFAULT_CUTOUT_BANK = ROOT / "data" / "asset_candidates" / "numista_current_cutout_bank_v1"
DEFAULT_TAXONOMY_DATA = ROOT / "data" / "cashsnap_v1" / "data.yaml"
DEFAULT_JSON_OUT = ROOT / "runs" / "cashsnap" / "currency_taxonomy_coverage_latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cutout-bank", type=Path, default=DEFAULT_CUTOUT_BANK)
    parser.add_argument("--taxonomy-data", type=Path, default=DEFAULT_TAXONOMY_DATA)
    parser.add_argument("--class-scope", choices=["operational", "official"], default="official")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any selected layer is incomplete.")
    return parser.parse_args()


def read_yolo_class_names(data_config: Path) -> tuple[list[str], list[str]]:
    resolved = resolve_repo_path(data_config).resolve()
    if not resolved.exists():
        return [], [f"taxonomy data config is missing: {repo_path(resolved)}"]
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return [], [f"taxonomy data config is malformed: {repo_path(resolved)}"]
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        rows: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                rows.append((int(key), str(value)))
            except (TypeError, ValueError):
                return [], [f"taxonomy data config has non-integer class id: {key!r}"]
        return [name for _, name in sorted(rows)], []
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names], []
    return [], [f"taxonomy data config has no usable names map: {repo_path(resolved)}"]


def model_coverage(data_config: Path, *, scope: str) -> dict[str, Any]:
    class_names, errors = read_yolo_class_names(data_config)
    target = class_names_for_scope(scope)
    present = [class_name for class_name in target if class_name in set(class_names)]
    return {
        "data": repo_path(resolve_repo_path(data_config).resolve()),
        "class_scope": scope,
        "target_class_names": target,
        "class_names": class_names,
        "present_class_names": present,
        "missing_class_names": [class_name for class_name in target if class_name not in set(class_names)],
        "outside_target_class_names": [class_name for class_name in class_names if class_name not in set(target)],
        "errors": errors,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    raw = numista_raw_coverage(args.metadata, scope=args.class_scope)
    cutout = cutout_bank_coverage(args.cutout_bank, scope=args.class_scope)
    model = model_coverage(args.taxonomy_data, scope=args.class_scope)
    target = class_names_for_scope(args.class_scope)
    blockers = []
    blockers.extend(f"raw current front/back missing: {name}" for name in raw["missing_current_front_back_class_names"])
    blockers.extend(f"active cutout front/back missing: {name}" for name in cutout["missing_front_back_class_names"])
    blockers.extend(f"model schema missing: {name}" for name in model["missing_class_names"])
    blockers.extend(raw.get("errors", []))
    blockers.extend(cutout.get("errors", []))
    blockers.extend(model.get("errors", []))
    return {
        "status": "pass" if not blockers else "blocked",
        "class_scope": args.class_scope,
        "official_sources": OFFICIAL_TAXONOMY_SOURCES,
        "official_current_class_names": OFFICIAL_CURRENT_CLASS_NAMES,
        "counts": {
            "target_classes": len(target),
            "raw_current_front_back_ready": len(raw["current_front_back_ready_class_names"]),
            "raw_any_front_back_ready": len(raw["any_front_back_ready_class_names"]),
            "active_cutout_front_back_ready": len(cutout["front_back_ready_class_names"]),
            "model_present": len(model["present_class_names"]),
        },
        "missing": {
            "raw_current_front_back": raw["missing_current_front_back_class_names"],
            "raw_any_front_back": raw["missing_any_front_back_class_names"],
            "active_cutout_front_back": cutout["missing_front_back_class_names"],
            "model_schema": model["missing_class_names"],
        },
        "blockers": blockers,
        "raw_numista": raw,
        "active_cutout_bank": cutout,
        "model": model,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    out = resolve_repo_path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = report["counts"]
    print(
        "currency_taxonomy_coverage="
        f"{report['status']} "
        f"scope={report['class_scope']} "
        f"target={counts['target_classes']} "
        f"raw_current={counts['raw_current_front_back_ready']} "
        f"raw_any={counts['raw_any_front_back_ready']} "
        f"cutout={counts['active_cutout_front_back_ready']} "
        f"model={counts['model_present']}"
    )
    for layer, missing in report["missing"].items():
        if missing:
            print(f"{layer}: {', '.join(missing)}")
    print(f"json: {repo_path(out)}")
    if args.strict and report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
