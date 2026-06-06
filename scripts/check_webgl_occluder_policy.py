#!/usr/bin/env python
"""Check WebGL packaged metadata for primitive occluder policy drift."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from webgl_constants import WEBGL_OCCLUDER_POLICIES


ROOT = Path(__file__).resolve().parents[1]
HAND_TOKENS = ("finger", "hand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL root.")
    parser.add_argument("--expected-policy", choices=sorted(WEBGL_OCCLUDER_POLICIES), default="")
    parser.add_argument("--forbid-hand-occluders", action="store_true")
    parser.add_argument("--require-zero-occluders", action="store_true")
    parser.add_argument("--allow-missing-policy", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {repo_rel(path)}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(path)}: expected JSON object")
    return data


def metadata_paths(root: Path) -> list[Path]:
    metadata_dir = root / "metadata"
    if not metadata_dir.exists():
        raise SystemExit(f"missing metadata dir: {repo_rel(metadata_dir)}")
    paths = sorted(metadata_dir.glob("variant_*_metadata.json"))
    if not paths:
        raise SystemExit(f"metadata dir has no variant metadata: {repo_rel(metadata_dir)}")
    return paths


def occluder_kind(row: object) -> str:
    return str(row.get("kind", "unknown")) if isinstance(row, dict) else "malformed"


def is_hand_kind(kind: str) -> bool:
    lowered = kind.lower()
    return any(token in lowered for token in HAND_TOKENS)


def policy_for(metadata: dict[str, Any], *, allow_missing: bool) -> str:
    policy = str(metadata.get("occluderPolicy", "missing"))
    if policy == "missing" and allow_missing:
        return "scene_default"
    return policy


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    policy_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    failures: list[str] = []
    hand_rows: list[str] = []
    occluder_rows: list[str] = []

    for path in metadata_paths(root):
        metadata = read_json(path)
        policy = policy_for(metadata, allow_missing=args.allow_missing_policy)
        policy_counts[policy] += 1
        scene_counts[str(metadata.get("sceneMode", "unknown"))] += 1
        if args.expected_policy and policy != args.expected_policy:
            failures.append(f"{repo_rel(path)} policy {policy!r} != {args.expected_policy!r}")
        occluders = metadata.get("occluders", [])
        if not isinstance(occluders, list):
            failures.append(f"{repo_rel(path)} occluders is not a list")
            continue
        if occluders:
            occluder_rows.append(repo_rel(path))
        for occluder in occluders:
            kind = occluder_kind(occluder)
            kind_counts[kind] += 1
            if is_hand_kind(kind):
                hand_rows.append(f"{repo_rel(path)}:{kind}")

    if args.require_zero_occluders and occluder_rows:
        failures.append(f"{len(occluder_rows)} metadata file(s) contain occluders")
    if args.forbid_hand_occluders and hand_rows:
        failures.append(f"{len(hand_rows)} hand/finger occluder instance(s) found")

    print(
        "webgl_occluder_policy="
        f"{'pass' if not failures else 'blocked'} "
        f"images={sum(policy_counts.values())} "
        f"policies={dict(sorted(policy_counts.items()))} "
        f"kinds={dict(sorted(kind_counts.items()))} "
        f"scenes={dict(sorted(scene_counts.items()))}"
    )
    for row in failures[:8]:
        print(f"- {row}")
    if len(failures) > 8:
        print(f"- ... {len(failures) - 8} more")
    if hand_rows and args.forbid_hand_occluders:
        print("hand_examples=" + ", ".join(hand_rows[:5]))
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
