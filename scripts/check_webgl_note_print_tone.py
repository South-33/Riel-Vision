#!/usr/bin/env python
"""Gate per-note print-tone metadata in WebGL packages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from webgl_constants import WEBGL_NOTE_PRINT_TONE_POLICIES


ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FIELDS = ("contrast", "saturation", "brightness", "shadowPull", "highlightPush")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--allow-missing", action="store_true", help="Allow legacy packages with no print-tone metadata.")
    parser.add_argument("--expected-policy", choices=sorted(WEBGL_NOTE_PRINT_TONE_POLICIES), default="")
    parser.add_argument("--min-notes", type=int, default=1)
    parser.add_argument("--min-mean-contrast", type=float, default=None)
    parser.add_argument("--max-mean-contrast", type=float, default=None)
    parser.add_argument("--min-contrast-range", type=float, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def numeric_range(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return max(rows) - min(rows)


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(rows) / len(rows)


def source_metadata_path(dataset_root: Path, row: dict) -> Path:
    raw = str(row.get("source_metadata", "")).strip()
    require(bool(raw), f"manifest row for variant {row.get('variant', '<unknown>')} is missing source_metadata")
    return dataset_root / raw


def main() -> int:
    args = parse_args()
    dataset_root = resolve(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    require(isinstance(manifest, list), "manifest.json must be a list")

    total_notes = 0
    missing_tone = 0
    policies: Counter[str] = Counter()
    values: dict[str, list[float]] = {field: [] for field in NUMERIC_FIELDS}

    for row in manifest:
        require(isinstance(row, dict), "manifest rows must be objects")
        meta = read_json(source_metadata_path(dataset_root, row))
        require(isinstance(meta, dict), f"{row.get('source_metadata')}: metadata must be an object")
        scene_config = meta.get("sceneConfig", {})
        require(isinstance(scene_config, dict), f"{row.get('source_metadata')}: sceneConfig must be an object")
        scene_policy = str(scene_config.get("notePrintTonePolicy", "off"))
        require(
            scene_policy in WEBGL_NOTE_PRINT_TONE_POLICIES,
            f"{row.get('source_metadata')}: invalid scene notePrintTonePolicy {scene_policy!r}",
        )
        assets = meta.get("assets", [])
        require(isinstance(assets, list), f"{row.get('source_metadata')}: metadata.assets must be a list")
        for asset in assets:
            require(isinstance(asset, dict), f"{row.get('source_metadata')}: asset rows must be objects")
            total_notes += 1
            tone = asset.get("printTone")
            if not isinstance(tone, dict):
                missing_tone += 1
                continue
            policy = str(tone.get("policy", scene_policy))
            require(
                policy in WEBGL_NOTE_PRINT_TONE_POLICIES,
                f"{row.get('source_metadata')}: invalid printTone policy {policy!r}",
            )
            policies[policy] += 1
            for field in NUMERIC_FIELDS:
                raw_value = tone.get(field)
                require(
                    isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool),
                    f"{row.get('source_metadata')}: printTone.{field} must be numeric",
                )
                values[field].append(float(raw_value))

    if total_notes == 0:
        print("ok: WebGL note print tone skipped (no notes)")
        return 0
    if missing_tone == total_notes and args.allow_missing:
        print(f"ok: WebGL note print tone skipped (legacy package: {total_notes} notes without printTone metadata)")
        return 0
    require(missing_tone == 0, f"{missing_tone}/{total_notes} notes are missing printTone metadata")
    require(total_notes >= args.min_notes, f"expected at least {args.min_notes} notes, got {total_notes}")

    if args.expected_policy:
        require(
            len(policies) == 1 and args.expected_policy in policies,
            f"expected note print tone policy {args.expected_policy}, got policies={dict(sorted(policies.items()))}",
        )

    mean_contrast = mean(values["contrast"])
    contrast_range = numeric_range(values["contrast"])
    if args.min_mean_contrast is not None:
        require(mean_contrast >= args.min_mean_contrast, f"mean contrast {mean_contrast:.4f} below {args.min_mean_contrast:.4f}")
    if args.max_mean_contrast is not None:
        require(mean_contrast <= args.max_mean_contrast, f"mean contrast {mean_contrast:.4f} above {args.max_mean_contrast:.4f}")
    if args.min_contrast_range is not None:
        require(contrast_range >= args.min_contrast_range, f"contrast range {contrast_range:.4f} below {args.min_contrast_range:.4f}")

    print(
        "ok: WebGL note print tone passed "
        f"({total_notes} notes, policies={dict(sorted(policies.items()))}, "
        f"mean_contrast={mean_contrast:.3f}, contrast_range={contrast_range:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
