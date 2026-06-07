from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    ROOT / "data" / "review" / "real_fan_candidate_proposal_review_v1" / "review.csv",
    ROOT / "data" / "review" / "roboflow_cuurecy_detection_is_oldcommon_highconf_failure_review_v1" / "manifest.csv",
    ROOT / "data" / "review" / "roboflow_cuurecy_detection_is_khr_5k_10k_partial_review_v1" / "manifest.csv",
    ROOT / "data" / "review" / "roboflow_cuurecy_detection_is_khr_20k_50k_partial_review_v1" / "manifest.csv",
    ROOT / "data" / "review" / "cashsnap_p1_oldcommon_partial_focus_review_v1" / "manifest.csv",
    ROOT / "data" / "review" / "p1_focus_v2_oldcommon_failure_review_v1" / "manifest.csv",
    ROOT / "data" / "review" / "cashsnap_old_common_khr_crop_review_v1" / "manifest.csv",
]
SELECTED_VALUES = {"1", "true", "yes", "y", "include", "included", "selected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CashSnap crop-review manifests.")
    parser.add_argument("manifest", nargs="*", type=Path, help="Review manifest CSVs. Defaults to the main curation packs.")
    parser.add_argument("--top", type=int, default=8, help="Maximum rows per grouped summary.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected(row: dict[str, str]) -> bool:
    return row.get("review_include", "").strip().lower() in SELECTED_VALUES


def first_value(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return "<blank>"


def summarize_counter(label: str, counter: Counter[str], top: int) -> None:
    if not counter:
        return
    values = ", ".join(f"{key}={count}" for key, count in counter.most_common(top))
    print(f"  {label}: {values}")


def summarize_manifest(path: Path, top: int) -> None:
    rows = read_rows(path)
    included = sum(1 for row in rows if selected(row))
    blanks = sum(1 for row in rows if not row.get("review_include", "").strip())
    crop_rows = [row for row in rows if row.get("crop_path", "").strip()]
    missing_crops = sum(1 for row in crop_rows if not resolve(Path(row["crop_path"])).exists())
    print(f"{repo_path(path)}")
    print(f"  rows={len(rows)} included={included} blank_review_include={blanks}")
    if crop_rows:
        print(f"  crop_paths={len(crop_rows)} missing_crop_paths={missing_crops}")
    summarize_counter(
        "classes",
        Counter(first_value(row, ["review_class", "target", "canonical_class", "class_name", "fragment_class", "detector_class"]) for row in rows),
        top,
    )
    summarize_counter("failure_pairs", Counter(row.get("failure_pair", "").strip() for row in rows if row.get("failure_pair", "").strip()), top)
    summarize_counter("sides", Counter(row.get("side", "").strip() for row in rows if row.get("side", "").strip()), top)
    summarize_counter("splits", Counter(row.get("split", "").strip() for row in rows if row.get("split", "").strip()), top)


def main() -> None:
    args = parse_args()
    manifests = [resolve(path) for path in args.manifest] if args.manifest else DEFAULT_MANIFESTS
    missing = [path for path in manifests if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing: {repo_path(path)}")
        raise SystemExit(1)
    for index, path in enumerate(manifests):
        if index:
            print()
        summarize_manifest(path, args.top)


if __name__ == "__main__":
    main()
