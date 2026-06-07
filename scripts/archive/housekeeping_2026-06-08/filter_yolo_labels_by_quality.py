from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUALITY = ROOT / "manifests" / "real_fan_benchmark_label_quality.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter YOLO labels to boxes marked fair-to-score in a quality manifest.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quality-manifest", default=str(DEFAULT_QUALITY))
    parser.add_argument("--include-quality", default="clear,partial_clear")
    parser.add_argument("--require-count-for-score", action="store_true", default=True)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "score", "keep"}


def main() -> None:
    args = parse_args()
    labels = resolve(args.labels)
    out = resolve(args.out)
    include_quality = {value.strip() for value in args.include_quality.replace(";", ",").split(",") if value.strip()}
    quality_rows = list(csv.DictReader(resolve(args.quality_manifest).open("r", newline="", encoding="utf-8")))
    key_path = repo_path(labels)
    keep_indices: set[int] = set()
    matched_quality_rows = 0
    for row in quality_rows:
        if row.get("label_path", "").replace("\\", "/") != key_path:
            continue
        matched_quality_rows += 1
        if row.get("quality", "").strip() not in include_quality:
            continue
        if args.require_count_for_score and not truthy(row.get("count_for_score", "")):
            continue
        keep_indices.add(int(row["label_index"]))

    source_lines = labels.read_text(encoding="utf-8").splitlines()
    if matched_quality_rows and max(keep_indices, default=-1) >= len(source_lines):
        raise SystemExit(f"Quality manifest references label index outside {key_path}")
    if not matched_quality_rows:
        raise SystemExit(f"No quality rows found for {key_path}")

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = [line for index, line in enumerate(source_lines) if index in keep_indices and line.strip()]
    out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"wrote {len(kept)}/{len(source_lines)} labels to {repo_path(out)}")


if __name__ == "__main__":
    main()
