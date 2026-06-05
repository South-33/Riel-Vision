from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "runs" / "cashsnap" / "mined_real_scoreable_dataset_latest" / "manifest.csv"
DEFAULT_OUT = ROOT / "runs" / "cashsnap" / "mined_real_browser_cases_latest.csv"
FIELDNAMES = [
    "case_id",
    "image",
    "labels",
    "min_same_class",
    "min_any_class",
    "max_count_error",
    "max_khr_error",
    "max_usd_error",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build browser-smoke cases from the mined-real scoreable manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gate-perfect", action="store_true", help="Write strict count/value/same-class gates per case.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def browser_path(path_text: str) -> str:
    return "/" + path_text.replace("\\", "/").lstrip("/")


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    rows = read_rows(args.manifest)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            boxes = row.get("boxes", "").strip()
            writer.writerow(
                {
                    "case_id": row["image_id"],
                    "image": browser_path(row["image_path"]),
                    "labels": row["label_path"].replace("\\", "/"),
                    "min_same_class": boxes if args.gate_perfect else "",
                    "min_any_class": boxes if args.gate_perfect else "",
                    "max_count_error": "0" if args.gate_perfect else "",
                    "max_khr_error": "0" if args.gate_perfect else "",
                    "max_usd_error": "0" if args.gate_perfect else "",
                    "notes": (
                        f"diagnostic mined real {row.get('benchmark_role', '')} "
                        f"source_split={row.get('source_split', '')} boxes={boxes}"
                    ),
                }
            )
    print(f"wrote {len(rows)} mined-real browser case(s) to {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
