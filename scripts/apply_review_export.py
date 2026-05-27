from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_FIELDS = ["review_include", "review_class", "review_notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge an exported CashSnap review CSV back into its source manifest.")
    parser.add_argument("--source", required=True, type=Path, help="Original review manifest CSV.")
    parser.add_argument("--export", required=True, type=Path, help="CSV exported from demo/review/.")
    parser.add_argument("--out", type=Path, default=None, help="Merged output CSV. Required unless --in-place is set.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite --source after merging.")
    parser.add_argument("--key", default="crop_id", help="Primary row key. Falls back to crop_path when blank or missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print merge summary without writing.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def row_key(row: dict[str, str], key: str) -> str:
    value = row.get(key, "").strip()
    if value:
        return f"{key}:{value}"
    fallback = row.get("crop_path", "").strip().replace("\\", "/")
    if fallback:
        return f"crop_path:{fallback}"
    return ""


def selected(row: dict[str, str]) -> bool:
    return bool(row.get("review_include", "").strip())


def main() -> None:
    args = parse_args()
    source = resolve(args.source)
    export = resolve(args.export)
    if args.in_place and args.out:
        raise SystemExit("Use either --in-place or --out, not both.")
    if not args.in_place and not args.out and not args.dry_run:
        raise SystemExit("Pass --out, --in-place, or --dry-run.")

    source_headers, source_rows = read_csv(source)
    export_headers, export_rows = read_csv(export)
    missing_fields = [field for field in REVIEW_FIELDS if field not in export_headers]
    if missing_fields:
        raise SystemExit(f"Export is missing review fields: {', '.join(missing_fields)}")

    export_by_key: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for row in export_rows:
        key = row_key(row, args.key)
        if not key:
            continue
        if key in export_by_key:
            duplicates.add(key)
        export_by_key[key] = row
    if duplicates:
        raise SystemExit(f"Export has duplicate keys, first duplicate: {sorted(duplicates)[0]}")

    matched = 0
    changed = 0
    for row in source_rows:
        key = row_key(row, args.key)
        exported = export_by_key.get(key)
        if exported is None:
            continue
        matched += 1
        for field in REVIEW_FIELDS:
            old_value = row.get(field, "")
            new_value = exported.get(field, "")
            if old_value != new_value:
                changed += 1
                row[field] = new_value

    selected_rows = sum(1 for row in source_rows if selected(row))
    print(
        f"source_rows={len(source_rows)} export_rows={len(export_rows)} matched_rows={matched} "
        f"changed_fields={changed} selected_rows={selected_rows}"
    )

    if args.dry_run:
        return

    out = source if args.in_place else resolve(args.out)
    headers = [*source_headers]
    for field in REVIEW_FIELDS:
        if field not in headers:
            headers.append(field)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(source_rows)
    print(f"wrote {repo_path(out)}")


if __name__ == "__main__":
    main()
