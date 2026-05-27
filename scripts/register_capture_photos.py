from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "manifests" / "real_partial_capture_inventory.csv"
FIELDNAMES = [
    "image_id",
    "local_path",
    "scene_type",
    "rights_status",
    "source_credit",
    "train_allowed",
    "label_status",
    "denominations",
    "notes",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a folder of CashSnap real partial-note captures.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder containing phone photos to register.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--scene-type", required=True, help="Scene bucket, e.g. single_khr, simple_overlap, hand_fan.")
    parser.add_argument(
        "--rights-status",
        default="own_photo",
        help="Rights marker for every row. Usable values include own_photo, rights_clear, public_domain, cc0.",
    )
    parser.add_argument("--source-credit", default="self", help="Credit/source value for every row.")
    parser.add_argument("--train-allowed", default="yes", choices=["yes", "no"], help="Whether rows may be used for training.")
    parser.add_argument("--label-status", default="unlabeled", help="Initial label status for every row.")
    parser.add_argument("--denominations", default="", help="Optional shared denomination list, e.g. KHR_5000;KHR_10000.")
    parser.add_argument("--notes", default="", help="Optional shared notes for every row.")
    parser.add_argument("--prefix", default="", help="Optional image_id prefix, e.g. capture_20260527.")
    parser.add_argument("--recursive", action="store_true", help="Scan images-dir recursively.")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without modifying the inventory.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "capture"


def iter_images(images_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in images_dir.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique_id(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def make_rows(args: argparse.Namespace, images: list[Path], existing: list[dict[str, str]]) -> list[dict[str, str]]:
    used_ids = {row.get("image_id", "") for row in existing}
    existing_paths = {row.get("local_path", "") for row in existing}
    rows: list[dict[str, str]] = []
    prefix = slug(args.prefix) if args.prefix else ""
    for image in images:
        local_path = repo_path(image)
        if local_path in existing_paths:
            continue
        base = slug(f"{prefix}_{image.stem}" if prefix else image.stem)
        rows.append(
            {
                "image_id": unique_id(base, used_ids),
                "local_path": local_path,
                "scene_type": args.scene_type,
                "rights_status": args.rights_status,
                "source_credit": args.source_credit,
                "train_allowed": args.train_allowed,
                "label_status": args.label_status,
                "denominations": args.denominations,
                "notes": args.notes,
            }
        )
    return rows


def write_inventory(path: Path, existing: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in [*existing, *rows]:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> None:
    args = parse_args()
    images_dir = resolve(args.images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {args.images_dir}")
    inventory = resolve(args.inventory)
    existing = read_existing(inventory)
    rows = make_rows(args, iter_images(images_dir, args.recursive), existing)
    print(f"new_rows={len(rows)} inventory={repo_path(inventory)}")
    for row in rows:
        print(",".join(row[field] for field in FIELDNAMES))
    if not args.dry_run:
        write_inventory(inventory, existing, rows)


if __name__ == "__main__":
    main()
