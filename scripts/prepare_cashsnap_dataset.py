"""
Remap and merge all Roboflow datasets + HF USD dataset into a single
canonical CashSnap v1 dataset under data/cashsnap_v1/.

Output layout (YOLO format):
  data/cashsnap_v1/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    data.yaml
    merge_report.json

Canonical 13-class map (0-indexed):
  0  USD_1       1  USD_5      2  USD_10    3  USD_20
  4  USD_50      5  USD_100
  6  KHR_500     7  KHR_1000   8  KHR_2000  9  KHR_5000
  10 KHR_10000  11 KHR_20000  12 KHR_50000

Usage:
    python scripts/prepare_cashsnap_dataset.py
    python scripts/prepare_cashsnap_dataset.py --dry-run   # just print report
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# -- Canonical class map --------------------------------------------------------
CANONICAL: dict[str, int] = {
    "USD_1": 0,
    "USD_5": 1,
    "USD_10": 2,
    "USD_20": 3,
    "USD_50": 4,
    "USD_100": 5,
    "KHR_500": 6,
    "KHR_1000": 7,
    "KHR_2000": 8,
    "KHR_5000": 9,
    "KHR_10000": 10,
    "KHR_20000": 11,
    "KHR_50000": 12,
}
ID_TO_NAME = {v: k for k, v in CANONICAL.items()}

# -- Per-source class remaps ----------------------------------------------------
# Map each source's raw class name -> canonical name (or None to drop)

# Khmer-US-currency v3: 11 classes
# Missing USD_5 (no '5-us' class in this dataset - covered by HF USD)
REMAP_KHMER_US = {
    "1-us":       "USD_1",
    "10-us":      "USD_10",
    "100-us":     "USD_100",
    "20-us":      "USD_20",
    "50-us":      "USD_50",
    # USD_5 not present in this dataset
    "100-riel":   None,          # KHR_100 - deferred from v1 scope
    "500-riel":   "KHR_500",
    "1000-riel":  "KHR_1000",
    "2000-riel":  "KHR_2000",
    "5000-riel":  "KHR_5000",
    "10000-riel": "KHR_10000",
    "20000-riel": "KHR_20000",
    # Note: this dataset's research summary listed 50000-riel but it does
    # not appear in the v3 yaml. Covered by Cambodia Currency Project.
}

# Cambodia Currency Project v2: 7 classes (all KHR)
REMAP_CAMBODIA = {
    "100_Riel":   None,          # KHR_100 - deferred
    "500_Riel":   "KHR_500",
    "1000_Riel":  "KHR_1000",
    "5000_Riel":  "KHR_5000",
    "10000_Riel": "KHR_10000",
    "20000_Riel": "KHR_20000",
    "50000_Riel": "KHR_50000",
}

# KHMER SCAN v1: 8 classes (all KHR, no objects class in this version)
REMAP_KHMER_SCAN = {
    "100_Riels":   None,          # KHR_100 - deferred
    "500_Riels":   "KHR_500",
    "1000_Riels":  "KHR_1000",
    "2000_Riels":  "KHR_2000",    # Only source of KHR_2000!
    "5000_Riels":  "KHR_5000",
    "10000_Riels": "KHR_10000",
    "20000_Riels": "KHR_20000",
    "50000_Riels": "KHR_50000",
}

# CashCountingXL v2: 34 classes
REMAP_CASHCOUNTINGXL = {
    "100USD": "USD_100", "100USD-Back": "USD_100", "100USD-Front": "USD_100",
    "10USD": "USD_10", "10USD-Back": "USD_10", "10USD-Front": "USD_10",
    "1USD": "USD_1", "1USD-Back": "USD_1", "1USD-Front": "USD_1",
    "20USD": "USD_20", "20USD-Back": "USD_20", "20USD-Front": "USD_20",
    "2USD": None,
    "50USD": "USD_50", "50USD-Back": "USD_50", "50USD-Front": "USD_50",
    "5USD": "USD_5", "5USD-Back": "USD_5", "5USD-Front": "USD_5",
}

# USD Total v1: 7 classes
REMAP_USD_TOTAL = {
    "1-dolar - v1 2024-10-13 8-11am": "USD_1",
    "50Dollar - v1 2024-10-20 10-48am": "USD_50",
    "USA 10 -Reviewed - v1 2024-10-13 8-20am": "USD_10",
    "USA 2 - Reviewed - v1 2024-10-20 10-26am": None,
    "USA 20 - Reviewed - v1 2024-10-20 10-41am": "USD_20",
    "USA 5 - v1 2024-10-20 10-30am": "USD_5",
    "Wilter- US dollar 100 - v1 2024-10-20 11-02am": "USD_100",
}

# BillsBank v1: 6 classes
REMAP_BILLSBANK = {
    "1Dollar": "USD_1",
    "5Dollar": "USD_5",
    "10Dollar": "USD_10",
    "20Dollar": "USD_20",
    "50Dollar": "USD_50",
    "100Dollar": "USD_100",
}

# Asian Currency Detection v1: 15 classes
REMAP_ASIAN_CURRENCY = {
    "cambodian riel - 100": None,
    "cambodian riel - 1-000": "KHR_1000",
    "cambodian riel - 5-000": "KHR_5000",
}

# HF USD Side Detection Dataset (already COCO, processed separately by prepare_hf_usd_yolo.py)
# Canonical class IDs directly (0=USD_1 .. 5=USD_100) - no remap needed here.

# -- Source definitions ---------------------------------------------------------
# Each entry: (source_dir, split_map, class_list, remap_dict)
# split_map maps source split folder name -> canonical split name
SOURCES = [
    {
        "name": "khmer_us_currency",
        "dir": DATA / "raw_datasets" / "roboflow_khmer_us_currency",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": True,
        "remap": REMAP_KHMER_US,
    },
    {
        "name": "cambodia_currency_project",
        "dir": DATA / "raw_datasets" / "roboflow_cambodia_currency_project",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": True,
        "remap": REMAP_CAMBODIA,
    },
    {
        "name": "khmer_scan",
        "dir": DATA / "raw_datasets" / "roboflow_khmer_scan",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": True,
        "remap": REMAP_KHMER_SCAN,
    },
    {
        "name": "hf_usd_side",
        "dir": DATA / "processed" / "hf_usd_side_yolo_canonical",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": False,   # already canonical 0-5
        "remap": None,
    },
    {
        "name": "cashcountingxl",
        "dir": DATA / "raw_datasets" / "roboflow_cashcountingxl",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": True,
        "remap": REMAP_CASHCOUNTINGXL,
    },
    {
        "name": "usd_total",
        "dir": DATA / "raw_datasets" / "roboflow_usd_total",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": True,
        "remap": REMAP_USD_TOTAL,
    },
    {
        "name": "billsbank",
        "dir": DATA / "raw_datasets" / "roboflow_billsbank",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": True,
        "remap": REMAP_BILLSBANK,
    },
    {
        "name": "asian_currency",
        "dir": DATA / "raw_datasets" / "roboflow_asian_currency_detection",
        "splits": {"train": "train", "valid": "val", "test": "test"},
        "class_list_from_yaml": True,
        "remap": REMAP_ASIAN_CURRENCY,
    },
]

OUT_DIR = DATA / "cashsnap_v1"


# -- Helpers --------------------------------------------------------------------

def load_duplicates_to_exclude(csv_path: Path) -> set[str]:
    """
    Read the exact_duplicates.csv report. For each hash group, keep the first
    file (preferring certain datasets) and add all others to the exclude set.
    """
    if not csv_path.exists():
        return set()

    hash_groups = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["cross_dataset"] == "True":
                # Ensure path separator matches local OS
                norm_path = Path(row["file"]).resolve().as_posix()
                hash_groups[row["hash"]].append((row["dataset"], norm_path))

    exclude = set()

    # Priority for which dataset gets to keep the file if there's a conflict
    # (lower index = higher priority)
    priority = [
        "hf_usd_side", "khmer_us_currency", "cambodia_currency_project",
        "khmer_scan", "billsbank", "usd_total", "cashcountingxl", "asian_currency"
    ]
    def get_priority(ds: str) -> int:
        try:
            return priority.index(ds)
        except ValueError:
            return 999

    for h, files in hash_groups.items():
        # Sort by dataset priority, then alphabetically to be deterministic
        files.sort(key=lambda x: (get_priority(x[0]), x[1]))
        # The first file is kept, the rest are excluded
        keeper = files[0]
        for ds, path in files[1:]:
            exclude.add(path)

    return exclude


def read_yaml_classes(source_dir: Path) -> list[str]:
    """Return ordered class list from data.yaml in source_dir."""
    for yaml_path in source_dir.rglob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        names: list[str] = []
        in_names = False
        for line in text.splitlines():
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                stripped = line.strip()
                if stripped.startswith("-"):
                    names.append(stripped.lstrip("- ").strip())
                elif stripped and not stripped.startswith("#"):
                    break
        if names:
            return names
    return []


def remap_label_file(
    src_label: Path,
    source_classes: list[str],
    remap: dict[str, str | None] | None,
) -> list[str]:
    """
    Read a YOLO label file and return remapped lines using canonical class IDs.
    Lines for dropped or unknown classes are omitted.
    If remap is None, treat class IDs as already canonical.
    """
    lines_out: list[str] = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        coords = parts[1:]
        if len(coords) != 4:
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            try:
                values = [float(value) for value in coords]
            except ValueError:
                continue
            xs = values[0::2]
            ys = values[1::2]
            x_min = max(0.0, min(xs))
            x_max = min(1.0, max(xs))
            y_min = max(0.0, min(ys))
            y_max = min(1.0, max(ys))
            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                continue
            coords = [
                f"{x_min + width / 2:.6f}",
                f"{y_min + height / 2:.6f}",
                f"{width:.6f}",
                f"{height:.6f}",
            ]

        if remap is None:
            # Already canonical - pass through if valid
            if class_id in ID_TO_NAME:
                lines_out.append(f"{class_id} {' '.join(coords)}")
            continue

        if class_id >= len(source_classes):
            continue
        raw_name = source_classes[class_id]
        canonical_name = remap.get(raw_name)
        if canonical_name is None:
            continue  # drop (deferred class or unknown)
        canonical_id = CANONICAL[canonical_name]
        lines_out.append(f"{canonical_id} {' '.join(coords)}")
    return lines_out


def copy_split(
    source_cfg: dict,
    split_src: str,
    split_dst: str,
    out_dir: Path,
    dry_run: bool,
    stats: dict,
    exclude_set: set[str],
) -> None:
    src_dir = source_cfg["dir"]
    remap = source_cfg["remap"]
    source_name = source_cfg["name"]

    img_src = src_dir / split_src / "images"
    lbl_src = src_dir / split_src / "labels"

    # Some Roboflow exports put images directly under the split folder
    if not img_src.exists():
        img_src = src_dir / split_src
    if not img_src.exists():
        img_src = src_dir / "images" / split_src
        lbl_src = src_dir / "labels" / split_src

    if not img_src.exists():
        print(f"  [SKIP] {source_name}/{split_src}: images dir not found at {img_src}")
        return

    source_classes = read_yaml_classes(src_dir) if source_cfg["class_list_from_yaml"] else []

    img_out = out_dir / "images" / split_dst
    lbl_out = out_dir / "labels" / split_dst
    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    dropped_labels = 0
    skipped_duplicates = 0
    boxes_by_class: dict[int, int] = defaultdict(int)

    for img_path in sorted(img_src.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        # Check against exclusion list (compare as posix paths)
        if img_path.resolve().as_posix() in exclude_set:
            skipped_duplicates += 1
            continue

        # Find corresponding label
        label_path = lbl_src / f"{img_path.stem}.txt" if lbl_src.exists() else None
        if label_path is None or not label_path.exists():
            # Try alongside the image
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                label_path = None

        remapped_lines: list[str] = []
        if label_path:
            remapped_lines = remap_label_file(label_path, source_classes, remap)

        # Prefix filename with source name to avoid collisions
        out_stem = f"{source_name}_{img_path.stem}"
        dst_img = img_out / f"{out_stem}{img_path.suffix}"
        dst_lbl = lbl_out / f"{out_stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")

        copied += 1
        if not remapped_lines:
            dropped_labels += 1
        for line in remapped_lines:
            cid = int(line.split()[0])
            boxes_by_class[cid] += 1

    key = f"{source_name}/{split_src}->{split_dst}"
    stats[key] = {
        "images": copied,
        "images_with_no_boxes": dropped_labels,
        "skipped_duplicates": skipped_duplicates,
        "boxes_by_class": {ID_TO_NAME[k]: v for k, v in sorted(boxes_by_class.items())},
    }
    print(f"  {key}: {copied} images, {sum(boxes_by_class.values())} boxes, {dropped_labels} empty labels, {skipped_duplicates} dups skipped")


def write_data_yaml(out_dir: Path) -> None:
    names_block = "\n".join(f"  {i}: {name}" for i, name in sorted(ID_TO_NAME.items()))
    yaml_text = f"""# CashSnap v1 merged dataset
path: {out_dir.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CANONICAL)}
names:
{names_block}
"""
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def class_balance_report(out_dir: Path) -> dict[str, int]:
    """Count total boxes per canonical class across all label files."""
    counts: dict[int, int] = defaultdict(int)
    for lbl in (out_dir / "labels").rglob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    return {ID_TO_NAME[k]: v for k, v in sorted(counts.items())}


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CashSnap datasets into canonical YOLO format.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing files.")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    dup_csv = DATA / "dedup" / "exact_duplicates.csv"
    exclude_set = load_duplicates_to_exclude(dup_csv)
    if exclude_set:
        print(f"Loaded {len(exclude_set)} cross-dataset duplicate files to exclude.\n")

    stats: dict = {}

    for source_cfg in SOURCES:
        src_dir = source_cfg["dir"]
        if not src_dir.exists():
            print(f"\n[MISSING] {source_cfg['name']}: {src_dir} - skipping")
            continue

        print(f"\n--- {source_cfg['name']} ---")
        if source_cfg["class_list_from_yaml"]:
            classes = read_yaml_classes(src_dir)
            print(f"  Source classes: {classes}")

        for split_src, split_dst in source_cfg["splits"].items():
            copy_split(source_cfg, split_src, split_dst, OUT_DIR, args.dry_run, stats, exclude_set)

    if not args.dry_run:
        write_data_yaml(OUT_DIR)
        balance = class_balance_report(OUT_DIR)
        report = {"sources": stats, "class_balance_all_splits": balance}
        (OUT_DIR / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n\n=== Class balance (all splits) ===")
        total = sum(balance.values())
        for cls, count in balance.items():
            bar = "#" * (count * 40 // max(balance.values()))
            print(f"  {cls:<12} {count:>5}  {bar}")
        print(f"  {'TOTAL':<12} {total:>5}")
        print(f"\nDataset written to: {OUT_DIR}")
        print(f"Merge report:       {OUT_DIR / 'merge_report.json'}")
        print("\nNext step: python scripts/check_yolo_dataset.py")
    else:
        print("\n[DRY RUN complete - no files written]")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
