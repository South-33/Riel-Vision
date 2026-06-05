from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO label files for CashSnap.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--json-out", default=None, help="Optional machine-readable summary output.")
    parser.add_argument(
        "--min-train-class-images",
        type=int,
        default=None,
        help="Require at least this many unique train images for every class.",
    )
    parser.add_argument(
        "--min-train-class-boxes",
        type=int,
        default=None,
        help="Require at least this many train boxes for every class.",
    )
    parser.add_argument("--fail-on-problems", action="store_true", help="Exit non-zero if label problems are found.")
    return parser.parse_args()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_summary() -> dict:
    return {
        "images": 0,
        "unique_images": 0,
        "duplicate_image_rows": 0,
        "background_images": 0,
        "unique_background_images": 0,
        "boxes": 0,
        "boxes_by_class": Counter(),
        "class_images": Counter(),
        "unique_class_images": Counter(),
        "problems": [],
    }


def count_split_dir(dataset_root: Path, split_path: str, class_count: int) -> dict:
    image_dir = split_root(dataset_root, split_path)
    images = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    return count_images(images, class_count)


def read_split_list(dataset_root: Path, split_path: str) -> list[Path]:
    list_path = split_root(dataset_root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        images.append(path if path.is_absolute() else dataset_root / path)
    return images


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def count_images(images: list[Path], class_count: int) -> dict:
    summary = empty_summary()
    counts: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    unique_class_images: Counter[int] = Counter()
    problems: list[str] = []
    seen_images: set[str] = set()

    for image in images:
        image_key = str(image.resolve())
        first_seen = image_key not in seen_images
        seen_images.add(image_key)
        label = label_path_for_image(image)
        if not label.exists():
            problems.append(f"Missing label: {label}")
            continue
        image_classes: set[int] = set()
        for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{label}:{line_no} expected 5 YOLO fields, found {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                problems.append(f"{label}:{line_no} contains non-numeric fields")
                continue
            if cls < 0 or cls >= class_count:
                problems.append(f"{label}:{line_no} class {cls} outside 0..{class_count - 1}")
            else:
                image_classes.add(cls)
            if any(value < 0 or value > 1 for value in values):
                problems.append(f"{label}:{line_no} box values must be normalized 0..1")
            counts[cls] += 1
        if image_classes:
            for cls in image_classes:
                class_images[cls] += 1
                if first_seen:
                    unique_class_images[cls] += 1
        else:
            summary["background_images"] += 1
            if first_seen:
                summary["unique_background_images"] += 1

    summary.update(
        {
            "images": len(images),
            "unique_images": len(seen_images),
            "duplicate_image_rows": len(images) - len(seen_images),
            "boxes": sum(counts.values()),
            "boxes_by_class": counts,
            "class_images": class_images,
            "unique_class_images": unique_class_images,
            "problems": problems,
        }
    )
    return summary


def count_split(dataset_root: Path, split_paths: str | list[str], class_count: int) -> dict:
    paths = split_paths if isinstance(split_paths, list) else [split_paths]
    total = empty_summary()
    for split_path in paths:
        resolved = split_root(dataset_root, split_path)
        if resolved.suffix.lower() == ".txt":
            summary = count_images(read_split_list(dataset_root, split_path), class_count)
        else:
            summary = count_split_dir(dataset_root, split_path, class_count)
        total["images"] += summary["images"]
        total["unique_images"] += summary["unique_images"]
        total["duplicate_image_rows"] += summary["duplicate_image_rows"]
        total["background_images"] += summary["background_images"]
        total["unique_background_images"] += summary["unique_background_images"]
        total["boxes"] += summary["boxes"]
        total["boxes_by_class"].update(summary["boxes_by_class"])
        total["class_images"].update(summary["class_images"])
        total["unique_class_images"].update(summary["unique_class_images"])
        total["problems"].extend(summary["problems"])
    return total


def split_source_records(dataset_root: Path, split_paths: str | list[str]) -> list[dict]:
    paths = split_paths if isinstance(split_paths, list) else [split_paths]
    records: list[dict] = []
    for split_path in paths:
        resolved = split_root(dataset_root, split_path)
        record: dict = {"path": repo_rel(resolved)}
        if resolved.suffix.lower() == ".txt":
            record["kind"] = "list"
            record["sha256"] = file_sha256(resolved)
        else:
            image_rows = sorted(
                repo_rel(path)
                for path in resolved.glob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            digest = hashlib.sha256()
            for row in image_rows:
                digest.update(row.encode("utf-8"))
                digest.update(b"\n")
            record["kind"] = "directory"
            record["image_count"] = len(image_rows)
            record["listing_sha256"] = digest.hexdigest()
        records.append(record)
    return records


def class_summary(names: dict, summary: dict) -> dict[str, dict[str, int]]:
    return {
        str(class_name): {
            "boxes": int(summary["boxes_by_class"][int(class_id)]),
            "image_rows": int(summary["class_images"][int(class_id)]),
            "unique_images": int(summary["unique_class_images"][int(class_id)]),
        }
        for class_id, class_name in names.items()
    }


def class_label(names: dict, class_id: int) -> str:
    return str(names.get(class_id, names.get(str(class_id), class_id)))


def json_ready_summary(names: dict, summary: dict) -> dict:
    return {
        "images": int(summary["images"]),
        "unique_images": int(summary["unique_images"]),
        "duplicate_image_rows": int(summary["duplicate_image_rows"]),
        "background_images": int(summary["background_images"]),
        "unique_background_images": int(summary["unique_background_images"]),
        "boxes": int(summary["boxes"]),
        "classes": class_summary(names, summary),
        "problems": list(summary["problems"]),
    }


def write_json(path_value: str, payload: dict) -> None:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.data)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = (config_path.parent / config["path"]).resolve()
    names = config["names"]
    class_count = len(names)
    split_summaries: dict[str, dict] = {}
    checks: list[dict] = []

    for split_key in ["train", "val", "test"]:
        split_path = config.get(split_key)
        if not split_path:
            continue
        summary = count_split(dataset_root, split_path, class_count)
        split_summaries[split_key] = summary
        print(
            f"{split_key}: {summary['images']} images, {summary['unique_images']} unique, "
            f"{summary['duplicate_image_rows']} duplicate rows, {summary['background_images']} backgrounds, "
            f"{summary['boxes']} boxes"
        )
        for class_id, class_name in names.items():
            class_index = int(class_id)
            print(
                f"  {class_name}: boxes={summary['boxes_by_class'][class_index]} "
                f"image_rows={summary['class_images'][class_index]} "
                f"unique_images={summary['unique_class_images'][class_index]}"
            )
        if summary["problems"]:
            print("  problems:")
            for problem in summary["problems"][:50]:
                print(f"    - {problem}")
            if len(summary["problems"]) > 50:
                print(f"    - ... {len(summary['problems']) - 50} more")

    train_summary = split_summaries.get("train")
    if train_summary and args.min_train_class_images is not None:
        failing = [
            class_label(names, class_id)
            for class_id in range(class_count)
            if train_summary["unique_class_images"][class_id] < args.min_train_class_images
        ]
        checks.append(
            {
                "name": "min_train_class_images",
                "passed": not failing,
                "threshold": args.min_train_class_images,
                "failed_classes": failing,
            }
        )
    if train_summary and args.min_train_class_boxes is not None:
        failing = [
            class_label(names, class_id)
            for class_id in range(class_count)
            if train_summary["boxes_by_class"][class_id] < args.min_train_class_boxes
        ]
        checks.append(
            {
                "name": "min_train_class_boxes",
                "passed": not failing,
                "threshold": args.min_train_class_boxes,
                "failed_classes": failing,
            }
        )
    problem_count = sum(len(summary["problems"]) for summary in split_summaries.values())
    if args.fail_on_problems:
        checks.append({"name": "label_problems", "passed": problem_count == 0, "problem_count": problem_count})
    passed = all(check["passed"] for check in checks)

    if args.json_out:
        payload = {
            "data": repo_rel(config_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "data_config_sha256": file_sha256(config_path),
            "dataset_root": repo_rel(dataset_root),
            "passed": passed,
            "checks": checks,
            "split_sources": {
                split_key: split_source_records(dataset_root, config[split_key])
                for split_key in split_summaries
            },
            "splits": {
                split_key: json_ready_summary(names, summary)
                for split_key, summary in split_summaries.items()
            },
        }
        write_json(args.json_out, payload)
        print(f"wrote_json={args.json_out}")
    if checks:
        verdict = "PASS" if passed else "FAIL"
        print(f"{verdict}: {len(checks)} check(s)")
        for check in checks:
            if not check["passed"]:
                print(f"  - {check['name']}: {', '.join(check.get('failed_classes', [])) or 'failed'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
