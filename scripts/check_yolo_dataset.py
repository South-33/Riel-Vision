from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO label files for CashSnap.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml", help="YOLO dataset YAML path.")
    return parser.parse_args()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def count_split_dir(dataset_root: Path, split_path: str, class_count: int) -> tuple[int, int, Counter[int], list[str]]:
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


def count_images(images: list[Path], class_count: int) -> tuple[int, int, Counter[int], list[str]]:
    counts: Counter[int] = Counter()
    problems: list[str] = []

    for image in images:
        label = label_path_for_image(image)
        if not label.exists():
            problems.append(f"Missing label: {label}")
            continue
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
            if any(value < 0 or value > 1 for value in values):
                problems.append(f"{label}:{line_no} box values must be normalized 0..1")
            counts[cls] += 1

    return len(images), sum(counts.values()), counts, problems


def count_split(dataset_root: Path, split_paths: str | list[str], class_count: int) -> tuple[int, int, Counter[int], list[str]]:
    paths = split_paths if isinstance(split_paths, list) else [split_paths]
    total_images = 0
    total_boxes = 0
    total_counts: Counter[int] = Counter()
    all_problems: list[str] = []
    for split_path in paths:
        resolved = split_root(dataset_root, split_path)
        if resolved.suffix.lower() == ".txt":
            image_count, box_count, counts, problems = count_images(read_split_list(dataset_root, split_path), class_count)
        else:
            image_count, box_count, counts, problems = count_split_dir(dataset_root, split_path, class_count)
        total_images += image_count
        total_boxes += box_count
        total_counts.update(counts)
        all_problems.extend(problems)
    return total_images, total_boxes, total_counts, all_problems


def main() -> None:
    args = parse_args()
    config_path = Path(args.data)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = (config_path.parent / config["path"]).resolve()
    names = config["names"]
    class_count = len(names)

    for split_key in ["train", "val", "test"]:
        split_path = config.get(split_key)
        if not split_path:
            continue
        image_count, box_count, counts, problems = count_split(dataset_root, split_path, class_count)
        print(f"{split_key}: {image_count} images, {box_count} boxes")
        for class_id, class_name in names.items():
            print(f"  {class_name}: {counts[int(class_id)]}")
        if problems:
            print("  problems:")
            for problem in problems[:50]:
                print(f"    - {problem}")
            if len(problems) > 50:
                print(f"    - ... {len(problems) - 50} more")


if __name__ == "__main__":
    main()
