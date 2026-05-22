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
    return dataset_root / split_path


def count_split(dataset_root: Path, split_path: str, class_count: int) -> tuple[int, int, Counter[int], list[str]]:
    image_dir = split_root(dataset_root, split_path)
    label_dir = Path(str(image_dir).replace(f"{Path.sep}images{Path.sep}", f"{Path.sep}labels{Path.sep}"))
    images = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    counts: Counter[int] = Counter()
    problems: list[str] = []

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            problems.append(f"Missing label: {label}")
            continue
        for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
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
