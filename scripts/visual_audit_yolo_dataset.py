from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets for YOLO dataset visual QA.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml")
    parser.add_argument("--out", default="data/audit")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33)
    return parser.parse_args()


def load_config(path: Path) -> tuple[Path, dict[int, str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = (path.parent / config["path"]).resolve()
    names = {int(key): value for key, value in config["names"].items()}
    return root, names


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def collect_samples(root: Path, names: dict[int, str]) -> dict[int, list[tuple[Path, list[float]]]]:
    samples: dict[int, list[tuple[Path, list[float]]]] = defaultdict(list)
    for split in ["train", "val", "test"]:
        label_dir = root / "labels" / split
        image_dir = root / "images" / split
        for label_path in label_dir.glob("*.txt"):
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                if class_id not in names:
                    continue
                samples[class_id].append((image_path, [float(value) for value in parts[1:]]))
    return samples


def draw_crop(image_path: Path, box: list[float], label: str) -> Image.Image:
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        x_center, y_center, box_w, box_h = box
        x1 = int((x_center - box_w / 2) * width)
        y1 = int((y_center - box_h / 2) * height)
        x2 = int((x_center + box_w / 2) * width)
        y2 = int((y_center + box_h / 2) * height)
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=max(2, width // 250))
        draw.rectangle([0, 0, min(width, 340), 28], fill="white")
        draw.text((4, 6), f"{label} | {image_path.name[:32]}", fill="black")
        image.thumbnail((260, 180))
        return image.copy()


def make_sheet(class_id: int, label: str, chosen: list[tuple[Path, list[float]]], out_dir: Path) -> None:
    thumbs = [draw_crop(image_path, box, label) for image_path, box in chosen]
    if not thumbs:
        return
    cols = 4
    cell_w, cell_h = 260, 180
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(thumb, (x, y))
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(out_dir / f"{class_id:02d}_{label}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    root, names = load_config(Path(args.data))
    samples = collect_samples(root, names)
    out_dir = Path(args.out)
    for class_id, label in names.items():
        available = samples.get(class_id, [])
        chosen = random.sample(available, min(args.per_class, len(available)))
        make_sheet(class_id, label, chosen, out_dir)
        print(f"{label}: {len(available)} available, wrote {len(chosen)}")


if __name__ == "__main__":
    main()
