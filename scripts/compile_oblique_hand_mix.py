#!/usr/bin/env python
"""Compile the oblique-fan champion mix plus hard oblique hand-occlusion dose."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]

BASE_OBLIQUE_FAN_TXT = (
    ROOT
    / "configs"
    / "generated_lists"
    / "webgl_ablation"
    / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_train.txt"
)
HAND_OCCLUSION_DIR = (
    ROOT
    / "data"
    / "synthetic"
    / "cashsnap_webgl_hard_oblique_hand_occlusion_candidate_v1"
    / "images"
    / "train"
)
OUT_TXT = (
    ROOT
    / "configs"
    / "generated_lists"
    / "webgl_ablation"
    / "cashsnap_production_pilot_v16_scaled2x_oblique_fan_handocc_train.txt"
)
ZERO_LABEL_MONEY_LIST = (
    ROOT
    / "configs"
    / "generated_lists"
    / "audit"
    / "cashsnap_zero_label_money_train_hardneg_broad240_v1.txt"
)
TRUE_EMPTY_LIST = (
    ROOT
    / "runs"
    / "cashsnap"
    / "empty_label_semantic_bridge_train_v1"
    / "lists"
    / "likely_true_empty.txt"
)

CLASS_NAMES = [
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]


def gather_images(directory: Path) -> list[str]:
    return [
        path.relative_to(ROOT).as_posix()
        for path in sorted(directory.glob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]


def read_image_list(path: Path) -> list[str]:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise SystemExit(f"missing image list: {resolved.relative_to(ROOT)}")
    return [
        line.strip().replace("\\", "/")
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def bounded_rows(rows: list[str], *, limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(rows):
        return rows
    sampled = list(rows)
    random.Random(seed).shuffle(sampled)
    return sorted(sampled[:limit])


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_class_ids(image: str) -> list[int]:
    label_path = ROOT / label_path_for_image(image)
    if not label_path.exists():
        return []
    class_ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            class_ids.append(int(float(line.split()[0])))
        except (IndexError, ValueError):
            continue
    return class_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=2, help="Repeat count for the hard hand-occlusion images.")
    parser.add_argument("--out", type=Path, default=OUT_TXT, help="Output train-list path.")
    parser.add_argument("--seed", type=int, default=20260618, help="Seed for bounded negative-row sampling.")
    parser.add_argument(
        "--zero-label-money-list",
        type=Path,
        default=ZERO_LABEL_MONEY_LIST,
        help="Train-side zero-label unknown-money list to add when --zero-label-money-repeat > 0.",
    )
    parser.add_argument(
        "--zero-label-money-max",
        type=int,
        default=0,
        help="Maximum unique zero-label money rows to add. 0 means add none; negative means all rows.",
    )
    parser.add_argument(
        "--zero-label-money-repeat",
        type=int,
        default=0,
        help="Repeat count for selected zero-label money rows.",
    )
    parser.add_argument(
        "--true-empty-list",
        type=Path,
        default=TRUE_EMPTY_LIST,
        help="Train-side likely true-empty list to add when --true-empty-repeat > 0.",
    )
    parser.add_argument(
        "--true-empty-max",
        type=int,
        default=0,
        help="Maximum unique true-empty rows to add. 0 means add none; negative means all rows.",
    )
    parser.add_argument(
        "--true-empty-repeat",
        type=int,
        default=0,
        help="Repeat count for selected true-empty rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    for name in ("zero_label_money_repeat", "true_empty_repeat"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")

    if not BASE_OBLIQUE_FAN_TXT.exists():
        raise SystemExit(f"missing base oblique fan list: {BASE_OBLIQUE_FAN_TXT.relative_to(ROOT)}")
    if not HAND_OCCLUSION_DIR.exists():
        raise SystemExit(f"missing hard hand-occlusion dir: {HAND_OCCLUSION_DIR.relative_to(ROOT)}")

    base_lines = [
        line.strip()
        for line in BASE_OBLIQUE_FAN_TXT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hand_images = gather_images(HAND_OCCLUSION_DIR)
    repeated_hand = hand_images * args.repeat
    zero_label_money_rows: list[str] = []
    true_empty_rows: list[str] = []
    if args.zero_label_money_repeat:
        all_zero_label_money = read_image_list(args.zero_label_money_list)
        zero_limit = len(all_zero_label_money) if args.zero_label_money_max < 0 else args.zero_label_money_max
        zero_label_money_rows = bounded_rows(all_zero_label_money, limit=zero_limit, seed=args.seed)
    if args.true_empty_repeat:
        all_true_empty = read_image_list(args.true_empty_list)
        empty_limit = len(all_true_empty) if args.true_empty_max < 0 else args.true_empty_max
        true_empty_rows = bounded_rows(all_true_empty, limit=empty_limit, seed=args.seed + 1)
    repeated_zero_label_money = zero_label_money_rows * args.zero_label_money_repeat
    repeated_true_empty = true_empty_rows * args.true_empty_repeat
    final_list = base_lines + repeated_hand + repeated_zero_label_money + repeated_true_empty

    out_txt = args.out if args.out.is_absolute() else ROOT / args.out
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(final_list) + "\n", encoding="utf-8")

    print(f"Loaded base oblique fan exposures: {len(base_lines)}")
    print(f"Found hard oblique hand-occlusion images: {len(hand_images)}")
    print(f"Hard hand-occlusion exposures: {len(repeated_hand)}")
    print(f"Zero-label money guard unique/exposures: {len(zero_label_money_rows)}/{len(repeated_zero_label_money)}")
    print(f"True-empty guard unique/exposures: {len(true_empty_rows)}/{len(repeated_true_empty)}")
    print(f"Total final exposures: {len(final_list)}")
    print(f"Wrote compiled training list to: {out_txt.relative_to(ROOT)}")

    counter: Counter[int] = Counter()
    for image in hand_images:
        counter.update(label_class_ids(image))
    print("\n--- Hard oblique hand-occlusion class distribution ---")
    for class_id, count in sorted(counter.items()):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls_{class_id}"
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
