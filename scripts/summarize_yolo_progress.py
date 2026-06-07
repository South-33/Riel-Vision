from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def results_csv(path: Path) -> Path:
    resolved = resolve(path)
    if resolved.is_dir():
        return resolved / "results.csv"
    return resolved


def read_rows(path: Path) -> list[dict[str, str]]:
    csv_path = results_csv(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summarize(path: Path, expected_epochs: int | None) -> str:
    rows = read_rows(path)
    csv_path = results_csv(path)
    if not rows:
        return f"progress: no results yet at {csv_path}"

    last = rows[-1]
    first = rows[0]
    epoch = int(as_float(last, "epoch") or len(rows))
    total = expected_epochs or "?"
    elapsed = as_float(last, "time")
    train_box = as_float(last, "train/box_loss")
    train_cls = as_float(last, "train/cls_loss")
    train_dfl = as_float(last, "train/dfl_loss")
    first_cls = as_float(first, "train/cls_loss")
    cls_delta = None if first_cls is None or train_cls is None else train_cls - first_cls
    map50 = as_float(last, "metrics/mAP50(B)")
    map5095 = as_float(last, "metrics/mAP50-95(B)")
    precision = as_float(last, "metrics/precision(B)")
    recall = as_float(last, "metrics/recall(B)")

    return (
        f"progress: epoch={epoch}/{total} elapsed={fmt(elapsed, 1)}s "
        f"train_box={fmt(train_box)} train_cls={fmt(train_cls)} "
        f"train_dfl={fmt(train_dfl)} cls_delta={fmt(cls_delta)} "
        f"mAP50={fmt(map50)} mAP50-95={fmt(map5095)} "
        f"precision={fmt(precision)} recall={fmt(recall)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize YOLO training progress from a run results.csv.")
    parser.add_argument("--run", required=True, type=Path, help="YOLO run directory or results.csv path.")
    parser.add_argument("--expected-epochs", type=int, default=None)
    parser.add_argument("--watch", action="store_true", help="Print progress repeatedly until interrupted.")
    parser.add_argument("--interval", type=float, default=300.0, help="Seconds between --watch updates.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.watch:
        print(summarize(args.run, args.expected_epochs), flush=True)
        return 0
    while True:
        print(summarize(args.run, args.expected_epochs), flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
