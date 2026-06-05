#!/usr/bin/env python
"""Build and run a geometry-focused real-vs-WebGL domain-gap audit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_TRAIN_LIST = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_v1_balanced_real_only_probe_train.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
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
BOX_METRICS = ["box_area", "box_width", "box_height", "box_aspect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--recipe-id", default="", help="Optional recipe id for output naming/report context.")
    parser.add_argument("--real-train-list", type=Path, default=DEFAULT_REAL_TRAIN_LIST)
    parser.add_argument("--synthetic-train-dir", default="images/train")
    parser.add_argument("--gate-preset", default="accepted_blend_geometry_v1")
    parser.add_argument("--stem", default="", help="Override output stem under runs/cashsnap.")
    parser.add_argument("--data-out", type=Path, default=None)
    parser.add_argument("--train-list-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--image-csv-out", type=Path, default=None)
    parser.add_argument("--box-csv-out", type=Path, default=None)
    parser.add_argument("--top-class-deltas", type=int, default=10)
    parser.add_argument("--fail-on-gap", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def default_stem(args: argparse.Namespace, root: Path) -> str:
    label = args.stem.strip()
    if label:
        return slug(label)
    if args.recipe_id.strip():
        return slug(f"{args.recipe_id}_{root.name}")
    return slug(root.name)


def read_real_rows(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing real train list: {repo_rel(path)}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"real train list has no rows: {repo_rel(path)}")
    return rows


def synthetic_images(root: Path, synthetic_train_dir: str) -> list[Path]:
    image_dir = root / synthetic_train_dir
    if not image_dir.exists():
        raise SystemExit(f"synthetic image dir missing: {repo_rel(image_dir)}")
    images = [path for path in sorted(image_dir.glob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise SystemExit(f"synthetic image dir has no images: {repo_rel(image_dir)}")
    return images


def write_domain_gap_inputs(
    *,
    root: Path,
    real_train_list: Path,
    synthetic_train_dir: str,
    train_list_out: Path,
    data_out: Path,
) -> None:
    real_rows = read_real_rows(real_train_list)
    synth_rows = [repo_rel(path) for path in synthetic_images(root, synthetic_train_dir)]
    train_list_out.parent.mkdir(parents=True, exist_ok=True)
    train_list_out.write_text("\n".join(real_rows + synth_rows) + "\n", encoding="utf-8")

    names_yaml = "\n".join(f"  {index}: {class_name}" for index, class_name in enumerate(CLASS_NAMES))
    data_out.parent.mkdir(parents=True, exist_ok=True)
    data_out.write_text(
        "\n".join(
            [
                f"path: {ROOT.as_posix()}",
                f"train: {repo_rel(train_list_out)}",
                "val: data/cashsnap_v1/images/val",
                "test: data/cashsnap_v1/images/test",
                "names:",
                names_yaml,
                "cashsnap_diagnostic:",
                "  purpose: WebGL recipe geometry domain-gap audit; not trainable/promoted",
                f"  synthetic_root: {repo_rel(root)}",
                f"  real_train_list: {repo_rel(real_train_list)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def mean_stat(section: dict[str, Any], metric: str) -> float | None:
    stats = section.get(metric)
    if not isinstance(stats, dict):
        return None
    value = stats.get("mean")
    return None if value is None else float(value)


def class_delta_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_family = payload.get("by_family", {})
    real_classes = by_family.get("real", {}).get("class_box_stats", {})
    synthetic_classes = by_family.get("synthetic", {}).get("class_box_stats", {})
    deltas = payload.get("deltas", {}).get("synthetic_minus_real", {}).get("class_box_stats", {})
    rows: list[dict[str, Any]] = []
    for class_name in sorted(set(real_classes) & set(synthetic_classes)):
        row: dict[str, Any] = {"class_name": class_name}
        severity = 0.0
        for metric in BOX_METRICS:
            real_mean = mean_stat(real_classes[class_name], metric)
            synthetic_mean = mean_stat(synthetic_classes[class_name], metric)
            delta = None
            if isinstance(deltas.get(class_name), dict):
                raw_delta = deltas[class_name].get(metric)
                delta = None if raw_delta is None else float(raw_delta)
            row[f"real_{metric}"] = real_mean
            row[f"synthetic_{metric}"] = synthetic_mean
            row[f"delta_{metric}"] = delta
            if delta is not None:
                severity = max(severity, abs(delta))
        row["severity"] = severity
        rows.append(row)
    return sorted(rows, key=lambda row: (-float(row["severity"]), str(row["class_name"])))


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def print_summary(payload: dict[str, Any], args: argparse.Namespace, root: Path, json_out: Path) -> None:
    gate = payload.get("domain_gap_gate", {})
    observed = gate.get("observed", {}) if isinstance(gate, dict) else {}
    by_family = payload.get("by_family", {})
    real = by_family.get("real", {})
    synthetic = by_family.get("synthetic", {})
    aggregate = payload.get("deltas", {}).get("synthetic_minus_real", {}).get("box_stats", {})

    print(f"root={repo_rel(root)}")
    if args.recipe_id.strip():
        print(f"recipe_id={args.recipe_id.strip()}")
    print(f"json={repo_rel(json_out)}")
    print(
        "counts="
        f"real_images:{int(real.get('images', 0) or 0)} "
        f"synthetic_images:{int(synthetic.get('images', 0) or 0)} "
        f"real_boxes:{int(real.get('boxes', 0) or 0)} "
        f"synthetic_boxes:{int(synthetic.get('boxes', 0) or 0)}"
    )
    print(
        "ratios="
        f"synthetic_image:{observed.get('synthetic_image_ratio', 'n/a')} "
        f"synthetic_box:{observed.get('synthetic_box_ratio', 'n/a')}"
    )
    print(
        "aggregate_box_delta="
        + " ".join(f"{metric}:{fmt(aggregate.get(metric))}" for metric in BOX_METRICS)
    )
    print("top_class_box_deltas:")
    for row in class_delta_rows(payload)[: max(0, args.top_class_deltas)]:
        print(
            f"- {row['class_name']}: "
            f"area {fmt(row['delta_box_area'])} "
            f"width {fmt(row['delta_box_width'])} "
            f"height {fmt(row['delta_box_height'])} "
            f"aspect {fmt(row['delta_box_aspect'])}"
        )
    if isinstance(gate, dict) and gate.get("requested"):
        print("domain_gap_gate=" + ("passed" if gate.get("passed") else "failed"))
        for failure in gate.get("failures", []):
            print(f"- {failure}")


def run_audit(args: argparse.Namespace, data_out: Path, json_out: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/audit_yolo_domain_gap.py",
        "--data",
        repo_rel(data_out),
        "--split",
        "train",
        "--json-out",
        repo_rel(json_out),
        "--gate-preset",
        args.gate_preset,
    ]
    if args.image_csv_out:
        cmd.extend(["--image-csv-out", repo_rel(resolve(args.image_csv_out))])
    if args.box_csv_out:
        cmd.extend(["--box-csv-out", repo_rel(resolve(args.box_csv_out))])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    if not root.exists():
        raise SystemExit(f"missing WebGL root: {repo_rel(root)}")
    real_train_list = resolve(args.real_train_list)
    stem = default_stem(args, root)
    train_list_out = resolve(args.train_list_out or Path("runs") / "cashsnap" / f"domain_gap_{stem}_geometry_train.txt")
    data_out = resolve(args.data_out or Path("runs") / "cashsnap" / f"domain_gap_{stem}_geometry_data.yaml")
    json_out = resolve(args.json_out or Path("runs") / "cashsnap" / f"domain_gap_{stem}_geometry.json")

    write_domain_gap_inputs(
        root=root,
        real_train_list=real_train_list,
        synthetic_train_dir=args.synthetic_train_dir,
        train_list_out=train_list_out,
        data_out=data_out,
    )
    run_audit(args, data_out, json_out)
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    print_summary(payload, args, root, json_out)

    gate = payload.get("domain_gap_gate", {})
    if args.fail_on_gap and isinstance(gate, dict) and not gate.get("passed", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
