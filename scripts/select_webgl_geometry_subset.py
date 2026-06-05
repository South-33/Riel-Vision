#!/usr/bin/env python
"""Select a WebGL image subset that best matches real-anchor YOLO geometry."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import audit_yolo_domain_gap as domain_gap


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
    parser.add_argument("--root", type=Path, action="append", required=True, help="Packaged WebGL root. Repeatable.")
    parser.add_argument("--count", type=int, required=True, help="Number of synthetic images to select.")
    parser.add_argument("--out-stem", required=True, help="Output stem under runs/cashsnap.")
    parser.add_argument("--real-train-list", type=Path, default=DEFAULT_REAL_TRAIN_LIST)
    parser.add_argument("--gate-preset", default="accepted_blend_geometry_v1", choices=sorted(domain_gap.DOMAIN_GAP_PRESETS))
    parser.add_argument("--min-per-class", type=int, default=1)
    parser.add_argument("--max-class-ratio", type=float, default=0.0)
    parser.add_argument("--max-class-spread", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2606)
    parser.add_argument("--restarts", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=1200)
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


def class_name_for_id(class_id: int) -> str:
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        raise ValueError(f"unknown class id {class_id}")
    return CLASS_NAMES[class_id]


def read_image_list(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing real train list: {repo_rel(path)}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"real train list has no rows: {repo_rel(path)}")
    return rows


def resolve_image_row(row: str) -> Path:
    path = Path(row)
    return path if path.is_absolute() else ROOT / path


def image_path_for_label(label: Path) -> Path:
    image_dir = label.parents[2] / "images" / label.parent.name
    for ext in IMAGE_EXTS:
        image = image_dir / f"{label.stem}{ext}"
        if image.exists():
            return image
    raise SystemExit(f"missing image for label: {repo_rel(label)}")


def read_box_rows(label: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label)}:{line_no} expected 5 YOLO fields, found {len(parts)}")
        class_id = int(parts[0])
        width = float(parts[3])
        height = float(parts[4])
        rows.append(
            {
                "class_name": class_name_for_id(class_id),
                "box_width": width,
                "box_height": height,
                "box_area": width * height,
                "box_aspect": width / height if height else None,
            }
        )
    return rows


def real_box_rows(real_rows: list[str]) -> list[dict[str, Any]]:
    names = {index: class_name for index, class_name in enumerate(CLASS_NAMES)}
    rows: list[dict[str, Any]] = []
    for row in real_rows:
        image = resolve_image_row(row)
        rows.extend(domain_gap.label_rows(image, names))
    if not rows:
        raise SystemExit("real train list has no boxes")
    return rows


def candidate_rows(roots: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for root in roots:
        labels_dir = root / "labels" / "train"
        if not labels_dir.exists():
            raise SystemExit(f"missing labels/train in {repo_rel(root)}")
        for label in sorted(labels_dir.glob("*.txt")):
            image = image_path_for_label(label)
            image_rel = repo_rel(image)
            if image_rel in seen_images:
                raise SystemExit(f"duplicate candidate image path: {image_rel}")
            boxes = read_box_rows(label)
            if not boxes:
                continue
            seen_images.add(image_rel)
            candidates.append(
                {
                    "image": image,
                    "image_rel": image_rel,
                    "root": repo_rel(root),
                    "variant": label.stem,
                    "boxes": boxes,
                }
            )
    if not candidates:
        raise SystemExit("no synthetic candidates found")
    return candidates


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_boxes(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, int]]:
    aggregate = {metric: mean([float(row[metric]) for row in rows if row.get(metric) is not None]) for metric in BOX_METRICS}
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["class_name"])].append(row)
    class_stats: dict[str, dict[str, float]] = {}
    class_counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_rows = by_class.get(class_name, [])
        class_counts[class_name] = len(class_rows)
        if class_rows:
            class_stats[class_name] = {
                metric: mean([float(row[metric]) for row in class_rows if row.get(metric) is not None])
                for metric in BOX_METRICS
            }
    if any(value is None for value in aggregate.values()):
        raise SystemExit("cannot summarize aggregate box stats")
    return aggregate, class_stats, class_counts


def selected_box_rows(candidates: list[dict[str, Any]], selected: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in selected:
        rows.extend(candidates[index]["boxes"])
    return rows


def ratio(max_count: int, min_count: int) -> float:
    return math.inf if min_count <= 0 else max_count / min_count


def score_selection(
    *,
    candidates: list[dict[str, Any]],
    selected: set[int],
    real_aggregate: dict[str, float],
    real_class_stats: dict[str, dict[str, float]],
    aggregate_limits: dict[str, float],
    class_limits: dict[str, float],
    min_per_class: int,
    max_class_ratio: float,
    max_class_spread: int,
) -> dict[str, Any]:
    synthetic_rows = selected_box_rows(candidates, selected)
    aggregate, class_stats, class_counts = summarize_boxes(synthetic_rows)
    failures: list[str] = []
    score = 0.0

    for metric, limit in aggregate_limits.items():
        delta = float(aggregate[metric]) - float(real_aggregate[metric])
        score += 3.0 * (abs(delta) / max(limit, 1e-9)) ** 2
        if abs(delta) > limit:
            failures.append(f"box_stats.{metric} delta {delta:.6f} exceeds abs limit {limit:.6f}")
            score += 100.0

    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    spread = max_count - min_count
    class_ratio = ratio(max_count, min_count)
    if min_count < min_per_class:
        failures.append(f"class count minimum {min_count} below {min_per_class}")
        score += 1000.0 * (min_per_class - min_count)
    if max_class_spread >= 0 and spread > max_class_spread:
        failures.append(f"class count spread {spread} exceeds {max_class_spread}")
        score += 500.0 * (spread - max_class_spread)
    if max_class_ratio > 0 and class_ratio > max_class_ratio:
        failures.append(f"class count ratio {class_ratio:.6f} exceeds {max_class_ratio:.6f}")
        score += 500.0 * (class_ratio - max_class_ratio)

    class_deltas: dict[str, dict[str, float | None]] = {}
    for class_name in CLASS_NAMES:
        class_deltas[class_name] = {}
        if class_name not in class_stats:
            failures.append(f"missing synthetic class {class_name}")
            score += 10000.0
            continue
        for metric, limit in class_limits.items():
            delta = float(class_stats[class_name][metric]) - float(real_class_stats[class_name][metric])
            class_deltas[class_name][metric] = delta
            weight = 5.0 if metric == "box_aspect" else 1.0
            score += weight * (abs(delta) / max(limit, 1e-9)) ** 2
            if abs(delta) > limit:
                failures.append(
                    f"class_box_stats.{class_name}.{metric} delta {delta:.6f} exceeds abs limit {limit:.6f}"
                )
                score += 50.0 * weight

    return {
        "score": score,
        "passed": not failures,
        "failures": failures,
        "synthetic_boxes": len(synthetic_rows),
        "class_counts": class_counts,
        "class_count_spread": spread,
        "class_count_ratio": class_ratio,
        "aggregate_deltas": {
            metric: float(aggregate[metric]) - float(real_aggregate[metric])
            for metric in aggregate_limits
        },
        "class_deltas": class_deltas,
    }


def search(candidates: list[dict[str, Any]], args: argparse.Namespace, real_stats: tuple[dict[str, float], dict[str, dict[str, float]], dict[str, int]]) -> tuple[set[int], dict[str, Any]]:
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.count > len(candidates):
        raise SystemExit(f"--count {args.count} exceeds candidate count {len(candidates)}")

    preset = domain_gap.DOMAIN_GAP_PRESETS[args.gate_preset]
    aggregate_limits = dict(preset.get("box", {}))
    class_limits = dict(preset.get("class_box", {}))
    if not class_limits:
        class_limits = dict(aggregate_limits)
    real_aggregate, real_class_stats, _real_class_counts = real_stats
    rng = random.Random(args.seed)

    def evaluate(selected: set[int]) -> dict[str, Any]:
        return score_selection(
            candidates=candidates,
            selected=selected,
            real_aggregate=real_aggregate,
            real_class_stats=real_class_stats,
            aggregate_limits=aggregate_limits,
            class_limits=class_limits,
            min_per_class=args.min_per_class,
            max_class_ratio=args.max_class_ratio,
            max_class_spread=args.max_class_spread,
        )

    initial = set(rng.sample(range(len(candidates)), args.count))
    best_selected = set(initial)
    best_metrics = evaluate(best_selected)

    for _restart in range(args.restarts):
        selected = set(rng.sample(range(len(candidates)), args.count))
        current_metrics = evaluate(selected)
        temperature = 6.0
        for _iteration in range(args.iterations):
            outgoing = rng.choice(tuple(selected))
            incoming = rng.randrange(len(candidates))
            if incoming in selected:
                continue
            candidate = set(selected)
            candidate.remove(outgoing)
            candidate.add(incoming)
            candidate_metrics = evaluate(candidate)
            current_score = float(current_metrics["score"])
            candidate_score = float(candidate_metrics["score"])
            accept = candidate_score < current_score
            if not accept:
                accept = rng.random() < math.exp((current_score - candidate_score) / max(0.5, temperature))
            if accept:
                selected = candidate
                current_metrics = candidate_metrics
                if candidate_score < float(best_metrics["score"]):
                    best_selected = set(selected)
                    best_metrics = dict(current_metrics)
            temperature *= 0.997
    return best_selected, best_metrics


def write_selected_outputs(
    *,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    selected: set[int],
    metrics: dict[str, Any],
    real_rows: list[str],
) -> tuple[Path, Path, Path, Path]:
    stem = slug(args.out_stem)
    out_dir = ROOT / "runs" / "cashsnap"
    train_list = out_dir / f"{stem}_train.txt"
    data_yaml = out_dir / f"{stem}_data.yaml"
    selection_json = out_dir / f"{stem}_selection.json"
    audit_json = out_dir / f"{stem}_geometry.json"
    selected_candidates = [candidates[index] for index in sorted(selected, key=lambda idx: candidates[idx]["image_rel"])]
    selected_rows = [str(row) for row in real_rows] + [str(row["image_rel"]) for row in selected_candidates]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_list.write_text("\n".join(selected_rows) + "\n", encoding="utf-8")
    names_yaml = "\n".join(f"  {index}: {class_name}" for index, class_name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {ROOT.as_posix()}",
                f"train: {repo_rel(train_list)}",
                "val: data/cashsnap_v1/images/val",
                "test: data/cashsnap_v1/images/test",
                "names:",
                names_yaml,
                "cashsnap_diagnostic:",
                "  purpose: Geometry-selected WebGL recipe subset; not trainable/promoted",
                f"  real_train_list: {repo_rel(resolve(args.real_train_list))}",
                "  synthetic_roots:",
                *[f"    - {repo_rel(resolve(root))}" for root in args.root],
                "",
            ]
        ),
        encoding="utf-8",
    )
    selection_json.write_text(
        json.dumps(
            {
                "out_stem": stem,
                "count": args.count,
                "gate_preset": args.gate_preset,
                "seed": args.seed,
                "restarts": args.restarts,
                "iterations": args.iterations,
                "roots": [repo_rel(resolve(root)) for root in args.root],
                "metrics": metrics,
                "selected_images": [str(row["image_rel"]) for row in selected_candidates],
                "selected_variants": [
                    {"root": str(row["root"]), "variant": str(row["variant"]), "image": str(row["image_rel"])}
                    for row in selected_candidates
                ],
                "outputs": {
                    "train_list": repo_rel(train_list),
                    "data_yaml": repo_rel(data_yaml),
                    "geometry_audit_json": repo_rel(audit_json),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return train_list, data_yaml, selection_json, audit_json


def run_geometry_audit(data_yaml: Path, audit_json: Path, gate_preset: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/audit_yolo_domain_gap.py",
        "--data",
        repo_rel(data_yaml),
        "--split",
        "train",
        "--json-out",
        repo_rel(audit_json),
        "--gate-preset",
        gate_preset,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(audit_json.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    roots = [resolve(root) for root in args.root]
    real_train_list = resolve(args.real_train_list)
    real_rows = read_image_list(real_train_list)
    real_stats = summarize_boxes(real_box_rows(real_rows))
    candidates = candidate_rows(roots)
    selected, metrics = search(candidates, args, real_stats)
    train_list, data_yaml, selection_json, audit_json = write_selected_outputs(
        args=args,
        candidates=candidates,
        selected=selected,
        metrics=metrics,
        real_rows=real_rows,
    )
    audit_payload = run_geometry_audit(data_yaml, audit_json, args.gate_preset)
    gate = audit_payload.get("domain_gap_gate", {})

    print(f"selected={len(selected)}/{len(candidates)}")
    print(f"train_list={repo_rel(train_list)}")
    print(f"selection_json={repo_rel(selection_json)}")
    print(f"geometry_json={repo_rel(audit_json)}")
    print(
        "selection_score="
        f"{float(metrics['score']):.3f} "
        f"selection_passed={bool(metrics['passed'])} "
        f"synthetic_boxes={metrics['synthetic_boxes']} "
        f"class_ratio={float(metrics['class_count_ratio']):.3f}"
    )
    print("domain_gap_gate=" + ("passed" if gate.get("passed") else "failed"))
    for failure in gate.get("failures", []):
        print(f"- {failure}")
    if args.fail_on_gap and not gate.get("passed", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
