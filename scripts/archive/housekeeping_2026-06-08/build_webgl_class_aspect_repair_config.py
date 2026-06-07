#!/usr/bin/env python
"""Build a surgical class-aspect repair config from clean WebGL candidate roots."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BASE = (
    ROOT
    / "configs"
    / "webgl_ablation"
    / "cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_puresynth_realval_v1.yaml"
)
DEFAULT_TARGETS = ROOT / "runs" / "cashsnap" / "domain_gap_filtered185_current_geometry_targets.json"
DEFAULT_CANDIDATE_ROOTS = [
    ROOT / "data" / "synthetic" / "cashsnap_webgl_clean_base_topdown_768x640_handled_probe_v2",
    ROOT / "data" / "synthetic" / "cashsnap_webgl_clean_base_square_classdiverse_postproc_geometry_selected96_v2",
    ROOT / "data" / "synthetic" / "cashsnap_webgl_clean_base_square_phoneauto_mixedcondition_postproc_pool_v1",
]
DEFAULT_TARGET_CLASSES = "USD_5,USD_10,USD_20,KHR_500"
DEFAULT_LIMITS = {
    "box_area": 0.30,
    "box_width": 0.40,
    "box_height": 0.40,
    "box_aspect": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--domain-gap-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--candidate-root", action="append", type=Path, default=[])
    parser.add_argument("--target-classes", default=DEFAULT_TARGET_CLASSES)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--iterations", type=int, default=75000)
    parser.add_argument("--churn-weight", type=float, default=0.015)
    parser.add_argument(
        "--selection-mode",
        choices=("replacement", "current_plus_candidates"),
        default="replacement",
        help=(
            "replacement samples a full new class distribution from candidate roots; "
            "current_plus_candidates samples from current rows plus candidate roots for capped low-churn repairs."
        ),
    )
    parser.add_argument(
        "--max-new-rows-per-class",
        type=int,
        default=None,
        help="Optional cap on non-current rows selected for each target class.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected YAML mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected JSON mapping")
    return data


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else dataset_root / path


def train_list_path(config_path: Path, config: dict[str, Any]) -> Path:
    train = config.get("train")
    if not isinstance(train, str):
        raise SystemExit(f"{repo_rel(config_path)} train split must be a .txt list for aspect repair")
    path = split_root(data_root(config_path, config), train)
    if path.suffix.lower() != ".txt":
        raise SystemExit(f"{repo_rel(config_path)} train split must point to a .txt list")
    return path


def read_image_list(path: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(line.replace("\\", "/"))
    return rows


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").replace(" ", ",").split(",") if item.strip()]


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if not isinstance(names, dict):
        raise SystemExit("base config names must be a mapping")
    return {int(class_id): str(class_name) for class_id, class_name in names.items()}


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root) / "images" / "train"
    if not image_dir.exists():
        raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
    return [
        repo_rel(path)
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def row_metric(image: str, names: dict[int, str], current_rows: set[str]) -> dict[str, Any] | None:
    label = resolve(label_path_for_image(image))
    if not label.exists():
        return None
    lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    parts = lines[0].split()
    if len(parts) != 5:
        return None
    class_name = names.get(int(parts[0]), str(parts[0]))
    width = float(parts[3])
    height = float(parts[4])
    return {
        "image": image,
        "class_name": class_name,
        "box_width": width,
        "box_height": height,
        "box_area": width * height,
        "box_aspect": width / max(height, 1e-9),
        "current": image in current_rows,
    }


def target_stats(path: Path, target_classes: list[str]) -> dict[str, dict[str, float]]:
    payload = read_json(path)
    rows = payload.get("class_targets", [])
    if not isinstance(rows, list):
        raise SystemExit("domain-gap targets JSON missing class_targets list")
    by_class: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        class_name = str(row.get("class_name", ""))
        if class_name not in target_classes:
            continue
        by_class[class_name] = {
            metric: float(row[f"real_{metric}"])
            for metric in DEFAULT_LIMITS
            if f"real_{metric}" in row
        }
    missing = [class_name for class_name in target_classes if class_name not in by_class]
    if missing:
        raise SystemExit(f"domain-gap targets missing classes: {', '.join(missing)}")
    return by_class


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in DEFAULT_LIMITS
    }


def score_rows(
    rows: list[dict[str, Any]],
    target: dict[str, float],
    churn_weight: float,
) -> tuple[float, dict[str, float]]:
    means = mean_metrics(rows)
    normalized = sum((abs(means[metric] - target[metric]) / DEFAULT_LIMITS[metric]) ** 2 for metric in DEFAULT_LIMITS)
    churn = sum(0 if bool(row["current"]) else 1 for row in rows) * churn_weight
    return normalized + churn, means


def random_sample(rng: random.Random, pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return rng.sample(pool, count)


def select_replacements(
    *,
    base_rows: list[str],
    candidate_roots: list[Path],
    class_names: dict[int, str],
    target_classes: list[str],
    targets: dict[str, dict[str, float]],
    iterations: int,
    seed: int,
    churn_weight: float,
    selection_mode: str,
    max_new_rows_per_class: int | None,
) -> tuple[list[str], dict[str, Any]]:
    base_set = set(base_rows)
    current_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in base_rows:
        metric = row_metric(image, class_names, base_set)
        if metric and metric["class_name"] in target_classes:
            current_by_class[str(metric["class_name"])].append(metric)
    for root in candidate_roots:
        for image in image_rows(root):
            metric = row_metric(image, class_names, base_set)
            if metric and metric["class_name"] in target_classes:
                candidate_by_class[str(metric["class_name"])].append(metric)

    rng = random.Random(seed)
    replacements_by_class: dict[str, list[str]] = {}
    reports: list[dict[str, Any]] = []
    for class_name in target_classes:
        current = current_by_class[class_name]
        raw_candidates = candidate_by_class[class_name]
        if selection_mode == "current_plus_candidates":
            raw_candidates = [row for row in raw_candidates if not bool(row["current"])]
        candidates = current + raw_candidates if selection_mode == "current_plus_candidates" else raw_candidates
        if not current:
            raise SystemExit(f"base list has no current rows for {class_name}")
        if len(candidates) < len(current):
            raise SystemExit(f"not enough candidate rows for {class_name}: {len(candidates)} < {len(current)}")
        if max_new_rows_per_class is not None and max_new_rows_per_class < 0:
            raise SystemExit("--max-new-rows-per-class must be non-negative")
        best_rows = list(current)
        best_score, best_means = score_rows(best_rows, targets[class_name], churn_weight)
        for _ in range(iterations):
            if selection_mode == "current_plus_candidates":
                max_new = len(current) if max_new_rows_per_class is None else max_new_rows_per_class
                max_new = min(max_new, len(current), len(raw_candidates))
                new_count = rng.randint(0, max_new)
                rows = random_sample(rng, raw_candidates, new_count) + random_sample(
                    rng,
                    current,
                    len(current) - new_count,
                )
            else:
                rows = random_sample(rng, candidates, len(current))
                new_rows = sum(0 if bool(row["current"]) else 1 for row in rows)
                if max_new_rows_per_class is not None and new_rows > max_new_rows_per_class:
                    continue
            score, means = score_rows(rows, targets[class_name], churn_weight)
            if score < best_score:
                best_score = score
                best_means = means
                best_rows = rows
        replacements_by_class[class_name] = [str(row["image"]) for row in best_rows]
        reports.append(
            {
                "class_name": class_name,
                "base_rows": len(current),
                "candidate_rows": len(raw_candidates),
                "selection_pool_rows": len(candidates),
                "replacement_rows": len(best_rows),
                "new_rows": sum(0 if bool(row["current"]) else 1 for row in best_rows),
                "score": round(float(best_score), 6),
                "target_means": {key: round(value, 6) for key, value in targets[class_name].items()},
                "selected_means": {key: round(value, 6) for key, value in best_means.items()},
                "selected_deltas": {
                    key: round(best_means[key] - targets[class_name][key], 6)
                    for key in DEFAULT_LIMITS
                },
                "replacement_image_rows": replacements_by_class[class_name],
            }
        )

    repaired_rows: list[str] = []
    used_replacement_index: dict[str, int] = defaultdict(int)
    for image in base_rows:
        metric = row_metric(image, class_names, base_set)
        if not metric or metric["class_name"] not in replacements_by_class:
            repaired_rows.append(image)
            continue
        class_name = str(metric["class_name"])
        index = used_replacement_index[class_name]
        repaired_rows.append(replacements_by_class[class_name][index])
        used_replacement_index[class_name] += 1
    if len(repaired_rows) != len(base_rows):
        raise SystemExit("repaired list length changed unexpectedly")
    if len(set(repaired_rows)) != len(repaired_rows):
        raise SystemExit("repaired list contains duplicate image rows")
    return repaired_rows, {"classes": reports, "target_classes": target_classes}


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base)
    base_config = read_yaml(base_path)
    base_train_list = train_list_path(base_path, base_config)
    base_rows = read_image_list(base_train_list)
    target_classes = parse_csv(args.target_classes)
    class_names = names_by_id(base_config)
    candidate_roots = [resolve(path) for path in (args.candidate_root or DEFAULT_CANDIDATE_ROOTS)]
    targets = target_stats(args.domain_gap_targets, target_classes)
    repaired_rows, report = select_replacements(
        base_rows=base_rows,
        candidate_roots=candidate_roots,
        class_names=class_names,
        target_classes=target_classes,
        targets=targets,
        iterations=args.iterations,
        seed=args.seed,
        churn_weight=args.churn_weight,
        selection_mode=args.selection_mode,
        max_new_rows_per_class=args.max_new_rows_per_class,
    )
    report.update(
        {
            "schema": "cashsnap_webgl_class_aspect_repair_v1",
            "base_config": repo_rel(base_path),
            "base_train_list": repo_rel(base_train_list),
            "candidate_roots": [repo_rel(path) for path in candidate_roots],
            "domain_gap_targets": repo_rel(resolve(args.domain_gap_targets)),
            "seed": int(args.seed),
            "iterations": int(args.iterations),
            "churn_weight": float(args.churn_weight),
            "selection_mode": str(args.selection_mode),
            "max_new_rows_per_class": args.max_new_rows_per_class,
            "selected_images": len(repaired_rows),
            "selected_unique_images": len(set(repaired_rows)),
        }
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    write_image_list(args.out_list, repaired_rows)
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(resolve(args.out_config).parent, ROOT)
    config["train"] = repo_rel(resolve(args.out_list))
    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["class_aspect_repair_base_config"] = repo_rel(base_path)
    sources["class_aspect_repair_base_train_list"] = repo_rel(base_train_list)
    sources["class_aspect_repair_candidate_roots"] = [repo_rel(path) for path in candidate_roots]
    config["cashsnap_sources"] = sources
    config["cashsnap_webgl_class_aspect_repair"] = report
    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    policy["intended_use"] = (
        "pure-synth clean visible-note TSTR probe testing surgical class-aspect replacement "
        "for USD_5, USD_10, USD_20, and KHR_500"
    )
    policy["promotion_rule"] = (
        "reject unless accepted-blend geometry gate passes and full/clean-visible real val/test preserve filtered185"
    )
    config["cashsnap_policy"] = policy
    write_yaml(args.out_config, config)
    print(json.dumps(report, indent=2))
    print(f"wrote_list={repo_rel(resolve(args.out_list))}")
    print(f"wrote_config={repo_rel(resolve(args.out_config))}")
    return 0


def rel_between(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
