#!/usr/bin/env python
"""Run diagnostic gates declared on a WebGL recipe catalog entry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from webgl_constants import WEBGL_NOTE_PRINT_TONE_POLICIES, WEBGL_OCCLUDER_POLICIES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
NOTE_CONDITION_POLICIES = {"mixed", "pristine_only", "heavy_wear", "wet_stress"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--recipe-id", required=True, help="Recipe id in the catalog.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def find_recipe(catalog: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    recipes = catalog.get("recipes", [])
    if not isinstance(recipes, list):
        raise SystemExit("catalog recipes must be a list")
    for row in recipes:
        if isinstance(row, dict) and row.get("id") == recipe_id:
            return row
    raise SystemExit(f"recipe not found: {recipe_id}")


def add_int_option(cmd: list[str], gate: dict[str, Any], key: str, option: str) -> None:
    if key in gate:
        cmd.extend([option, str(int(gate[key]))])


def add_float_option(cmd: list[str], gate: dict[str, Any], key: str, option: str) -> None:
    if key in gate:
        cmd.extend([option, str(float(gate[key]))])


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"diagnostic gate command failed with exit code {completed.returncode}")


def run_class_distribution_gate(root: Path, gate: dict[str, Any]) -> None:
    expected_classes = gate.get("expected_classes", [])
    if not isinstance(expected_classes, list) or not expected_classes:
        raise SystemExit("class_distribution.expected_classes must be a non-empty list")
    cmd = [
        sys.executable,
        "scripts/check_webgl_class_distribution.py",
        "--root",
        str(root),
        "--expected-classes",
        ",".join(str(item).strip() for item in expected_classes if str(item).strip()),
    ]
    add_int_option(cmd, gate, "min_images", "--min-images")
    add_int_option(cmd, gate, "min_total", "--min-total")
    add_int_option(cmd, gate, "min_per_class", "--min-per-class")
    add_int_option(cmd, gate, "max_class_spread", "--max-class-spread")
    add_float_option(cmd, gate, "max_class_ratio", "--max-class-ratio")
    if gate.get("allow_extra_classes"):
        cmd.append("--allow-extra-classes")
    run(cmd)


def run_count_stress_gate(root: Path, gate: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_count_stress.py",
        "--root",
        str(root),
    ]
    add_int_option(cmd, gate, "min_images", "--min-images")
    add_int_option(cmd, gate, "min_repeat_images", "--min-repeat-images")
    add_int_option(cmd, gate, "min_max_same_class", "--min-max-same-class")
    add_int_option(cmd, gate, "min_kept_split_parent_count", "--min-kept-split-parent-count")
    add_int_option(cmd, gate, "min_all_split_parent_count", "--min-all-split-parent-count")
    add_int_option(cmd, gate, "min_naive_kept_fragment_overcount", "--min-naive-kept-fragment-overcount")
    add_int_option(cmd, gate, "min_naive_all_fragment_overcount", "--min-naive-all-fragment-overcount")
    run(cmd)


def run_note_condition_diversity_gate(root: Path, gate: dict[str, Any], recipe: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_note_condition_diversity.py",
        "--root",
        str(root),
    ]
    expected_policy = str(gate.get("expected_policy", recipe.get("note_condition_policy", ""))).strip()
    if expected_policy:
        if expected_policy not in NOTE_CONDITION_POLICIES:
            raise SystemExit(f"note_condition_diversity.expected_policy must be one of {sorted(NOTE_CONDITION_POLICIES)}")
        cmd.extend(["--expected-policy", expected_policy])
    add_int_option(cmd, gate, "min_notes", "--min-notes")
    add_int_option(cmd, gate, "min_profiles", "--min-profiles")
    add_float_option(cmd, gate, "min_dirtiness_range", "--min-dirtiness-range")
    add_float_option(cmd, gate, "min_crinkle_range", "--min-crinkle-range")
    add_float_option(cmd, gate, "min_wetness_range", "--min-wetness-range")
    add_int_option(cmd, gate, "min_dirty_notes", "--min-dirty-notes")
    add_int_option(cmd, gate, "min_pristine_notes", "--min-pristine-notes")
    add_int_option(cmd, gate, "min_wet_notes", "--min-wet-notes")
    run(cmd)


def run_note_print_tone_gate(root: Path, gate: dict[str, Any], recipe: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_note_print_tone.py",
        "--root",
        str(root),
    ]
    expected_policy = str(gate.get("expected_policy", recipe.get("note_print_tone_policy", ""))).strip()
    if expected_policy:
        if expected_policy not in WEBGL_NOTE_PRINT_TONE_POLICIES:
            raise SystemExit(f"note_print_tone.expected_policy must be one of {sorted(WEBGL_NOTE_PRINT_TONE_POLICIES)}")
        cmd.extend(["--expected-policy", expected_policy])
    add_int_option(cmd, gate, "min_notes", "--min-notes")
    add_float_option(cmd, gate, "min_mean_contrast", "--min-mean-contrast")
    add_float_option(cmd, gate, "max_mean_contrast", "--max-mean-contrast")
    add_float_option(cmd, gate, "min_contrast_range", "--min-contrast-range")
    if gate.get("allow_missing"):
        cmd.append("--allow-missing")
    run(cmd)


def run_occluder_policy_gate(root: Path, gate: dict[str, Any], recipe: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_occluder_policy.py",
        "--root",
        str(root),
    ]
    expected_policy = str(gate.get("expected_policy", recipe.get("occluder_policy", ""))).strip()
    if expected_policy:
        if expected_policy not in WEBGL_OCCLUDER_POLICIES:
            raise SystemExit(f"occluder_policy.expected_policy must be one of {sorted(WEBGL_OCCLUDER_POLICIES)}")
        cmd.extend(["--expected-policy", expected_policy])
    if gate.get("forbid_hand_occluders"):
        cmd.append("--forbid-hand-occluders")
    if gate.get("require_zero_occluders"):
        cmd.append("--require-zero-occluders")
    if gate.get("allow_missing_policy"):
        cmd.append("--allow-missing-policy")
    run(cmd)


def run_hard_negative_diversity_gate(root: Path, gate: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_hard_negative_diversity.py",
        "--root",
        str(root),
    ]
    add_int_option(cmd, gate, "min_images", "--min-images")
    add_int_option(cmd, gate, "min_total_props", "--min-total-props")
    add_int_option(cmd, gate, "min_prop_kinds", "--min-prop-kinds")
    add_int_option(cmd, gate, "min_textured_props", "--min-textured-props")
    if gate.get("require_zero_assets"):
        cmd.append("--require-zero-assets")
    run(cmd)


def add_metric_limit_options(cmd: list[str], gate: dict[str, Any], key: str, option: str) -> None:
    raw_limits = gate.get(key, {})
    if raw_limits in (None, ""):
        return
    if not isinstance(raw_limits, dict):
        raise SystemExit(f"domain_gap.{key} must be an object")
    for metric, value in sorted(raw_limits.items()):
        cmd.extend([option, f"{metric}={float(value)}"])


def write_domain_gap_data_yaml(config_out: Path, train_list: Path, root: Path, gate: dict[str, Any]) -> None:
    synthetic_train_dir = root / str(gate.get("synthetic_train_dir", "images/train"))
    if not synthetic_train_dir.exists():
        raise SystemExit(f"domain_gap synthetic train dir missing: {repo_rel(synthetic_train_dir)}")
    synthetic_images = [
        path
        for path in sorted(synthetic_train_dir.glob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not synthetic_images:
        raise SystemExit(f"domain_gap synthetic train dir has no images: {repo_rel(synthetic_train_dir)}")

    real_train_list = resolve(Path(str(gate.get("real_train_list", ""))))
    if not real_train_list.exists():
        raise SystemExit(f"domain_gap real_train_list missing: {repo_rel(real_train_list)}")
    real_rows = [line.strip() for line in real_train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not real_rows:
        raise SystemExit(f"domain_gap real_train_list has no rows: {repo_rel(real_train_list)}")

    train_rows = real_rows + [repo_rel(path) for path in synthetic_images]
    train_list.parent.mkdir(parents=True, exist_ok=True)
    train_list.write_text("\n".join(train_rows) + "\n", encoding="utf-8")

    names_yaml = "\n".join(f"  {index}: {class_name}" for index, class_name in enumerate(CLASS_NAMES))
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(
        "\n".join(
            [
                f"path: {ROOT.as_posix()}",
                f"train: {repo_rel(train_list)}",
                "val: data/cashsnap_v1/images/val",
                "test: data/cashsnap_v1/images/test",
                "names:",
                names_yaml,
                "cashsnap_diagnostic:",
                "  purpose: WebGL recipe diagnostic domain-gap gate; not trainable/promoted",
                f"  synthetic_root: {repo_rel(root)}",
                f"  real_train_list: {repo_rel(real_train_list)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_domain_gap_gate(root: Path, gate: dict[str, Any], recipe_id: str) -> None:
    stem = recipe_id.replace("/", "_")
    train_list = resolve(Path(str(gate.get("train_list_out", f"runs/cashsnap/domain_gap_{stem}_diagnostic_train.txt"))))
    config_out = resolve(Path(str(gate.get("config_out", f"runs/cashsnap/domain_gap_{stem}_diagnostic_data.yaml"))))
    json_out = resolve(Path(str(gate.get("json_out", f"runs/cashsnap/domain_gap_{stem}_diagnostic_train.json"))))
    write_domain_gap_data_yaml(config_out, train_list, root, gate)

    cmd = [
        sys.executable,
        "scripts/audit_yolo_domain_gap.py",
        "--data",
        repo_rel(config_out),
        "--split",
        "train",
        "--json-out",
        repo_rel(json_out),
    ]
    preset = str(gate.get("preset", "")).strip()
    if preset:
        cmd.extend(["--gate-preset", preset])
    add_metric_limit_options(cmd, gate, "max_abs_image_delta", "--max-abs-image-delta")
    add_metric_limit_options(cmd, gate, "max_abs_box_delta", "--max-abs-box-delta")
    add_metric_limit_options(cmd, gate, "max_abs_class_box_delta", "--max-abs-class-box-delta")
    for class_name in gate.get("class_box_delta_classes", []):
        cmd.extend(["--class-box-delta-class", str(class_name)])
    for key, option in (
        ("max_synthetic_image_ratio", "--max-synthetic-image-ratio"),
        ("max_synthetic_box_ratio", "--max-synthetic-box-ratio"),
        ("max_synthetic_class_box_ratio", "--max-synthetic-class-box-ratio"),
    ):
        if key in gate:
            cmd.extend([option, str(float(gate[key]))])
    if gate.get("fail_on_gap", True):
        cmd.append("--fail-on-gap")
    run(cmd)


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    catalog = read_json(resolve(args.catalog))
    recipe = find_recipe(catalog, args.recipe_id)
    gates = recipe.get("diagnostic_gates", {})
    if not gates:
        print(f"ok: {args.recipe_id} declares no diagnostic gates")
        return 0
    if not isinstance(gates, dict):
        raise SystemExit(f"{args.recipe_id}: diagnostic_gates must be an object")

    class_distribution = gates.get("class_distribution")
    if class_distribution is not None:
        if not isinstance(class_distribution, dict):
            raise SystemExit(f"{args.recipe_id}: class_distribution gate must be an object")
        run_class_distribution_gate(root, class_distribution)

    count_stress = gates.get("count_stress")
    if count_stress is not None:
        if not isinstance(count_stress, dict):
            raise SystemExit(f"{args.recipe_id}: count_stress gate must be an object")
        run_count_stress_gate(root, count_stress)

    note_condition_diversity = gates.get("note_condition_diversity")
    if note_condition_diversity is not None:
        if not isinstance(note_condition_diversity, dict):
            raise SystemExit(f"{args.recipe_id}: note_condition_diversity gate must be an object")
        run_note_condition_diversity_gate(root, note_condition_diversity, recipe)

    note_print_tone = gates.get("note_print_tone")
    if note_print_tone is not None:
        if not isinstance(note_print_tone, dict):
            raise SystemExit(f"{args.recipe_id}: note_print_tone gate must be an object")
        run_note_print_tone_gate(root, note_print_tone, recipe)

    occluder_policy = gates.get("occluder_policy")
    if occluder_policy is not None:
        if not isinstance(occluder_policy, dict):
            raise SystemExit(f"{args.recipe_id}: occluder_policy gate must be an object")
        run_occluder_policy_gate(root, occluder_policy, recipe)

    hard_negative_diversity = gates.get("hard_negative_diversity")
    if hard_negative_diversity is not None:
        if not isinstance(hard_negative_diversity, dict):
            raise SystemExit(f"{args.recipe_id}: hard_negative_diversity gate must be an object")
        run_hard_negative_diversity_gate(root, hard_negative_diversity)

    domain_gap = gates.get("domain_gap")
    if domain_gap is not None:
        if not isinstance(domain_gap, dict):
            raise SystemExit(f"{args.recipe_id}: domain_gap gate must be an object")
        run_domain_gap_gate(root, domain_gap, args.recipe_id)

    print(f"ok: {args.recipe_id} diagnostic gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
