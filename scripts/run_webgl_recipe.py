#!/usr/bin/env python
"""Run a named WebGL synthetic recipe from the recipe catalog."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from hardware_profile import (
    HEADROOM_MAX_GPU_MEM_PERCENT,
    HEADROOM_MAX_PERCENT,
    HEADROOM_MAX_RAM_PERCENT,
    HEADROOM_MIN_FREE_RAM_GB,
    HEADROOM_RESUME_PERCENT,
    WEBGL_CHECK_JOBS,
    WEBGL_RENDERER_BATCH_SIZE,
    WEBGL_RENDER_JOBS,
)
from webgl_constants import (
    WEBGL_ASSET_QUALITY_POLICIES,
    WEBGL_ASSET_SIDE_POLICIES,
    WEBGL_CAMERA_ISP_POLICIES,
    WEBGL_CAMERA_PROFILES,
    WEBGL_CLEAN_ORIENTATION_POLICIES,
    WEBGL_NOTE_CONDITION_POLICIES,
    WEBGL_NOTE_PRINT_TONE_POLICIES,
    WEBGL_NEGATIVE_PROP_POLICIES,
    WEBGL_OCCLUDER_POLICIES,
    WEBGL_SCENE_MODES,
    WEBGL_STACK_POSE_POLICIES,
    WEBGL_TEXTURE_QA_EFFECTS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
RUNNABLE_SCENE_MODES = WEBGL_SCENE_MODES
STATUS_TO_BATCH_STATUS = {
    "planned": "diagnostic",
    "smoke_ready": "smoke",
    "label_policy_ready": "diagnostic",
    "diagnostic": "diagnostic",
    "trainable-candidate": "trainable-candidate",
    "promoted": "trainable-candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--start-variant", type=int, default=0)
    parser.add_argument("--count", type=int, default=None, help="Rendered pool count; defaults to recipe render_pool_count or 4.")
    parser.add_argument("--scene-mode", default="", help="Override the first runnable scene mode from the catalog.")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--visual-scale", default="2", help="Visual WebGL supersampling scale.")
    parser.add_argument("--browser-executable", type=Path, default=None, help="Optional Chromium/Edge executable override.")
    parser.add_argument("--asset-side-policy", default="", help="Override catalog asset-side sampling policy.")
    parser.add_argument("--asset-quality-policy", default="", help="Override catalog asset quality/current-design sampling policy.")
    parser.add_argument("--camera-profile", default="", help="Override catalog WebGL camera/FOV/framing profile.")
    parser.add_argument("--camera-isp-policy", default="", help="Override catalog camera/ISP-style RGB postprocess policy.")
    parser.add_argument("--stack-pose-policy", default="", help="Override catalog class-conditioned stack pose policy.")
    parser.add_argument("--clean-orientation-policy", default="", help="Override catalog class-conditioned clean-scene orientation policy.")
    parser.add_argument("--class-sequence", default="", help="Override catalog class sequence for supported scene modes.")
    parser.add_argument("--note-condition-policy", default="", help="Override catalog per-note dirt/crinkle/wetness policy.")
    parser.add_argument("--lens-distortion-policy", default="", help="Override catalog shared radial lens-warp policy.")
    parser.add_argument("--note-print-tone-policy", default="", help="Override catalog per-note print dynamic-range policy.")
    parser.add_argument("--texture-qa-effects", default="", help="Override texture_qa effect-ladder stage.")
    parser.add_argument("--occluder-policy", default="", help="Override catalog primitive occluder policy.")
    parser.add_argument("--negative-prop-policy", default="", help="Override catalog zero-label negative prop policy.")
    parser.add_argument("--artifact-status", choices=["smoke", "diagnostic", "trainable-candidate"], default="")
    parser.add_argument("--background-dir", type=Path, default=None)
    parser.add_argument("--environment-dir", type=Path, default=None, help="Optional equirectangular environment map directory for visual lighting/reflections.")
    parser.add_argument(
        "--environment-bank-config",
        type=Path,
        default=Path("configs/synthetic_recipes/cashsnap_webgl_environment_banks_v1.json"),
        help="Review registry forwarded when --environment-dir is used.",
    )
    parser.add_argument("--headroom-max-percent", default=str(int(HEADROOM_MAX_PERCENT)))
    parser.add_argument("--headroom-resume-percent", default=str(int(HEADROOM_RESUME_PERCENT)))
    parser.add_argument("--headroom-max-ram-percent", default=str(int(HEADROOM_MAX_RAM_PERCENT)))
    parser.add_argument("--headroom-max-gpu-mem-percent", default=str(int(HEADROOM_MAX_GPU_MEM_PERCENT)))
    parser.add_argument("--min-free-ram-gb", default=str(int(HEADROOM_MIN_FREE_RAM_GB)))
    parser.add_argument("--preflight-timeout", default="120")
    parser.add_argument("--render-jobs", type=int, default=WEBGL_RENDER_JOBS, help="Forwarded WebGL render subprocess concurrency.")
    parser.add_argument("--renderer-batch-size", type=int, default=WEBGL_RENDERER_BATCH_SIZE, help="Forwarded variants per Node/WebGL renderer process.")
    parser.add_argument("--check-jobs", type=int, default=WEBGL_CHECK_JOBS, help="Forwarded rendered variant smoke-check concurrency.")
    parser.add_argument("--check-mode", choices=["in-process", "subprocess"], default="subprocess", help="Forwarded rendered variant smoke-check execution mode.")
    parser.add_argument("--shared-browser", action="store_true", help="Forwarded: reuse one headless browser across renderer subprocesses.")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-yolo-check", action="store_true")
    parser.add_argument("--skip-label-view-check", action="store_true")
    parser.add_argument("--balanced-subset-count", type=int, default=None, help="Package this many rendered variants after balanced visible-count selection; 0 disables catalog selection.")
    parser.add_argument("--balanced-subset-classes", default="", help="Override catalog balanced-subset classes.")
    parser.add_argument("--balanced-subset-min-per-class", type=int, default=None)
    parser.add_argument("--balanced-subset-max-class-spread", type=int, default=None)
    parser.add_argument("--balanced-subset-max-class-ratio", type=float, default=None)
    parser.add_argument("--balanced-subset-max-combinations", type=int, default=None)
    parser.add_argument("--run-diagnostic-gates", action="store_true", help="Run catalog-declared diagnostic gates after rendering.")
    parser.add_argument("--skip-smoke-gate", action="store_true")
    parser.add_argument("--skip-trainable-gate", action="store_true")
    parser.add_argument("--train-views", default="detect", help="Comma-separated train views for trainable-candidate gates.")
    parser.add_argument("--require-visual-note-quality", action="store_true", help="Forward readable-note pixel/softness QA to the trainable gate.")
    parser.add_argument(
        "--fragment-review-policy",
        choices=["auto", "diagnostic", "ignore"],
        default="auto",
        help="auto ignores review-required fragments only when fragment is a selected trainable-candidate view.",
    )
    parser.add_argument("--allow-zero-visible-trainable", action="store_true", help="Allow zero visible banknotes in a trainable-candidate gate.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing catalog: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def find_recipe(catalog: dict, recipe_id: str) -> dict:
    for row in catalog.get("recipes", []):
        if row.get("id") == recipe_id:
            return row
    raise SystemExit(f"recipe not found: {recipe_id}")


def choose_scene_mode(recipe: dict, override: str) -> str:
    if override:
        if override not in RUNNABLE_SCENE_MODES:
            raise SystemExit(f"--scene-mode must be one of {sorted(RUNNABLE_SCENE_MODES)}")
        return override
    for scene_mode in recipe.get("scene_modes", []):
        if scene_mode in RUNNABLE_SCENE_MODES:
            return str(scene_mode)
    raise SystemExit(f"{recipe['id']}: no runnable scene_modes; current modes={recipe.get('scene_modes', [])}")


def parse_train_views(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()}


def choose_asset_side_policy(recipe: dict, override: str) -> str:
    policy = override or str(recipe.get("asset_side_policy", "any"))
    if policy not in WEBGL_ASSET_SIDE_POLICIES:
        raise SystemExit(f"--asset-side-policy must be one of {sorted(WEBGL_ASSET_SIDE_POLICIES)}")
    return policy


def choose_camera_profile(recipe: dict, override: str) -> str:
    profile = override or str(recipe.get("camera_profile", "generic_phone_jitter"))
    if profile not in WEBGL_CAMERA_PROFILES:
        raise SystemExit(f"--camera-profile must be one of {sorted(WEBGL_CAMERA_PROFILES)}")
    return profile


def choose_stack_pose_policy(recipe: dict, override: str) -> str:
    policy = override or str(recipe.get("stack_pose_policy", "default"))
    if policy not in WEBGL_STACK_POSE_POLICIES:
        raise SystemExit(f"--stack-pose-policy must be one of {sorted(WEBGL_STACK_POSE_POLICIES)}")
    return policy


def choose_clean_orientation_policy(recipe: dict, override: str) -> str:
    policy = override or str(recipe.get("clean_orientation_policy", "default"))
    if policy not in WEBGL_CLEAN_ORIENTATION_POLICIES:
        raise SystemExit(f"--clean-orientation-policy must be one of {sorted(WEBGL_CLEAN_ORIENTATION_POLICIES)}")
    return policy


def normalize_class_sequence(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def normalize_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def normalize_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def main() -> int:
    args = parse_args()
    catalog = read_json(resolve(args.catalog))
    recipe = find_recipe(catalog, args.recipe_id)
    scene_mode = choose_scene_mode(recipe, args.scene_mode)
    asset_side_policy = choose_asset_side_policy(recipe, args.asset_side_policy)
    asset_quality_policy = args.asset_quality_policy.strip() or str(recipe.get("asset_quality_policy", "latest_design")).strip() or "latest_design"
    if asset_quality_policy not in WEBGL_ASSET_QUALITY_POLICIES:
        raise SystemExit(f"--asset-quality-policy must be one of {sorted(WEBGL_ASSET_QUALITY_POLICIES)}")
    camera_profile = choose_camera_profile(recipe, args.camera_profile)
    camera_isp_policy = args.camera_isp_policy.strip() or str(recipe.get("camera_isp_policy", "default")).strip() or "default"
    if camera_isp_policy not in WEBGL_CAMERA_ISP_POLICIES:
        raise SystemExit(f"unsupported camera_isp_policy: {camera_isp_policy}")
    stack_pose_policy = choose_stack_pose_policy(recipe, args.stack_pose_policy)
    clean_orientation_policy = choose_clean_orientation_policy(recipe, args.clean_orientation_policy)
    class_sequence = args.class_sequence.strip() or normalize_class_sequence(recipe.get("class_sequence", ""))
    note_condition_policy = args.note_condition_policy.strip() or str(recipe.get("note_condition_policy", "mixed")).strip() or "mixed"
    if note_condition_policy not in WEBGL_NOTE_CONDITION_POLICIES:
        raise SystemExit(f"unsupported note_condition_policy: {note_condition_policy}")
    lens_distortion_policy = args.lens_distortion_policy.strip() or str(recipe.get("lens_distortion_policy", "off")).strip() or "off"
    if lens_distortion_policy not in {"off", "phone_mild"}:
        raise SystemExit(f"unsupported lens_distortion_policy: {lens_distortion_policy}")
    note_print_tone_policy = args.note_print_tone_policy.strip() or str(recipe.get("note_print_tone_policy", "off")).strip() or "off"
    if note_print_tone_policy not in WEBGL_NOTE_PRINT_TONE_POLICIES:
        raise SystemExit(f"unsupported note_print_tone_policy: {note_print_tone_policy}")
    texture_qa_effects = args.texture_qa_effects.strip() or str(recipe.get("texture_qa_effects", "flat")).strip() or "flat"
    if texture_qa_effects not in WEBGL_TEXTURE_QA_EFFECTS:
        raise SystemExit(f"unsupported texture_qa_effects: {texture_qa_effects}")
    occluder_policy = args.occluder_policy.strip() or str(recipe.get("occluder_policy", "scene_default")).strip() or "scene_default"
    if occluder_policy not in WEBGL_OCCLUDER_POLICIES:
        raise SystemExit(f"unsupported occluder_policy: {occluder_policy}")
    negative_prop_policy = args.negative_prop_policy.strip() or str(recipe.get("negative_prop_policy", "classic")).strip() or "classic"
    if negative_prop_policy not in WEBGL_NEGATIVE_PROP_POLICIES:
        raise SystemExit(f"unsupported negative_prop_policy: {negative_prop_policy}")
    count = int(args.count if args.count is not None else recipe.get("render_pool_count", 4))
    if count < 1:
        raise SystemExit("--count must be positive")
    artifact_status = args.artifact_status or STATUS_TO_BATCH_STATUS.get(str(recipe.get("artifact_status")), "diagnostic")
    balanced_subset = recipe.get("balanced_subset", {})
    if not isinstance(balanced_subset, dict):
        balanced_subset = {}
    balanced_subset_count = (
        args.balanced_subset_count
        if args.balanced_subset_count is not None
        else normalize_optional_int(balanced_subset.get("count"))
    )
    balanced_subset_count = int(balanced_subset_count or 0)
    if balanced_subset_count > count:
        raise SystemExit(f"--balanced-subset-count {balanced_subset_count} exceeds --count {count}")
    balanced_subset_classes = (
        args.balanced_subset_classes.strip()
        or normalize_class_sequence(balanced_subset.get("classes", ""))
    )
    balanced_subset_min_per_class = (
        args.balanced_subset_min_per_class
        if args.balanced_subset_min_per_class is not None
        else normalize_optional_int(balanced_subset.get("min_per_class"))
    )
    balanced_subset_max_class_spread = (
        args.balanced_subset_max_class_spread
        if args.balanced_subset_max_class_spread is not None
        else normalize_optional_int(balanced_subset.get("max_class_spread"))
    )
    balanced_subset_max_class_ratio = (
        args.balanced_subset_max_class_ratio
        if args.balanced_subset_max_class_ratio is not None
        else normalize_optional_float(balanced_subset.get("max_class_ratio"))
    )
    balanced_subset_max_combinations = (
        args.balanced_subset_max_combinations
        if args.balanced_subset_max_combinations is not None
        else normalize_optional_int(balanced_subset.get("max_combinations"))
    )
    fragment_review_policy = args.fragment_review_policy
    if fragment_review_policy == "auto":
        train_views = parse_train_views(args.train_views)
        fragment_review_policy = (
            "ignore"
            if artifact_status == "trainable-candidate" and "fragment" in train_views
            else "diagnostic"
        )
    out_root = args.out_root or Path("data") / "synthetic" / f"{slug(args.recipe_id)}_v{args.start_variant}_{args.start_variant + count - 1}"

    cmd = [
        sys.executable,
        "scripts/render_webgl_variant_batch.py",
        "--out-root",
        str(out_root),
        "--start-variant",
        str(args.start_variant),
        "--count",
        str(count),
        "--scene-mode",
        scene_mode,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--visual-scale",
        str(args.visual_scale),
        "--asset-side-policy",
        asset_side_policy,
        "--asset-quality-policy",
        asset_quality_policy,
        "--camera-profile",
        camera_profile,
        "--camera-isp-policy",
        camera_isp_policy,
        "--stack-pose-policy",
        stack_pose_policy,
        "--clean-orientation-policy",
        clean_orientation_policy,
        "--occluder-policy",
        occluder_policy,
        "--negative-prop-policy",
        negative_prop_policy,
        "--texture-qa-effects",
        texture_qa_effects,
        "--recipe-name",
        args.recipe_id,
        "--artifact-status",
        artifact_status,
        "--fragment-review-policy",
        fragment_review_policy,
        "--intended-use",
        str(recipe.get("intended_use", "")),
        "--notes",
        f"promotion_gate={recipe.get('promotion_gate', '')}; current_blocker={recipe.get('current_blocker', '')}",
        "--headroom-max-percent",
        args.headroom_max_percent,
        "--headroom-resume-percent",
        args.headroom_resume_percent,
        "--headroom-max-ram-percent",
        args.headroom_max_ram_percent,
        "--headroom-max-gpu-mem-percent",
        args.headroom_max_gpu_mem_percent,
        "--min-free-ram-gb",
        args.min_free_ram_gb,
        "--preflight-timeout",
        args.preflight_timeout,
        "--render-jobs",
        str(args.render_jobs),
        "--renderer-batch-size",
        str(args.renderer_batch_size),
        "--check-jobs",
        str(args.check_jobs),
        "--check-mode",
        args.check_mode,
    ]
    if class_sequence:
        cmd.extend(["--class-sequence", class_sequence])
    if note_condition_policy != "mixed":
        cmd.extend(["--note-condition-policy", note_condition_policy])
    if lens_distortion_policy != "off":
        cmd.extend(["--lens-distortion-policy", lens_distortion_policy])
    if note_print_tone_policy != "off":
        cmd.extend(["--note-print-tone-policy", note_print_tone_policy])
    if balanced_subset_count > 0:
        cmd.extend(["--balanced-subset-count", str(balanced_subset_count)])
        if balanced_subset_classes:
            cmd.extend(["--balanced-subset-classes", balanced_subset_classes])
        if balanced_subset_min_per_class is not None:
            cmd.extend(["--balanced-subset-min-per-class", str(balanced_subset_min_per_class)])
        if balanced_subset_max_class_spread is not None:
            cmd.extend(["--balanced-subset-max-class-spread", str(balanced_subset_max_class_spread)])
        if balanced_subset_max_class_ratio is not None:
            cmd.extend(["--balanced-subset-max-class-ratio", str(balanced_subset_max_class_ratio)])
        if balanced_subset_max_combinations is not None:
            cmd.extend(["--balanced-subset-max-combinations", str(balanced_subset_max_combinations)])
    if args.background_dir:
        cmd.extend(["--background-dir", str(args.background_dir)])
    if args.environment_dir:
        cmd.extend(["--environment-dir", str(args.environment_dir)])
        cmd.extend(["--environment-bank-config", str(args.environment_bank_config)])
    if args.browser_executable:
        cmd.extend(["--browser-executable", str(args.browser_executable)])
    if args.shared_browser:
        cmd.append("--shared-browser")
    if args.skip_render:
        cmd.append("--skip-render")
    if args.skip_yolo_check:
        cmd.append("--skip-yolo-check")
    if args.skip_label_view_check:
        cmd.append("--skip-label-view-check")

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if args.run_diagnostic_gates:
        gate_cmd = [
            sys.executable,
            "scripts/check_webgl_recipe_diagnostic_gates.py",
            "--root",
            str(out_root),
            "--recipe-id",
            args.recipe_id,
            "--catalog",
            str(resolve(args.catalog)),
        ]
        print(" ".join(gate_cmd), flush=True)
        subprocess.run(gate_cmd, cwd=ROOT, check=True)
    if artifact_status == "smoke" and not args.skip_smoke_gate:
        gate_cmd = [
            sys.executable,
            "scripts/check_webgl_smoke_gate.py",
            "--root",
            str(out_root),
            "--require-recipe",
            args.recipe_id,
        ]
        if scene_mode != "auto":
            gate_cmd.extend(["--require-scene-mode", scene_mode])
        gate_cmd.extend(["--require-asset-side-policy", asset_side_policy])
        gate_cmd.extend(["--require-camera-profile", camera_profile])
        print(" ".join(gate_cmd), flush=True)
        subprocess.run(gate_cmd, cwd=ROOT, check=True)
    if artifact_status == "trainable-candidate" and not args.skip_trainable_gate:
        gate_cmd = [
            sys.executable,
            "scripts/check_webgl_trainable_candidate_gate.py",
            "--root",
            str(out_root),
            "--require-recipe",
            args.recipe_id,
            "--train-views",
            args.train_views,
        ]
        if scene_mode != "auto":
            gate_cmd.extend(["--require-scene-mode", scene_mode])
        gate_cmd.extend(["--require-asset-side-policy", asset_side_policy])
        gate_cmd.extend(["--require-camera-profile", camera_profile])
        if args.allow_zero_visible_trainable:
            gate_cmd.append("--allow-zero-visible")
        if args.require_visual_note_quality:
            gate_cmd.append("--require-visual-note-quality")
        print(" ".join(gate_cmd), flush=True)
        subprocess.run(gate_cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
