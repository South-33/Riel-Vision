#!/usr/bin/env python
"""Run the WebGL trainable-candidate operations sequence."""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_trainable_candidates_v1.json"
DEFAULT_MIX = ROOT / "configs" / "cashsnap_webgl_trainable_candidates_mix.yaml"
DEFAULT_REVIEW_OUT = ROOT / "data" / "review" / "webgl_trainable_candidates_visual_review_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--mix-out", type=Path, default=DEFAULT_MIX)
    parser.add_argument("--review-out", type=Path, default=DEFAULT_REVIEW_OUT)
    parser.add_argument("--skip-render", action="store_true", help="Repackage/check existing rendered variants.")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument(
        "--max-review-images-per-recipe",
        type=int,
        default=12,
        help="Sample this many packaged scenes per recipe for visual review; 0 means include every image.",
    )
    parser.add_argument(
        "--review-selection-policy",
        choices=["easy", "random"],
        default="easy",
        help="Choose lower-clutter scenes first, or random scenes, when sampling the visual review pack.",
    )
    parser.add_argument("--require-visual-review", action="store_true", help="Require review.csv to have accepted rows only.")
    parser.add_argument("--skip-p1-readiness", action="store_true")
    parser.add_argument("--train-smoke", action="store_true", help="Run a tiny headroom training smoke on the candidate mix.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--visual-scale", default=None, help="Override every suite recipe visual_scale. Defaults to each suite row.")
    parser.add_argument("--browser-executable", type=Path, default=None, help="Optional Chromium/Edge executable override.")
    parser.add_argument("--headroom-max-percent", default=str(int(HEADROOM_MAX_PERCENT)))
    parser.add_argument("--headroom-resume-percent", default=str(int(HEADROOM_RESUME_PERCENT)))
    parser.add_argument("--headroom-max-ram-percent", default=str(int(HEADROOM_MAX_RAM_PERCENT)))
    parser.add_argument("--headroom-max-gpu-mem-percent", default=str(int(HEADROOM_MAX_GPU_MEM_PERCENT)))
    parser.add_argument("--min-free-ram-gb", default=str(int(HEADROOM_MIN_FREE_RAM_GB)))
    parser.add_argument("--preflight-timeout", default="120")
    parser.add_argument("--render-jobs", type=int, default=WEBGL_RENDER_JOBS)
    parser.add_argument("--renderer-batch-size", type=int, default=WEBGL_RENDERER_BATCH_SIZE)
    parser.add_argument("--check-jobs", type=int, default=WEBGL_CHECK_JOBS)
    parser.add_argument("--check-mode", choices=["in-process", "subprocess"], default="subprocess")
    parser.add_argument("--shared-browser", action="store_true", help="Reuse one headless browser per recipe render batch.")
    return parser.parse_args()


def rel(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    return resolved.relative_to(ROOT).as_posix()


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    suite = rel(args.suite)
    mix_out = rel(args.mix_out)
    review_out = rel(args.review_out)

    run([sys.executable, "scripts/check_webgl_trainable_candidate_suite.py", "--suite", suite], args.dry_run)

    suite_cmd = [
        sys.executable,
        "scripts/run_webgl_trainable_candidate_suite.py",
        "--suite",
        suite,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
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
    if args.visual_scale is not None:
        suite_cmd.extend(["--visual-scale", str(args.visual_scale)])
    if args.skip_render:
        suite_cmd.append("--skip-render")
    if args.browser_executable:
        suite_cmd.extend(["--browser-executable", str(args.browser_executable)])
    if args.shared_browser:
        suite_cmd.append("--shared-browser")
    if args.dry_run:
        suite_cmd.append("--dry-run")
    run(suite_cmd, args.dry_run)

    run(
        [
            sys.executable,
            "scripts/build_webgl_mix_yaml.py",
            "--suite",
            suite,
            "--gate-kind",
            "trainable-candidate",
            "--out",
            mix_out,
        ],
        args.dry_run,
    )

    if not args.skip_review_pack:
        run(
            [
                sys.executable,
                "scripts/make_webgl_visual_review_pack.py",
                "--suite",
                suite,
                "--out-dir",
                review_out,
                "--max-images-per-recipe",
                str(args.max_review_images_per_recipe),
                "--selection-policy",
                args.review_selection_policy,
            ],
            args.dry_run,
        )
        review_check = [
            sys.executable,
            "scripts/check_webgl_visual_review.py",
            "--review-csv",
            str(Path(review_out) / "review.csv"),
        ]
        if args.require_visual_review:
            review_check.append("--require-accepted")
        run(review_check, args.dry_run)

    if not args.skip_p1_readiness:
        run(
            [
                sys.executable,
                "scripts/check_webgl_p1_readiness.py",
                "--smoke-mix",
                mix_out,
            ],
            args.dry_run,
        )

    if args.train_smoke:
        run(
            [
                sys.executable,
                "scripts/bench_train_with_headroom.py",
                "--data",
                mix_out,
                "--name",
                "webgl_trainable_candidates_tiny_train_probe",
                "--epochs",
                "1",
                "--max-train-batches",
                "2",
                "--imgsz",
                "416",
                "--batch",
                "2",
                "--workers",
                "0",
                "--quiet",
                "--no-val",
                "--exist-ok",
            ],
            args.dry_run,
        )

    print("ok: WebGL trainable-candidate pipeline command sequence completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
