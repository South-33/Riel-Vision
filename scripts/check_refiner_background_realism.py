from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from local_runtime import ROOT, configure_project_cache


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare refiner output background pixels against strict no-note "
            "CashSnap background patches. Use this after preservation checks "
            "and before visual QA or YOLO probes."
        )
    )
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--background-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-background-rows", type=int, default=0)
    parser.add_argument("--max-pixels-per-image", type=int, default=120_000)
    parser.add_argument("--max-luma-mean-shift", type=float, default=0.18)
    parser.add_argument("--max-saturation-mean-shift", type=float, default=0.12)
    parser.add_argument("--max-colorfulness-mean-shift", type=float, default=0.10)
    parser.add_argument("--min-luma-std-ratio", type=float, default=0.45)
    parser.add_argument("--min-hue-entropy-ratio", type=float, default=0.50)
    parser.add_argument("--max-dominant-hue-fraction", type=float, default=0.72)
    parser.add_argument("--fail-on-violations", action="store_true")
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing manifest: {repo_rel(path)}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{repo_rel(path)}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"empty manifest: {repo_rel(path)}")
    return rows


def find_refined_image(refined_root: Path, source_image: Path) -> Path | None:
    stem = source_image.stem
    candidates = [refined_root / source_image.name, refined_root / "images" / "train" / source_image.name]
    for ext in IMAGE_EXTS:
        candidates.append(refined_root / f"{stem}{ext}")
        candidates.append(refined_root / "images" / "train" / f"{stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {shape}: {repo_rel(path)}")
    return mask


def downsample_pixels(pixels: np.ndarray, max_pixels: int) -> np.ndarray:
    if pixels.shape[0] <= max_pixels:
        return pixels
    stride = max(1, math.ceil(pixels.shape[0] / max_pixels))
    return pixels[::stride][:max_pixels]


def image_features(rgb: np.ndarray, pixel_mask: np.ndarray | None, max_pixels: int) -> dict[str, float | int]:
    if pixel_mask is None:
        pixels = rgb.reshape(-1, 3)
    else:
        pixels = rgb[pixel_mask]
    if pixels.shape[0] < 512:
        raise ValueError("not enough background pixels for realism stats")
    pixels = downsample_pixels(pixels, max_pixels).astype(np.float32) / 255.0
    luma = pixels @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    hsv = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    saturation = hsv[:, 1]
    hue = hsv[:, 0]
    saturated = saturation > 0.05
    if saturated.any():
        hist, _ = np.histogram(hue[saturated], bins=18, range=(0.0, 360.0))
        probs = hist.astype(np.float32) / max(1, int(hist.sum()))
        nonzero = probs[probs > 0]
        hue_entropy = float(-(nonzero * np.log2(nonzero)).sum() / math.log2(18))
        dominant_hue_fraction = float(probs.max())
    else:
        hue_entropy = 0.0
        dominant_hue_fraction = 1.0
    channel_std = pixels.std(axis=0)
    colorfulness = float(channel_std.mean())
    return {
        "pixels": int(pixels.shape[0]),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "colorfulness": colorfulness,
        "hue_entropy": hue_entropy,
        "dominant_hue_fraction": dominant_hue_fraction,
    }


def summarize_feature_rows(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    keys = [
        key
        for key, value in rows[0].items()
        if key != "pixels" and isinstance(value, (float, int))
    ]
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        summary[key] = {
            "mean": round(float(values.mean()), 6),
            "p05": round(float(np.percentile(values, 5)), 6),
            "p50": round(float(np.percentile(values, 50)), 6),
            "p95": round(float(np.percentile(values, 95)), 6),
        }
    return summary


def candidate_features(rows: list[dict[str, Any]], refined_root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        source_image = resolve_repo_path(Path(str(row["source_image"])))
        refined_image = find_refined_image(refined_root, source_image)
        if refined_image is None:
            failures.append({"id": row.get("id", source_image.stem), "violation": "missing_refined_image"})
            continue
        try:
            rgb = load_rgb(refined_image)
            source_rgb = load_rgb(source_image)
            if source_rgb.shape != rgb.shape:
                raise ValueError(f"source/refined shape mismatch: {source_rgb.shape} vs {rgb.shape}")
            note_mask = load_mask(resolve_repo_path(Path(str(row["note_mask"]))), rgb.shape[:2])
            edge_mask = load_mask(resolve_repo_path(Path(str(row["edge_band_mask"]))), rgb.shape[:2])
            protected = note_mask | edge_mask
            features = image_features(rgb, ~protected, args.max_pixels_per_image)
            background_delta = np.abs(rgb[~protected].astype(np.float32) - source_rgb[~protected].astype(np.float32)).mean() / 255.0
            features["source_background_l1"] = float(background_delta)
        except (OSError, ValueError) as exc:
            failures.append({"id": row.get("id", source_image.stem), "violation": str(exc)})
            continue
        features.update({"id": row.get("id", source_image.stem), "image": repo_rel(refined_image)})
        feature_rows.append(features)
    return feature_rows, failures


def background_features(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        image_path = resolve_repo_path(Path(str(row["image"])))
        try:
            rgb = load_rgb(image_path)
            features = image_features(rgb, None, args.max_pixels_per_image)
        except (OSError, ValueError) as exc:
            failures.append({"image": repo_rel(image_path), "violation": str(exc)})
            continue
        features.update({"image": repo_rel(image_path)})
        feature_rows.append(features)
    return feature_rows, failures


def compare(candidate: dict[str, dict[str, float]], target: dict[str, dict[str, float]], args: argparse.Namespace) -> tuple[dict[str, float], list[str]]:
    comparisons = {
        "luma_mean_shift": abs(candidate["luma_mean"]["mean"] - target["luma_mean"]["mean"]),
        "saturation_mean_shift": abs(candidate["saturation_mean"]["mean"] - target["saturation_mean"]["mean"]),
        "colorfulness_mean_shift": abs(candidate["colorfulness"]["mean"] - target["colorfulness"]["mean"]),
        "luma_std_ratio": candidate["luma_std"]["mean"] / max(1e-6, target["luma_std"]["mean"]),
        "hue_entropy_ratio": candidate["hue_entropy"]["mean"] / max(1e-6, target["hue_entropy"]["mean"]),
        "dominant_hue_fraction": candidate["dominant_hue_fraction"]["mean"],
    }
    comparisons = {key: round(float(value), 6) for key, value in comparisons.items()}
    violations: list[str] = []
    if comparisons["luma_mean_shift"] > args.max_luma_mean_shift:
        violations.append("luma_mean_shift_over_limit")
    if comparisons["saturation_mean_shift"] > args.max_saturation_mean_shift:
        violations.append("saturation_mean_shift_over_limit")
    if comparisons["colorfulness_mean_shift"] > args.max_colorfulness_mean_shift:
        violations.append("colorfulness_mean_shift_over_limit")
    if comparisons["luma_std_ratio"] < args.min_luma_std_ratio:
        violations.append("luma_std_ratio_under_limit")
    if comparisons["hue_entropy_ratio"] < args.min_hue_entropy_ratio:
        violations.append("hue_entropy_ratio_under_limit")
    if comparisons["dominant_hue_fraction"] > args.max_dominant_hue_fraction:
        violations.append("dominant_hue_fraction_over_limit")
    return comparisons, violations


def main() -> int:
    configure_project_cache()
    args = parse_args()
    synthetic_manifest = resolve_repo_path(args.synthetic_manifest)
    refined_root = resolve_repo_path(args.refined_root)
    background_manifest = resolve_repo_path(args.background_manifest)
    if not refined_root.exists():
        raise SystemExit(f"missing refined root: {repo_rel(refined_root)}")
    candidate_rows = read_jsonl(synthetic_manifest)
    background_rows = read_jsonl(background_manifest)
    if args.max_rows > 0:
        candidate_rows = candidate_rows[: args.max_rows]
    if args.max_background_rows > 0:
        background_rows = background_rows[: args.max_background_rows]

    candidate_feature_rows, candidate_failures = candidate_features(candidate_rows, refined_root, args)
    background_feature_rows, background_failures = background_features(background_rows, args)
    if not candidate_feature_rows:
        raise SystemExit("no candidate background rows")
    if not background_feature_rows:
        raise SystemExit("no target background rows")

    candidate_summary = summarize_feature_rows(candidate_feature_rows)
    target_summary = summarize_feature_rows(background_feature_rows)
    comparisons, violations = compare(candidate_summary, target_summary, args)
    failures = candidate_failures + background_failures
    if failures:
        violations.append("feature_extraction_failures")
    summary = {
        "schema": "cashsnap_refiner_background_realism_v1",
        "synthetic_manifest": repo_rel(synthetic_manifest),
        "refined_root": repo_rel(refined_root),
        "background_manifest": repo_rel(background_manifest),
        "candidate_rows": len(candidate_feature_rows),
        "target_background_rows": len(background_feature_rows),
        "candidate_summary": candidate_summary,
        "target_background_summary": target_summary,
        "comparisons": comparisons,
        "thresholds": {
            "max_luma_mean_shift": args.max_luma_mean_shift,
            "max_saturation_mean_shift": args.max_saturation_mean_shift,
            "max_colorfulness_mean_shift": args.max_colorfulness_mean_shift,
            "min_luma_std_ratio": args.min_luma_std_ratio,
            "min_hue_entropy_ratio": args.min_hue_entropy_ratio,
            "max_dominant_hue_fraction": args.max_dominant_hue_fraction,
        },
        "violations": sorted(set(violations)),
        "status": "pass" if not violations else "fail",
        "failures": failures[:25],
    }
    json_out = resolve_repo_path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": repo_rel(json_out),
                "status": summary["status"],
                "violations": summary["violations"],
                "comparisons": comparisons,
            },
            sort_keys=True,
        )
    )
    if args.fail_on_violations and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
