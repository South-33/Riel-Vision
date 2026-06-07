#!/usr/bin/env python
"""Compare mined real background-FP visuals with synthetic negative roots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

import audit_yolo_crop_visual_domain_gap as crop_gap
import audit_yolo_domain_gap as domain_gap


ROOT = Path(__file__).resolve().parents[1]
IMAGE_STAT_KEYS = domain_gap.IMAGE_STAT_KEYS
CROP_STAT_KEYS = crop_gap.CROP_STAT_KEYS
BOX_STAT_KEYS = ["box_area", "box_width", "box_height", "box_aspect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-fp-json", action="append", default=[], type=Path)
    parser.add_argument("--real-review-manifest", action="append", default=[], type=Path)
    parser.add_argument("--model-label", default="", help="Optional model_label filter for --real-fp-json rows.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence filter for --real-fp-json rows.")
    parser.add_argument(
        "--source-contains",
        action="append",
        default=[],
        help="Optional substring filter for real FP source labels. Repeatable.",
    )
    parser.add_argument(
        "--synthetic-root",
        action="append",
        default=[],
        type=Path,
        help="Synthetic root containing manifest.json. Repeatable.",
    )
    parser.add_argument(
        "--synthetic-crop-mode",
        default="full",
        choices=["full", "center80", "none"],
        help="Crop proxy for zero-label synthetic negatives.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--image-csv-out", type=Path, default=None)
    parser.add_argument("--crop-csv-out", type=Path, default=None)
    parser.add_argument("--top-deltas", type=int, default=8)
    parser.add_argument("--min-real-crops", type=int, default=1)
    parser.add_argument("--min-synthetic-crops", type=int, default=1)
    parser.add_argument(
        "--max-abs-image-delta",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Limit abs(synthetic-real) image-stat mean delta.",
    )
    parser.add_argument(
        "--max-abs-crop-delta",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Limit abs(synthetic-real) crop-stat mean delta.",
    )
    parser.add_argument("--fail-on-gap", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def approx_equal(left: float, right: float, epsilon: float = 1e-9) -> bool:
    return abs(left - right) <= epsilon


def row_matches_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.model_label and str(row.get("model_label")) != args.model_label:
        return False
    if args.conf is not None and not approx_equal(float(row.get("conf", -1.0)), args.conf):
        return False
    if args.source_contains:
        source = str(row.get("image_root", ""))
        if not any(token in source for token in args.source_contains):
            return False
    return True


def review_manifests_from_fp_json(path: Path, args: argparse.Namespace) -> list[Path]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {repo_rel(path)}")
    manifests: list[Path] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict) or not row_matches_filters(row, args):
            continue
        review = row.get("review", {})
        manifest = review.get("manifest") if isinstance(review, dict) else None
        if manifest:
            manifests.append(resolve(Path(str(manifest))))
    return manifests


def load_review_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        resolved = resolve(path)
        if not resolved.exists():
            raise SystemExit(f"missing real review manifest: {repo_rel(resolved)}")
        manifest_rows = load_json(resolved)
        if not isinstance(manifest_rows, list):
            raise SystemExit(f"expected review manifest list: {repo_rel(resolved)}")
        for row in manifest_rows:
            if not isinstance(row, dict):
                continue
            image = str(row.get("image", ""))
            crop = str(row.get("crop", ""))
            bbox = json.dumps(row.get("bbox_xyxy", []), sort_keys=True)
            key = (image, crop, bbox)
            if not image or key in seen:
                continue
            seen.add(key)
            rows.append({**row, "review_manifest": repo_rel(resolved)})
    return rows


def clipped_xyxy(bbox: list[Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox)
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def box_stats(left: int, top: int, right: int, bottom: int, width: int, height: int) -> dict[str, float]:
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    return {
        "box_area": box_width * box_height,
        "box_width": box_width,
        "box_height": box_height,
        "box_aspect": (right - left) / (bottom - top) if bottom > top else 0.0,
    }


def center_crop_box(width: int, height: int, frac: float) -> tuple[int, int, int, int]:
    crop_w = max(1, int(round(width * frac)))
    crop_h = max(1, int(round(height * frac)))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return left, top, min(width, left + crop_w), min(height, top + crop_h)


def real_rows(review_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    for row in review_rows:
        image_path = resolve(Path(str(row["image"])))
        if not image_path.exists():
            raise SystemExit(f"missing real FP image: {repo_rel(image_path)}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        image_rows.append(
            {
                "family": "real",
                "group": "real_mined_fp",
                "image": repo_rel(image_path),
                "review_manifest": row.get("review_manifest", ""),
                "class": row.get("class", ""),
                "confidence": row.get("confidence", ""),
                **domain_gap.image_stats(image_path),
            }
        )

        crop_path = resolve(Path(str(row.get("crop", "")))) if row.get("crop") else None
        crop_image = None
        if crop_path and crop_path.exists():
            with Image.open(crop_path) as handle:
                crop_image = handle.convert("RGB")
        else:
            box = clipped_xyxy(list(row.get("bbox_xyxy", [])), width, height)
            if box is not None:
                crop_image = image.crop(box)
        if crop_image is None:
            continue
        box = clipped_xyxy(list(row.get("bbox_xyxy", [])), width, height) or (0, 0, width, height)
        crop_rows.append(
            {
                "family": "real",
                "group": "real_mined_fp",
                "image": repo_rel(image_path),
                "crop": repo_rel(crop_path) if crop_path and crop_path.exists() else "",
                "class": row.get("class", ""),
                "confidence": row.get("confidence", ""),
                **box_stats(*box, width, height),
                **crop_gap.crop_stats(crop_image),
            }
        )
    return image_rows, crop_rows


def synthetic_manifest_rows(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing synthetic manifest: {repo_rel(manifest_path)}")
    rows = load_json(manifest_path)
    if not isinstance(rows, list):
        raise SystemExit(f"expected manifest list: {repo_rel(manifest_path)}")
    return [row for row in rows if isinstance(row, dict)]


def synthetic_prop_summary(root: Path, manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prop_kinds: Counter[str] = Counter()
    hardness: Counter[str] = Counter()
    total_props = 0
    for row in manifest_rows:
        metadata = row.get("source_metadata")
        if not metadata:
            continue
        metadata_path = root / str(metadata)
        if not metadata_path.exists():
            continue
        payload = load_json(metadata_path)
        for prop in payload.get("occluders", []):
            if not isinstance(prop, dict):
                continue
            total_props += 1
            prop_kinds[str(prop.get("propKind", prop.get("kind", "unknown")))] += 1
            hardness[str(prop.get("negativeConfusionHardness", "none"))] += 1
    return {
        "total_props": total_props,
        "prop_kinds": dict(sorted(prop_kinds.items())),
        "negative_confusion_hardness": dict(sorted(hardness.items())),
    }


def synthetic_rows(roots: list[Path], crop_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    prop_summaries: dict[str, Any] = {}
    for raw_root in roots:
        root = resolve(raw_root)
        manifest_rows = synthetic_manifest_rows(root)
        group = "synthetic:" + root.name
        prop_summaries[repo_rel(root)] = synthetic_prop_summary(root, manifest_rows)
        for row in manifest_rows:
            image_value = row.get("image")
            if not image_value:
                continue
            image_path = root / str(image_value)
            if not image_path.exists():
                raise SystemExit(f"missing synthetic image: {repo_rel(image_path)}")
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            width, height = image.size
            image_rows.append(
                {
                    "family": "synthetic",
                    "group": group,
                    "image": repo_rel(image_path),
                    "variant": row.get("variant", ""),
                    **domain_gap.image_stats(image_path),
                }
            )
            if crop_mode == "none":
                continue
            box = (0, 0, width, height) if crop_mode == "full" else center_crop_box(width, height, 0.80)
            crop_rows.append(
                {
                    "family": "synthetic",
                    "group": group,
                    "image": repo_rel(image_path),
                    "crop": crop_mode,
                    "class": "unknown_banknote_proxy",
                    "confidence": "",
                    **box_stats(*box, width, height),
                    **crop_gap.crop_stats(image.crop(box)),
                }
            )
    return image_rows, crop_rows, prop_summaries


def summarize(rows: list[dict[str, Any]], keys: list[str], count_key: str) -> dict[str, Any]:
    groups = sorted({str(row["group"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})

    def group_summary(group_key: str, group_value: str) -> dict[str, Any]:
        group_rows = [row for row in rows if str(row[group_key]) == group_value]
        return {
            count_key: len(group_rows),
            "stats": domain_gap.summarize_numeric(group_rows, keys),
        }

    return {
        "by_family": {family: group_summary("family", family) for family in families},
        "by_group": {group: group_summary("group", group) for group in groups},
    }


def deltas(real: dict[str, Any], synthetic: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    real_stats = real.get("stats", {}) if isinstance(real, dict) else {}
    synthetic_stats = synthetic.get("stats", {}) if isinstance(synthetic, dict) else {}
    for key in keys:
        real_mean = real_stats.get(key, {}).get("mean") if isinstance(real_stats, dict) else None
        synthetic_mean = synthetic_stats.get(key, {}).get("mean") if isinstance(synthetic_stats, dict) else None
        out[key] = None if real_mean is None or synthetic_mean is None else synthetic_mean - real_mean
    return out


def parse_metric_limits(specs: list[str], valid_keys: list[str], label: str) -> dict[str, float]:
    limits: dict[str, float] = {}
    valid = set(valid_keys)
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"{label} limit must be METRIC=VALUE, got {spec!r}")
        metric, raw_value = (part.strip() for part in spec.split("=", 1))
        if metric not in valid:
            raise SystemExit(f"unknown {label} metric {metric!r}; expected one of {sorted(valid)}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"{label} limit for {metric!r} must be numeric, got {raw_value!r}") from exc
        if value < 0:
            raise SystemExit(f"{label} limit for {metric!r} must be non-negative")
        limits[metric] = value
    return limits


def gate(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image_limits = parse_metric_limits(args.max_abs_image_delta, IMAGE_STAT_KEYS, "image")
    crop_limits = parse_metric_limits(args.max_abs_crop_delta, CROP_STAT_KEYS, "crop")
    requested = bool(args.fail_on_gap or image_limits or crop_limits)
    failures: list[str] = []
    image_real = payload["image_summary"]["by_family"].get("real", {})
    image_synth = payload["image_summary"]["by_family"].get("synthetic", {})
    crop_real = payload["crop_summary"]["by_family"].get("real", {})
    crop_synth = payload["crop_summary"]["by_family"].get("synthetic", {})

    if requested and int(crop_real.get("crops", 0) or 0) < args.min_real_crops:
        failures.append(f"real crop count {crop_real.get('crops', 0)} below minimum {args.min_real_crops}")
    if requested and int(crop_synth.get("crops", 0) or 0) < args.min_synthetic_crops:
        failures.append(
            f"synthetic crop count {crop_synth.get('crops', 0)} below minimum {args.min_synthetic_crops}"
        )

    image_deltas = payload["deltas"]["synthetic_minus_real"]["image_stats"]
    crop_deltas = payload["deltas"]["synthetic_minus_real"]["crop_stats"]
    for metric, limit in image_limits.items():
        delta = image_deltas.get(metric)
        if delta is None:
            failures.append(f"image_stats.{metric} is unavailable")
        elif abs(float(delta)) > limit:
            failures.append(f"image_stats.{metric} delta {float(delta):.6f} exceeds abs limit {limit:.6f}")
    for metric, limit in crop_limits.items():
        delta = crop_deltas.get(metric)
        if delta is None:
            failures.append(f"crop_stats.{metric} is unavailable")
        elif abs(float(delta)) > limit:
            failures.append(f"crop_stats.{metric} delta {float(delta):.6f} exceeds abs limit {limit:.6f}")

    return {
        "requested": requested,
        "passed": not failures,
        "failures": failures,
        "limits": {"image": image_limits, "crop": crop_limits},
        "image_counts": {
            "real": image_real.get("images", 0),
            "synthetic": image_synth.get("images", 0),
        },
        "crop_counts": {
            "real": crop_real.get("crops", 0),
            "synthetic": crop_synth.get("crops", 0),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_text(deltas_by_metric: dict[str, Any], metrics: list[str]) -> str:
    return " ".join(
        f"{metric} {float(deltas_by_metric[metric]):+.3f}"
        for metric in metrics
        if deltas_by_metric.get(metric) is not None
    )


def main() -> int:
    args = parse_args()
    if not args.real_fp_json and not args.real_review_manifest:
        raise SystemExit("provide --real-fp-json or --real-review-manifest")
    if not args.synthetic_root:
        raise SystemExit("provide at least one --synthetic-root")

    review_paths = [resolve(path) for path in args.real_review_manifest]
    for fp_json in args.real_fp_json:
        review_paths.extend(review_manifests_from_fp_json(resolve(fp_json), args))
    review_paths = sorted(set(review_paths))
    if not review_paths:
        raise SystemExit("no real review manifests matched the requested filters")

    real_image_rows, real_crop_rows = real_rows(load_review_rows(review_paths))
    synth_image_rows, synth_crop_rows, prop_summaries = synthetic_rows(args.synthetic_root, args.synthetic_crop_mode)
    image_rows = real_image_rows + synth_image_rows
    crop_rows = real_crop_rows + synth_crop_rows

    image_summary = summarize(image_rows, IMAGE_STAT_KEYS, "images")
    crop_summary = summarize(crop_rows, CROP_STAT_KEYS + BOX_STAT_KEYS, "crops")
    image_real = image_summary["by_family"].get("real", {})
    image_synth = image_summary["by_family"].get("synthetic", {})
    crop_real = crop_summary["by_family"].get("real", {})
    crop_synth = crop_summary["by_family"].get("synthetic", {})
    payload = {
        "schema": "cashsnap_negative_fp_visual_gap_v1",
        "real_review_manifests": [repo_rel(path) for path in review_paths],
        "synthetic_roots": [repo_rel(resolve(path)) for path in args.synthetic_root],
        "synthetic_crop_mode": args.synthetic_crop_mode,
        "image_summary": image_summary,
        "crop_summary": crop_summary,
        "synthetic_prop_summaries": prop_summaries,
        "deltas": {
            "synthetic_minus_real": {
                "image_stats": deltas(image_real, image_synth, IMAGE_STAT_KEYS),
                "crop_stats": deltas(crop_real, crop_synth, CROP_STAT_KEYS + BOX_STAT_KEYS),
            }
        },
    }
    payload["negative_visual_gap_gate"] = gate(payload, args)

    if args.json_out:
        out = resolve(args.json_out)
        write_json(out, payload)
        print(f"wrote_json={repo_rel(out)}")
    if args.image_csv_out:
        out = resolve(args.image_csv_out)
        write_csv(out, image_rows)
        print(f"wrote_image_csv={repo_rel(out)}")
    if args.crop_csv_out:
        out = resolve(args.crop_csv_out)
        write_csv(out, crop_rows)
        print(f"wrote_crop_csv={repo_rel(out)}")

    print(
        "images="
        f"real:{image_real.get('images', 0)} synthetic:{image_synth.get('images', 0)} "
        f"crops=real:{crop_real.get('crops', 0)} synthetic:{crop_synth.get('crops', 0)}"
    )
    image_delta = payload["deltas"]["synthetic_minus_real"]["image_stats"]
    crop_delta = payload["deltas"]["synthetic_minus_real"]["crop_stats"]
    print(
        "image_deltas: "
        + metric_text(image_delta, ["luma_mean", "luma_std", "luma_p05", "luma_p95", "saturation_std", "sharpness_grad_var"])
    )
    print(
        "crop_deltas: "
        + metric_text(
            crop_delta,
            ["box_area", "luma_mean", "luma_std", "luma_p05", "luma_p95", "saturation_std", "sharpness_grad_var"],
        )
    )
    gate_result = payload["negative_visual_gap_gate"]
    if gate_result["requested"]:
        print("negative_visual_gap_gate=" + ("passed" if gate_result["passed"] else "failed"))
        for failure in gate_result["failures"][: args.top_deltas]:
            print(f"- {failure}")
    return 1 if args.fail_on_gap and not gate_result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
