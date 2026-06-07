"""Audit synthetic note composite boundary strength from target-anchor metadata."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: str | Path) -> str:
    value = resolve(path)
    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return value.resolve().as_posix()


def load_records(root: Path, max_images: int, seed: int) -> list[dict[str, Any]]:
    metadata_path = root / "metadata" / "train.jsonl"
    if not metadata_path.exists():
        raise SystemExit(f"missing metadata jsonl: {repo_rel(metadata_path)}")
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record.get("quad_xy") and record.get("image")]
    if max_images > 0 and len(records) > max_images:
        rng = random.Random(seed)
        records = rng.sample(records, max_images)
    return records


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p05": percentile(values, 5),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def mask_from_quad(quad: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(quad, dtype=np.float32).round().astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def band_masks(mask: np.ndarray, band_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_size = max(3, band_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel)
    dilated = cv2.dilate(mask, kernel)
    inside = (mask > 0) & (eroded == 0)
    outside = (dilated > 0) & (mask == 0)
    boundary = inside | outside
    return inside, outside, boundary


def audit_record(root: Path, record: dict[str, Any], band_px: int) -> dict[str, Any] | None:
    image_path = root / record["image"]
    if not image_path.exists():
        return None
    with Image.open(image_path).convert("RGB") as image:
        rgb = np.asarray(image).astype(np.float32) / 255.0
    mask = mask_from_quad(record["quad_xy"], (rgb.shape[1], rgb.shape[0]))
    quad = np.asarray(record["quad_xy"], dtype=np.float32)
    quad_width = float(quad[:, 0].max() - quad[:, 0].min())
    quad_height = float(quad[:, 1].max() - quad[:, 1].min())
    quad_area = float((mask > 0).sum())
    inside, outside, boundary = band_masks(mask, band_px)
    if inside.sum() < 24 or outside.sum() < 24:
        return None
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    inside_rgb = rgb[inside]
    outside_rgb = rgb[outside]
    inside_mean = inside_rgb.mean(axis=0)
    outside_mean = outside_rgb.mean(axis=0)
    color_step = float(np.abs(inside_mean - outside_mean).mean())
    inside_grad = float(grad[inside].mean())
    outside_grad = float(grad[outside].mean())
    boundary_grad = float(grad[boundary].mean())
    grad_ratio = boundary_grad / max(0.0001, outside_grad)
    return {
        "image": record["image"],
        "class_name": record.get("class_name"),
        "composite_policy": record.get("composite_policy", ""),
        "shadow_policy": record.get("shadow_policy", ""),
        "boundary_grad_mean": boundary_grad,
        "inside_grad_mean": inside_grad,
        "outside_grad_mean": outside_grad,
        "boundary_to_outside_grad_ratio": grad_ratio,
        "edge_color_step_mean": color_step,
        "quad_width_px": quad_width,
        "quad_height_px": quad_height,
        "quad_area_px": quad_area,
        "inside_band_pixels": int(inside.sum()),
        "outside_band_pixels": int(outside.sum()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "image",
        "class_name",
        "composite_policy",
        "shadow_policy",
        "boundary_grad_mean",
        "inside_grad_mean",
        "outside_grad_mean",
        "boundary_to_outside_grad_ratio",
        "edge_color_step_mean",
        "quad_width_px",
        "quad_height_px",
        "quad_area_px",
        "inside_band_pixels",
        "outside_band_pixels",
    ]
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            values.append('"' + text.replace('"', '""') + '"' if "," in text else text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve(args.root)
    records = load_records(root, args.max_images, args.seed)
    rows: list[dict[str, Any]] = []
    for record in records:
        row = audit_record(root, record, args.band_px)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("no auditable rows")
    metrics = [
        "boundary_grad_mean",
        "inside_grad_mean",
        "outside_grad_mean",
        "boundary_to_outside_grad_ratio",
        "edge_color_step_mean",
        "quad_width_px",
        "quad_height_px",
        "quad_area_px",
    ]
    summary = {metric: summarize([float(row[metric]) for row in rows]) for metric in metrics}
    by_policy: dict[str, dict[str, Any]] = {}
    for policy in sorted({str(row.get("composite_policy") or "unknown") for row in rows}):
        policy_rows = [row for row in rows if str(row.get("composite_policy") or "unknown") == policy]
        by_policy[policy] = {
            metric: summarize([float(row[metric]) for row in policy_rows])
            for metric in metrics
        }
    return {
        "schema": "cashsnap_synthetic_composite_edge_audit_v1",
        "root": repo_rel(root),
        "records": len(records),
        "audited": len(rows),
        "band_px": args.band_px,
        "policy_counts": dict(Counter(str(row.get("composite_policy") or "unknown") for row in rows)),
        "summary": summary,
        "by_policy": by_policy,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit synthetic composite edge/contact metrics from quad metadata.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--band-px", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit(args)
    json_out = resolve(args.json_out)
    csv_out = resolve(args.csv_out) if args.csv_out else json_out.with_suffix(".csv")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_out, payload["rows"])
    ratio = payload["summary"]["boundary_to_outside_grad_ratio"]["mean"]
    color = payload["summary"]["edge_color_step_mean"]["mean"]
    print(
        "composite_edge_audit="
        f"{repo_rel(json_out)} audited={payload['audited']} "
        f"boundary_ratio_mean={ratio:.4f} edge_color_step_mean={color:.4f}"
    )
    print(f"csv={repo_rel(csv_out)}")


if __name__ == "__main__":
    main()
