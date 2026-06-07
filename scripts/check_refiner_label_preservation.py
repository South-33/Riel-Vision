from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from local_runtime import ROOT, configure_project_cache


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether synthetic->real refiner outputs preserve locked note details. "
            "Run this before treating refined images as YOLO/OCR training data."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Readiness synthetic_manifest.jsonl.")
    parser.add_argument(
        "--refined-root",
        type=Path,
        required=True,
        help="Directory containing refined images. The checker searches this root and images/train by source filename.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--max-detail-l1", type=float, default=8.0)
    parser.add_argument("--max-note-l1", type=float, default=18.0)
    parser.add_argument("--max-edge-l1", type=float, default=45.0)
    parser.add_argument("--max-fail-fraction", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional leading manifest row cap for tiny smokes.")
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
    candidates = [
        refined_root / source_image.name,
        refined_root / "images" / "train" / source_image.name,
        refined_root / source_image.relative_to(ROOT) if source_image.is_relative_to(ROOT) else refined_root / source_image.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    stem = source_image.stem
    for ext in IMAGE_EXTS:
        candidate = refined_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        candidate = refined_root / "images" / "train" / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def masked_l1(source: np.ndarray, refined: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.abs(source[mask] - refined[mask]).mean())


def check_row(row: dict[str, Any], refined_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_path = resolve_repo_path(Path(str(row["source_image"])))
    refined_path = find_refined_image(refined_root, source_path)
    result: dict[str, Any] = {
        "id": row.get("id", source_path.stem),
        "class_name": row.get("class_name", ""),
        "source_image": repo_rel(source_path),
        "refined_image": repo_rel(refined_path) if refined_path is not None else "",
        "status": "pass",
        "violations": [],
    }
    if refined_path is None:
        result["status"] = "fail"
        result["violations"].append("missing_refined_image")
        return result
    source_rgb = load_rgb(source_path)
    refined_rgb = load_rgb(refined_path)
    if source_rgb.shape != refined_rgb.shape:
        result["status"] = "fail"
        result["violations"].append("shape_changed")
        result["source_shape"] = list(source_rgb.shape)
        result["refined_shape"] = list(refined_rgb.shape)
        return result

    detail_mask = load_mask(resolve_repo_path(Path(str(row["detail_lock_mask"]))))
    note_mask = load_mask(resolve_repo_path(Path(str(row["note_mask"]))))
    edge_mask = load_mask(resolve_repo_path(Path(str(row["edge_band_mask"]))))
    if detail_mask.shape != source_rgb.shape[:2] or note_mask.shape != source_rgb.shape[:2] or edge_mask.shape != source_rgb.shape[:2]:
        result["status"] = "fail"
        result["violations"].append("mask_shape_mismatch")
        return result

    detail_l1 = masked_l1(source_rgb, refined_rgb, detail_mask)
    note_l1 = masked_l1(source_rgb, refined_rgb, note_mask)
    edge_l1 = masked_l1(source_rgb, refined_rgb, edge_mask)
    result.update(
        {
            "detail_l1": round(detail_l1, 4),
            "note_l1": round(note_l1, 4),
            "edge_l1": round(edge_l1, 4),
            "detail_pixels": int(detail_mask.sum()),
            "note_pixels": int(note_mask.sum()),
            "edge_pixels": int(edge_mask.sum()),
        }
    )
    if detail_l1 > args.max_detail_l1:
        result["violations"].append("detail_l1_over_limit")
    if note_l1 > args.max_note_l1:
        result["violations"].append("note_l1_over_limit")
    if edge_l1 > args.max_edge_l1:
        result["violations"].append("edge_l1_over_limit")
    if result["violations"]:
        result["status"] = "fail"
    return result


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, manifest: Path, refined_root: Path) -> dict[str, Any]:
    failures = [row for row in rows if row["status"] != "pass"]
    fail_fraction = len(failures) / max(1, len(rows))
    metrics = {}
    for key in ("detail_l1", "note_l1", "edge_l1"):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            metrics[key] = {
                "mean": round(float(np.mean(values)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
                "max": round(float(np.max(values)), 4),
            }
    return {
        "manifest": repo_rel(manifest),
        "refined_root": repo_rel(refined_root),
        "rows": len(rows),
        "failures": len(failures),
        "fail_fraction": round(fail_fraction, 6),
        "max_fail_fraction": args.max_fail_fraction,
        "thresholds": {
            "max_detail_l1": args.max_detail_l1,
            "max_note_l1": args.max_note_l1,
            "max_edge_l1": args.max_edge_l1,
        },
        "metrics": metrics,
        "status": "pass" if fail_fraction <= args.max_fail_fraction else "fail",
        "failed_rows": failures[:25],
    }


def main() -> int:
    configure_project_cache()
    args = parse_args()
    if args.max_fail_fraction < 0 or args.max_fail_fraction > 1:
        raise SystemExit("--max-fail-fraction must be between 0 and 1")
    manifest = resolve_repo_path(args.manifest)
    refined_root = resolve_repo_path(args.refined_root)
    if not refined_root.exists():
        raise SystemExit(f"missing refined root: {repo_rel(refined_root)}")
    manifest_rows = read_jsonl(manifest)
    if args.max_rows > 0:
        manifest_rows = manifest_rows[: args.max_rows]
    row_results = [check_row(row, refined_root, args) for row in manifest_rows]
    summary = summarize(row_results, args, manifest, refined_root)
    json_out = resolve_repo_path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": repo_rel(json_out), "status": summary["status"], "failures": summary["failures"]}, sort_keys=True))
    if args.fail_on_violations and summary["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
