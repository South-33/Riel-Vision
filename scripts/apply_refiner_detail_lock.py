from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from local_runtime import ROOT, configure_project_cache


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore original note detail inside refiner outputs using readiness masks. "
            "This keeps learned background/edge/camera changes while forcing denomination evidence "
            "back to the trusted source pixels."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--lock-mask",
        choices=("detail_lock", "note", "note_edge"),
        default="detail_lock",
        help=(
            "Mask used for source-pixel restoration. detail_lock preserves edge freedom; "
            "note restores the whole note; note_edge restores the note plus contact edge band."
        ),
    )
    parser.add_argument("--feather-px", type=float, default=1.5, help="Blur radius for the restoration alpha.")
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
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def load_alpha(path: Path, size: tuple[int, int], feather_px: float) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != size:
            raise ValueError(f"mask size {mask.size} does not match image size {size}: {repo_rel(path)}")
        if feather_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        arr = np.asarray(mask, dtype=np.float32) / 255.0
    return arr[:, :, None]


def load_mask_array(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != size:
            raise ValueError(f"mask size {mask.size} does not match image size {size}: {repo_rel(path)}")
        return np.asarray(mask, dtype=np.uint8)


def load_lock_alpha(row: dict[str, Any], size: tuple[int, int], args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    if args.lock_mask == "detail_lock":
        mask_keys = ["detail_lock_mask"]
    elif args.lock_mask == "note":
        mask_keys = ["note_mask"]
    else:
        mask_keys = ["note_mask", "edge_band_mask"]

    combined = np.zeros((size[1], size[0]), dtype=np.uint8)
    for mask_key in mask_keys:
        mask = load_mask_array(resolve_repo_path(Path(str(row[mask_key]))), size)
        combined = np.maximum(combined, mask)
    mask_image = Image.fromarray(combined, mode="L")
    if args.feather_px > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=args.feather_px))
    alpha = np.asarray(mask_image, dtype=np.float32) / 255.0
    return alpha[:, :, None], mask_keys


def apply_lock(row: dict[str, Any], refined_root: Path, out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
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
    source = load_rgb(source_path)
    refined = load_rgb(refined_path)
    if source.shape != refined.shape:
        result["status"] = "fail"
        result["violations"].append("shape_changed")
        result["source_shape"] = list(source.shape)
        result["refined_shape"] = list(refined.shape)
        return result
    alpha, mask_keys = load_lock_alpha(row, (source.shape[1], source.shape[0]), args)
    locked = source * alpha + refined * (1.0 - alpha)
    out_path = out_root / f"{source_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(locked, 0, 255).astype(np.uint8), mode="RGB").save(out_path)
    result["locked_image"] = repo_rel(out_path)
    result["lock_masks"] = [row[mask_key] for mask_key in mask_keys]
    return result


def main() -> int:
    configure_project_cache()
    args = parse_args()
    manifest = resolve_repo_path(args.manifest)
    refined_root = resolve_repo_path(args.refined_root)
    out_root = resolve_repo_path(args.out_root)
    if not refined_root.exists():
        raise SystemExit(f"missing refined root: {repo_rel(refined_root)}")
    rows = read_jsonl(manifest)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    results = [apply_lock(row, refined_root, out_root, args) for row in rows]
    failures = [row for row in results if row["status"] != "pass"]
    summary = {
        "manifest": repo_rel(manifest),
        "refined_root": repo_rel(refined_root),
        "out_root": repo_rel(out_root),
        "rows": len(results),
        "failures": len(failures),
        "lock_mask": args.lock_mask,
        "feather_px": args.feather_px,
        "status": "pass" if not failures else "fail",
        "failed_rows": failures[:25],
        "outputs": results,
    }
    json_out = resolve_repo_path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": repo_rel(json_out), "status": summary["status"], "rows": len(results), "failures": len(failures)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
