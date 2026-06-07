from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from hardware_profile import headroom_defaults, ram_available_gb, read_gpu
from local_runtime import ROOT, configure_project_cache


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a small mask-preserving synthetic->real refiner readiness pack. "
            "This does not train a refiner; it builds manifests, note/detail/edge masks, "
            "a visual mask preview, and a CUDA/hardware readiness summary."
        )
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path("data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_v1"),
        help="Synthetic YOLO root with images/train, labels/train, and metadata/train.jsonl.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata JSONL. Defaults to <synthetic-root>/metadata/train.jsonl.",
    )
    parser.add_argument(
        "--real-image-root",
        type=Path,
        default=Path("data/cashsnap_v1/images/train"),
        help="Train-only real CashSnap images used as the unpaired target domain.",
    )
    parser.add_argument(
        "--real-label-root",
        type=Path,
        default=Path("data/cashsnap_v1/labels/train"),
        help="Optional train label root for target-domain audit references.",
    )
    parser.add_argument(
        "--background-root",
        type=Path,
        default=Path("data/backgrounds/cashsnap_v1_no_note_patches_strict_v1"),
        help="Train-only strict no-note patches for compositor/refiner background references.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-synthetic", type=int, default=52, help="Balanced synthetic smoke rows. <=0 keeps all.")
    parser.add_argument("--max-real", type=int, default=160, help="Random train-only real target rows. <=0 keeps all.")
    parser.add_argument("--max-backgrounds", type=int, default=64, help="Random no-note background rows. <=0 keeps all.")
    parser.add_argument("--edge-band-px", type=int, default=10, help="Boundary band retained as the refiner freedom zone.")
    parser.add_argument("--detail-erode-px", type=int, default=10, help="Erosion used for the strong interior/detail lock mask.")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--preview-thumb-width", type=int, default=240)
    parser.add_argument("--allow-cpu", action="store_true", help="Do not fail if CUDA is unavailable.")
    parser.add_argument("--min-free-vram-gb", type=float, default=1.5)
    parser.add_argument("--min-free-ram-gb", type=float, default=1.0)
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"missing image root: {repo_rel(root)}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing metadata JSONL: {repo_rel(path)}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{repo_rel(path)}:{line_no}: invalid JSON: {exc}") from exc
        if not row.get("image") or not row.get("label") or not row.get("class_name") or not row.get("quad_xy"):
            continue
        rows.append(row)
    if not rows:
        raise SystemExit(f"no usable synthetic rows in {repo_rel(path)}")
    return rows


def balanced_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda row: str(row.get("image", "")))
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["class_name"])].append(row)
    for class_rows in by_class.values():
        rng.shuffle(class_rows)

    selected: list[dict[str, Any]] = []
    class_names = sorted(by_class)
    cursor = 0
    while len(selected) < limit and any(by_class.values()):
        class_name = class_names[cursor % len(class_names)]
        cursor += 1
        if by_class[class_name]:
            selected.append(by_class[class_name].pop())
    return sorted(selected, key=lambda row: str(row.get("image", "")))


def random_sample(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    rng = random.Random(seed)
    chosen = list(paths)
    rng.shuffle(chosen)
    return sorted(chosen[:limit])


def mask_from_quad(quad_xy: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(quad_xy, dtype=np.float32).round().astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def make_lock_masks(note_mask: np.ndarray, detail_erode_px: int, edge_band_px: int) -> tuple[np.ndarray, np.ndarray]:
    erode_size = max(1, int(detail_erode_px) * 2 + 1)
    band_size = max(1, int(edge_band_px) * 2 + 1)
    erode_kernel = np.ones((erode_size, erode_size), dtype=np.uint8)
    band_kernel = np.ones((band_size, band_size), dtype=np.uint8)
    detail_lock = cv2.erode(note_mask, erode_kernel)
    dilated = cv2.dilate(note_mask, band_kernel)
    eroded = cv2.erode(note_mask, band_kernel)
    edge_band = np.where((dilated > 0) & (eroded == 0), 255, 0).astype(np.uint8)
    return detail_lock, edge_band


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def hardware_summary(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "require_cuda": not args.allow_cpu,
        "min_free_vram_gb": args.min_free_vram_gb,
        "min_free_ram_gb": args.min_free_ram_gb,
    }
    try:
        import torch
    except ImportError:
        summary["torch_import_ok"] = False
        summary["cuda_available"] = False
        if not args.allow_cpu:
            raise SystemExit("torch is not importable; refiner work would not be CUDA-ready")
        warnings.append("torch is not importable; CPU-only fallback allowed by --allow-cpu")
        return summary, warnings

    cuda_available = bool(torch.cuda.is_available())
    summary.update(
        {
            "torch_import_ok": True,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": cuda_available,
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
    )
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        summary.update(
            {
                "cuda_device": "0",
                "cuda_device_name": torch.cuda.get_device_name(0),
                "torch_total_vram_gb": round(float(props.total_memory) / (1024**3), 3),
            }
        )
    elif not args.allow_cpu:
        raise SystemExit("CUDA is not available; pass --allow-cpu only for non-training dry runs")
    else:
        warnings.append("CUDA is unavailable; CPU-only fallback allowed by --allow-cpu")

    gpu = read_gpu()
    if gpu is not None:
        summary.update(
            {
                "nvidia_smi_name": gpu.name,
                "nvidia_smi_total_vram_gb": round(gpu.mem_total_mb / 1024.0, 3),
                "nvidia_smi_used_vram_gb": round(gpu.mem_used_mb / 1024.0, 3),
                "nvidia_smi_free_vram_gb": round(gpu.mem_free_gb, 3),
                "nvidia_smi_util_percent": gpu.util_percent,
            }
        )
        if gpu.mem_free_gb < args.min_free_vram_gb:
            message = (
                f"free VRAM {gpu.mem_free_gb:.2f} GB is below --min-free-vram-gb "
                f"{args.min_free_vram_gb:.2f} GB"
            )
            if args.allow_cpu:
                warnings.append(message)
            else:
                raise SystemExit(message)
    else:
        warnings.append("nvidia-smi snapshot unavailable")

    ram_gb = ram_available_gb()
    if ram_gb is not None:
        summary["available_ram_gb"] = round(ram_gb, 3)
        if ram_gb < args.min_free_ram_gb:
            raise SystemExit(f"free RAM {ram_gb:.2f} GB is below --min-free-ram-gb {args.min_free_ram_gb:.2f} GB")

    free_vram = float(summary.get("nvidia_smi_free_vram_gb", summary.get("torch_total_vram_gb", 0.0)) or 0.0)
    recommended_resolution = 512 if free_vram >= 5.5 else 384
    if free_vram and free_vram < 3.0:
        recommended_resolution = 256
    if ram_gb is not None and ram_gb < 4.0:
        recommended_resolution = min(recommended_resolution, 384)
        warnings.append(
            f"available RAM is {ram_gb:.2f} GB; prefer 384px or smaller smoke runs until the machine is freer"
        )
    if ram_gb is not None and ram_gb < 2.0:
        recommended_resolution = min(recommended_resolution, 256)
    summary["recommended_refiner"] = {
        "device": "cuda:0" if cuda_available else "cpu",
        "batch": 1,
        "precision": "fp16" if cuda_available else "fp32",
        "amp": bool(cuda_available),
        "resolution": recommended_resolution,
        "workers": 0,
        "gradient_accumulation": 1 if recommended_resolution <= 384 else 2,
        "cache_root": repo_rel(ROOT / ".cache_runtime"),
    }
    summary["headroom_guard"] = {
        "wrapper": "scripts/run_with_headroom.py",
        **headroom_defaults(),
        "use_for": "wrap every long refiner training/inference command so the PC remains usable",
        "example_prefix": (
            "rl python scripts\\run_with_headroom.py --memory-action pause -- "
            "python <refiner_train_or_infer.py>"
        ),
    }
    return summary, warnings


def build_synthetic_pack(rows: list[dict[str, Any]], synthetic_root: Path, out_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str], list[str]]:
    manifest_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    warnings: list[str] = []
    mask_root = out_dir / "masks"

    for row in rows:
        image_path = synthetic_root / str(row["image"])
        label_path = synthetic_root / str(row["label"])
        if not image_path.exists():
            warnings.append(f"missing synthetic image: {repo_rel(image_path)}")
            continue
        if not label_path.exists():
            warnings.append(f"missing synthetic label: {repo_rel(label_path)}")
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        note_mask = mask_from_quad(row["quad_xy"], (width, height))
        detail_lock, edge_band = make_lock_masks(note_mask, args.detail_erode_px, args.edge_band_px)
        if int((note_mask > 0).sum()) <= 0:
            warnings.append(f"empty note mask: {repo_rel(image_path)}")
            continue
        if int((detail_lock > 0).sum()) <= 0:
            warnings.append(f"empty detail lock mask: {repo_rel(image_path)}")
            continue

        stem = image_path.stem
        note_mask_path = mask_root / "note" / f"{stem}.png"
        detail_mask_path = mask_root / "detail_lock" / f"{stem}.png"
        edge_mask_path = mask_root / "edge_band" / f"{stem}.png"
        write_mask(note_mask_path, note_mask)
        write_mask(detail_mask_path, detail_lock)
        write_mask(edge_mask_path, edge_band)

        class_name = str(row["class_name"])
        class_counts[class_name] += 1
        manifest_rows.append(
            {
                "id": stem,
                "class_name": class_name,
                "source_image": repo_rel(image_path),
                "label": repo_rel(label_path),
                "asset": row.get("asset", ""),
                "asset_side": row.get("asset_side", ""),
                "background": row.get("background", ""),
                "geometry_source": row.get("geometry_source", ""),
                "quad_xy": row.get("quad_xy", []),
                "composite_policy": row.get("composite_policy", ""),
                "shadow_policy": row.get("shadow_policy", ""),
                "note_mask": repo_rel(note_mask_path),
                "detail_lock_mask": repo_rel(detail_mask_path),
                "edge_band_mask": repo_rel(edge_mask_path),
                "lock_policy": {
                    "identity_l1": "note_mask",
                    "strong_identity_l1": "detail_lock_mask",
                    "edge_freedom": "edge_band_mask",
                    "forbid_mutation": [
                        "denomination_numbers",
                        "Khmer/USD design details",
                        "note_geometry",
                        "class_identity",
                        "label_box_location",
                    ],
                },
            }
        )
    return manifest_rows, class_counts, warnings


def build_real_manifest(image_root: Path, label_root: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_path in random_sample(image_files(image_root), limit, seed):
        label_path = label_root / f"{image_path.stem}.txt"
        rows.append(
            {
                "image": repo_rel(image_path),
                "label": repo_rel(label_path) if label_path.exists() else "",
                "target_domain": "cashsnap_train",
            }
        )
    return rows


def build_background_manifest(background_root: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    paths = [path for path in image_files(background_root) if "_train" in path.stem]
    return [{"image": repo_rel(path), "target_domain": "cashsnap_strict_no_note_train_patch"} for path in random_sample(paths, limit, seed)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def overlay_preview(image_path: Path, note_mask_path: Path, detail_mask_path: Path, edge_mask_path: Path, thumb_width: int) -> Image.Image:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    scale = thumb_width / max(1, rgb.width)
    thumb_height = max(1, int(round(rgb.height * scale)))
    rgb = rgb.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
    overlays = []
    for mask_path, color in [
        (note_mask_path, (255, 220, 0)),
        (detail_mask_path, (0, 220, 120)),
        (edge_mask_path, (255, 40, 40)),
    ]:
        with Image.open(mask_path) as mask_image:
            mask = mask_image.convert("L").resize(rgb.size, Image.Resampling.NEAREST)
        tint = Image.new("RGB", rgb.size, color)
        overlays.append((tint, mask))
    out = rgb.copy()
    for tint, mask in overlays:
        out = Image.composite(Image.blend(out, tint, 0.35), out, mask)
    return out


def write_preview(out_dir: Path, manifest_rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    if args.preview_count <= 0 or not manifest_rows:
        return ""
    rows = manifest_rows[: args.preview_count]
    thumbs = [
        overlay_preview(
            ROOT / row["source_image"],
            ROOT / row["note_mask"],
            ROOT / row["detail_lock_mask"],
            ROOT / row["edge_band_mask"],
            args.preview_thumb_width,
        )
        for row in rows
    ]
    cols = min(4, len(thumbs))
    rows_count = int(np.ceil(len(thumbs) / cols))
    label_h = 36
    sheet = Image.new("RGB", (cols * args.preview_thumb_width, rows_count * (thumbs[0].height + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        col = index % cols
        row_index = index // cols
        x = col * args.preview_thumb_width
        y = row_index * (thumb.height + label_h)
        sheet.paste(thumb, (x, y))
        draw.text((x + 6, y + thumb.height + 4), rows[index]["class_name"], fill=(20, 20, 20))
    preview_path = out_dir / "preview" / "mask_lock_contact.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_path, quality=92)
    return repo_rel(preview_path)


def write_readme(out_dir: Path, summary: dict[str, Any]) -> str:
    recommended = summary["hardware"].get("recommended_refiner", {})
    lines = [
        "# CashSnap Refiner Readiness Pack",
        "",
        "Purpose: smoke-test a label-preserving synthetic-to-real refiner without touching val/test data.",
        "",
        "Masks:",
        "- `note_mask`: whole visible note; use for identity/self-regularization.",
        "- `detail_lock_mask`: eroded note interior; use for the strongest denomination/detail lock.",
        "- `edge_band_mask`: contact boundary band; allow limited refiner freedom here.",
        "",
        "Hardware posture:",
        f"- device: `{recommended.get('device', 'unknown')}`",
        f"- resolution: `{recommended.get('resolution', 'unknown')}`",
        f"- batch: `{recommended.get('batch', 'unknown')}`",
        f"- precision: `{recommended.get('precision', 'unknown')}`",
        f"- workers: `{recommended.get('workers', 'unknown')}`",
        "",
        "Headroom:",
        "- Wrap long refiner training/inference commands with `scripts/run_with_headroom.py`.",
        "- The wrapper caps CPU/RAM/GPU-memory defaults at 95% and pauses or exits before the PC freezes.",
        "",
        "Do not train a refiner on CashSnap val/test. Do not accept any output that mutates class evidence.",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return repo_rel(path)


def main() -> int:
    configure_project_cache()
    args = parse_args()
    if args.edge_band_px < 1 or args.detail_erode_px < 1:
        raise SystemExit("--edge-band-px and --detail-erode-px must be >= 1")
    synthetic_root = resolve_repo_path(args.synthetic_root)
    metadata_path = resolve_repo_path(args.metadata) if args.metadata else synthetic_root / "metadata" / "train.jsonl"
    real_image_root = resolve_repo_path(args.real_image_root)
    real_label_root = resolve_repo_path(args.real_label_root)
    background_root = resolve_repo_path(args.background_root)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hardware, hardware_warnings = hardware_summary(args)
    all_rows = load_metadata(metadata_path)
    synthetic_rows = balanced_sample(all_rows, args.max_synthetic, args.seed)
    synthetic_manifest, class_counts, synthetic_warnings = build_synthetic_pack(synthetic_rows, synthetic_root, out_dir, args)
    if not synthetic_manifest:
        raise SystemExit("no synthetic rows survived readiness pack creation")
    real_manifest = build_real_manifest(real_image_root, real_label_root, args.max_real, args.seed + 11)
    background_manifest = build_background_manifest(background_root, args.max_backgrounds, args.seed + 23)
    preview_path = write_preview(out_dir, synthetic_manifest, args)

    synthetic_manifest_path = out_dir / "synthetic_manifest.jsonl"
    real_manifest_path = out_dir / "real_target_manifest.jsonl"
    background_manifest_path = out_dir / "background_manifest.jsonl"
    write_jsonl(synthetic_manifest_path, synthetic_manifest)
    write_jsonl(real_manifest_path, real_manifest)
    write_jsonl(background_manifest_path, background_manifest)

    warnings = hardware_warnings + synthetic_warnings
    summary = {
        "synthetic_root": repo_rel(synthetic_root),
        "metadata": repo_rel(metadata_path),
        "real_image_root": repo_rel(real_image_root),
        "real_label_root": repo_rel(real_label_root),
        "background_root": repo_rel(background_root),
        "out_dir": repo_rel(out_dir),
        "synthetic_rows": len(synthetic_manifest),
        "real_target_rows": len(real_manifest),
        "background_rows": len(background_manifest),
        "synthetic_class_counts": dict(sorted(class_counts.items())),
        "edge_band_px": args.edge_band_px,
        "detail_erode_px": args.detail_erode_px,
        "preview": preview_path,
        "manifests": {
            "synthetic": repo_rel(synthetic_manifest_path),
            "real_target": repo_rel(real_manifest_path),
            "background": repo_rel(background_manifest_path),
        },
        "hardware": hardware,
        "warnings": warnings,
    }
    readme_path = write_readme(out_dir, summary)
    summary["readme"] = readme_path
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(json.dumps({"summary": repo_rel(summary_path), "synthetic_rows": len(synthetic_manifest), "real_target_rows": len(real_manifest), "background_rows": len(background_manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
