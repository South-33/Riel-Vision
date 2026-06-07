from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import ExifTags, Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EXIF_NAMES = {
    "Make",
    "Model",
    "LensModel",
    "FocalLength",
    "FocalLengthIn35mmFilm",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "PhotographicSensitivity",
    "DateTimeOriginal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-stage detector/classifier stack on a photo folder and build a human review pack."
    )
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--detector",
        default="runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.pt",
    )
    parser.add_argument(
        "--classifier",
        default="runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--det-threshold", type=float, default=0.17)
    parser.add_argument("--nms-iou", type=float, default=0.85)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan images-dir recursively, preserving parent folders in output names.",
    )
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def project_env() -> dict[str, str]:
    env = os.environ.copy()
    cache_root = ROOT / ".cache_runtime"
    env.setdefault("XDG_CACHE_HOME", str(cache_root))
    env.setdefault("TORCH_HOME", str(cache_root / "torch"))
    env.setdefault("HF_HOME", str(cache_root / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    return env


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def image_paths(images_dir: Path, limit: int, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = sorted(path for path in images_dir.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit else paths


def output_stem(images_dir: Path, image_path: Path, recursive: bool) -> str:
    source = image_path.relative_to(images_dir).with_suffix("") if recursive else Path(image_path.stem)
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(source.parts)).strip("_") or "image"


def jsonable_exif_value(value: object) -> object:
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        denominator = getattr(value, "denominator")
        return None if denominator == 0 else float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return [jsonable_exif_value(item) for item in value]
    return value


def image_camera_metadata(image_path: Path) -> dict[str, object]:
    with Image.open(image_path) as image:
        record: dict[str, object] = {
            "image_path": str(image_path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
        }
        exif = image.getexif()
        if not exif:
            record["has_exif"] = False
            return record

        record["has_exif"] = True
        for tag_id, value in exif.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if name in EXIF_NAMES:
                record[name] = jsonable_exif_value(value)
        return record


def write_camera_metadata(images: list[Path], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            handle.write(json.dumps(image_camera_metadata(image_path), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    images_dir = resolve(args.images_dir)
    out_dir = resolve(args.out_dir)
    raw_dir = out_dir / "proposals_raw"
    fused_dir = out_dir / "proposals_fused"
    preview_dir = out_dir / "previews"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fused_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    env = project_env()
    images = image_paths(images_dir, args.limit, args.recursive)
    write_camera_metadata(images, out_dir / "camera_metadata.jsonl")

    pairs: list[tuple[Path, Path]] = []
    for image_path in images:
        stem = output_stem(images_dir, image_path, args.recursive)
        raw_csv = raw_dir / f"{stem}_raw.csv"
        raw_preview = preview_dir / f"{stem}_raw.jpg"
        fused_csv = fused_dir / f"{stem}_fused.csv"
        fused_preview = preview_dir / f"{stem}_fused.jpg"
        run(
            [
                sys.executable,
                "scripts/classify_yolo_proposals.py",
                "--image",
                str(image_path),
                "--detector",
                str(resolve(args.detector)),
                "--classifier",
                str(resolve(args.classifier)),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
                "--iou",
                str(args.iou),
                "--agnostic-nms",
                "--padding",
                str(args.padding),
                "--out-csv",
                str(raw_csv),
                "--out-preview",
                str(raw_preview),
                "--device",
                args.device,
            ],
            env,
        )
        run(
            [
                sys.executable,
                "scripts/fuse_two_stage_csv.py",
                "--csv",
                str(raw_csv),
                "--out",
                str(fused_csv),
                "--det-threshold",
                str(args.det_threshold),
                "--nms-iou",
                str(args.nms_iou),
                "--nms-score-column",
                "detector_conf",
                "--image",
                str(image_path),
                "--out-preview",
                str(fused_preview),
            ],
            env,
        )
        pairs.append((image_path, fused_csv))

    if not pairs:
        raise SystemExit(f"No images found in {images_dir}")
    review_cmd = [
        sys.executable,
        "scripts/build_proposal_review_pack.py",
        "--out-dir",
        str(out_dir / "review_pack"),
    ]
    for image_path, fused_csv in pairs:
        review_cmd.extend(["--item", str(image_path), str(fused_csv)])
    run(review_cmd, env)
    print(f"wrote capture review pipeline outputs to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
