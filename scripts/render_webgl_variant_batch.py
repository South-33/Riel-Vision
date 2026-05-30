#!/usr/bin/env python
"""Render and check a small deterministic WebGL variant batch."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--out-root", type=Path, default=Path("data/synthetic/cashsnap_webgl_variant_batch_smoke"))
    parser.add_argument("--start-variant", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scene-mode", choices=["auto", "stack", "fan"], default="auto")
    parser.add_argument("--background-dir", type=Path, help="Optional reviewed-clean background image directory.")
    parser.add_argument("--skip-render", action="store_true", help="Only recheck/contact-sheet existing outputs.")
    parser.add_argument("--skip-yolo-check", action="store_true", help="Do not run check_yolo_dataset.py on the packaged dataset.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def render_variant(variant: int, out_dir: Path, scene_mode: str, background_dir: Path | None) -> None:
    cmd = [
        sys.executable,
        "scripts/run_with_headroom.py",
        "--max-percent",
        "90",
        "--resume-percent",
        "82",
        "--max-ram-percent",
        "90",
        "--max-gpu-mem-percent",
        "90",
        "--min-free-ram-gb",
        "3",
        "--preflight-timeout",
        "120",
        "--",
        "node",
        "renderers/webgl/src/render-smoke.mjs",
        "--variant",
        str(variant),
        "--scene-mode",
        scene_mode,
        "--out-dir",
        str(out_dir),
    ]
    if background_dir is not None:
        cmd.extend(["--background-dir", str(background_dir)])
    run(cmd)


def check_variant(out_dir: Path) -> None:
    run([sys.executable, "scripts/check_webgl_smoke_output.py", "--out-dir", str(out_dir)])


def write_contact_sheet(variant_dirs: list[tuple[int, Path]], out_path: Path) -> None:
    cell_w, cell_h = 320, 240
    header_h = 30
    row_h = cell_h + header_h
    sheet = Image.new("RGB", (cell_w * 2, row_h * len(variant_dirs)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for row, (variant, out_dir) in enumerate(variant_dirs):
        y = row * row_h
        visual = Image.open(out_dir / "visual.png").convert("RGB").resize((cell_w, cell_h))
        mask = Image.open(out_dir / "id.png").convert("RGB").resize((cell_w, cell_h))
        sheet.paste(visual, (0, y + header_h))
        sheet.paste(mask, (cell_w, y + header_h))
        draw.text((8, y + 8), f"variant {variant} visual", fill=(255, 255, 255))
        draw.text((cell_w + 8, y + 8), f"variant {variant} id", fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_yolo_dataset(variant_dirs: list[tuple[int, Path]], out_root: Path) -> Path:
    images_dir = out_root / "images" / "train"
    labels_dir = out_root / "labels" / "train"
    ids_dir = out_root / "ids" / "train"
    metadata_dir = out_root / "metadata"
    for directory in (images_dir, labels_dir, ids_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = []
    for variant, out_dir in variant_dirs:
        stem = f"variant_{variant:04d}"
        image_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"
        id_path = ids_dir / f"{stem}.png"
        boxes_path = metadata_dir / f"{stem}_visible_boxes.json"
        audit_path = metadata_dir / f"{stem}_layer_audit.json"
        source_metadata_path = metadata_dir / f"{stem}_metadata.json"
        shutil.copyfile(out_dir / "visual.png", image_path)
        shutil.copyfile(out_dir / "labels_visible.txt", label_path)
        shutil.copyfile(out_dir / "id.png", id_path)
        shutil.copyfile(out_dir / "visible_boxes.json", boxes_path)
        shutil.copyfile(out_dir / "layer_audit.json", audit_path)
        shutil.copyfile(out_dir / "metadata.json", source_metadata_path)
        manifest.append(
            {
                "variant": variant,
                "image": str(image_path.relative_to(out_root)),
                "label": str(label_path.relative_to(out_root)),
                "id": str(id_path.relative_to(out_root)),
                "visible_boxes": str(boxes_path.relative_to(out_root)),
                "layer_audit": str(audit_path.relative_to(out_root)),
            }
        )

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/train",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")

    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    variant_dirs: list[tuple[int, Path]] = []

    for variant in range(args.start_variant, args.start_variant + args.count):
        out_dir = out_root / f"variant_{variant:04d}"
        if not args.skip_render:
            render_variant(variant, out_dir, args.scene_mode, args.background_dir)
        check_variant(out_dir)
        variant_dirs.append((variant, out_dir))

    contact_sheet = out_root / "contact_sheet.png"
    write_contact_sheet(variant_dirs, contact_sheet)
    data_yaml = write_yolo_dataset(variant_dirs, out_root)
    if not args.skip_yolo_check:
        run([sys.executable, "scripts/check_yolo_dataset.py", "--data", str(data_yaml)])
    print(f"wrote {contact_sheet.relative_to(ROOT)}")
    print(f"wrote {data_yaml.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
