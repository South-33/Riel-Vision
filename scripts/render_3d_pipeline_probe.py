"""Render a small CashSnap synthetic proof dataset from a 3D pipeline config.

This is the P0 renderer scaffold, not the final photoreal WebGL renderer. It
proves the contract: config in, visual pass, ID pass, visible-only labels,
metadata, and QA sheets out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("render_3d_pipeline_probe.py requires opencv-python/cv2") from exc


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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Asset:
    path: Path
    class_name: str
    side: str
    status: str
    title: str
    years: str
    note_id: str


@dataclass
class InstancePlan:
    asset: Asset
    color: tuple[int, int, int]
    quad: np.ndarray
    full_area: int


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "data" / "synthetic").resolve()
    if allowed not in resolved.parents and resolved != allowed:
        raise SystemExit(f"Refusing to clean outside {allowed}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def choose_weighted(items: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    total = sum(float(item.get("weight", 0.0)) for item in items)
    pick = rng.random() * total
    acc = 0.0
    for item in items:
        acc += float(item.get("weight", 0.0))
        if pick <= acc:
            return item
    return items[-1]


def rand_range(value: list[float] | list[int], rng: random.Random) -> float:
    return rng.uniform(float(value[0]), float(value[1]))


def rand_int_range(value: list[float] | list[int], rng: random.Random) -> int:
    return rng.randint(int(value[0]), int(value[1]))


def load_assets(config: dict[str, Any]) -> list[Asset]:
    assets_config = config["assets"]
    manifest_path = repo_path(assets_config["banknote_manifest"])
    classes = set(assets_config["classes"])
    sides = set(assets_config["sides"])
    statuses = set(assets_config["status"])
    assets: list[Asset] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("class_name") not in classes:
                continue
            if row.get("side") not in sides:
                continue
            if row.get("status") not in statuses:
                continue
            path = repo_path(row.get("asset_path", ""))
            if not path.exists():
                continue
            assets.append(
                Asset(
                    path=path,
                    class_name=str(row.get("class_name", "")),
                    side=str(row.get("side", "")),
                    status=str(row.get("status", "")),
                    title=str(row.get("title", "")),
                    years=str(row.get("years", "")),
                    note_id=str(row.get("note_id", "")),
                )
            )
    if not assets:
        raise SystemExit("Config selected zero renderable assets.")
    return assets


def make_background(width: int, height: int, rng: random.Random, mode: str) -> Image.Image:
    base_colors = {
        "diffuse_indoor": (165, 145, 119),
        "warm_shop_bulb": (154, 120, 79),
        "daylight_window": (136, 152, 145),
        "phone_flash": (170, 170, 160),
    }
    color = base_colors.get(mode, (145, 135, 120))
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = color
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 7, arr.shape)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(18):
        y = rng.randint(0, height - 1)
        shade = tuple(max(0, min(255, c + rng.randint(-16, 16))) for c in color)
        draw.line((0, y, width, y + rng.randint(-16, 16)), fill=shade, width=rng.randint(1, 4))
    return image


def note_color(instance_index: int) -> tuple[int, int, int]:
    value = instance_index + 1
    return (value & 255, (value >> 8) & 255, (value >> 16) & 255)


def rotated_quad(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
        dtype=np.float32,
    )
    theta = math.radians(angle_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float32)


def jitter_quad(quad: np.ndarray, rng: random.Random, amount: float) -> np.ndarray:
    jitter = np.array([[rng.uniform(-amount, amount), rng.uniform(-amount, amount)] for _ in range(4)], dtype=np.float32)
    return quad + jitter


def plan_instances(
    config: dict[str, Any],
    assets: list[Asset],
    layout: dict[str, Any],
    camera: dict[str, Any],
    rng: random.Random,
) -> list[InstancePlan]:
    width, height = [int(v) for v in camera["resolution"]]
    count = rand_int_range(layout["notes_per_scene"], rng)
    layout_name = str(layout["name"])
    plans: list[InstancePlan] = []
    class_assets: dict[str, list[Asset]] = {}
    for asset in assets:
        class_assets.setdefault(asset.class_name, []).append(asset)

    for index in range(count):
        class_name = rng.choice(sorted(class_assets))
        asset = rng.choice(class_assets[class_name])
        with Image.open(asset.path) as image:
            aw, ah = image.size
        aspect = aw / max(1, ah)
        note_w = rng.uniform(width * 0.30, width * 0.58)
        note_h = note_w / aspect

        if layout_name == "shop_stack":
            cx = width * 0.46 + (index - count / 2) * rng.uniform(22, 48) + rng.uniform(-45, 45)
            cy = height * 0.50 + (index - count / 2) * rng.uniform(10, 28) + rng.uniform(-35, 35)
            angle = rng.uniform(-18, 18)
        elif layout_name == "radial_fan":
            sweep = rand_range(layout.get("fan_angle_degrees", [25, 80]), rng)
            angle = -sweep / 2 + sweep * index / max(1, count - 1) + rng.uniform(-4, 4)
            pivot = np.array([width * 0.36, height * 0.72], dtype=np.float32)
            offset = np.array([note_w * 0.36, -note_h * 0.40], dtype=np.float32)
            theta = math.radians(angle)
            rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
            cx, cy = pivot + offset @ rot.T
        elif layout_name == "edge_partial":
            cx = rng.choice([-0.05 * width, 1.05 * width, rng.uniform(width * 0.25, width * 0.75)])
            cy = rng.choice([-0.02 * height, 1.02 * height, rng.uniform(height * 0.25, height * 0.75)])
            angle = rng.uniform(-45, 45)
        else:
            cx = rng.uniform(width * 0.30, width * 0.72)
            cy = rng.uniform(height * 0.30, height * 0.72)
            angle = rng.uniform(-38, 38)

        quad = rotated_quad(float(cx), float(cy), note_w, note_h, angle)
        quad = jitter_quad(quad, rng, amount=min(note_w, note_h) * rng.uniform(0.00, 0.06))
        plans.append(InstancePlan(asset=asset, color=note_color(index), quad=quad, full_area=int(note_w * note_h)))
    return plans


def warp_rgba(image: Image.Image, quad: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    src = np.array([[0, 0], [image.width - 1, 0], [image.width - 1, image.height - 1], [0, image.height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    rgba = np.array(image.convert("RGBA"))
    return cv2.warpPerspective(rgba, matrix, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def alpha_composite_np(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
    base_rgb = base[:, :, :3].astype(np.float32)
    layer_rgb = layer[:, :, :3].astype(np.float32)
    out_rgb = layer_rgb * alpha + base_rgb * (1.0 - alpha)
    base[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    return base


def apply_material(image: Image.Image, material: dict[str, Any], rng: random.Random) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba).astype(np.float32)
    jitter = rand_range(material.get("color_jitter", [1.0, 1.0]), rng)
    arr[:, :, :3] *= jitter
    dirt_alpha = rand_range(material.get("dirt_alpha", [0.0, 0.0]), rng)
    if dirt_alpha > 0:
        dirt = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 28, arr[:, :, :3].shape)
        arr[:, :, :3] = arr[:, :, :3] * (1 - dirt_alpha) + np.clip(arr[:, :, :3] + dirt, 0, 255) * dirt_alpha
    arr[:, :, :3] = np.clip(arr[:, :, :3], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def render_scene(
    config: dict[str, Any],
    assets: list[Asset],
    scene_index: int,
    split: str,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]], list[str], list[str]]:
    layout = choose_weighted(config["layouts"], rng)
    camera = choose_weighted(config["camera_profiles"], rng)
    material = choose_weighted(config["material_presets"], rng)
    lighting_modes = config.get("lighting", {}).get("modes", ["diffuse_indoor"])
    lighting = rng.choice(lighting_modes)
    width, height = [int(v) for v in camera["resolution"]]
    visual = make_background(width, height, rng, lighting).convert("RGBA")
    visual_arr = np.array(visual)
    id_arr = np.zeros((height, width, 3), dtype=np.uint8)
    plans = plan_instances(config, assets, layout, camera, rng)

    for plan in plans:
        with Image.open(plan.asset.path) as raw:
            note = apply_material(raw, material, rng)
        warped = warp_rgba(note, plan.quad, (width, height))
        alpha = warped[:, :, 3]
        if alpha.max() == 0:
            continue
        plan.full_area = int((alpha > 8).sum())

        shadow_alpha = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=rng.uniform(5, 13)))
        shadow = np.zeros((height, width, 4), dtype=np.uint8)
        shadow[:, :, :3] = 0
        shadow[:, :, 3] = (np.array(shadow_alpha).astype(np.float32) * rng.uniform(0.18, 0.34)).astype(np.uint8)
        shift_x = rng.randint(3, 12)
        shift_y = rng.randint(4, 14)
        shadow = np.roll(shadow, shift=(shift_y, shift_x), axis=(0, 1))
        visual_arr = alpha_composite_np(visual_arr, shadow)
        visual_arr = alpha_composite_np(visual_arr, warped)

        mask = alpha > 8
        id_arr[mask] = plan.color

    visual = Image.fromarray(visual_arr[:, :, :3], "RGB")
    id_image = Image.fromarray(id_arr, "RGB")
    instances: list[dict[str, Any]] = []
    label_lines: list[str] = []
    obb_label_lines: list[str] = []
    min_pixels = int(config["label_policy"]["min_visible_pixels"])
    min_ratio = float(config["label_policy"]["min_visibility_ratio_for_denoms"])
    for instance_id, plan in enumerate(plans):
        mask = np.all(id_arr == np.array(plan.color, dtype=np.uint8), axis=2)
        ys, xs = np.where(mask)
        visible_pixels = int(mask.sum())
        visibility_ratio = visible_pixels / max(1, plan.full_area)
        exported = visible_pixels >= min_pixels and visibility_ratio >= min_ratio
        bbox = ""
        obb = ""
        if visible_pixels:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            bbox = [x1, y1, x2, y2]
            points = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            obb = [[round(float(x), 2), round(float(y), 2)] for x, y in box]
            if exported:
                cx = ((x1 + x2) / 2) / width
                cy = ((y1 + y2) / 2) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                label_lines.append(f"{CLASS_TO_ID[plan.asset.class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                norm_points = []
                for point_x, point_y in obb:
                    norm_points.extend([point_x / width, point_y / height])
                obb_values = " ".join(f"{value:.6f}" for value in norm_points)
                obb_label_lines.append(f"{CLASS_TO_ID[plan.asset.class_name]} {obb_values}")
        instances.append(
            {
                "instance_id": instance_id,
                "class_name": plan.asset.class_name,
                "side": plan.asset.side,
                "title": plan.asset.title,
                "years": plan.asset.years,
                "note_id": plan.asset.note_id,
                "source_asset": str(plan.asset.path.relative_to(ROOT)),
                "id_color": plan.color,
                "layout_mode": layout["name"],
                "camera_profile": camera["name"],
                "material_preset": material["name"],
                "lighting": lighting,
                "split": split,
                "full_area": plan.full_area,
                "visible_pixels": visible_pixels,
                "visibility_ratio": round(visibility_ratio, 6),
                "exported_label": exported,
                "bbox_xyxy": bbox,
                "obb": obb,
            }
        )
    return visual, id_image, instances, label_lines, obb_label_lines


def make_contact_sheet(image_paths: list[Path], out_path: Path, title: str) -> None:
    if not image_paths:
        return
    thumbs: list[Image.Image] = []
    for path in image_paths[:24]:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((220, 160))
            tile = Image.new("RGB", (220, 184), "white")
            tile.paste(thumb, ((220 - thumb.width) // 2, 0))
            ImageDraw.Draw(tile).text((4, 164), path.name[:28], fill="black")
            thumbs.append(tile)
    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 220, rows * 184 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), title, fill="black")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 220, 28 + (index // cols) * 184))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def overlay_id(visual_path: Path, id_path: Path, out_path: Path) -> None:
    with Image.open(visual_path).convert("RGBA") as visual, Image.open(id_path).convert("RGB") as ids:
        color = np.array(ids)
        mask = color.sum(axis=2) > 0
        overlay = np.zeros((ids.height, ids.width, 4), dtype=np.uint8)
        overlay[:, :, :3] = color
        overlay[:, :, 3] = (mask.astype(np.uint8) * 90)
        layer = Image.fromarray(overlay, "RGBA")
        visual.alpha_composite(layer)
        visual.convert("RGB").save(out_path, quality=92)


def write_data_yaml(out_dir: Path) -> None:
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "3d_pipeline" / "proof_p0_renderer_smoke.json")
    parser.add_argument("--out", type=Path, help="Override output root from config.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    config_path = repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    out_dir = repo_path(args.out or config["output_root"])
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["images/train", "images/val", "labels/train", "labels/val", "masks/train", "masks/val", "metadata", "qa"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    for subdir in ["labels_obb/train", "labels_obb/val"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(config["seed"]))
    assets = load_assets(config)
    image_paths: list[Path] = []
    overlay_paths: list[Path] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    exported_class_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    visibility_ratios: list[float] = []
    exported_labels = 0
    total_instances = 0
    metadata_path = out_dir / "metadata" / "scenes.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        for scene_index in range(int(config["scene_count"])):
            split = "val" if scene_index % 5 == 0 else "train"
            visual, id_image, instances, label_lines, obb_label_lines = render_scene(config, assets, scene_index, split, rng)
            stem = f"{config['name']}_{scene_index:05d}"
            image_path = out_dir / "images" / split / f"{stem}.jpg"
            id_path = out_dir / "masks" / split / f"{stem}_id.png"
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            obb_label_path = out_dir / "labels_obb" / split / f"{stem}.txt"
            visual.save(image_path, quality=rng.randint(*config.get("postprocess", {}).get("jpeg_quality", [82, 92])))
            id_image.save(id_path)
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            obb_label_path.write_text("\n".join(obb_label_lines) + ("\n" if obb_label_lines else ""), encoding="utf-8")
            overlay_path = out_dir / "qa" / f"{stem}_overlay.jpg"
            overlay_id(image_path, id_path, overlay_path)
            image_paths.append(image_path)
            overlay_paths.append(overlay_path)
            split_counts[split] += 1
            for instance in instances:
                total_instances += 1
                class_name = str(instance["class_name"])
                class_counts[class_name] += 1
                layout_counts[str(instance["layout_mode"])] += 1
                visibility_ratios.append(float(instance["visibility_ratio"]))
                if instance["exported_label"]:
                    exported_labels += 1
                    exported_class_counts[class_name] += 1
            metadata_handle.write(
                json.dumps(
                    {
                        "scene_index": scene_index,
                        "split": split,
                        "image": str(image_path.relative_to(ROOT)),
                        "id_mask": str(id_path.relative_to(ROOT)),
                        "label": str(label_path.relative_to(ROOT)),
                        "obb_label": str(obb_label_path.relative_to(ROOT)),
                        "instances": instances,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    write_data_yaml(out_dir)
    sorted_visibility = sorted(visibility_ratios)
    stats = {
        "config": str(config_path.relative_to(ROOT)),
        "scene_count": int(config["scene_count"]),
        "splits": dict(sorted(split_counts.items())),
        "instances": total_instances,
        "exported_labels": exported_labels,
        "classes_all": dict(sorted(class_counts.items())),
        "classes_exported": dict(sorted(exported_class_counts.items())),
        "layouts": dict(sorted(layout_counts.items())),
        "visibility_ratio": {
            "min": sorted_visibility[0] if sorted_visibility else None,
            "p50": sorted_visibility[len(sorted_visibility) // 2] if sorted_visibility else None,
            "max": sorted_visibility[-1] if sorted_visibility else None,
        },
    }
    (out_dir / "qa" / "label_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_contact_sheet(image_paths, out_dir / "qa" / "contact_sheet.jpg", config["name"])
    make_contact_sheet(overlay_paths, out_dir / "qa" / "mask_overlay_contact.jpg", f"{config['name']} masks")
    print(f"rendered {config['scene_count']} scenes to {out_dir}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
