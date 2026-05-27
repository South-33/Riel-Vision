from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from local_runtime import configure_project_cache

configure_project_cache()

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO proposals, then reclassify each crop with a fragment classifier.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS for detector proposals.")
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-preview", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_classifier(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def crop_with_padding(image: Image.Image, xyxy: tuple[float, float, float, float], padding: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = xyxy
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    return image.crop((left, top, right, bottom))


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xyxy: tuple[float, float, float, float], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    label_y = max(0, y1 - text_height - 8)
    draw.rectangle((x1, label_y, x1 + text_width + 8, label_y + text_height + 8), fill=color)
    draw.text((x1 + 4, label_y + 4), text, fill=(0, 0, 0), font=font)


def main() -> None:
    args = parse_args()
    image_path = resolve(args.image)
    detector_path = resolve(args.detector)
    classifier_path = resolve(args.classifier)
    checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=False)
    class_names: list[str] = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 224))
    device = choose_device(args.device)

    classifier = build_classifier(len(class_names)).to(device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    detector = YOLO(str(detector_path))
    result = detector.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        verbose=False,
    )[0]
    detector_names = {int(key): value for key, value in detector.names.items()}
    with Image.open(image_path).convert("RGB") as image:
        rows: list[dict[str, str]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            detector_cls = boxes.cls.cpu().numpy().astype(int)
            detector_conf = boxes.conf.cpu().numpy().astype(float)
            crops = [transform(crop_with_padding(image, tuple(box), args.padding)) for box in xyxy]
            batch = torch.stack(crops).to(device) if crops else torch.empty(0)
            with torch.no_grad():
                probs = torch.softmax(classifier(batch), dim=1).cpu() if crops else torch.empty((0, len(class_names)))
            for index, box in enumerate(xyxy):
                best_prob, best_idx = probs[index].max(dim=0)
                rows.append(
                    {
                        "index": str(index),
                        "x1": f"{box[0]:.1f}",
                        "y1": f"{box[1]:.1f}",
                        "x2": f"{box[2]:.1f}",
                        "y2": f"{box[3]:.1f}",
                        "detector_class": detector_names.get(int(detector_cls[index]), str(int(detector_cls[index]))),
                        "detector_conf": f"{detector_conf[index]:.4f}",
                        "fragment_class": class_names[int(best_idx)],
                        "fragment_conf": f"{float(best_prob):.4f}",
                        "fragment_probs": json.dumps(
                            {class_names[i]: round(float(probs[index, i]), 4) for i in range(len(class_names))}
                        ),
                    }
                )

        out_csv = resolve(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "index",
            "x1",
            "y1",
            "x2",
            "y2",
            "detector_class",
            "detector_conf",
            "fragment_class",
            "fragment_conf",
            "fragment_probs",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if args.out_preview:
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            font = load_font(max(20, preview.width // 95))
            colors = [(20, 180, 220), (230, 80, 180), (180, 230, 30), (255, 170, 30)]
            for row in rows:
                box = tuple(float(row[key]) for key in ["x1", "y1", "x2", "y2"])
                text = f"{row['fragment_class']} {float(row['fragment_conf']):.2f}"
                draw_label(draw, box, text, colors[int(row["index"]) % len(colors)], font)
            out_preview = resolve(args.out_preview)
            out_preview.parent.mkdir(parents=True, exist_ok=True)
            preview.save(out_preview, quality=92)
    print(f"wrote {len(rows)} proposal classifications to {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
