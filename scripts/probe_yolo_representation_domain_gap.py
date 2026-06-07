#!/usr/bin/env python
"""Probe YOLO internal features for real-vs-synthetic domain gaps.

The goal is not to make another visual proxy. It is to ask the detector itself:
at which layers do real and synthetic examples become easy to separate?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from local_runtime import configure_project_cache


configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_LAYERS = "0,1,2,4,6,8,10,13,16,19,22"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="YOLO weights path.")
    parser.add_argument("--real-data", required=True, type=Path, help="Real YOLO dataset YAML.")
    parser.add_argument("--real-split", default="test", help="Split key in --real-data.")
    parser.add_argument("--synthetic-data", required=True, type=Path, help="Synthetic YOLO dataset YAML.")
    parser.add_argument("--synthetic-split", default="train", help="Split key in --synthetic-data.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", default=DEFAULT_LAYERS, help="Comma/space separated YOLO layer indexes.")
    parser.add_argument("--nearest-layer", default=None, help="Layer index to use for uncovered-image exports.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', '0', or CUDA device string.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=20, help="Per-domain cap for class-balanced sampling.")
    parser.add_argument("--min-per-class", type=int, default=2, help="Minimum rows per domain for class gap rows.")
    parser.add_argument("--max-total", type=int, default=0, help="Optional global cap after class-balanced sampling.")
    parser.add_argument("--include-background", action="store_true", help="Include empty-label rows as BACKGROUND.")
    parser.add_argument("--no-class-balance", action="store_true", help="Use random equal-domain sampling instead.")
    parser.add_argument("--top-k", type=int, default=40, help="Rows to export for farthest-nearest examples.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "item"


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = [(ROOT / "runs").resolve()]
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise SystemExit(f"Refusing to clean outside runs/: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_layers(raw: str) -> list[int]:
    layers = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if token:
            layers.append(int(token))
    if not layers:
        raise SystemExit("--layers did not contain any layer indexes")
    return layers


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Dataset YAML must be a mapping: {path}")
    return payload


def read_names(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names")
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    raise SystemExit("Dataset YAML must include names as a list or mapping")


def dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = Path(str(config.get("path", ".")))
    return raw_root if raw_root.is_absolute() else (config_path.parent / raw_root).resolve()


def split_root(root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else root / path


def read_split_list(root: Path, split_path: str) -> list[Path]:
    list_path = split_root(root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line)
        images.append(image if image.is_absolute() else root / image)
    return images


def split_images(root: Path, split_value: str | list[str]) -> list[Path]:
    split_values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for value in split_values:
        path = split_root(root, str(value))
        if path.suffix.lower() == ".txt":
            images.extend(read_split_list(root, str(value)))
        else:
            images.extend(
                sorted(
                    item
                    for item in path.glob("*")
                    if item.is_file() and item.suffix.lower() in IMAGE_EXTS
                )
            )
    return images


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_label_rows(image: Path) -> list[dict[str, float | int]]:
    label = label_path_for_image(image)
    if not label.exists():
        return []
    rows = []
    for line_no, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{label}:{line_no} expected 5 YOLO fields, found {len(parts)}")
        cls = int(parts[0])
        x, y, w, h = (float(value) for value in parts[1:])
        rows.append({"class_id": cls, "x": x, "y": y, "w": w, "h": h, "area": w * h})
    return rows


def primary_class(labels: list[dict[str, float | int]]) -> int | None:
    if not labels:
        return None
    return int(max(labels, key=lambda row: float(row["area"]))["class_id"])


def dataset_records(
    *,
    data_path: Path,
    split: str,
    domain: str,
    include_background: bool,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    config = read_yaml(data_path)
    names = read_names(config)
    root = dataset_root(data_path, config)
    if split not in config:
        raise SystemExit(f"{data_path} has no split key {split!r}")

    records: list[dict[str, Any]] = []
    for image in split_images(root, config[split]):
        labels = read_label_rows(image)
        cls = primary_class(labels)
        if cls is None and not include_background:
            continue
        class_name = "BACKGROUND" if cls is None else names.get(cls, str(cls))
        records.append(
            {
                "domain": domain,
                "image": image,
                "image_rel": repo_rel(image),
                "label": label_path_for_image(image),
                "class_id": -1 if cls is None else cls,
                "class_name": class_name,
                "box_count": len(labels),
                "labels": labels,
            }
        )
    return records, names


def sample_records(
    real_records: list[dict[str, Any]],
    synthetic_records: list[dict[str, Any]],
    *,
    seed: int,
    max_per_class: int,
    max_total: int,
    class_balance: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    for rows in (real_records, synthetic_records):
        rng.shuffle(rows)

    if not class_balance:
        count = min(len(real_records), len(synthetic_records))
        if max_total > 0:
            count = min(count, max_total // 2)
        selected = real_records[:count] + synthetic_records[:count]
        rng.shuffle(selected)
        return selected, {
            "mode": "equal_domain_random",
            "per_domain": count,
            "input_real": len(real_records),
            "input_synthetic": len(synthetic_records),
        }

    real_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    synthetic_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in real_records:
        real_by_class[int(row["class_id"])].append(row)
    for row in synthetic_records:
        synthetic_by_class[int(row["class_id"])].append(row)

    selected: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    for class_id in sorted(set(real_by_class) & set(synthetic_by_class)):
        real_rows = real_by_class[class_id]
        synthetic_rows = synthetic_by_class[class_id]
        count = min(len(real_rows), len(synthetic_rows))
        if max_per_class > 0:
            count = min(count, max_per_class)
        if count <= 0:
            continue
        selected.extend(real_rows[:count])
        selected.extend(synthetic_rows[:count])
        per_class.append(
            {
                "class_id": class_id,
                "class_name": real_rows[0]["class_name"],
                "real": len(real_rows),
                "synthetic": len(synthetic_rows),
                "selected_per_domain": count,
            }
        )

    if max_total > 0 and len(selected) > max_total:
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(int(row["class_id"]), str(row["domain"]))].append(row)
        cap = max(1, max_total // max(2, len(grouped)))
        capped = []
        for key in sorted(grouped):
            capped.extend(grouped[key][:cap])
        selected = capped[:max_total]

    rng.shuffle(selected)
    return selected, {
        "mode": "class_balanced_equal_domain",
        "input_real": len(real_records),
        "input_synthetic": len(synthetic_records),
        "selected": len(selected),
        "per_class": per_class,
    }


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def load_letterboxed_tensor(path: Path, imgsz: int) -> torch.Tensor:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(imgsz / width, imgsz / height)
    resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = image.resize(resized, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    left = (imgsz - resized[0]) // 2
    top = (imgsz - resized[1]) // 2
    canvas.paste(image, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def pooled_feature(output: Any) -> torch.Tensor | None:
    if isinstance(output, (list, tuple)):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        if not tensors:
            return None
        output = tensors[0]
    if not isinstance(output, torch.Tensor):
        return None
    tensor = output.detach().float()
    if tensor.ndim == 4:
        return F.adaptive_avg_pool2d(tensor, 1).flatten(1).cpu()
    if tensor.ndim == 3:
        return tensor.mean(dim=-1).cpu()
    if tensor.ndim == 2:
        return tensor.cpu()
    return tensor.flatten(1).cpu() if tensor.ndim > 1 else None


def extract_features(
    *,
    model_path: Path,
    records: list[dict[str, Any]],
    layers: list[int],
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    yolo = YOLO(str(model_path))
    model = yolo.model.to(device).eval()
    modules = list(model.model)
    for layer in layers:
        if layer < 0 or layer >= len(modules):
            raise SystemExit(f"Layer {layer} outside model range 0..{len(modules) - 1}")

    captured: dict[int, torch.Tensor] = {}
    features: dict[str, list[np.ndarray]] = {str(layer): [] for layer in layers}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            feature = pooled_feature(output)
            if feature is not None:
                captured[layer_index] = feature

        return hook

    for layer in layers:
        handles.append(modules[layer].register_forward_hook(make_hook(layer)))

    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch_records = records[start : start + batch_size]
                batch = torch.stack(
                    [load_letterboxed_tensor(Path(row["image"]), imgsz) for row in batch_records],
                    dim=0,
                ).to(device)
                captured.clear()
                _ = model(batch)
                for layer in layers:
                    if layer not in captured:
                        raise SystemExit(f"Layer {layer} did not emit a tensor feature")
                    features[str(layer)].append(captured[layer].numpy())
    finally:
        for handle in handles:
            handle.remove()

    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in features.items()}


def safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or len(y) < 2:
        return {"mmd2": float("nan"), "gamma": float("nan")}
    combined = np.vstack([x, y])
    distances = pairwise_distances(combined, metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0]
    median_sq = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(2.0 * median_sq, 1e-12)
    kxx = np.exp(-gamma * pairwise_distances(x, x, metric="sqeuclidean"))
    kyy = np.exp(-gamma * pairwise_distances(y, y, metric="sqeuclidean"))
    kxy = np.exp(-gamma * pairwise_distances(x, y, metric="sqeuclidean"))
    return {"mmd2": float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean()), "gamma": gamma}


def coral_distance(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    x_cov = (x_centered.T @ x_centered) / max(len(x) - 1, 1)
    y_cov = (y_centered.T @ y_centered) / max(len(y) - 1, 1)
    dim = max(x.shape[1], 1)
    return float(np.sum((x_cov - y_cov) ** 2) / (4.0 * dim * dim))


def domain_accuracy(x: np.ndarray, domains: np.ndarray, seed: int) -> dict[str, float | int | None]:
    counts = Counter(int(value) for value in domains)
    min_class = min(counts.values()) if counts else 0
    if min_class < 4 or len(counts) != 2:
        return {"mean": None, "std": None, "folds": 0}
    folds = min(5, min_class)
    estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, x, domains, cv=cv, scoring="accuracy")
    mean = float(scores.mean())
    return {
        "mean": mean,
        "std": float(scores.std()),
        "folds": folds,
        "proxy_a_distance": float(max(0.0, min(2.0, 4.0 * mean - 2.0))),
    }


def layer_metrics(
    *,
    features: np.ndarray,
    records: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    domains = np.array([0 if row["domain"] == "real" else 1 for row in records], dtype=np.int64)
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    real_x = x[domains == 0]
    synthetic_x = x[domains == 1]

    real_centroid = real_x.mean(axis=0)
    synthetic_centroid = synthetic_x.mean(axis=0)
    centroid_l2 = float(np.linalg.norm(real_centroid - synthetic_centroid))
    within_real = np.linalg.norm(real_x - real_centroid, axis=1).mean() if len(real_x) else 0.0
    within_synthetic = np.linalg.norm(synthetic_x - synthetic_centroid, axis=1).mean() if len(synthetic_x) else 0.0
    pooled_within = max(float((within_real + within_synthetic) / 2.0), 1e-12)
    mmd = mmd_rbf(real_x, synthetic_x)
    return (
        {
            "dim": int(features.shape[1]),
            "domain_accuracy": domain_accuracy(x, domains, seed),
            "centroid_l2": centroid_l2,
            "centroid_l2_over_within": float(centroid_l2 / pooled_within),
            "centroid_cosine_distance": safe_cosine_distance(real_centroid, synthetic_centroid),
            "within_l2_mean": {
                "real": float(within_real),
                "synthetic": float(within_synthetic),
            },
            "mmd_rbf": mmd,
            "coral_distance": coral_distance(real_x, synthetic_x),
        },
        x,
    )


def per_class_gaps(
    *,
    x: np.ndarray,
    records: list[dict[str, Any]],
    min_per_class: int,
) -> list[dict[str, Any]]:
    rows = []
    class_ids = sorted({int(row["class_id"]) for row in records})
    for class_id in class_ids:
        indexes = [i for i, row in enumerate(records) if int(row["class_id"]) == class_id]
        real_indexes = [i for i in indexes if records[i]["domain"] == "real"]
        synthetic_indexes = [i for i in indexes if records[i]["domain"] == "synthetic"]
        if len(real_indexes) < min_per_class or len(synthetic_indexes) < min_per_class:
            continue
        real_x = x[real_indexes]
        synthetic_x = x[synthetic_indexes]
        real_centroid = real_x.mean(axis=0)
        synthetic_centroid = synthetic_x.mean(axis=0)
        distances = pairwise_distances(real_x, synthetic_x, metric="euclidean")
        rows.append(
            {
                "class_id": class_id,
                "class_name": records[indexes[0]]["class_name"],
                "real": len(real_indexes),
                "synthetic": len(synthetic_indexes),
                "centroid_l2": float(np.linalg.norm(real_centroid - synthetic_centroid)),
                "centroid_cosine_distance": safe_cosine_distance(real_centroid, synthetic_centroid),
                "real_to_synthetic_nearest_l2_mean": float(distances.min(axis=1).mean()),
                "real_to_synthetic_nearest_l2_max": float(distances.min(axis=1).max()),
                "synthetic_to_real_nearest_l2_mean": float(distances.min(axis=0).mean()),
            }
        )
    rows.sort(key=lambda row: row["real_to_synthetic_nearest_l2_mean"], reverse=True)
    return rows


def nearest_exports(
    *,
    x: np.ndarray,
    records: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real_indexes = [i for i, row in enumerate(records) if row["domain"] == "real"]
    synthetic_indexes = [i for i, row in enumerate(records) if row["domain"] == "synthetic"]
    distances = pairwise_distances(x[real_indexes], x[synthetic_indexes], metric="euclidean")

    real_rows = []
    for row_pos, record_index in enumerate(real_indexes):
        nearest_pos = int(np.argmin(distances[row_pos]))
        nearest_index = synthetic_indexes[nearest_pos]
        real_rows.append(
            {
                "domain": "real",
                "image": records[record_index]["image_rel"],
                "class_name": records[record_index]["class_name"],
                "box_count": records[record_index]["box_count"],
                "nearest_synthetic": records[nearest_index]["image_rel"],
                "nearest_synthetic_class": records[nearest_index]["class_name"],
                "nearest_l2": float(distances[row_pos, nearest_pos]),
            }
        )
    real_rows.sort(key=lambda row: row["nearest_l2"], reverse=True)

    synthetic_rows = []
    for col_pos, record_index in enumerate(synthetic_indexes):
        nearest_pos = int(np.argmin(distances[:, col_pos]))
        nearest_index = real_indexes[nearest_pos]
        synthetic_rows.append(
            {
                "domain": "synthetic",
                "image": records[record_index]["image_rel"],
                "class_name": records[record_index]["class_name"],
                "box_count": records[record_index]["box_count"],
                "nearest_real": records[nearest_index]["image_rel"],
                "nearest_real_class": records[nearest_index]["class_name"],
                "nearest_l2": float(distances[nearest_pos, col_pos]),
            }
        )
    synthetic_rows.sort(key=lambda row: row["nearest_l2"], reverse=True)
    return real_rows[:top_k], synthetic_rows[:top_k]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# YOLO Representation Domain Gap",
        "",
        f"- Model: `{payload['model']}`",
        f"- Real: `{payload['real_data']}` split `{payload['real_split']}`",
        f"- Synthetic: `{payload['synthetic_data']}` split `{payload['synthetic_split']}`",
        f"- Sample mode: `{payload['sampling']['mode']}`; selected `{payload['sampling'].get('selected', len(payload['records']))}` rows",
        "",
        "## Layer Domain Separability",
        "",
        "| Layer | Dim | Domain acc | Proxy A-distance | Centroid/within | MMD2 | CORAL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["layer_table"]:
        acc = row["domain_accuracy"]["mean"]
        pad = row["domain_accuracy"].get("proxy_a_distance")
        lines.append(
            "| {layer} | {dim} | {acc} | {pad} | {centroid} | {mmd} | {coral} |".format(
                layer=row["layer"],
                dim=row["dim"],
                acc="" if acc is None else f"{acc:.3f}",
                pad="" if pad is None else f"{pad:.3f}",
                centroid=f"{row['centroid_l2_over_within']:.3f}",
                mmd=f"{row['mmd_rbf']['mmd2']:.4f}",
                coral=f"{row['coral_distance']:.4f}",
            )
        )
    lines.extend(["", f"## Worst Classes At Layer {payload['nearest_layer']}", ""])
    lines.extend(["| Class | Real | Synthetic | Nearest gap mean | Centroid L2 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in payload["per_class_gaps"][:15]:
        lines.append(
            f"| {row['class_name']} | {row['real']} | {row['synthetic']} | "
            f"{row['real_to_synthetic_nearest_l2_mean']:.3f} | {row['centroid_l2']:.3f} |"
        )
    lines.extend(["", "## Most Uncovered Real Images", ""])
    lines.extend(["| Class | Nearest L2 | Image | Nearest synthetic |", "| --- | ---: | --- | --- |"])
    for row in payload["top_uncovered_real"][:20]:
        lines.append(
            f"| {row['class_name']} | {row['nearest_l2']:.3f} | "
            f"`{row['image']}` | `{row['nearest_synthetic']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = resolve(args.model)
    real_data = resolve(args.real_data)
    synthetic_data = resolve(args.synthetic_data)
    out_dir = resolve(args.out_dir)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    nearest_layer = str(args.nearest_layer if args.nearest_layer is not None else layers[-1])
    if int(nearest_layer) not in layers:
        raise SystemExit("--nearest-layer must be one of --layers")

    real_records, names = dataset_records(
        data_path=real_data,
        split=args.real_split,
        domain="real",
        include_background=args.include_background,
    )
    synthetic_records, synthetic_names = dataset_records(
        data_path=synthetic_data,
        split=args.synthetic_split,
        domain="synthetic",
        include_background=args.include_background,
    )
    if names != synthetic_names:
        print("[warn] real and synthetic class-name maps differ; using row-level names", file=sys.stderr)

    records, sampling = sample_records(
        real_records,
        synthetic_records,
        seed=args.seed,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
        class_balance=not args.no_class_balance,
    )
    if len(records) < 8:
        raise SystemExit(f"Not enough sampled records for a representation probe: {len(records)}")

    device = choose_device(args.device)
    print(
        f"[probe] model={repo_rel(model_path)} records={len(records)} layers={layers} "
        f"imgsz={args.imgsz} batch={args.batch} device={device}",
        flush=True,
    )
    features_by_layer = extract_features(
        model_path=model_path,
        records=records,
        layers=layers,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=device,
    )

    layer_payloads: dict[str, Any] = {}
    standardized_by_layer: dict[str, np.ndarray] = {}
    layer_table = []
    for layer in [str(value) for value in layers]:
        metrics, standardized = layer_metrics(features=features_by_layer[layer], records=records, seed=args.seed)
        layer_payloads[layer] = metrics
        standardized_by_layer[layer] = standardized
        layer_table.append({"layer": layer, **metrics})

    nearest_x = standardized_by_layer[nearest_layer]
    class_gaps = per_class_gaps(x=nearest_x, records=records, min_per_class=args.min_per_class)
    top_real, top_synthetic = nearest_exports(x=nearest_x, records=records, top_k=args.top_k)

    records_payload = [
        {
            "domain": row["domain"],
            "image": row["image_rel"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "box_count": row["box_count"],
        }
        for row in records
    ]
    payload = {
        "schema": "cashsnap_yolo_representation_domain_gap_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": repo_rel(model_path),
        "real_data": repo_rel(real_data),
        "real_split": args.real_split,
        "synthetic_data": repo_rel(synthetic_data),
        "synthetic_split": args.synthetic_split,
        "imgsz": args.imgsz,
        "layers": layers,
        "nearest_layer": nearest_layer,
        "sampling": sampling,
        "records": records_payload,
        "input_counts": {
            "real": len(real_records),
            "synthetic": len(synthetic_records),
            "real_by_class": dict(Counter(row["class_name"] for row in real_records)),
            "synthetic_by_class": dict(Counter(row["class_name"] for row in synthetic_records)),
        },
        "layers_metrics": layer_payloads,
        "layer_table": layer_table,
        "per_class_gaps": class_gaps,
        "top_uncovered_real": top_real,
        "top_uncovered_synthetic": top_synthetic,
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "top_uncovered_real.csv", top_real)
    write_csv(out_dir / "top_uncovered_synthetic.csv", top_synthetic)
    write_csv(out_dir / "per_class_gaps.csv", class_gaps)
    write_markdown(out_dir / "summary.md", payload)
    print(f"ok: wrote {repo_rel(json_path)}", flush=True)


if __name__ == "__main__":
    main()
