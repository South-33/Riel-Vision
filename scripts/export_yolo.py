from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CashSnap YOLO checkpoint.")
    parser.add_argument("--model", required=True, help="YOLO checkpoint path.")
    parser.add_argument("--format", required=True, help="Ultralytics export format, e.g. onnx, ncnn, tflite.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default=None, help="Ultralytics device selector, e.g. cpu or 0.")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset when applicable.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic input shape when supported.")
    parser.add_argument("--simplify", action="store_true", help="Simplify exported graph when supported.")
    parser.add_argument("--half", action="store_true", help="Export FP16 when supported.")
    parser.add_argument("--int8", action="store_true", help="Export INT8 when supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_from_root(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "half": args.half,
        "int8": args.int8,
    }
    if args.device is not None:
        export_args["device"] = args.device
    if args.opset is not None:
        export_args["opset"] = args.opset

    YOLO(str(model_path)).export(**export_args)


if __name__ == "__main__":
    main()
