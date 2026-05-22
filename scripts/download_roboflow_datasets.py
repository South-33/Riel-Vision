"""
Download the three Roboflow datasets needed for CashSnap.

Usage:
    python scripts/download_roboflow_datasets.py

Reads ROBOFLOW_API_KEY from .env (or environment).
Each dataset is saved under data/raw_datasets/roboflow_<name>/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Load .env file into os.environ if present."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# Dataset definitions: (workspace_id, project_id, version, output_name)
# Versions verified via API - use clean/raw versions, not heavily pre-augmented ones.
# Khmer-US-currency v3 = 1,782 raw images (best base; v5+ add augmentation we don't want pre-baked)
# Cambodia Currency Project v2 = only available version (~552-615 images, 7 KHR classes)
# KHMER SCAN v1 = 145 images with train/val/test splits
DATASETS = [
    (
        "robot-yfusg",
        "khmer-us-currency-jofw1",
        3,                              # v3 = 1,782 clean images (train:1254 val:348 test:180)
        "roboflow_khmer_us_currency",
    ),
    (
        "khmer-riel-classification-computer-vision",
        "cambodia-currency-project",
        2,                              # v2 = only public version (~615 images, 7 KHR classes)
        "roboflow_cambodia_currency_project",
    ),
    (
        "test-gl3sj",
        "khmer-scan",
        1,                              # v1 = 145 images (train:102 val:29 test:14)
        "roboflow_khmer_scan",
    ),
]

EXPORT_FORMAT = "yolov8"  # YOLO TXT format, compatible with YOLO11n / YOLOv8n


def download_dataset(
    api_key: str,
    workspace: str,
    project_id: str,
    version: int,
    output_name: str,
) -> Path:
    from roboflow import Roboflow  # type: ignore

    out_dir = ROOT / "data" / "raw_datasets" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {workspace}/{project_id} v{version}")
    print(f"  -> {out_dir}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    dataset = project.version(version).download(
        model_format=EXPORT_FORMAT,
        location=str(out_dir),
        overwrite=True,
    )
    print(f"  OK: {dataset.location}")
    return out_dir


def main() -> None:
    load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    failed: list[str] = []
    for workspace, project_id, version, output_name in DATASETS:
        try:
            download_dataset(api_key, workspace, project_id, version, output_name)
        except Exception as exc:
            print(f"\n  FAIL {project_id}: {exc}", file=sys.stderr)
            failed.append(project_id)

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(DATASETS)} datasets downloaded successfully.")
        print("\nNext step: run  python scripts/prepare_roboflow_datasets.py")


if __name__ == "__main__":
    main()
