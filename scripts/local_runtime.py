from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".cache_runtime"


def configure_project_cache() -> None:
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
    os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "huggingface" / "transformers"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
