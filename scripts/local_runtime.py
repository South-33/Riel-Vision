from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".cache_runtime"


def configure_project_cache() -> None:
    cache_paths = {
        "XDG_CACHE_HOME": CACHE_ROOT,
        "TORCH_HOME": CACHE_ROOT / "torch",
        "HF_HOME": CACHE_ROOT / "huggingface",
        "TRANSFORMERS_CACHE": CACHE_ROOT / "huggingface" / "transformers",
        "YOLO_CONFIG_DIR": CACHE_ROOT / "ultralytics",
        "ULTRALYTICS_CONFIG_DIR": CACHE_ROOT / "ultralytics",
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
        "PIP_CACHE_DIR": CACHE_ROOT / "pip",
        "NUMBA_CACHE_DIR": CACHE_ROOT / "numba",
    }
    temp_path = CACHE_ROOT / "tmp"
    for path in [*cache_paths.values(), temp_path]:
        path.mkdir(parents=True, exist_ok=True)
    for key, path in cache_paths.items():
        os.environ.setdefault(key, str(path))

    # Windows defaults TEMP/TMP to the user profile on C:. Force temp-heavy ML
    # child processes into the repo-local runtime cache instead.
    os.environ["TMP"] = str(temp_path)
    os.environ["TEMP"] = str(temp_path)
    os.environ["TMPDIR"] = str(temp_path)
