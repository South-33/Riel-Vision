from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cashsnap_two_stage_oldcommon_browser_stack.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CashSnap browser/mobile stack artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-total-mb", type=float, default=20.0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def check_artifact(label: str, path_text: str) -> tuple[float, list[str]]:
    path = resolve(Path(path_text))
    if not path.exists():
        return 0.0, [f"{label}: missing {path_text}"]
    if not path.is_file():
        return 0.0, [f"{label}: not a file {path_text}"]
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"{label}: {path.relative_to(ROOT).as_posix()} {size_mb:.2f} MB")
    return size_mb, []


def main() -> None:
    args = parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    errors: list[str] = []
    total_mb = 0.0
    for label, section in [
        ("detector", config.get("detector", {})),
        ("fragment_classifier", config.get("fragment_classifier", {})),
    ]:
        size_mb, artifact_errors = check_artifact(label, section.get("path", ""))
        total_mb += size_mb
        errors.extend(artifact_errors)
    print(f"detector_classes={len(config.get('detector', {}).get('classes', []))}")
    print(f"fragment_classifier_classes={len(config.get('fragment_classifier', {}).get('classes', []))}")
    print(f"total_artifacts={total_mb:.2f} MB")
    if total_mb > args.max_total_mb:
        errors.append(f"total artifact size {total_mb:.2f} MB exceeds {args.max_total_mb:.2f} MB")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("browser stack artifact check passed")


if __name__ == "__main__":
    main()
