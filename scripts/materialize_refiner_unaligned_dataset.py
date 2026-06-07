from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from local_runtime import ROOT, configure_project_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a small unpaired synthetic->real dataset from a refiner readiness pack. "
            "The output follows the common CUT/FastCUT trainA/trainB layout while preserving "
            "labels and masks for CashSnap-specific gates."
        )
    )
    parser.add_argument(
        "--readiness-dir",
        type=Path,
        default=Path("runs/cashsnap/refiner_readiness_poisson_contact_v1"),
        help="Directory created by build_refiner_readiness_pack.py.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to <readiness-dir>/cut_unaligned_smoke.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--img2img-turbo-test-count", type=int, default=8)
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing manifest: {repo_rel(path)}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{repo_rel(path)}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"empty manifest: {repo_rel(path)}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"missing source file: {repo_rel(src)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def image_name(index: int, row: dict[str, Any], key: str) -> str:
    src = Path(str(row[key]))
    suffix = src.suffix.lower() or ".jpg"
    row_id = str(row.get("id") or src.stem)
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in row_id)
    return f"{index:04d}_{safe_id}{suffix}"


def materialize_synthetic(rows: list[dict[str, Any]], out_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        name = image_name(index, row, "source_image")
        stem = Path(name).stem
        dst_image = out_root / "trainA" / name
        copy_file(resolve_repo_path(Path(str(row["source_image"]))), dst_image)
        dst_label = out_root / "labelsA" / f"{stem}.txt"
        copy_file(resolve_repo_path(Path(str(row["label"]))), dst_label)
        dst_note = out_root / "masks" / "note" / f"{stem}.png"
        dst_detail = out_root / "masks" / "detail_lock" / f"{stem}.png"
        dst_edge = out_root / "masks" / "edge_band" / f"{stem}.png"
        copy_file(resolve_repo_path(Path(str(row["note_mask"]))), dst_note)
        copy_file(resolve_repo_path(Path(str(row["detail_lock_mask"]))), dst_detail)
        copy_file(resolve_repo_path(Path(str(row["edge_band_mask"]))), dst_edge)
        manifest.append(
            {
                "id": row.get("id", stem),
                "class_name": row.get("class_name", ""),
                "source_image": repo_rel(dst_image),
                "label": repo_rel(dst_label),
                "note_mask": repo_rel(dst_note),
                "detail_lock_mask": repo_rel(dst_detail),
                "edge_band_mask": repo_rel(dst_edge),
                "trainA_image": repo_rel(dst_image),
                "original_source_image": row.get("source_image", ""),
                "source_label": row.get("label", ""),
            }
        )
    return manifest


def materialize_real(rows: list[dict[str, Any]], out_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        name = image_name(index, row, "image")
        dst_image = out_root / "trainB" / name
        copy_file(resolve_repo_path(Path(str(row["image"]))), dst_image)
        manifest.append(
            {
                "target_domain": row.get("target_domain", "cashsnap_train"),
                "trainB_image": repo_rel(dst_image),
                "source_image": row.get("image", ""),
                "source_label": row.get("label", ""),
            }
        )
    return manifest


def copy_layout_file(src: Path, dst: Path) -> str:
    copy_file(src, dst)
    return repo_rel(dst)


def materialize_img2img_turbo_layout(
    synthetic_manifest: list[dict[str, Any]],
    real_manifest: list[dict[str, Any]],
    out_root: Path,
    test_count: int,
) -> dict[str, Any]:
    train_a: list[str] = []
    train_b: list[str] = []
    test_a: list[str] = []
    test_b: list[str] = []

    for row in synthetic_manifest:
        src = resolve_repo_path(Path(str(row["trainA_image"])))
        dst = out_root / "train_A" / src.name
        train_a.append(copy_layout_file(src, dst))
    for row in real_manifest:
        src = resolve_repo_path(Path(str(row["trainB_image"])))
        dst = out_root / "train_B" / src.name
        train_b.append(copy_layout_file(src, dst))

    for row in synthetic_manifest[: max(0, test_count)]:
        src = resolve_repo_path(Path(str(row["trainA_image"])))
        dst = out_root / "test_A" / src.name
        test_a.append(copy_layout_file(src, dst))
    for row in real_manifest[: max(0, test_count)]:
        src = resolve_repo_path(Path(str(row["trainB_image"])))
        dst = out_root / "test_B" / src.name
        test_b.append(copy_layout_file(src, dst))

    fixed_prompt_a = "synthetic composited CashSnap banknote photo"
    fixed_prompt_b = "real CashSnap phone photo of banknotes on retail surfaces"
    (out_root / "fixed_prompt_a.txt").write_text(fixed_prompt_a + "\n", encoding="utf-8")
    (out_root / "fixed_prompt_b.txt").write_text(fixed_prompt_b + "\n", encoding="utf-8")

    return {
        "dataset_folder": repo_rel(out_root),
        "train_A_count": len(train_a),
        "train_B_count": len(train_b),
        "test_A_count": len(test_a),
        "test_B_count": len(test_b),
        "fixed_prompt_a": fixed_prompt_a,
        "fixed_prompt_b": fixed_prompt_b,
    }


def write_readme(out_root: Path, synthetic_count: int, real_count: int) -> None:
    lines = [
        "# CashSnap Refiner Unaligned Smoke Dataset",
        "",
        "Layout:",
        "- `trainA/`: synthetic Poisson/contact source images.",
        "- `trainB/`: train-only real CashSnap target-domain images.",
        "- `labelsA/`: unchanged YOLO labels for the synthetic images.",
        "- `masks/`: note, detail-lock, and edge-band masks for preservation gates.",
        "- `train_A/`, `train_B/`, `test_A/`, `test_B/`: CycleGAN-Turbo/img2img-turbo layout.",
        "- `fixed_prompt_a.txt`, `fixed_prompt_b.txt`: fixed domain captions for CycleGAN-Turbo.",
        "",
        f"Counts: trainA `{synthetic_count}`, trainB `{real_count}`.",
        "",
        "Run long refiner training through `scripts/run_with_headroom.py`; do not use CashSnap val/test.",
        "After inference, run `scripts/check_refiner_label_preservation.py` before any YOLO/OCR training.",
    ]
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_project_cache()
    args = parse_args()
    readiness_dir = resolve_repo_path(args.readiness_dir)
    if not readiness_dir.exists():
        raise SystemExit(f"missing readiness dir: {repo_rel(readiness_dir)}")
    out_root = resolve_repo_path(args.out_root) if args.out_root else readiness_dir / "cut_unaligned_smoke"
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    synthetic_rows = read_jsonl(readiness_dir / "synthetic_manifest.jsonl")
    real_rows = read_jsonl(readiness_dir / "real_target_manifest.jsonl")
    synthetic_manifest = materialize_synthetic(synthetic_rows, out_root)
    real_manifest = materialize_real(real_rows, out_root)
    synthetic_manifest_path = out_root / "synthetic_manifest.jsonl"
    write_jsonl(synthetic_manifest_path, synthetic_manifest)
    img2img_turbo_layout = materialize_img2img_turbo_layout(
        synthetic_manifest,
        real_manifest,
        out_root,
        args.img2img_turbo_test_count,
    )

    combined = {
        "readiness_dir": repo_rel(readiness_dir),
        "out_root": repo_rel(out_root),
        "trainA_count": len(synthetic_manifest),
        "trainB_count": len(real_manifest),
        "checker_manifest": repo_rel(synthetic_manifest_path),
        "synthetic_manifest": synthetic_manifest,
        "real_manifest": real_manifest,
        "cut_fastcut_layout": {
            "dataroot": repo_rel(out_root),
            "synthetic_domain": "trainA",
            "real_domain": "trainB",
        },
        "img2img_turbo_layout": img2img_turbo_layout,
    }
    summary_path = out_root / "dataset_manifest.json"
    summary_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root, len(synthetic_manifest), len(real_manifest))
    print(json.dumps({"summary": repo_rel(summary_path), "trainA": len(synthetic_manifest), "trainB": len(real_manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
