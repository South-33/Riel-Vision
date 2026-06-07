from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from local_runtime import ROOT, configure_project_cache


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize preservation-gated refiner outputs as a standard YOLO "
            "synthetic dataset root with images/train, labels/train, data.yaml, "
            "and a manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Refiner synthetic_manifest.jsonl.")
    parser.add_argument("--refined-root", type=Path, required=True, help="Directory containing gated refined images.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output YOLO dataset root.")
    parser.add_argument(
        "--reference-data-yaml",
        type=Path,
        required=True,
        help="Reference data.yaml to copy class names from.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional leading row cap for smokes.")
    parser.add_argument("--candidate-id", default="", help="Optional provenance id written into metadata.")
    parser.add_argument("--lock-policy", default="note_edge_hard", help="Refiner lock policy written into metadata.")
    parser.add_argument("--clean", action="store_true", help="Delete the output root before materializing.")
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


def find_refined_image(refined_root: Path, source_image: Path) -> Path | None:
    stem = source_image.stem
    candidates = [refined_root / source_image.name, refined_root / "images" / "train" / source_image.name]
    for ext in IMAGE_EXTS:
        candidates.append(refined_root / f"{stem}{ext}")
        candidates.append(refined_root / "images" / "train" / f"{stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_names(reference_data_yaml: Path) -> dict[int, str]:
    if not reference_data_yaml.exists():
        raise SystemExit(f"missing reference data.yaml: {repo_rel(reference_data_yaml)}")
    document = yaml.safe_load(reference_data_yaml.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{repo_rel(reference_data_yaml)} must contain a YAML mapping")
    names = document.get("names")
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    raise SystemExit(f"{repo_rel(reference_data_yaml)} missing names list or mapping")


def infer_dataset_root(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1):
        if parts[idx].lower() == "images" and idx + 1 < len(parts) and parts[idx + 1].lower() == "train":
            return Path(*parts[:idx])
    return None


def load_metadata_by_image(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    roots: set[Path] = set()
    for row in rows:
        original = row.get("original_source_image")
        if not original:
            continue
        root = infer_dataset_root(resolve_repo_path(Path(str(original))))
        if root is not None:
            roots.add(root)
    metadata: dict[str, dict[str, Any]] = {}
    for root in sorted(roots):
        metadata_path = root / "metadata" / "train.jsonl"
        if not metadata_path.exists():
            continue
        for line_no, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{repo_rel(metadata_path)}:{line_no}: invalid JSON: {exc}") from exc
            image = str(record.get("image", ""))
            if image:
                metadata[(root / image).resolve().as_posix()] = record
                metadata[Path(image).stem] = record
    return metadata


def copy_row(row: dict[str, Any], refined_root: Path, out_root: Path) -> dict[str, Any]:
    source_image = resolve_repo_path(Path(str(row["source_image"])))
    refined_image = find_refined_image(refined_root, source_image)
    source_label = resolve_repo_path(Path(str(row.get("label") or row.get("source_label"))))
    result: dict[str, Any] = {
        "id": row.get("id", source_image.stem),
        "class_name": row.get("class_name", ""),
        "source_image": repo_rel(source_image),
        "source_label": repo_rel(source_label),
        "original_source_image": row.get("original_source_image", ""),
        "refined_image": repo_rel(refined_image) if refined_image is not None else "",
        "status": "pass",
        "violations": [],
    }
    if refined_image is None:
        result["status"] = "fail"
        result["violations"].append("missing_refined_image")
        return result
    if not source_label.exists():
        result["status"] = "fail"
        result["violations"].append("missing_source_label")
        return result

    image_out = out_root / "images" / "train" / f"{source_image.stem}{refined_image.suffix.lower()}"
    label_out = out_root / "labels" / "train" / f"{source_image.stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(refined_image, image_out)
    shutil.copy2(source_label, label_out)
    result["output_image"] = repo_rel(image_out)
    result["output_label"] = repo_rel(label_out)
    return result


def write_metadata(
    out_root: Path,
    manifest_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    candidate_id: str,
    lock_policy: str,
) -> int:
    metadata_by_image = load_metadata_by_image(manifest_rows)
    metadata_rows: list[dict[str, Any]] = []
    for source_row, output_row in zip(manifest_rows, output_rows):
        if output_row.get("status") != "pass":
            continue
        original_image = resolve_repo_path(Path(str(source_row.get("original_source_image", ""))))
        source_record = metadata_by_image.get(original_image.resolve().as_posix()) or metadata_by_image.get(original_image.stem)
        if not source_record:
            if not source_row.get("quad_xy"):
                continue
            source_record = dict(source_row)
        output_image = Path(str(output_row["output_image"]))
        output_label = Path(str(output_row["output_label"]))
        record = dict(source_record)
        record["source_composite_policy"] = record.get("composite_policy", "")
        record["composite_policy"] = f"refiner_{lock_policy}"
        record["image"] = (Path("images") / "train" / output_image.name).as_posix()
        record["label"] = (Path("labels") / "train" / output_label.name).as_posix()
        record["split"] = "train"
        record["refiner_candidate_id"] = candidate_id
        record["refiner_lock_policy"] = lock_policy
        record["refiner_source_image"] = output_row.get("source_image", "")
        record["refiner_original_source_image"] = output_row.get("original_source_image", "")
        record["refiner_raw_image"] = output_row.get("refined_image", "")
        metadata_rows.append(record)

    metadata_path = out_root / "metadata" / "train.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in metadata_rows) + "\n",
        encoding="utf-8",
    )
    return len(metadata_rows)


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label_path = resolve_repo_path(Path(str(row.get("output_label", ""))))
        if row.get("status") != "pass" or not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                counts[parts[0]] = counts.get(parts[0], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def write_data_yaml(out_root: Path, names: dict[int, str]) -> None:
    payload = {
        "path": out_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/train",
        "test": "images/train",
        "names": names,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    configure_project_cache()
    args = parse_args()
    manifest = resolve_repo_path(args.manifest)
    refined_root = resolve_repo_path(args.refined_root)
    out_root = resolve_repo_path(args.out_root)
    reference_data_yaml = resolve_repo_path(args.reference_data_yaml)
    if not refined_root.exists():
        raise SystemExit(f"missing refined root: {repo_rel(refined_root)}")
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    rows = read_jsonl(manifest)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    names = load_names(reference_data_yaml)
    outputs = [copy_row(row, refined_root, out_root) for row in rows]
    failures = [row for row in outputs if row["status"] != "pass"]
    metadata_rows = 0
    if not failures:
        write_data_yaml(out_root, names)
        metadata_rows = write_metadata(
            out_root,
            rows,
            outputs,
            candidate_id=args.candidate_id or out_root.name,
            lock_policy=args.lock_policy,
        )

    summary = {
        "manifest": repo_rel(manifest),
        "refined_root": repo_rel(refined_root),
        "reference_data_yaml": repo_rel(reference_data_yaml),
        "out_root": repo_rel(out_root),
        "data_yaml": repo_rel(out_root / "data.yaml"),
        "rows": len(outputs),
        "failures": len(failures),
        "metadata_rows": metadata_rows,
        "class_instance_counts": label_counts(outputs),
        "status": "pass" if not failures else "fail",
        "failed_rows": failures[:25],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in outputs) + "\n",
        encoding="utf-8",
    )
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"data_yaml": summary["data_yaml"], "failures": len(failures), "rows": len(outputs), "status": summary["status"]}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
