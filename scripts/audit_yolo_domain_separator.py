#!/usr/bin/env python
"""Train lightweight real-vs-synthetic separators on domain-gap CSV features."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FAMILY_LABELS = {"real": 0, "synthetic": 1}
NON_FEATURE_COLUMNS = {
    "family",
    "image",
    "label_index",
    "source_group",
    "source_family",
    "split",
    "class_id",
    "class_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-csv", type=Path, default=None)
    parser.add_argument("--box-csv", type=Path, default=None)
    parser.add_argument("--crop-csv", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--max-auc", type=float, default=None, help="Optional fail gate for cross-val ROC AUC.")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing CSV: {repo_rel(resolved)}")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def numeric_features(rows: list[dict[str, str]]) -> list[str]:
    candidates = [key for key in rows[0] if key not in NON_FEATURE_COLUMNS]
    features = []
    for key in candidates:
        values = []
        for row in rows:
            value = row.get(key, "")
            if value == "":
                continue
            try:
                values.append(float(value))
            except ValueError:
                values = []
                break
        if values:
            features.append(key)
    return features


def row_family(row: dict[str, str]) -> str:
    family = str(row.get("source_family", "")).strip()
    if family in FAMILY_LABELS:
        return family
    role = str(row.get("family", "")).strip()
    if role == "reference":
        return "real"
    if role == "candidate":
        return "synthetic"
    group = str(row.get("source_group", "")).strip()
    if group == "real":
        return "real"
    if group.startswith("synthetic"):
        return "synthetic"
    return ""


def feature_matrix(rows: list[dict[str, str]], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    kept_rows = []
    labels = []
    for row in rows:
        family = row_family(row)
        if family not in FAMILY_LABELS:
            continue
        values = []
        usable = True
        for feature in features:
            try:
                values.append(float(row.get(feature, "")))
            except ValueError:
                usable = False
                break
        if usable:
            kept_rows.append(values)
            labels.append(FAMILY_LABELS[family])
    return np.asarray(kept_rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def family_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        family: int((labels == label).sum())
        for family, label in FAMILY_LABELS.items()
    }


def feature_means(rows: list[dict[str, str]], features: list[str]) -> dict[str, dict[str, float | None]]:
    means: dict[str, dict[str, float | None]] = {}
    for feature in features:
        family_values: dict[str, list[float]] = {"real": [], "synthetic": []}
        for row in rows:
            family = row_family(row)
            if family not in family_values:
                continue
            try:
                family_values[family].append(float(row.get(feature, "")))
            except ValueError:
                continue
        real_values = family_values["real"]
        synth_values = family_values["synthetic"]
        real_mean = float(np.mean(real_values)) if real_values else None
        synth_mean = float(np.mean(synth_values)) if synth_values else None
        means[feature] = {
            "real_mean": real_mean,
            "synthetic_mean": synth_mean,
            "synthetic_minus_real": None
            if real_mean is None or synth_mean is None
            else float(synth_mean - real_mean),
        }
    return means


def evaluate_separator(rows: list[dict[str, str]], *, name: str, top_k: int) -> dict[str, Any]:
    if not rows:
        return {"name": name, "available": False, "reason": "no rows"}
    features = numeric_features(rows)
    if not features:
        return {"name": name, "available": False, "reason": "no numeric features"}
    x_values, labels = feature_matrix(rows, features)
    counts = family_counts(labels)
    min_family_count = min(counts.values())
    if min_family_count < 2:
        return {
            "name": name,
            "available": False,
            "reason": "need at least two rows per family",
            "family_counts": counts,
        }
    splits = min(5, min_family_count)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2606)
    probabilities = cross_val_predict(model, x_values, labels, cv=cv, method="predict_proba")[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    model.fit(x_values, labels)
    logistic = model.named_steps["logisticregression"]
    coefficients = logistic.coef_[0]
    means = feature_means(rows, features)
    top_features = []
    for feature, coefficient in sorted(
        zip(features, coefficients, strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:top_k]:
        row = dict(means[feature])
        row.update(
            {
                "feature": feature,
                "coefficient_synthetic_positive": float(coefficient),
                "abs_coefficient": abs(float(coefficient)),
            }
        )
        top_features.append(row)
    return {
        "name": name,
        "available": True,
        "rows": int(labels.shape[0]),
        "family_counts": counts,
        "features": features,
        "cv_splits": splits,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "top_features": top_features,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    tasks = []
    if args.image_csv:
        tasks.append(("image_stats", args.image_csv))
    if args.box_csv:
        tasks.append(("box_geometry", args.box_csv))
    if args.crop_csv:
        tasks.append(("crop_stats", args.crop_csv))
    if not tasks:
        raise SystemExit("provide --image-csv, --box-csv, and/or --crop-csv")

    results = []
    for name, path in tasks:
        rows = read_csv_rows(path)
        result = evaluate_separator(rows, name=name, top_k=args.top_k)
        result["source_csv"] = repo_rel(resolve(path))
        results.append(result)

    failing = [
        row
        for row in results
        if args.max_auc is not None and row.get("available") and float(row.get("roc_auc", 0.0)) > args.max_auc
    ]
    payload = {
        "schema": "cashsnap_domain_separator_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "max_auc": args.max_auc,
        "passed": not failing,
        "results": results,
        "failures": [
            {"name": row["name"], "roc_auc": row["roc_auc"], "threshold": args.max_auc}
            for row in failing
        ],
    }
    out = resolve(args.json_out)
    write_json(out, payload)
    print(f"wrote_json={repo_rel(out)}")
    for row in results:
        if not row.get("available"):
            print(f"{row['name']}: unavailable reason={row.get('reason')}")
            continue
        top = ", ".join(
            f"{feature['feature']}({feature['coefficient_synthetic_positive']:+.3f})"
            for feature in row.get("top_features", [])[:5]
        )
        print(
            f"{row['name']}: auc={row['roc_auc']:.3f} "
            f"balanced_acc={row['balanced_accuracy']:.3f} rows={row['rows']} top={top}"
        )
    return 0 if payload["passed"] or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
