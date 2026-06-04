from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC = "metrics/mAP50-95(B)"


def resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two val_yolo.py metrics JSON files.")
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate metrics JSON.")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric key under the JSON 'results' object.")
    parser.add_argument("--min-delta", type=float, default=None, help="Require candidate - baseline >= this value.")
    parser.add_argument("--max-drop", type=float, default=0.0, help="Allow at most this much negative delta.")
    parser.add_argument("--json-out", default=None, help="Optional machine-readable comparison output.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0, even when the comparison fails.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def result_metric(document: dict, metric: str) -> float:
    results = document.get("results", {})
    if not isinstance(results, dict) or metric not in results:
        raise KeyError(f"missing results metric {metric!r}")
    return float(results[metric])


def per_class_by_name(document: dict) -> dict[str, dict]:
    rows = document.get("per_class", [])
    if not isinstance(rows, list):
        return {}
    by_name = {}
    for row in rows:
        if isinstance(row, dict) and row.get("class_name") is not None:
            by_name[str(row["class_name"])] = row
    return by_name


def compare_per_class(baseline: dict, candidate: dict, metric_name: str = "map50_95") -> list[dict]:
    baseline_rows = per_class_by_name(baseline)
    candidate_rows = per_class_by_name(candidate)
    rows = []
    for class_name in sorted(set(baseline_rows) | set(candidate_rows)):
        baseline_value = baseline_rows.get(class_name, {}).get(metric_name)
        candidate_value = candidate_rows.get(class_name, {}).get(metric_name)
        row = {
            "class_name": class_name,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": None,
        }
        if baseline_value is not None and candidate_value is not None:
            row["delta"] = float(candidate_value) - float(baseline_value)
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline_path = resolve(args.baseline)
    candidate_path = resolve(args.candidate)
    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)
    baseline_value = result_metric(baseline, args.metric)
    candidate_value = result_metric(candidate, args.metric)
    delta = candidate_value - baseline_value

    checks = [
        {
            "name": "max_drop",
            "passed": delta >= -args.max_drop,
            "threshold": -args.max_drop,
            "actual_delta": delta,
        }
    ]
    if args.min_delta is not None:
        checks.append(
            {
                "name": "min_delta",
                "passed": delta >= args.min_delta,
                "threshold": args.min_delta,
                "actual_delta": delta,
            }
        )

    passed = all(check["passed"] for check in checks)
    payload = {
        "passed": passed,
        "metric": args.metric,
        "baseline_path": repo_rel(baseline_path),
        "candidate_path": repo_rel(candidate_path),
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "checks": checks,
        "per_class_map50_95": compare_per_class(baseline, candidate),
    }
    if args.json_out:
        write_json(resolve(args.json_out), payload)

    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict}: {args.metric} baseline={baseline_value:.6f} "
        f"candidate={candidate_value:.6f} delta={delta:+.6f}"
    )
    if args.json_out:
        print(f"wrote_json={repo_rel(resolve(args.json_out))}")
    return 0 if passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
