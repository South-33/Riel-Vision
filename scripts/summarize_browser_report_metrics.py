from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_report(path_text: str) -> tuple[Path, dict]:
    path = Path(path_text)
    resolved = path if path.is_absolute() else ROOT / path
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def metric_line(path: Path, report: dict) -> str:
    cases = report.get("cases", [])
    total_cases = len(cases)
    gt = pred = same = any_match = 0
    abs_count = abs_khr = abs_usd = rejected = 0
    perfect = 0
    failures = len(report.get("failures", []) or [])
    for case in cases:
        evaluation = case.get("evaluation") or {}
        gt_count = int(evaluation.get("gtCount", 0) or 0)
        pred_count = int(evaluation.get("predCount", case.get("totalCount", 0)) or 0)
        same_count = int(evaluation.get("matchedSameClass", 0) or 0)
        any_count = int(evaluation.get("matchedAnyClass", 0) or 0)
        count_error = int(evaluation.get("countError", pred_count - gt_count) or 0)
        khr_error = int(evaluation.get("khrValueError", 0) or 0)
        usd_error = int(evaluation.get("usdValueError", 0) or 0)
        gt += gt_count
        pred += pred_count
        same += same_count
        any_match += any_count
        abs_count += abs(count_error)
        abs_khr += abs(khr_error)
        abs_usd += abs(usd_error)
        rejected += int((case.get("debug") or {}).get("rejectedProposals", 0) or 0)
        if same_count == gt_count and count_error == 0 and khr_error == 0 and usd_error == 0:
            perfect += 1
    return (
        f"{path.relative_to(ROOT)}: cases={total_cases} failures={failures} "
        f"perfect={perfect}/{total_cases} gt={gt} pred={pred} same={same} any={any_match} "
        f"abs_count={abs_count} abs_khr={abs_khr} abs_usd={abs_usd} rejected={rejected}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: browser_report_metrics.py REPORT [REPORT ...]")
    for arg in sys.argv[1:]:
        path, report = load_report(arg)
        print(metric_line(path, report))


if __name__ == "__main__":
    main()
