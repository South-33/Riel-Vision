from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


SOURCE_ORDER = {"final": 0, "detector": 1, "fragment": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize final/detector/fragment source metrics from run_browser_smoke_cases.py JSON reports."
    )
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path, help="Optional JSON summary output.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_report(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected JSON object")
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit(f"{repo_path(resolved)}: expected cases array")
    data["_source"] = repo_path(resolved)
    return data


def case_role(case: dict[str, Any]) -> str:
    notes = str(case.get("notes", ""))
    marker = "diagnostic mined real "
    if marker in notes:
        return notes.split(marker, 1)[1].split(" ", 1)[0] or "unknown"
    case_id = str(case.get("caseId", ""))
    if case_id:
        return re.sub(r"_v\d+$", "", case_id)
    return "unknown"


def source_rows(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {"final": evaluation}
    sources = evaluation.get("sources", {})
    if isinstance(sources, dict):
        for name, row in sources.items():
            if name == "final" or not isinstance(row, dict):
                continue
            rows[str(name)] = row
    return rows


def add_metric(counter: Counter[str], key: str, value: Any) -> None:
    try:
        counter[key] += int(value or 0)
    except (TypeError, ValueError):
        return


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for case in report.get("cases", []):
        if not isinstance(case, dict):
            continue
        evaluation = case.get("evaluation", {})
        if not isinstance(evaluation, dict):
            continue
        role = case_role(case)
        gt_count = int(evaluation.get("gtCount", 0) or 0)
        rejected = int((case.get("debug", {}) or {}).get("rejectedProposals", 0) or 0)
        for source, row in source_rows(evaluation).items():
            counter = totals[(role, source)]
            counter["cases"] += 1
            counter["gt"] += gt_count
            counter["rejected"] += rejected if source == "final" else 0
            add_metric(counter, "pred", row.get("predCount"))
            add_metric(counter, "same", row.get("matchedSameClass"))
            add_metric(counter, "any", row.get("matchedAnyClass"))
            counter["abs_count_error"] += abs(int(row.get("countError", 0) or 0))
            if source == "final":
                counter["abs_khr_error"] += abs(int(evaluation.get("khrValueError", 0) or 0))
                counter["abs_usd_error"] += abs(int(evaluation.get("usdValueError", 0) or 0))

    rows: list[dict[str, Any]] = []
    for (role, source), counter in sorted(
        totals.items(), key=lambda item: (item[0][0], SOURCE_ORDER.get(item[0][1], 99), item[0][1])
    ):
        gt = int(counter["gt"])
        same = int(counter["same"])
        any_match = int(counter["any"])
        rows.append(
            {
                "role": role,
                "source": source,
                "cases": int(counter["cases"]),
                "gt": gt,
                "pred": int(counter["pred"]),
                "same": same,
                "any": any_match,
                "recall_same": (same / gt) if gt else 1.0,
                "recall_any": (any_match / gt) if gt else 1.0,
                "abs_count_error": int(counter["abs_count_error"]),
                "abs_khr_error": int(counter["abs_khr_error"]),
                "abs_usd_error": int(counter["abs_usd_error"]),
                "rejected": int(counter["rejected"]),
            }
        )
    return {
        "source": report.get("_source", ""),
        "schema": report.get("schema", ""),
        "generated_at_utc": report.get("generated_at_utc", ""),
        "rows": rows,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"report={summary['source']}")
    print("role,source,cases,gt,pred,same,any,recall_same,recall_any,abs_count_error,abs_khr_error,abs_usd_error,rejected")
    for row in summary["rows"]:
        print(
            ",".join(
                [
                    str(row["role"]),
                    str(row["source"]),
                    str(row["cases"]),
                    str(row["gt"]),
                    str(row["pred"]),
                    str(row["same"]),
                    str(row["any"]),
                    f"{row['recall_same']:.4f}",
                    f"{row['recall_any']:.4f}",
                    str(row["abs_count_error"]),
                    str(row["abs_khr_error"]),
                    str(row["abs_usd_error"]),
                    str(row["rejected"]),
                ]
            )
        )


def main() -> int:
    args = parse_args()
    summaries = [summarize_report(read_report(path)) for path in args.reports]
    for index, summary in enumerate(summaries):
        if index:
            print()
        print_summary(summary)
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"reports": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
