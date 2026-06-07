from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "manifests" / "browser_smoke_cases.csv"
DEFAULT_OUT_DIR = ROOT / ".agent" / "browser_smoke_cases"
DEFAULT_BROWSER_APP = ROOT / "demo" / "browser" / "app.js"
DEFAULT_BROWSER_INDEX = ROOT / "demo" / "browser" / "index.html"
DEFAULT_SMOKE_CDP = ROOT / "scripts" / "smoke_browser_demo_cdp.cjs"
NUMERIC_FIELDS = ["min_same_class", "min_any_class", "max_count_error", "max_khr_error", "max_usd_error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labeled CashSnap browser smoke cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout-ms", default="120000")
    parser.add_argument(
        "--subprocess-grace-seconds",
        type=float,
        default=30.0,
        help="Extra seconds beyond --timeout-ms before Python kills a stuck Node/Edge smoke subprocess.",
    )
    parser.add_argument("--port-base", type=int, default=8877, help="First local HTTP port for smoke cases.")
    parser.add_argument("--debug-port-base", type=int, default=9323, help="First Edge DevTools port for smoke cases.")
    parser.add_argument("--jobs", type=int, default=1, help="Browser smoke subprocesses to run concurrently.")
    parser.add_argument(
        "--retry-failed-sequential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --jobs > 1, rerun no-summary infrastructure failures sequentially before reporting.",
    )
    parser.add_argument("--edge", default="", help="Optional Edge executable path forwarded to the node smoke script.")
    parser.add_argument("--proposal-conf", default="", help="Optional browser detector proposal confidence override.")
    parser.add_argument("--detector-override", default="", help="Optional detector-vs-fragment fusion threshold override.")
    parser.add_argument(
        "--detector-model",
        "--detector-path",
        dest="detector_model",
        default="",
        help="Optional detector ONNX path served from the repo root.",
    )
    parser.add_argument(
        "--fragment-classifier-model",
        "--fragment-classifier-path",
        "--fragment-model",
        "--fragment-path",
        dest="fragment_classifier_model",
        default="",
        help="Optional fragment classifier ONNX path served from the repo root.",
    )
    parser.add_argument("--stack-config", "--config", dest="stack_config", default="", help="Optional stack config JSON path.")
    parser.add_argument(
        "--reject-fragment-disagreement",
        action="store_true",
        help="Reject browser proposals when detector and fragment classifier disagree.",
    )
    parser.add_argument(
        "--fragment-disagreement-min-conf",
        default="",
        help="Optional minimum fragment confidence for --reject-fragment-disagreement.",
    )
    parser.add_argument(
        "--unclassified-min-conf",
        default="",
        help="Reject browser proposals below this detector confidence when they are not fragment-classified.",
    )
    parser.add_argument("--nms-iou", default="", help="Optional fusion NMS IoU override.")
    parser.add_argument("--crop-padding", default="", help="Optional fragment crop padding override.")
    parser.add_argument("--summary-json", type=Path, help="Optional aggregate JSON summary output path.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the case manifest without launching Edge.")
    parser.add_argument("--no-artifacts", action="store_true", help="Do not write per-case screenshots or detection CSVs.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> dict[str, object]:
    resolved = resolve(path)
    row: dict[str, object] = {"path": repo_path(resolved), "exists": resolved.exists()}
    if resolved.is_file():
        row["sha256"] = file_sha256(resolved)
    return row


def browser_asset_path(value: str) -> Path:
    raw = value.strip()
    if raw.startswith("/"):
        return ROOT / raw.lstrip("/")
    return resolve(Path(raw))


def repo_asset_arg(value: str, label: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    path = browser_asset_path(raw)
    if not path.exists():
        raise ValueError(f"{label} not found: {raw}")
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be under repo root for browser serving: {raw}") from exc


def browser_stack_config_path(override: str = "") -> Path:
    if override.strip():
        return browser_asset_path(override)
    app_text = DEFAULT_BROWSER_APP.read_text(encoding="utf-8")
    match = re.search(r'STACK_CONFIG_URL\s*=\s*"([^"]+)"', app_text)
    if not match:
        raise ValueError(f"{repo_path(DEFAULT_BROWSER_APP)}: missing STACK_CONFIG_URL")
    raw = match.group(1)
    if raw.startswith("/"):
        return ROOT / raw.lstrip("/")
    return resolve(Path(raw))


def read_browser_stack_config(path: Path) -> dict:
    resolved = resolve(path)
    if not resolved.exists():
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def browser_input_fingerprints(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    stack_path = browser_stack_config_path(args.stack_config)
    stack_config = read_browser_stack_config(stack_path)
    fingerprints: dict[str, dict[str, object]] = {
        "case_manifest": fingerprint_file(args.cases),
        "smoke_runner": fingerprint_file(Path(__file__)),
        "smoke_cdp": fingerprint_file(DEFAULT_SMOKE_CDP),
        "browser_app": fingerprint_file(DEFAULT_BROWSER_APP),
        "browser_index": fingerprint_file(DEFAULT_BROWSER_INDEX),
        "browser_stack_config": fingerprint_file(stack_path),
    }
    detector_path = args.detector_model.strip() or str((stack_config.get("detector") or {}).get("path", "")).strip()
    if detector_path:
        fingerprints["detector_model"] = fingerprint_file(browser_asset_path(detector_path))
    fragment_path = args.fragment_classifier_model.strip() or str((stack_config.get("fragment_classifier") or {}).get("path", "")).strip()
    if fragment_path:
        fingerprints["fragment_classifier_model"] = fingerprint_file(browser_asset_path(fragment_path))
    return fingerprints


def read_cases(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def repo_or_url_path(value: str) -> Path:
    if value.startswith("/"):
        return ROOT / value.lstrip("/")
    return resolve(Path(value))


def validate_case(case: dict[str, str], index: int) -> None:
    prefix = f"case row {index + 2}"
    for field in ["case_id", "image", "labels"]:
        if not case.get(field, "").strip():
            raise ValueError(f"{prefix}: missing {field}")
    image_path = repo_or_url_path(case["image"].strip())
    if not image_path.exists():
        raise ValueError(f"{case['case_id']}: image not found: {case['image']}")
    labels_path = resolve(Path(case["labels"].strip()))
    if not labels_path.exists():
        raise ValueError(f"{case['case_id']}: labels not found: {case['labels']}")
    for field in NUMERIC_FIELDS:
        value = case.get(field, "").strip()
        if not value:
            continue
        try:
            float(value)
        except ValueError as exc:
            raise ValueError(f"{case['case_id']}: {field} must be numeric, got {value!r}") from exc


def add_optional_number(command: list[str], flag: str, value: str) -> None:
    value = value.strip()
    if value:
        command.extend([flag, value])


def parse_summary(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("smoke output did not contain a JSON summary")
    return json.loads(stdout[start : end + 1])


def browser_stress_report(args: argparse.Namespace, summaries: list[dict], failures: list[str]) -> dict[str, object]:
    return {
        "schema": "cashsnap_browser_synthetic_stress_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "case_count": len(summaries),
        "case_manifest": repo_path(resolve(args.cases)),
        "input_fingerprints": browser_input_fingerprints(args),
        "settings": {
            "timeout_ms": int(args.timeout_ms),
            "subprocess_grace_seconds": args.subprocess_grace_seconds,
            "edge": args.edge,
            "proposal_conf": args.proposal_conf,
            "detector_override": args.detector_override,
            "detector_model": repo_asset_arg(args.detector_model, "detector model") if args.detector_model else "",
            "fragment_classifier_model": repo_asset_arg(args.fragment_classifier_model, "fragment classifier model")
            if args.fragment_classifier_model
            else "",
            "stack_config": repo_asset_arg(args.stack_config, "stack config") if args.stack_config else "",
            "reject_fragment_disagreement": args.reject_fragment_disagreement,
            "fragment_disagreement_min_conf": args.fragment_disagreement_min_conf,
            "unclassified_min_conf": args.unclassified_min_conf,
            "nms_iou": args.nms_iou,
            "crop_padding": args.crop_padding,
            "jobs": args.jobs,
            "retry_failed_sequential": args.retry_failed_sequential,
        },
        "failures": failures,
        "cases": summaries,
    }


def command_for_case(case: dict[str, str], args: argparse.Namespace, index: int) -> list[str]:
    command = [
        "node",
        "scripts/smoke_browser_demo_cdp.cjs",
        "--image",
        case["image"].strip(),
        "--labels",
        case["labels"].strip(),
        "--port",
        str(args.port_base + index),
        "--debug-port",
        str(args.debug_port_base + index),
        "--timeout-ms",
        args.timeout_ms,
    ]
    add_optional_number(command, "--min-same-class", case.get("min_same_class", ""))
    add_optional_number(command, "--min-any-class", case.get("min_any_class", ""))
    add_optional_number(command, "--max-count-error", case.get("max_count_error", ""))
    add_optional_number(command, "--max-khr-error", case.get("max_khr_error", ""))
    add_optional_number(command, "--max-usd-error", case.get("max_usd_error", ""))
    if args.edge:
        command.extend(["--edge", args.edge])
    if args.proposal_conf:
        command.extend(["--proposal-conf", args.proposal_conf])
    if args.detector_override:
        command.extend(["--detector-override", args.detector_override])
    if args.detector_model:
        command.extend(["--detector-model", repo_asset_arg(args.detector_model, "detector model")])
    if args.fragment_classifier_model:
        command.extend(
            ["--fragment-classifier-model", repo_asset_arg(args.fragment_classifier_model, "fragment classifier model")]
        )
    if args.stack_config:
        command.extend(["--stack-config", repo_asset_arg(args.stack_config, "stack config")])
    if args.reject_fragment_disagreement:
        command.append("--reject-fragment-disagreement")
    if args.fragment_disagreement_min_conf:
        command.extend(["--fragment-disagreement-min-conf", args.fragment_disagreement_min_conf])
    if args.unclassified_min_conf:
        command.extend(["--unclassified-min-conf", args.unclassified_min_conf])
    if args.nms_iou:
        command.extend(["--nms-iou", args.nms_iou])
    if args.crop_padding:
        command.extend(["--crop-padding", args.crop_padding])
    if not args.no_artifacts:
        out_dir = resolve(args.out_dir)
        case_id = case["case_id"]
        command.extend(
            [
                "--screenshot",
                str(out_dir / f"{case_id}.png"),
                "--out-csv",
                str(out_dir / f"{case_id}.csv"),
                "--out-json",
                str(out_dir / f"{case_id}.json"),
            ]
        )
    return command


def run_case(index: int, case: dict[str, str], args: argparse.Namespace) -> tuple[int, dict | None, str | None, str, str]:
    case_id = case["case_id"]
    command = command_for_case(case, args, index)
    timeout_seconds = int(args.timeout_ms) / 1000.0 + max(0.0, args.subprocess_grace_seconds)
    print(
        f"{case_id}: starting browser smoke port={args.port_base + index} "
        f"debug_port={args.debug_port_base + index}",
        flush=True,
    )
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        return index, None, f"{case_id}: subprocess timeout after {timeout_seconds:.0f}s", stdout, stderr
    try:
        summary = parse_summary(result.stdout)
    except ValueError as exc:
        failure = f"{case_id}: exit {result.returncode}" if result.returncode else f"{case_id}: {exc}"
        return index, None, failure, result.stdout, result.stderr
    summary["caseId"] = case_id
    summary["notes"] = case.get("notes", "")
    summary["smokeExitCode"] = result.returncode
    if result.stderr.strip():
        summary["smokeError"] = result.stderr.strip()
    failure = f"{case_id}: exit {result.returncode}" if result.returncode else None
    return index, summary, failure, result.stdout, result.stderr


def print_case_summary(summary: dict) -> None:
    evaluation = summary.get("evaluation") or {}
    print(
        f"{summary.get('caseId')}: count={summary.get('totalCount')} "
        f"khr={summary.get('khrValue')} usd={summary.get('usdValue')} "
        f"same={evaluation.get('matchedSameClass')}/{evaluation.get('gtCount')} "
        f"any={evaluation.get('matchedAnyClass')}/{evaluation.get('gtCount')} "
        f"count_error={evaluation.get('countError')} "
        f"khr_error={evaluation.get('khrValueError')} usd_error={evaluation.get('usdValueError')}"
    )


def main() -> None:
    args = parse_args()
    for value, label in [
        (args.detector_model, "detector model"),
        (args.fragment_classifier_model, "fragment classifier model"),
        (args.stack_config, "stack config"),
    ]:
        repo_asset_arg(value, label)
    cases = read_cases(args.cases)
    if not cases:
        raise SystemExit(f"no smoke cases found in {resolve(args.cases).relative_to(ROOT)}")
    for index, case in enumerate(cases):
        validate_case(case, index)
    if args.validate_only:
        print(f"validated {len(cases)} browser smoke case(s)")
        return
    if not args.no_artifacts:
        resolve(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    failures: list[str] = []
    summaries_by_index: dict[int, dict] = {}
    if args.jobs == 1:
        results = [run_case(index, case, args) for index, case in enumerate(cases)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_case, index, case, args) for index, case in enumerate(cases)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        retry_indexes = [index for index, summary, _failure, _stdout, _stderr in results if summary is None]
        if args.retry_failed_sequential and retry_indexes:
            print(f"retrying {len(retry_indexes)} browser smoke infrastructure failure(s) sequentially")
            replacements = {index: run_case(index, cases[index], args) for index in retry_indexes}
            fixed_results = []
            for row in results:
                fixed_results.append(replacements.get(row[0], row))
            results = fixed_results

    for index, summary, failure, stdout, stderr in sorted(results, key=lambda row: row[0]):
        if summary is None:
            print(stdout, end="")
            print(stderr, end="")
            if failure:
                failures.append(failure)
            continue
        summaries_by_index[index] = summary
        print_case_summary(summary)
        if failure:
            if stderr:
                print(stderr, end="")
            failures.append(failure)

    summaries = [summaries_by_index[index] for index in sorted(summaries_by_index)]
    summary_json = args.summary_json
    if summary_json is None and not args.no_artifacts:
        summary_json = resolve(args.out_dir) / "summary.json"
    if summary_json is not None:
        summary_path = resolve(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(browser_stress_report(args, summaries, failures), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {repo_path(summary_path)}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
