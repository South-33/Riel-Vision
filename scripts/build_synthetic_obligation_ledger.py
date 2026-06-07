"""Build a synthetic-data obligation ledger from real-transfer evidence.

The ledger is a guardrail against proxy work: it converts scattered scorecards,
error reviews, background false positives, crop-stat gaps, and manual visual QA
notes into concrete generation obligations.  A future synthetic recipe should
name which obligations it attacks before it earns a model run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POSITIVE_ERROR_REVIEWS = [
    "runs/cashsnap/bridge_core13_error_review_filtered185_vs_bridge_leader_conf005/summary.json",
]
DEFAULT_BACKGROUND_FP = [
    "runs/cashsnap/background_fp_real_empty_filtered185_vs_camera_isp_context_billae_train260_conf005_top50.json",
    "runs/cashsnap/transfer_scorecard_unknownsoftfp3_repair_vs_fixedstep_filtered185_clean_transfer.json",
]
DEFAULT_SYNTH_CROP_STATS = [
    "runs/cashsnap/crop_visual_stats_target_anchor_latest_v1_train.json",
    "runs/cashsnap/crop_visual_stats_target_anchor_realfgstyle_v1_train.json",
]
DEFAULT_REAL_CROP_STATS = "runs/cashsnap/crop_visual_stats_cashsnap_test_for_target_anchor.json"
DEFAULT_SCORECARD = "runs/cashsnap/synthetic_dataset_scorecard_agent4_probe.json"

GEOMETRY_METRICS = {"crop_width_px", "crop_height_px", "crop_area_px"}
APPEARANCE_METRICS = {
    "luma_mean",
    "luma_p05",
    "luma_p95",
    "luma_std",
    "saturation_mean",
    "saturation_std",
    "sharpness_grad_var",
}
METRIC_LIMITS = {
    "crop_width_px": 96.0,
    "crop_height_px": 96.0,
    "crop_area_px": 50000.0,
    "luma_mean": 0.035,
    "luma_p05": 0.050,
    "luma_p95": 0.050,
    "luma_std": 0.045,
    "saturation_mean": 0.035,
    "saturation_std": 0.030,
    "sharpness_grad_var": 0.003,
}
SCORECARD_CATEGORY = {
    "currency_taxonomy_scope": "taxonomy_and_assets",
    "target_condition_coverage": "condition_coverage",
    "real_anchor_and_holdout": "validation_bridge",
    "real_capture_requirements": "validation_bridge",
    "edge_case_inventory": "condition_coverage",
    "real_train_class_coverage": "rare_class_support",
    "hard_negatives": "near_negative_discrimination",
    "mixed_cross_currency_bridge": "condition_coverage",
    "visible_note_geometry_gap": "geometry_transfer",
    "diagnostic_real_utility": "validation_bridge",
    "browser_synthetic_stress": "deploy_stress",
    "browser_mined_real_stress": "deploy_stress",
    "real_utility_gate": "promotion_gate",
    "readiness_freshness": "governance",
}
SCORECARD_P0 = {
    "hard_negatives",
    "visible_note_geometry_gap",
    "diagnostic_real_utility",
    "real_utility_gate",
}
SCORECARD_P1 = {
    "currency_taxonomy_scope",
    "real_anchor_and_holdout",
    "real_capture_requirements",
    "real_train_class_coverage",
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: str | Path) -> str:
    value = resolve(path)
    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return value.resolve().as_posix()


def load_json(path: str | Path) -> dict[str, Any] | None:
    value = resolve(path)
    if not value.exists():
        return None
    return json.loads(value.read_text(encoding="utf-8"))


def slug(value: str, max_len: int = 96) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    if not cleaned:
        return f"obligation_{digest}"
    keep = max(12, max_len - len(digest) - 1)
    return f"{cleaned[:keep].strip('_')}_{digest}"


def short_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.3f}"
    return str(value)


def add_obligation(
    obligations: list[dict[str, Any]],
    seen: set[str],
    *,
    key: str,
    priority: int,
    category: str,
    title: str,
    evidence: dict[str, Any],
    required_action: str,
    sources: list[str],
) -> None:
    obligation_id = slug(key)
    if obligation_id in seen:
        return
    seen.add(obligation_id)
    obligations.append(
        {
            "id": obligation_id,
            "priority": priority,
            "category": category,
            "status": "open",
            "title": title,
            "evidence": evidence,
            "required_action": required_action,
            "sources": sources,
        }
    )


def ingest_scorecard(path: str, obligations: list[dict[str, Any]], seen: set[str]) -> list[str]:
    payload = load_json(path)
    if not payload:
        return []
    source = repo_rel(path)
    for axis in payload.get("axes", []):
        if axis.get("status") == "pass":
            continue
        name = str(axis.get("name", "unknown_axis"))
        blockers = [str(value) for value in axis.get("blockers", [])]
        priority = 0 if name in SCORECARD_P0 else 1 if name in SCORECARD_P1 else 2
        add_obligation(
            obligations,
            seen,
            key=f"scorecard {name}",
            priority=priority,
            category=SCORECARD_CATEGORY.get(name, "scorecard_blocker"),
            title=f"Clear scorecard blocker: {name}",
            evidence={
                "status": axis.get("status"),
                "summary": axis.get("summary"),
                "blocker_count": len(blockers),
                "top_blockers": blockers[:8],
            },
            required_action=str(axis.get("next_action") or "Define a generation or validation repair for this axis."),
            sources=[source],
        )
    return [source]


def ingest_positive_error_review(
    path: str,
    obligations: list[dict[str, Any]],
    seen: set[str],
    *,
    min_gt: int,
    max_recall: float,
    min_wrong_pair: int,
) -> list[str]:
    payload = load_json(path)
    if not payload:
        return []
    source = repo_rel(path)
    for split_key, summary in payload.get("summaries", {}).items():
        if not isinstance(summary, dict):
            continue
        error_types = summary.get("error_types", {})
        missed = int(error_types.get("missed_gt", 0) or 0)
        gt = int(summary.get("gt", 0) or 0)
        if gt and missed / max(1, gt) >= 0.50:
            add_obligation(
                obligations,
                seen,
                key=f"positive missed {source} {split_key}",
                priority=0,
                category="positive_recall",
                title=f"Recover broad positive recall on {split_key}",
                evidence={
                    "gt": gt,
                    "tp": summary.get("tp", 0),
                    "missed_gt": missed,
                    "miss_rate": round(missed / max(1, gt), 4),
                    "predictions": summary.get("predictions", 0),
                },
                required_action=(
                    "Mine representative missed positives and create synthetic obligations for their "
                    "camera/foreground/scale conditions before another train run."
                ),
                sources=[source],
            )
        for class_name, row in sorted((summary.get("by_class") or {}).items()):
            class_gt = int(row.get("gt", 0) or 0)
            recall = float(row.get("recall_at_iou", 0.0) or 0.0)
            if class_gt >= min_gt and recall <= max_recall:
                add_obligation(
                    obligations,
                    seen,
                    key=f"class recall {source} {split_key} {class_name}",
                    priority=0,
                    category="class_identity_recall",
                    title=f"Recover {class_name} recall on {split_key}",
                    evidence={
                        "gt": class_gt,
                        "tp": row.get("tp", 0),
                        "recall_at_iou": recall,
                        "predictions": row.get("predictions", 0),
                    },
                    required_action=(
                        "Generate or refine same-class positives that preserve denomination detail "
                        "under real phone blur/compression and target-anchor geometry."
                    ),
                    sources=[source],
                )
        for pair, count in sorted(
            (summary.get("wrong_class_pairs") or {}).items(),
            key=lambda item: int(item[1]),
            reverse=True,
        ):
            count_int = int(count)
            if count_int < min_wrong_pair:
                continue
            add_obligation(
                obligations,
                seen,
                key=f"wrong class {source} {split_key} {pair}",
                priority=1,
                category="class_confusion",
                title=f"Disambiguate {pair} on {split_key}",
                evidence={"wrong_class_pair": pair, "count": count_int},
                required_action=(
                    "Add a targeted class-contrast obligation: same geometry/background, "
                    "paired denominations, and label-preserving camera degradation."
                ),
                sources=[source],
            )
    return [source]


def ingest_background_fp(
    path: str,
    obligations: list[dict[str, Any]],
    seen: set[str],
    *,
    min_images_with_fp: int,
) -> list[str]:
    payload = load_json(path)
    if not payload:
        return []
    source = repo_rel(path)
    for index, row in enumerate(payload.get("rows", [])):
        images_with_fp = int(row.get("images_with_fp", 0) or 0)
        detections = int(row.get("detections", 0) or 0)
        if images_with_fp < min_images_with_fp and detections < min_images_with_fp:
            continue
        by_class = row.get("by_class") or {}
        top_classes = sorted(by_class.items(), key=lambda item: int(item[1]), reverse=True)[:8]
        add_obligation(
            obligations,
            seen,
            key=f"background fp row {source} {index} {row.get('model_label')} {row.get('image_root')}",
            priority=0,
            category="near_negative_discrimination",
            title=f"Reduce real empty-frame false positives for {row.get('model_label', 'model')}",
            evidence={
                "image_root": row.get("image_root"),
                "images": row.get("images"),
                "images_with_fp": images_with_fp,
                "detections": detections,
                "fp_per_image": row.get("fp_per_image"),
                "top_classes": [{"class_name": name, "count": count} for name, count in top_classes],
                "top_images": row.get("top", [])[:5],
            },
            required_action=(
                "Build realistic zero-label near-negative data from the top real empty-frame modes: "
                "foreign/unknown banknotes, receipts/cards, target-like partials, and matching surfaces."
            ),
            sources=[source],
        )
    for row in payload.get("background_fp", []):
        passed = bool(row.get("passed", False))
        if passed:
            continue
        add_obligation(
            obligations,
            seen,
            key=f"background fp scorecard {source} {row.get('split')}",
            priority=0,
            category="near_negative_discrimination",
            title=f"Fix background-FP regression on {row.get('split')} split",
            evidence={
                "baseline_detections": row.get("baseline_detections"),
                "candidate_detections": row.get("candidate_detections"),
                "detection_delta": row.get("detection_delta"),
                "images_with_fp_delta": row.get("images_with_fp_delta"),
                "top_class_deltas": row.get("top_class_deltas", [])[:8],
            },
            required_action=(
                "Any candidate that improves positive recall must also include near-negative pressure "
                "or a guardrail proving real-empty false positives do not regress."
            ),
            sources=[source],
        )
    return [source]


def family_stats(payload: dict[str, Any], family: str) -> dict[str, Any] | None:
    entry = (payload.get("by_family") or {}).get(family)
    return entry if isinstance(entry, dict) else None


def mean_metric(stats: dict[str, Any], metric: str) -> float | None:
    row = stats.get(metric)
    if isinstance(row, dict) and isinstance(row.get("mean"), (float, int)):
        return float(row["mean"])
    return None


def source_label_from_stats(path: str, payload: dict[str, Any]) -> str:
    data = str(payload.get("data", ""))
    if data:
        return Path(data).stem
    return Path(path).stem


def ingest_crop_stats(
    real_path: str,
    synth_paths: list[str],
    obligations: list[dict[str, Any]],
    seen: set[str],
) -> list[str]:
    real_payload = load_json(real_path)
    if not real_payload:
        return []
    real_family = family_stats(real_payload, "real")
    if not real_family:
        return []
    real_stats = real_family.get("crop_stats") or {}
    sources = [repo_rel(real_path)]
    for synth_path in synth_paths:
        synth_payload = load_json(synth_path)
        if not synth_payload:
            continue
        sources.append(repo_rel(synth_path))
        synth_family = family_stats(synth_payload, "synthetic")
        if not synth_family:
            continue
        synth_stats = synth_family.get("crop_stats") or {}
        label = source_label_from_stats(synth_path, synth_payload)
        metric_rows: list[dict[str, Any]] = []
        for metric in sorted(GEOMETRY_METRICS | APPEARANCE_METRICS):
            real_value = mean_metric(real_stats, metric)
            synth_value = mean_metric(synth_stats, metric)
            if real_value is None or synth_value is None:
                continue
            delta = synth_value - real_value
            limit = METRIC_LIMITS.get(metric)
            if limit is not None and abs(delta) >= limit:
                metric_rows.append(
                    {
                        "metric": metric,
                        "synthetic_mean": round(synth_value, 6),
                        "real_mean": round(real_value, 6),
                        "delta": round(delta, 6),
                        "limit": limit,
                    }
                )
        geometry = [row for row in metric_rows if row["metric"] in GEOMETRY_METRICS]
        appearance = [row for row in metric_rows if row["metric"] in APPEARANCE_METRICS]
        if geometry:
            add_obligation(
                obligations,
                seen,
                key=f"crop geometry {label}",
                priority=0,
                category="geometry_transfer",
                title=f"Repair visible crop geometry gap for {label}",
                evidence={"metric_deltas": geometry[:10]},
                required_action=(
                    "Generate target-anchor positives with real-sized visible notes before scaling; "
                    "thin-edge partials must stay a protected slice, not define clean geometry."
                ),
                sources=[repo_rel(real_path), repo_rel(synth_path)],
            )
        if appearance:
            add_obligation(
                obligations,
                seen,
                key=f"crop appearance {label}",
                priority=0,
                category="image_formation",
                title=f"Repair crop appearance/ISP gap for {label}",
                evidence={"metric_deltas": appearance[:10]},
                required_action=(
                    "Change the image-formation mechanism: coupled blur/focus, local tone, "
                    "sensor noise, compression, contact shadow, and edge blending. Do not rely "
                    "on global mean/std matching."
                ),
                sources=[repo_rel(real_path), repo_rel(synth_path)],
            )
    return sources


def ingest_visual_failures(
    visual_failures: list[str],
    obligations: list[dict[str, Any]],
    seen: set[str],
) -> list[str]:
    sources: list[str] = []
    for index, value in enumerate(visual_failures):
        parts = value.split("|", 2)
        if len(parts) == 3:
            label, source, note = parts
        elif len(parts) == 2:
            label, source = parts
            note = "Manual visual QA failure."
        else:
            label = f"visual_failure_{index}"
            source = value
            note = "Manual visual QA failure."
        sources.append(source)
        add_obligation(
            obligations,
            seen,
            key=f"visual failure {label} {source}",
            priority=0,
            category="image_formation",
            title=f"Fix visual QA failure: {label}",
            evidence={"source": source, "note": note},
            required_action=(
                "Add a full-size visual QA gate and change the compositor/refiner until the "
                "foreground, edge, contact shadow, blur, and compression are physically coherent."
            ),
            sources=[source],
        )
    return sources


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    obligations = payload["obligations"]
    lines = [
        "# Synthetic Obligation Ledger",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "This ledger ranks the real-transfer obligations that the next synthetic generator must attack.",
        "",
        "## Summary",
        "",
    ]
    for key, value in sorted(payload["summary"]["by_priority"].items(), key=lambda item: int(item[0])):
        lines.append(f"- P{key}: {value}")
    lines.append("")
    for key, value in sorted(payload["summary"]["by_category"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Open Obligations", ""])
    for row in obligations:
        lines.append(f"### P{row['priority']} {row['id']}")
        lines.append("")
        lines.append(f"- Category: `{row['category']}`")
        lines.append(f"- Title: {row['title']}")
        lines.append(f"- Required action: {row['required_action']}")
        evidence = row.get("evidence", {})
        if evidence:
            compact = json.dumps(evidence, sort_keys=True)
            if len(compact) > 900:
                compact = compact[:897] + "..."
            lines.append(f"- Evidence: `{compact}`")
        if row.get("sources"):
            lines.append("- Sources: " + ", ".join(f"`{source}`" for source in row["sources"]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    obligations: list[dict[str, Any]] = []
    seen: set[str] = set()
    sources: list[str] = []
    if args.no_default_evidence:
        if args.scorecard == DEFAULT_SCORECARD:
            args.scorecard = None
        args.positive_error_review = [
            value for value in args.positive_error_review if value not in DEFAULT_POSITIVE_ERROR_REVIEWS
        ]
        args.background_fp = [value for value in args.background_fp if value not in DEFAULT_BACKGROUND_FP]
        if args.real_crop_stats == DEFAULT_REAL_CROP_STATS:
            args.real_crop_stats = None
        args.synthetic_crop_stats = [
            value for value in args.synthetic_crop_stats if value not in DEFAULT_SYNTH_CROP_STATS
        ]
    if args.scorecard:
        sources.extend(ingest_scorecard(args.scorecard, obligations, seen))
    for path in args.positive_error_review:
        sources.extend(
            ingest_positive_error_review(
                path,
                obligations,
                seen,
                min_gt=args.min_gt,
                max_recall=args.max_recall,
                min_wrong_pair=args.min_wrong_pair,
            )
        )
    for path in args.background_fp:
        sources.extend(
            ingest_background_fp(
                path,
                obligations,
                seen,
                min_images_with_fp=args.min_images_with_fp,
            )
        )
    if args.real_crop_stats:
        sources.extend(ingest_crop_stats(args.real_crop_stats, args.synthetic_crop_stats, obligations, seen))
    sources.extend(ingest_visual_failures(args.visual_failure, obligations, seen))
    obligations.sort(key=lambda row: (row["priority"], row["category"], row["id"]))
    by_priority = Counter(str(row["priority"]) for row in obligations)
    by_category = Counter(row["category"] for row in obligations)
    return {
        "schema": "cashsnap_synthetic_obligation_ledger_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {
            "min_gt": args.min_gt,
            "max_recall": args.max_recall,
            "min_wrong_pair": args.min_wrong_pair,
            "min_images_with_fp": args.min_images_with_fp,
        },
        "sources": sorted(set(sources)),
        "summary": {
            "obligations": len(obligations),
            "by_priority": dict(sorted(by_priority.items(), key=lambda item: int(item[0]))),
            "by_category": dict(sorted(by_category.items())),
        },
        "obligations": obligations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic-data obligation ledger from real-transfer evidence.")
    parser.add_argument(
        "--no-default-evidence",
        action="store_true",
        help="Use only evidence paths supplied on this command line.",
    )
    parser.add_argument("--scorecard", default=DEFAULT_SCORECARD)
    parser.add_argument("--positive-error-review", action="append", default=list(DEFAULT_POSITIVE_ERROR_REVIEWS))
    parser.add_argument("--background-fp", action="append", default=list(DEFAULT_BACKGROUND_FP))
    parser.add_argument("--real-crop-stats", default=DEFAULT_REAL_CROP_STATS)
    parser.add_argument("--synthetic-crop-stats", action="append", default=list(DEFAULT_SYNTH_CROP_STATS))
    parser.add_argument(
        "--visual-failure",
        action="append",
        default=[],
        help="Manual visual QA failure as label|source|note. Repeatable.",
    )
    parser.add_argument("--min-gt", type=int, default=6)
    parser.add_argument("--max-recall", type=float, default=0.10)
    parser.add_argument("--min-wrong-pair", type=int, default=3)
    parser.add_argument("--min-images-with-fp", type=int, default=25)
    parser.add_argument("--json-out", default="runs/cashsnap/synthetic_obligation_ledger_latest.json")
    parser.add_argument("--md-out", default="runs/cashsnap/synthetic_obligation_ledger_latest.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args)
    json_out = resolve(args.json_out)
    md_out = resolve(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_out, payload)
    print(
        "synthetic_obligation_ledger="
        f"{repo_rel(json_out)} obligations={payload['summary']['obligations']} "
        f"p0={payload['summary']['by_priority'].get('0', 0)} "
        f"categories={len(payload['summary']['by_category'])}"
    )
    print(f"markdown={repo_rel(md_out)}")


if __name__ == "__main__":
    main()
