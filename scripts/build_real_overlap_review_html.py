#!/usr/bin/env python
"""Build a local HTML reviewer for a real-overlap review CSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = Path("runs/cashsnap/real_overlap_focus_review_packet_v1/focus_review_packet_v1.csv")
DEFAULT_OUT_HTML = Path("runs/cashsnap/real_overlap_focus_review_packet_v1/focus_review_packet_v1_review.html")

USABLE_ROUTES = [
    "",
    "trusted_overlap_eval",
    "train_anchor_candidate",
    "hard_negative_context",
    "partial_policy_unclear",
    "unknown_or_foreign",
    "exclude_duplicate_or_flat",
    "exclude",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_OUT_HTML)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def csv_escape(value: str) -> str:
    if any(ch in value for ch in [",", "\"", "\n", "\r"]):
        return "\"" + value.replace("\"", "\"\"") + "\""
    return value


def field_list(fields: list[str]) -> list[str]:
    required = ["review_decision", "usable_as", "final_route", "review_notes"]
    output = list(fields)
    for field in required:
        if field not in output:
            output.append(field)
    return output


def default_route(row: dict[str, str]) -> str:
    return row.get("suggested_usable_as", "") or row.get("usable_as", "")


def row_to_payload(row: dict[str, str], index: int, html_dir: Path) -> dict[str, Any]:
    image = row.get("image", "")
    image_path = resolve(image)
    overlays = [value for value in row.get("model_error_top_overlays", "").split("|") if value.strip()]
    return {
        "index": index,
        "image_src": rel_between(html_dir, image_path) if image_path.exists() else image,
        "overlay_srcs": [
            rel_between(html_dir, resolve(overlay)) if resolve(overlay).exists() else overlay for overlay in overlays[:4]
        ],
        "default_route": default_route(row),
        "row": row,
    }


def render_html(*, rows: list[dict[str, str]], fields: list[str], review_csv: Path, out_html: Path) -> str:
    html_dir = out_html.parent
    payload = [row_to_payload(row, index, html_dir) for index, row in enumerate(rows)]
    fields = field_list(fields)
    decisions = ["", "accepted"]
    escaped_rows = json.dumps(payload, ensure_ascii=True)
    escaped_fields = json.dumps(fields, ensure_ascii=True)
    escaped_routes = json.dumps(USABLE_ROUTES, ensure_ascii=True)
    escaped_decisions = json.dumps(decisions, ensure_ascii=True)
    title = "CashSnap Focused Overlap Review"
    source = repo_rel(resolve(review_csv))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light; font-family: Arial, sans-serif; }}
  body {{ margin: 0; background: #f6f7f8; color: #1d2329; }}
  header {{ position: sticky; top: 0; z-index: 4; background: #fff; border-bottom: 1px solid #d7dde3; padding: 12px 16px; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  .meta {{ font-size: 13px; color: #53616f; }}
  .actions {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }}
  .help {{ background: #eef4ff; border: 1px solid #c7d8ff; border-radius: 8px; padding: 10px 12px; margin-top: 10px; font-size: 14px; line-height: 1.4; }}
  .help strong {{ color: #173a8a; }}
  button, select, textarea {{ font: inherit; }}
  button {{ border: 1px solid #9aa7b3; background: #fff; padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
  button.primary {{ background: #2459d6; color: #fff; border-color: #2459d6; }}
  main {{ padding: 14px; display: grid; gap: 14px; }}
  .card {{ background: #fff; border: 1px solid #d7dde3; border-radius: 8px; overflow: hidden; }}
  .row-head {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #e4e8ed; }}
  .title {{ font-weight: 700; }}
  .tags {{ font-size: 12px; color: #53616f; margin-top: 4px; }}
  .body {{ display: grid; grid-template-columns: minmax(260px, 420px) 1fr; gap: 12px; padding: 12px; }}
  img.main {{ width: 100%; max-height: 420px; object-fit: contain; background: #f0f2f4; border: 1px solid #d7dde3; }}
  .overlays {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-top: 8px; }}
  .overlays img {{ width: 100%; max-height: 180px; object-fit: contain; border: 1px solid #d7dde3; background: #f0f2f4; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, minmax(180px, 1fr)); gap: 10px; }}
  label {{ display: grid; gap: 4px; font-size: 12px; color: #53616f; }}
  .readonly {{ font-size: 13px; color: #1d2329; background: #f6f7f8; border: 1px solid #d7dde3; padding: 7px; border-radius: 6px; min-height: 18px; }}
  textarea {{ min-height: 70px; resize: vertical; }}
  .wide {{ grid-column: 1 / -1; }}
  .quick {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .quick button {{ font-size: 12px; padding: 5px 7px; }}
  @media (max-width: 850px) {{ .body {{ grid-template-columns: 1fr; }} .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Source: <code>{html.escape(source)}</code>. Export a CSV after choosing decisions.</div>
  <div class="help">
    <strong>Your job:</strong> decide whether each photo is safe to use. You are not drawing boxes.
    Use the row buttons: <strong>accept suggested</strong> for a real usable money scene,
    <strong>flat/duplicate</strong> for catalog/scans/front-back layouts,
    <strong>partial unclear</strong> when a human could not confidently count it,
    <strong>unknown/foreign</strong> for out-of-scope money, and <strong>exclude</strong> for junk.
    Leaving a row blank is okay; blank rows are ignored.
  </div>
  <div class="actions">
    <button class="primary" id="download">Download reviewed CSV</button>
    <button id="clear">Clear decisions</button>
    <span class="meta" id="count"></span>
  </div>
</header>
<main id="rows"></main>
<script>
const payload = {escaped_rows};
const fields = {escaped_fields};
const routes = {escaped_routes};
const decisions = {escaped_decisions};

function el(tag, attrs = {{}}, children = []) {{
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {{
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }}
  for (const child of children) node.appendChild(child);
  return node;
}}

function makeSelect(values, selected, className) {{
  const select = el("select", {{ class: className }});
  for (const value of values) {{
    const option = el("option", {{ value, text: value || "(blank)" }});
    if (value === selected) option.selected = true;
    select.appendChild(option);
  }}
  return select;
}}

function setRow(card, decision, route, noteSuffix = "") {{
  card.querySelector(".review-decision").value = decision;
  card.querySelector(".usable-as").value = route;
  card.querySelector(".final-route").value = route;
  if (noteSuffix) {{
    const notes = card.querySelector(".review-notes");
    notes.value = notes.value ? notes.value + "; " + noteSuffix : noteSuffix;
  }}
  updateCount();
}}

function clearRow(card) {{
  card.querySelector(".review-decision").value = "";
  card.querySelector(".usable-as").value = "";
  card.querySelector(".final-route").value = "";
  card.querySelector(".review-notes").value = "";
  updateCount();
}}

function render() {{
  const root = document.getElementById("rows");
  for (const item of payload) {{
    const row = item.row;
    const card = el("section", {{ class: "card", "data-index": item.index }});
    const head = el("div", {{ class: "row-head" }}, [
      el("div", {{}}, [
        el("div", {{ class: "title", text: `${{row.focus_review_id || item.index + 1}} · ${{row.focus_bucket || row.packet_bucket || ""}}` }}),
        el("div", {{ class: "tags", text: `${{row.split}} · ${{row.source_group}} · ${{row.classes}} · errors=${{row.model_error_total || "0"}}` }})
      ]),
      el("div", {{ class: "tags", text: row.suggested_usable_as ? `suggested: ${{row.suggested_usable_as}}` : "" }})
    ]);
    const imageBlock = el("div", {{}}, [
      el("img", {{ class: "main", src: item.image_src, loading: "lazy" }})
    ]);
    if (item.overlay_srcs.length) {{
      const overlayGrid = el("div", {{ class: "overlays" }});
      for (const src of item.overlay_srcs) overlayGrid.appendChild(el("img", {{ src, loading: "lazy" }}));
      imageBlock.appendChild(overlayGrid);
    }}

    const decision = makeSelect(decisions, row.review_decision || "", "review-decision");
    const usable = makeSelect(routes, row.usable_as || "", "usable-as");
    const finalRoute = makeSelect(routes, row.final_route || "", "final-route");
    const notes = el("textarea", {{ class: "review-notes" }});
    notes.value = row.review_notes || "";
    const info = el("div", {{ class: "grid" }}, [
      el("label", {{ text: "review_decision" }}, [decision]),
      el("label", {{ text: "usable_as" }}, [usable]),
      el("label", {{ text: "final_route" }}, [finalRoute]),
      el("label", {{ text: "suggested route" }}, [el("div", {{ class: "readonly", text: row.suggested_usable_as || "" }})]),
      el("label", {{ text: "focus reason", class: "wide" }}, [el("div", {{ class: "readonly", text: row.focus_reason || "" }})]),
      el("label", {{ text: "model errors", class: "wide" }}, [el("div", {{ class: "readonly", text: row.model_error_types || "" }})]),
      el("label", {{ text: "notes", class: "wide" }}, [notes]),
      el("div", {{ class: "quick wide" }}, [
        el("button", {{ type: "button", text: "accept suggested" }}),
        el("button", {{ type: "button", text: "flat/duplicate" }}),
        el("button", {{ type: "button", text: "partial unclear" }}),
        el("button", {{ type: "button", text: "unknown/foreign" }}),
        el("button", {{ type: "button", text: "exclude" }})
      ])
    ]);
    const quick = info.querySelectorAll(".quick button");
    quick[0].addEventListener("click", () => setRow(card, "accepted", item.default_route || "exclude", "accepted suggested route"));
    quick[1].addEventListener("click", () => setRow(card, "accepted", "exclude_duplicate_or_flat", "flat/catalog/duplicate"));
    quick[2].addEventListener("click", () => setRow(card, "accepted", "partial_policy_unclear", "partial policy unclear"));
    quick[3].addEventListener("click", () => setRow(card, "accepted", "unknown_or_foreign", "unknown/out-of-scope/foreign"));
    quick[4].addEventListener("click", () => setRow(card, "accepted", "exclude", "excluded"));
    for (const input of [decision, usable, finalRoute, notes]) input.addEventListener("change", updateCount);
    card.appendChild(head);
    card.appendChild(el("div", {{ class: "body" }}, [imageBlock, info]));
    root.appendChild(card);
  }}
  updateCount();
}}

function currentRows() {{
  return payload.map((item) => {{
    const card = document.querySelector(`[data-index="${{item.index}}"]`);
    return {{
      ...item.row,
      review_decision: card.querySelector(".review-decision").value,
      usable_as: card.querySelector(".usable-as").value,
      final_route: card.querySelector(".final-route").value,
      review_notes: card.querySelector(".review-notes").value
    }};
  }});
}}

function csvEscape(value) {{
  value = value == null ? "" : String(value);
  return /[",\\n\\r]/.test(value) ? `"${{value.replaceAll('"', '""')}}"` : value;
}}

function downloadCsv() {{
  const lines = [fields.map(csvEscape).join(",")];
  for (const row of currentRows()) lines.push(fields.map((field) => csvEscape(row[field] || "")).join(","));
  const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv" }});
  const link = el("a", {{ href: URL.createObjectURL(blob), download: "focus_review_packet_v1_reviewed.csv" }});
  document.body.appendChild(link);
  link.click();
  link.remove();
}}

function updateCount() {{
  const rows = currentRows();
  const accepted = rows.filter((row) => row.review_decision === "accepted").length;
  document.getElementById("count").textContent = `${{accepted}} / ${{rows.length}} rows accepted/reviewed`;
}}

document.getElementById("download").addEventListener("click", downloadCsv);
document.getElementById("clear").addEventListener("click", () => {{
  for (const card of document.querySelectorAll(".card")) clearRow(card);
}});
render();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    review_csv = resolve(args.review_csv)
    out_html = resolve(args.out_html)
    rows, fields = read_csv(review_csv)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_html(rows=rows, fields=fields, review_csv=review_csv, out_html=out_html), encoding="utf-8")
    print(f"review_html={repo_rel(out_html)} rows={len(rows)}")


if __name__ == "__main__":
    main()
