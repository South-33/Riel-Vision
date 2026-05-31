from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "manifests" / "real_fan_benchmark_sources.csv"
DEFAULT_TASKS = ROOT / "manifests" / "real_fan_benchmark_label_tasks.csv"
DEFAULT_DRAFT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "drafts"
DEFAULT_OUT = ROOT / "data" / "real_fan_benchmark" / "review_index.html"
DEFAULT_FOCUS_IMAGE_ID = "real_overlap_0003_commons_shop_5k_10k_20k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local review index for CashSnap benchmark candidates.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--draft-label-dir", type=Path, default=DEFAULT_DRAFT_LABEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--focus-image-id", default=DEFAULT_FOCUS_IMAGE_ID)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    return resolve(path).relative_to(ROOT).as_posix()


def local_href(path: Path, out_dir: Path) -> str:
    relative = os.path.relpath(resolve(path), out_dir)
    return Path(relative).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_yolo_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))


def preview_path(image_id: str, image_path: Path) -> Path:
    audit = ROOT / "data" / "real_fan_benchmark" / "previews" / f"{image_id}_draft_label_audit.jpg"
    draft = ROOT / "data" / "real_fan_benchmark" / "previews" / "draft_labels" / f"{image_id}.jpg"
    if audit.exists():
        return audit
    if draft.exists():
        return draft
    return resolve(image_path)


def command_for(image_id: str) -> str:
    return (
        "rl python scripts\\promote_real_benchmark_label.py "
        f"--image-id {image_id} "
        "--confirm-reviewed --reviewed-by Venom "
        '--review-notes "Human reviewed local draft-label overlay; boxes and classes accepted."'
    )


def row_html(source: dict[str, str], task: dict[str, str], draft_dir: Path, out_dir: Path, focus_image_id: str) -> str:
    image_id = source["image_id"]
    image_path = resolve(Path(source["local_path"]))
    draft_label = draft_dir / f"{image_id}.txt"
    draft_path = repo_path(draft_label) if draft_label.exists() else None
    draft_count = count_yolo_rows(draft_label)
    status = source.get("label_status") or task.get("label_status", "")
    role = source.get("benchmark_role") or task.get("benchmark_role", "")
    preview = preview_path(image_id, image_path)
    review_ready = draft_count > 0
    is_focus = image_id == focus_image_id
    item_classes = "item focus" if is_focus else "item"
    command = command_for(image_id)
    draft_button = (
        f'<a class="button secondary" href="{html.escape(local_href(draft_label, out_dir))}">Open draft label</a>'
        if draft_path
        else ""
    )
    command_block = (
        f"""
          <div class="command">
            <div class="command-title">After you visually accept this draft, run:</div>
            <textarea readonly onclick="this.select()">{html.escape(command)}</textarea>
          </div>
        """
        if review_ready
        else ""
    )
    badge = "Ready to unblock P1" if review_ready else "Needs labels"
    return f"""
      <article class="{item_classes}">
        <a class="thumb" href="{html.escape(local_href(preview, out_dir))}">
          <img src="{html.escape(local_href(preview, out_dir))}" alt="{html.escape(image_id)} review preview" />
        </a>
        <div class="body">
          <div class="badge">{html.escape(badge)}</div>
          <h2>{html.escape(image_id)}</h2>
          <p><strong>Status:</strong> {html.escape(status)} / <strong>Draft boxes:</strong> {draft_count}</p>
          <p><strong>Priority:</strong> {html.escape(task.get("priority", ""))} / <strong>Benchmark:</strong> {html.escape(source.get("benchmark_status", ""))} / <strong>Role:</strong> {html.escape(role)}</p>
          <p><strong>Review rule:</strong> {html.escape(task.get("annotation_rule", ""))}</p>
          <p>{html.escape(task.get("notes", source.get("notes", "")))}</p>
          <div class="actions">
            <a class="button" href="{html.escape(local_href(preview, out_dir))}">Open overlay</a>
            <a class="button secondary" href="{html.escape(local_href(image_path, out_dir))}">Open original</a>
            {draft_button}
          </div>
          {command_block}
        </div>
      </article>
    """


def main() -> None:
    args = parse_args()
    sources = read_csv(args.sources)
    tasks = {row["image_id"]: row for row in read_csv(args.tasks)}
    draft_dir = resolve(args.draft_label_dir)
    out = resolve(args.out)
    out_dir = out.parent
    ordered_sources = sorted(
        sources,
        key=lambda source: (
            source["image_id"] != args.focus_image_id,
            -count_yolo_rows(draft_dir / f"{source['image_id']}.txt"),
            int(tasks.get(source["image_id"], {}).get("priority") or 999),
            source["image_id"],
        ),
    )
    items = "\n".join(
        row_html(source, tasks.get(source["image_id"], {}), draft_dir, out_dir, args.focus_image_id)
        for source in ordered_sources
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CashSnap Benchmark Review</title>
    <style>
      :root {{ font-family: "Segoe UI", system-ui, sans-serif; color: #151817; background: #f3f5f2; }}
      body {{ margin: 0; padding: 18px; }}
      main {{ width: min(1200px, 100%); margin: 0 auto; }}
      h1 {{ margin: 0 0 14px; font-size: 26px; letter-spacing: 0; }}
      .lede {{ margin: 0 0 16px; color: #47524d; max-width: 880px; }}
      .item {{ display: grid; grid-template-columns: minmax(280px, 420px) 1fr; gap: 16px; padding: 14px; margin-bottom: 14px; border: 1px solid #cfd6d1; border-radius: 8px; background: white; }}
      .item.focus {{ border-color: #151817; box-shadow: 0 1px 10px rgba(21, 24, 23, 0.12); }}
      .thumb img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #eef1ee; }}
      h2 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
      p {{ margin: 0 0 8px; color: #47524d; }}
      .badge {{ display: inline-block; margin-bottom: 8px; border-radius: 6px; background: #e7f3ea; color: #1e6332; padding: 5px 8px; font-weight: 700; font-size: 13px; }}
      .button {{ display: inline-block; border-radius: 6px; background: #151817; color: white; padding: 8px 11px; text-decoration: none; margin: 0 6px 6px 0; }}
      .button.secondary {{ background: #e7ebe8; color: #151817; }}
      .command {{ margin-top: 10px; }}
      .command-title {{ font-weight: 700; margin-bottom: 6px; }}
      textarea {{ width: 100%; min-height: 74px; box-sizing: border-box; resize: vertical; border: 1px solid #cfd6d1; border-radius: 6px; padding: 9px; font-family: Consolas, "Courier New", monospace; font-size: 13px; }}
      @media (max-width: 760px) {{ .item {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <h1>CashSnap Benchmark Review</h1>
      <p class="lede">Use this page for the quick visual check before promoting a draft real benchmark label. The first card is the current P1 unblock candidate. If the overlay boxes/classes look right, run the command shown on that card.</p>
      {items}
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
