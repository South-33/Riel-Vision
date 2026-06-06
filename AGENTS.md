This is the project's AGENTS.md

## Notes
- `model.md` is the working brain for plans, active decisions, asset rankings, command posture, and durable experiment results; update it when direction changes.
- Keep `model.md` honest: prune stale phase labels, achieved TODOs, and outdated advice during normal work instead of letting old plans accumulate.
- Preferred doc shape is one project `AGENTS.md`, one working `model.md`, and one user-facing `README.md`; fold scattered active docs back into those files or archive/delete them when they stop earning their keep.
- Work directly on `master`/mainline for this repo unless the user explicitly asks for a branch.
- Keep generated YOLO/training outputs under repo-local ignored `runs/`, not user-home fallback paths.
- Do not add new active model-planning docs under `docs/`; archive/reference material can live there, but active model memory belongs in `model.md`.
- Before claiming a currency class exists or is missing, run `scripts/check_currency_taxonomy_coverage.py`; raw Numista, active cutout bank, and YOLO schema coverage intentionally differ.
