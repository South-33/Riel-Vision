# KhmerCurrencyOCR

KhmerCurrencyOCR is the research repo behind CashSnap, a lightweight computer-vision banknote counter for mixed Cambodian riel (KHR) and US dollar (USD) photos.

The goal is practical retail counting from one casual phone image: separated notes, overlapping stacks, handheld fans, partial notes, and hand/finger occlusion. Counterfeit detection and authenticity checks are intentionally out of scope.

## Current Status

This project is still research/prototype work. Clean visible-note synthetic transfer is the active milestone; dense overlap, fan layouts, and partial-note counting are not solved yet.

The current production path is:

- train a small phone/browser-deployable detector from controlled synthetic clean-note data
- prove transfer on real clean-positive, labeled-positive, and empty-frame guardrails
- mine failures into targeted synthetic counterexamples instead of blindly scaling data
- only after clean transfer is credible, move into overlap/fan/hand curricula and fragment fusion

Synthetic data is being built as a controlled experiment generator, not as a shortcut around real validation. "Perfect" synthetic data here means dense, exact-label, controllable coverage whose knobs survive real transfer checks.

## Repository Shape

The project intentionally keeps its active written memory small:

- `README.md` is the user-facing overview.
- `AGENTS.md` is the short project entry note for coding agents.
- `model.md` is the working brain for current model direction, trusted assets, known blockers, and durable results.

Most datasets, generated synthetic images, model weights, caches, and run outputs are intentionally git-ignored.

Useful folders:

- `configs/synthetic_recipes/` contains governed WebGL recipe catalogs and source/artifact policies.
- `configs/webgl_ablation/` and `configs/generated_lists/` contain reproducible experiment configs and generated train lists.
- `data/synthetic/`, `data/external_negatives/`, `runs/`, `tmp/`, and `.cache_runtime/` are local generated/runtime areas and should stay out of commits unless a manifest/config is intentionally promoted.
- `docs/research/` is reference material; `docs/archive/` is old working memory. Active model direction belongs in `model.md`.

## Synthetic Pipeline

The current WebGL pipeline can render banknote scenes through local Microsoft Edge using Three.js. It emits:

- RGB visual render
- exact flat-color ID mask
- visible-only YOLO detect labels
- OBB sidecar labels with rejection metadata
- fragment/evidence labels
- ignored-fragment metadata for below-threshold components
- per-batch `qa/summary.json`
- per-batch `recipe.json`

Trainable-candidate gates check label integrity, geometry, appearance diversity, note-condition diversity, texture/source policy, and zero-label hard-negative behavior. Renderer quality is necessary, but model-side real transfer remains the promotion authority.

## Quick Start

This repo is developed on Windows with Python and Node tooling. Prefer `pnpm` for Node work.

```powershell
python -m pip install -r requirements.txt
cd renderers\webgl
pnpm install
cd ..\..
```

For active model work, read `model.md` first. Long rendering or training jobs should use the repo headroom wrappers so the laptop remains usable.

Project Python entry points configure repo-local runtime storage through `scripts/local_runtime.py`; Ultralytics, Torch, Matplotlib, pip/Numba caches, and temp files are directed under `.cache_runtime/` for train/eval/probe runs.

## Public Data Note

Currency imagery and public datasets can have licensing, reproduction, split-leakage, and current-design caveats. This repo treats public and synthetic data as research inputs only; final quality claims need reviewed real phone captures and real held-out benchmarks.

## Project Scope

In scope:

- KHR + USD denomination detection/counting
- phone/browser-deployable model paths
- synthetic data with exact labels
- real transfer validation slices for clean, empty, near-negative, and later fan/overlap/hand stress

Out of scope:

- counterfeit detection
- authentication/security claims
- training on the real fan benchmark
- broad unreviewed data scraping as a substitute for validation
