# CashSnap Docs Map

Use this file as the first stop when the repo feels noisy. The project has many historical probes; the active path is narrower.

## Current Active Path

1. Read `docs/cashsnap-active-plan.md` for the current north star and phase order.
2. Treat Numista `in_circulation` as the clean KHR scan/metadata backbone; treat Roboflow/Commons/public phone data as review/domain-stress material until circulation scope is checked.
3. Keep scoreable real labels explicit with `manifests/real_fan_benchmark_label_quality.csv`; do not evaluate on ambiguous fragments.
4. Prioritize targeted rights-clear phone captures for `KHR_5000` portrait-plus-number overlap and `KHR_20000` thin/edge cases before more generic partial training.
5. Keep 3D/WebGL parked behind the proof gate in `docs/synthetic-strategy-evaluation.md`.

## Start Here

- `docs/model-card-cashsnap-two-stage-oldcommon.md`: current deployable diagnostic stack, known limits, and browser smoke evidence.
- `docs/cashsnap-active-plan.md`: current phase order, parked ideas, and immediate milestones.
- `docs/real-fan-capture-guide.md`: current capture and proposal-review workflow for scoreable real partial/fan evidence.
- `docs/p1-fragment-curation-runbook.md`: historical/current curation loop for old/common `KHR_5000`/`KHR_20000` collapse; useful, but not a substitute for targeted rights-clear captures.
- `docs/synthetic-strategy-evaluation.md`: current 2D/2.5D vs 3D decision memo.
- `docs/synthetic-harness-runbook.md`: scan-bank audit, 2.5D generation, QA, and detector-probe commands.
- `docs/data-prep.md`: source inventory, data rules, and generated artifact notes.

## Decision Memos

- `docs/fan-failure-analysis.md`: why the hard fan failure is mainly slice geometry/data, not a simple NMS or tiling issue.
- `docs/fragment-classifier-plan.md`: detector-plus-fragment-classifier architecture and historical classifier results.
- `docs/cashsnap-partial-banknote-synthetic-roadmap.md`: renderer-agnostic roadmap and promotion gates.
- `docs/3d-scene-composition-pipeline.md`: draft WebGL/Three.js renderer plan.
- `docs/archive/`: older baseline/model-first/compositor plans; useful for history, not the next command source.

## Data And Source References

- `docs/khr-circulation-scope.md`: KHR design/version scope.
- `docs/roboflow-cuurecy-detection-audit.md`: public Roboflow segmentation lead audit.
- `docs/data-source-leads.md`: external data leads and blocked/manual sources.
- `docs/research/README.md`: index for research handoff PDFs.
- `docs/real-fan-benchmark.md`: fixed benchmark rules.
- `docs/real-fan-capture-guide.md`: capture and proposal-review workflow.
- `docs/mobile-export.md`: ONNX/NCNN/TFLite export notes.

## Generated Artifacts

Most model runs, datasets, review packs, contact sheets, and downloaded source exports live under ignored `data/`, `runs/`, and `data/audit/` paths. Treat those as reproducible or local working artifacts unless a doc explicitly says they are the current benchmark or deployed stack.

Do not train on `data/real_fan_benchmark/`. Benchmark images can produce draft labels and review hints, but training/calibration data must come from non-benchmark captures or reviewed public-source crops.
