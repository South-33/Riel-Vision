# CashSnap Model Brain

This is the living working memory for model and synthetic-data decisions. Keep it
short, current, and decision-oriented. Old detail belongs in `docs/archive/`,
registries, or the folder structure itself.

Major history snapshots:
- `docs/archive/model_brain_pre_housekeeping_2026-06-08.md`
- `docs/archive/model_brain_pre_cleanup_2026-06-07.md`
- `docs/archive/model_brain_pre_compact_2026-06-07.md`
- `docs/archive/model_brain_full_history_2026-06-06.md`

## Shape Rule

Keep this shape simple and stable:

1. Yardstick And Posture.
2. Research Frame: Current State, Tested Ideas, Untested Ideas.
3. Promotion Gates.
4. Validation, Labels, And Scope.
5. Repo Hygiene.

Do not split the live context into separate "current read", "current bet",
"next move", and "evidence" sections. Keep those inside `Research Frame` as
current state, tested ideas, and untested ideas. If an idea cannot name the
expected effect, the guardrail, and what would kill it, leave the space open
instead of adding filler.

Do not use this file as an artifact index. Folder placement, archive folders,
JSON registries, generated-list locations, and `rg` should answer "where is that
file?" This file should answer what we believe, what is blocked, what not to
repeat without a new reason, which ideas look promising, and what gates decide
promotion.

Keep `model.md` live. Whenever direction, evidence, blockers, or candidate
ideas change, update this file in the same pass: prune stale advice, remove
achieved or rejected ideas, and rewrite the research frame instead of appending a
new mini-changelog. A stale `model.md` is a repo bug, not harmless history.

This file is context, not a command queue. A future agent should read it,
challenge it, inspect the current repo/results, and choose the best next step by
their own judgment.

## Yardstick And Posture

North star: build a small phone/browser-deployable model that counts mixed USD
and Khmer riel from one casual retail photo.

Current phase: synthetic data must transfer to real clean-visible notes before
the project seriously moves into overlap/fan/hand curricula.

Hard clean-base target: `0.82-0.85` full real test mAP50-95. This is realistic
because a near-size real-trained control has already reached `0.819153` on full
real test.

Current reality:
- Fast fixed-step filtered185 reference: `0.075907`.
- Slow synthetic-only filtered185 reference: `0.142098`.
- Current synthetic-only fixed-step leader, target-anchor latest-design
  transplant: `0.144740`.

Distance to target is roughly `+0.68` to `+0.71` mAP50-95. Small gains under
`0.25` are mechanism clues, not a trajectory toward done.

A useful candidate direction should answer at least one of these:
- Does it reduce real positive misses at usable confidence?
- Does it reduce giant/full-frame or empty-frame false positives?
- Does it protect weak/high-value classes instead of trading them away?
- Does it reduce real-vs-synth representation/domain separability for the right
  reason?
- Does it expose a missing real validation bridge, label policy, or harness
  limitation that must be fixed before scale?

Working posture:
- Be a good researcher, not a comfortable executor.
- Be bold but bounded. Prefer experiments that can fail loudly and teach the next
  direction over safe micro-tweaks that only make one proxy look tidier.
- Be willing to redo, restructure, or replace the harness when the harness blocks
  the right question.
- Research first when the path is fuzzy: read the code, docs, prior artifacts,
  papers, or web sources as needed. Preserve only conclusions that change a
  decision.
- Run a Builder/Skeptic pass before non-trivial direction changes: why could this
  create a regime change, and why might it be too small, misleading, already
  disproven, or proxy work?
- Ask the uncomfortable questions: What would make this fail? Has that already
  happened? Are we measuring what is easy instead of what matters? What simple
  obvious idea are we avoiding? What would collapse the most uncertainty?
- Do not chase pretty contact sheets, row-count comfort, or one-seed wins.
  Promotion is real/deploy utility under guardrails.

## Research Frame

Use this section as working state, not a pep talk. Keep untested ideas short and
sharp; if the next good idea is unclear, say that instead of padding the list.

### Current State

- Synthetic-to-real clean-visible transfer is the blocker. Synthetic self-eval,
  package QA, geometry gates, and visual contact sheets can all pass while real
  transfer still fails.
- Current synthetic-only fixed-step leader is the target-anchor latest-design
  transplant at `0.144740` mAP50-95. It is still roughly `+0.68` to `+0.71`
  short of the hard clean-base target.
- The leader is not close operationally: it misses `669/817` real-test GT boxes
  at `conf=0.05` and fires on `174/748` empty test images.
- Reverse-transfer asymmetry is the core warning. A real-trained model reads the
  synthetic blend well (`0.866141`), while the pure-synth model learns synthetic
  self-eval (`0.914675`) but transfers poorly (`0.142098`).
- Representation probes show synthetic-vs-real separability at early and late
  layers. Camera/image-formation statistics, context, extent, and background
  rejection are still the main suspects.
- The own-photo capture bridge is empty, so rare/high-value class claims and
  mixed USD/KHR retail scenes are still under-validated.

### Tested Ideas

- **Target-anchor latest-design transplant: useful clue, not foundation.** It is
  the current synthetic-only leader because it combines train-only CashSnap
  no-note pixels, real CashSnap geometry, and latest-design target assets. The
  result is too weak to scale directly.
- **Poisson/contact image formation: partial positive clue.** Low-batch transfer
  improved (`0.028350 -> 0.042401`), but empty-frame FPs worsened, so this needs
  background/negative pressure before it can matter.
- **Source-context and multi-instance replacement: plausible but unsafe.** They
  are the strongest representation mechanism so far, but source remnants,
  inpaint scars, label safety, and real-transfer proof still block promotion.
- **Reduced mosaic: curriculum clue only.** `mosaic=0.75` improved small
  bounded-real behavior, but class/threshold guards still failed.
- **Broad stat matching, strict geometry matching, and accepted-blend polishing:
  not enough.** These improved proxies and contact sheets but did not prove real
  transfer. Do not revisit without a specific failure mechanism.
- **Stylized dark/unknown/hard-negative rows: not a fix.** They can be too easy
  or suppressive and risk hurting positives unless tied to real FP failures and
  matched controls.
- **Overlap/fragment/two-stage detectors: not current clean-base work.** They
  improve some mined-real fan/overlap recall but overcount and hallucinate more.
  Keep them archived until clean-visible transfer is credible.
- **Refiner/editor outputs: harness lesson, not trainable data.** SD-Turbo
  note-edge locking proved label-preservation mechanics can work. FastCUT/CUT
  and CycleGAN-Turbo remain diagnostic paths only; raw learned outputs are not
  trainable without full preservation and transfer gates.

### Untested Ideas

- **Failure-led obligation set.** Build the next candidate from the current
  real-test misses and empty-frame FPs, not from renderer aesthetics. It should
  improve real positive recall at low confidence without increasing empty-frame
  FPs; otherwise the obligation design is wrong.
- **Train-side mined-real near-negative curriculum.** Use only train-side mined
  false-positive shapes to design or dose realistic negatives, with val/test
  leakage blocked. Kill it if positives drop or FPs do not improve at
  `conf=0.05`.
- **Audited source-context replacement.** Revisit source-context or
  multi-instance replacement only with strict source-remnant/inpaint-scar audits
  and hard recomposition of labels/edges. Kill it on any remnant leakage or
  background-FP regression.
- **Own-photo bridge before rare-class claims.** Capture or label the missing
  retail scenes: mixed USD+KHR stacks, hard `KHR_50000`, `KHR_5000/KHR_20000`
  thin slices, same-denomination fans, and no-note paper props. Without this,
  rare/high-value progress claims stay weak.

## Promotion Gates

A synthetic axis is credible only when it improves or preserves:
- full real val/test;
- clean-visible val/test;
- labeled-positive and geometry-stress slices;
- protected classes, especially riel and high-value notes;
- real empty-frame FP detections and images-with-FP at `conf=0.05`,
  `imgsz=416`, `batch=1`, `device=0`;
- max per-class mAP50-95 drop `<=0.05`, unless explicitly waived;
- at least one seed repeat for serious promotion, more for large claims.

Synthetic package gates are necessary filters, not promotion authority. Self-eval
preservation is not enough. For low-memory probes, use the lightweight transfer
scorecard over multiple confidence thresholds and require no recall regression
plus no FP/background regression.

Clean-base can move toward overlap/fan/hand only when synthetic-only `yolo26n`
is near the target line, clean-visible and labeled-positive test are `>=0.75`,
protected riel passes, real-empty FPs are no worse than control, and the result
survives seed repeat or a slow-promotion run.

## Validation, Labels, And Scope

Validation:
- Full real val/test includes many empty-label images; always pair aggregate AP
  with empty-frame FP probes.
- Roboflow core-13 bridge is a positive KHR/USD judge for the current detector,
  but it is stretched and lacks background pressure.
- Roboflow official21 partial bridge preserves official classes present in the
  source, including `KHR_100`, but current 13-class weights cannot evaluate it.
- Mined-real stress is a warning slice, not release proof. It currently has `17`
  ready stress images and `35` scoreable boxes with narrow class coverage.
- Own-photo capture bridge is empty. High-value gaps are hand fan,
  same-denomination fan, KHR_5000/KHR_20000 thin slices, KHR_5000 face/number
  overlap, KHR_50000 hard positives, mixed USD+KHR stacks, no-note backgrounds,
  and non-banknote paper props.

Labels and class scope:
- Visible evidence is authoritative.
- Detector labels are visible-instance AABBs.
- OBB/quadrilateral metadata is for audits and future oriented/fusion work, not
  today's direct YOLO detect label.
- Fragment/evidence labels are not physical-note counts; count fusion must map
  fragments back to parent notes.
- Human-unidentifiable slivers should be ignore/unknown, not forced
  denominations.
- Zero-label hard-negative roots must remain zero-label; do not silently turn
  foreign/unknown notes into target classes.
- Current active detector scope is 13 operational classes, not all official
  USD/KHR. Run `scripts/check_currency_taxonomy_coverage.py` before class-scope
  claims.
- `KHR_100` is official KHR but outside the current core-13 detector.
- `KHR_50` remains blocked for v1 operational training unless real retail/bank
  capture evidence or an explicit product requirement justifies it.
- Trainable WebGL target-note renders must pass the approved texture-asset gate.

## Repo Hygiene

Documentation:
- Preferred doc shape is one project `AGENTS.md`, this working `model.md`, and
  one user-facing `README.md`.
- No long path inventories, append-only changelogs, stale "active" labels, or
  command dumps here.
- When a script, config, or dataset is no longer active, make that visible in the
  folder or registry. Before moving code/configs, check imports, CLI references,
  docs, and workflow callers with `rg`.

Runtime and harness:
- Work on `master` unless the user asks for a branch.
- Prefer `rl` command prefixes in RunLong.
- Use repo-local runtime storage through `scripts/local_runtime.py`.
- Keep YOLO train/eval caches and generated outputs under repo-local ignored
  paths.
- YOLO promotion posture: train `batch=64`, `workers=0`, `device=0`,
  `cache=false`; eval `batch=64`, `workers=2`; background-FP guardrail
  `batch=1`.
- In Codex/RunLong memory pressure, use smaller diagnostic batches but label them
  clearly. Do not compare low-batch diagnostics to b64 promotion parity.
- WebGL default posture remains `--render-jobs 2 --renderer-batch-size 32
  --check-jobs 4`.
- `cache=disk` is rejected for YOLO probes because it created large `.npy` caches
  and slowed throughput.

Canonical checks:

```powershell
rl python scripts\check_currency_taxonomy_coverage.py
rl python scripts\check_data_lifecycle_registry.py
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
rl python scripts\check_yolo_transfer_guardrails.py --help
```
