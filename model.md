# CashSnap Model Brain

This is the living working memory for model and synthetic-data decisions. Keep it
short, current, and decision-oriented. Old details belong in `docs/archive/`,
registries, or the folder structure itself.

Major history snapshots:
- `docs/archive/model_brain_pre_housekeeping_2026-06-08.md`
- `docs/archive/model_brain_pre_cleanup_2026-06-07.md`
- `docs/archive/model_brain_pre_compact_2026-06-07.md`
- `docs/archive/model_brain_full_history_2026-06-06.md`

## Shape Contract

Keep this shape stable unless the project itself changes:

1. Yardstick.
2. Working posture.
3. Current decision.
4. Current read.
5. Durable evidence.
6. Next bet.
7. Contracts, gates, validation, policy, and harness posture.

Do not invent a new `model.md` format every time the work gets messy. Update the
sections in place, prune stale detail, and archive old history when it stops
guiding decisions.

## Yardstick

North star: build a small phone/browser-deployable model that counts mixed USD
and Khmer riel from one casual retail photo.

Current phase: synthetic data must transfer to real clean-visible notes before
the project seriously moves into overlap/fan/hand curricula.

Hard clean-base target: `0.82-0.85` full real test mAP50-95. This is not
fantasy; a near-size real-trained control has already reached `0.819153` on full
real test.

Current reality:
- Fast fixed-step filtered185 reference: `0.075907`.
- Slow synthetic-only filtered185 reference: `0.142098`.
- Current synthetic-only fixed-step leader, target-anchor latest-design
  transplant: `0.144740`.

Distance to target is roughly `+0.68` to `+0.71` mAP50-95. Treat small gains
under `0.25` as mechanism clues, not as a trajectory toward done.

A useful next move should answer at least one of these:
- Does it reduce real positive misses at usable confidence?
- Does it reduce giant/full-frame or empty-frame false positives?
- Does it protect weak/high-value classes instead of trading them away?
- Does it reduce real-vs-synth representation/domain separability for the right
  reason?
- Does it expose a missing real validation bridge, label policy, or harness
  limitation that must be fixed before scale?

If a line of work cannot plausibly close a meaningful part of the target gap,
turn its lesson into a guardrail and move on.

## Working Posture

Be a good researcher, not a comfortable executor.

- Be bold but bounded. Prefer experiments that can fail loudly and teach the next
  direction over safe micro-tweaks that only make one proxy look tidier.
- Be willing to redo, restructure, or replace the harness when the harness is the
  thing blocking the right question.
- Research first when the path is fuzzy: read the code, docs, prior artifacts,
  papers, or web sources as needed. Preserve only conclusions that change a
  decision.
- Run a Builder/Skeptic pass before non-trivial direction changes:
  Builder says why this could create a regime change; Skeptic says why it may be
  too small, misleading, already disproven, or just proxy work.
- Ask the uncomfortable questions: What would make this fail? Has that already
  happened? Are we measuring what is easy instead of what matters? What simple
  obvious idea are we avoiding? What would collapse the most uncertainty?
- Do not chase pretty contact sheets, row-count comfort, or one-seed wins.
  Promotion is real/deploy utility under guardrails.
- Keep the laptop and repo usable. If a job is too heavy, shrink, split, or
  improve the harness instead of waiting forever.

## Current Decision

Stop treating `model.md` as an artifact index. Folder placement, archive folders,
JSON registries, generated-list locations, and `rg` should answer "where is that
file?" This file should answer:

- what we believe;
- what is blocked;
- what not to repeat;
- which experiment would collapse meaningful uncertainty next;
- which gates decide promotion.

When a script, config, or dataset is no longer active, organize the repo so that
state is visible in the folder or registry. Before moving code/configs, check
imports, CLI references, docs, and workflow callers with `rg`; update callers or
archive only when the move will not silently break active workflows.

## Current Read

- Synthetic transfer is the bottleneck. Synthetic self-eval and package QA can
  pass while real transfer fails.
- Target-anchor latest worked because it combined train-only CashSnap no-note
  pixels, real CashSnap geometry, and latest-design target assets. It is a
  direction signal, not a trainable foundation.
- The current leader misses `669/817` real-test GT boxes at `conf=0.05` and fires
  on `174/748` empty test images. It is not close.
- Detector representation gaps show synthetic-vs-real separability at early and
  late layers. Camera/image-formation statistics, context, extent, and background
  rejection are major suspects.
- Source-context replacement is the strongest representation mechanism so far,
  but label safety and inpaint scars remain blockers. Strict clean/fallback
  variants are diagnostics only; model-side transfer remained weak or blocked by
  memory.
- Reduced mosaic is a useful curriculum clue. `mosaic=0.75` improved small
  bounded-real behavior but still failed class/threshold guards. Treat it as a
  training-shortcut clue, not a finish-line path.
- Realistic near-negatives are necessary now. Existing stylized WebGL
  unknown/hard-negative roots are too easy or suppressive. Mined-real negatives
  should teach failure shape, not become leakage-prone training data.
- The refiner harness is useful, but raw learned/AI outputs are not trainable.
  Any refiner must preserve note and edge evidence by design or by hard
  recomposition until stricter gates prove relaxation safe.
- Full CashSnap empty-label frames are hard-negative diagnostics, not clean
  positive canvases. Many contain visible foreign/unknown/target-like notes.
- Current active detector scope is 13 operational classes, not all official
  USD/KHR. Run the taxonomy coverage check before class-scope claims.

## Durable Evidence

- Reverse-transfer asymmetry is the core warning: a real-trained model reads the
  synthetic blend well (`0.866141`), while the pure-synth model learns synthetic
  self-eval (`0.914675`) but transfers poorly (`0.142098`).
- Bridge-calibrated and strict-geometry packages proved that prettier proxy stats
  can train worse detectors. Domain separators and visual-gap audits are warning
  lights, not judges.
- Latest-design asset policy matters. All-manifest target-anchor sampling mixed
  old/current appearances and failed class guards.
- Poisson/contact image formation isolated a useful low-batch positive-transfer
  clue (`0.028350 -> 0.042401`), but worsened empty-frame FPs. Keep the
  compositor lesson; do not repeat small negative-dose loops around it.
- Close-up pose/scale reduced some FPs but hurt positive transfer, especially
  `KHR_10000`; do not use it as a direct replacement.
- Naive duplicated fusion, broad stat matching, dark negative row banks, and
  tiny row-dose hill-climbing have repeatedly failed matched controls.
- Multi-instance replacement and source-context branches are mechanism probes.
  Stock/catalog contexts can improve proxies while being the wrong target
  domain; phone-context and final detector audits are mandatory.
- Old overlap-stage detectors improve mined-real fan/overlap recall but
  overcount and hallucinate more. They support staged curriculum thinking, not
  overlap promotion.

## Next Bet

Work on an obligation-driven sim-to-real rebuild:

1. Start from real failure clusters: missed positives, giant/full-frame FPs,
   weak classes, rare capture modes, and mined-real stress failures.
2. Build candidate data that names which obligation it attacks.
3. Verify with representation gap, positive-error review, background-FP review,
   lightweight real transfer, and class guards before scale.
4. Prefer changes that could produce a regime shift: image formation, context,
   calibration/curriculum, realistic near-negatives, or a label-preserving
   refiner. Avoid comfort loops that only improve a proxy.

Current concrete direction:
- Diagnose and repair full-frame/extent/background hallucination alongside
  positive recall.
- Use low-dose diversified realistic near-negatives as a teacher for synthetic
  negative design, while avoiding val/test leakage.
- Keep source-context replacement and multi-instance replacement as bounded
  mechanism branches only after strict source-remnant audits pass.
- Explore refiner/editor models only through the preservation-first harness.

## Refiner Contract

Raw learned outputs are not trainable data.

Trainable refined data must pass:
- exact label and metadata preservation;
- note/detail/edge preservation, or explicit proof that relaxation is safe;
- full-size visual QA, not only contact sheets;
- composite-edge, crop/geometry, and background-realism audits;
- real-trained detector consistency;
- fixed-step `yolo26n` transfer;
- background-FP and per-class guardrails.

FastCUT/CUT and CycleGAN-Turbo are memory/diagnostic paths. SD-Turbo note-edge
locking proved the harness can preserve labels, but the current candidate is not
promoted. Prompt-based editors are allowed only as small gated smokes because
denomination details can mutate.

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

## Validation Slices

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
- Protected real fan/overlap/hand proof is still missing.

## Label And Class Policy

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
- `KHR_100` is official KHR but outside the current core-13 detector.
- `KHR_50` remains blocked for v1 operational training unless real retail/bank
  capture evidence or an explicit product requirement justifies it.
- Trainable WebGL target-note renders must pass the approved texture-asset gate.

## Organization

Preferred doc shape is one project `AGENTS.md`, this working `model.md`, and one
user-facing `README.md`.

Keep this file lean:
- No long path inventories.
- No append-only changelog.
- No stale "active" labels after a decision is rejected.
- No command dump except tiny canonical checks.

Use repository structure for status:
- Active/reused scripts should remain easy to find in `scripts/` or a clear
  domain subfolder.
- Historical probes should move to archive folders only after callers and docs
  are checked.
- Generated roots stay in ignored data/run locations and are recreated from
  configs when possible.
- New data roots must be registered or classified in the data lifecycle registry
  before training/rendering use.

## Harness Posture

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

## Canonical Checks

```powershell
rl python scripts\check_currency_taxonomy_coverage.py
rl python scripts\check_data_lifecycle_registry.py
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
rl python scripts\check_yolo_transfer_guardrails.py --help
```
