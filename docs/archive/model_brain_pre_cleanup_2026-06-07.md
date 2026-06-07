# CashSnap Model Brain

This is the active brain for model and synthetic-data work. Keep it compact,
current, and decision-oriented. Move stale detail to `docs/archive/`.

Archives:
- Full older history: `docs/archive/model_brain_full_history_2026-06-06.md`
- Pre-compaction brain: `docs/archive/model_brain_pre_compact_2026-06-07.md`

## Mission

Build a small phone/browser-deployable model that counts mixed USD and Khmer
riel from one casual retail photo.

Clean-visible synthetic transfer comes first. Do not move seriously into
overlap/fan/hand until a synthetic-only `yolo26n` can learn normal visible notes
well enough to transfer to real photos.

Hard product requirement later: partial notes, thin edges, overlaps, fans,
stacks, and hand occlusions must count when a human can see denomination
evidence. Counterfeit/authenticity detection is out of scope.

## Target Line

Use one hard scoreboard line:

**Clean-base target: `0.82-0.85` full real test mAP50-95.**

Why: the near-same-size real-trained `p96_bg96` probe reaches `0.819153` on
full real test and `0.866141` on the 185-image synthetic blend. If synthetic
data is truly camera-real and broad enough, synthetic-only training should
approach that real-transfer scale.

Current reality:

- Slow synthetic-only filtered185 full real test: `0.142098`.
- Fast fixed-step filtered185 full real test: `0.075907`.
- Current synthetic-only fixed-step aggregate leader:
  target-anchor latest-design transplant `0.144740`; it passes the per-class
  guard versus fast filtered185.
- Old WebGL repair best full real test: `0.116104`.
- Bill-AE context fixed-step full real test: `0.093702`.
- Roboflow v10+v3 core-13 bridge fixed-step reference: val `0.223646`,
  test `0.225793`.
- Bill-AE context on the same bridge: val `0.220701`, test `0.254125`,
  with per-class guard failures. Not promoted.
- Bridgeprotect support-dose probe is rejected: it trails the bridge leader on
  Roboflow core-13 val/test and worsens CashSnap full real test.
- Bridge-calibrated renderer policies now exist and produce a real regime
  change in appearance/geometry audits, but the calibrated pack is still a
  diagnostic, not trainable proof.
- Bridge-calibrated selected-v1 is rejected as trainable evidence: fixed-step
  full real test drops filtered185 `0.075907 -> 0.048918`; Roboflow core-13
  bridge val/test drop `0.223646/0.225793 -> 0.126261/0.121071`; bridge error
  review shows zero recall on every USD class. Lesson: matching crop/box summary
  stats without enough class identity/support/diversity can make a prettier but
  weaker learner.
- Roboflow bridge-only real baseline, partial epoch-31 checkpoint after host RAM
  stopped a 50-epoch run: Roboflow bridge test `0.946364`, CashSnap val
  `0.487398`, CashSnap test `0.445459`. The bridge is learnable and useful,
  but its stretched/positive-only distribution still does not match CashSnap.
  Weak CashSnap-test classes: `KHR_50000`, `USD_5`, `KHR_20000`, then larger
  USD classes. This is a real validation bridge, not the final data domain.

Perspective: we are around `9-14` trying to reach `82-85`, not `80` trying to
reach `85`. Small `+0.01` or `+0.02` wins below `0.25` are not progress toward
done; they are only diagnostic clues. Treat them as evidence about a mechanism,
not as a ladder. Stay in step-change mode: rethink data distribution,
camera/ISP realism, target-vs-non-target currency discrimination, curriculum,
training strategy, or validation.

Small incremental polish becomes reasonable only near `0.75+` with guardrails
passing.

## Operating Posture

Before adding another small experiment, ask:

- Is this the best decision I can make, or just the easiest safe next step?
- Where are we relative to the `0.82-0.85` line?
- Is the latest gain production-relevant or only a local metric bump?
- If this gain repeated ten times, would it solve the problem? If not, what
  mechanism would?
- What would make this direction fail, and has that already shown up?
- Are we optimizing a proxy because it is easy to measure?
- What simple obvious idea have we overlooked?
- What experiment would collapse the most uncertainty?

For non-trivial direction changes, run two voices:

- Builder: why this bet could create a regime change.
- Skeptic: why this is likely still too small, misleading, or already disproven.

Final action should survive that argument. Do not wait for the user to ask
"why"; critique the path, research outside the repo when useful, and update this
file when the bet changes.

Do not be precious about the current repo shape, renderer assumptions, model
harness, or curriculum. If a stronger path needs a rewrite, a new abstraction,
a discarded axis, or a rebuilt validation bridge, do it. The goal is the finish
line, not preserving the route that got us to `0.09-0.14`.

Visual QA note: use vision deliberately. Prefer opening several clear,
full-size/simple-scene images over relying on one compressed contact sheet,
because small contact-sheet tiles hide rendering flaws and make visual reasoning
harder.

## Current Read

- Renderer mechanics and package QA are strong enough for bounded probes. They
  are not perfect, and renderer gates are not transfer proof.
- The main blocker is real transfer, not synthetic self-eval. Real validation is
  the judge; synthetic data is the engine.
- Target-anchored transplant is the first clear recent mechanism signal:
  real CashSnap no-note pixels plus real CashSnap geometry plus latest-design
  target assets beats the WebGL tune/repair track on aggregate fixed-step
  transfer. This is still only `0.144740`, so it is a direction signal, not a
  finish-line signal.
- Full CashSnap empty-label frames are hard negatives, not clean positive
  canvases. Many contain visible foreign/unknown/target-like notes. Use strict
  no-note patches for positive transplants; use full empty frames for
  background-FP and target-vs-non-target pressure.
- Latest-design texture policy matters. All-manifest target-anchor sampling
  improved aggregate but failed class guards, likely by mixing old/current
  class appearances. Keep `latest_design` as the default trainable target-note
  policy unless a specific asset experiment says otherwise.
- Naive duplicated fusion is currently a trap. A 50/50 repeated mix of
  target-anchor latest plus the best WebGL repair failed under the memory-safer
  batch-32 fixed-step probe and collapsed precision. Do not promote or repeat
  that recipe blindly.
- Target-anchor real-foreground-style v1 now exists, but is not promoted:
  it builds and validates, and crop stats move some foreground metrics toward
  real while worsening others. Model proof is currently blocked by host RAM
  before a fair fixed-step run can finish.
- Visual QA on the same real-foreground-style v1 batch found the dominant
  failure signature is still obvious: target notes look like crisp stickers on
  camera backgrounds, with hard foreground/background sharpness and contact-edge
  mismatch. First-order real-crop stats did not remove the pasted-cutout
  shortcut; do not make "retry realfgstyle" the main direction.
- The recent trajectory is not linearly headed to `0.82-0.85`. Target-anchor
  latest is the only clear recent step (`0.075907 -> 0.144740` fixed-step), and
  stat-matched selection plus naive fusion both regressed. Treat this as a
  mechanism clue, not a scalable improvement rate.
- True camera-real photorealism is a valid high-leverage bet. The warning is
  only against pretty contact sheets that do not match phone-photo pixels.
- Real "background" frames include visible out-of-scope/foreign currency-like
  notes with empty labels. Synthetic training must learn target-vs-non-target
  banknote discrimination, not just detect any readable note.
- Existing `unknown_currency_*_smoke_v1` synthetic negatives are stylized line
  props, not realistic foreign banknotes; do not treat them as sufficient for
  the real empty-label problem.
- `data/cashsnap_v1` empty-label frames are useful hard-negative diagnostics,
  not a clean positive-quality judge.
- The positive validation bridge is now
  `data/processed/roboflow_khmer_us_currency_core13_bridge_v1`: deduped
  Roboflow v10+v3, 2,333 images, all 13 operational classes covered.
- Official KHR taxonomy is broader than the active detector. The repo maps all
  NBC-listed Cambodian banknote denominations in circulation as of 2026-06-07:
  `KHR_50`, `KHR_100`, `KHR_200`, `KHR_500`, `KHR_1000`, `KHR_2000`,
  `KHR_5000`, `KHR_10000`, `KHR_15000`, `KHR_20000`, `KHR_30000`,
  `KHR_50000`, `KHR_100000`, `KHR_200000`. Current model schema covers only
  the operational subset (`KHR_500` through `KHR_50000` except rare official
  extras), plus USD except `USD_2`.
- Before class-scope claims, run `scripts/check_currency_taxonomy_coverage.py`.

## Evidence

Reverse-transfer asymmetry:

- Real-only `p96_bg96` scores `0.866141` on the 185-image synthetic blend.
- The 185-image pure-synth model scores `0.914675` on synthetic self-eval but
  only `0.142098` on full real test.
- Synthetic images are readable to a real-trained model; synthetic training
  learns shortcuts that do not transfer back to real.

Target-anchor transplant signal:

- `scripts/build_cashsnap_target_anchor_transplant.py` builds a pure-synthetic
  YOLO train root by pasting approved synthetic cutouts onto strict train-only
  CashSnap no-note background patches, using real CashSnap train label geometry
  for box scale/placement. This gives synthetic labels while anchoring
  background pixels and object geometry in the target camera domain.
- Smoke visual QA found full empty-label CashSnap frames often include
  unlabeled real/foreign/target-like notes. That would poison positive
  transplant labels, so the builder defaults to
  `data/backgrounds/cashsnap_v1_no_note_patches_strict_v1` and only `*_train`
  patches.
- All-manifest target-anchor MVP:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_mvp_puresynth_realval_v1.yaml`
  has 1,248 images and improves fixed-step full real test
  `0.075907 -> 0.119088`, but fails the per-class guard on `KHR_5000`
  (`-0.139`) and `USD_10` (`-0.058`). Diagnostic only.
- Latest-design target-anchor:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml`
  has 1,248 images, 96 per class, and exactly two latest-design assets per
  class. Fixed-step full real test improves `0.075907 -> 0.144740` and passes
  the per-class guard; worst class drop is `USD_10 -0.015`.
- Remaining weak classes in latest-design target-anchor are still severe:
  `KHR_20000=0.033`, `KHR_50000=0.040`, `USD_5=0.030`, `USD_50=0.025`.
  The foreground still looks pasted/cutout-like, so the next high-leverage
  attack is foreground camera-style transfer from real same-class crops, not
  larger row counts.
- Real-foreground-style target-anchor v1:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_realfgstyle_puresynth_realval_v1.yaml`
  applies capped same-class CashSnap train crop stats to the synthetic
  foreground. It builds 1,248 images and passes dataset validation. Against
  CashSnap real test crop means, it improves `luma_p05` delta
  `+0.080 -> +0.066`, `saturation_mean` `-0.040 -> -0.033`,
  `saturation_std` `-0.028 -> -0.025`, and sharpness slightly, but worsens
  `luma_mean` `-0.014 -> -0.025` and `luma_p95` `-0.045 -> -0.056`.
  User-supplied visual QA samples from this batch
  (`cashsnap_target_anchor_000029_usd_20.jpg` and
  `cashsnap_target_anchor_000018_khr_50000.jpg`) still show a crisp pasted-note
  signature. This downgrades realfgstyle from "plausible next v2" to
  "diagnostic only." Fixed-step attempts at b64, b32, and b16 were stopped by
  host RAM pressure, but do not spend the next serious run here unless the
  compositor/image-formation mechanism changes.
- Poisson/contact compositor smoke:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_smoke_puresynth_realval.yaml`
  is a 26-image, 2/class target-anchor smoke using `composite_policy=poisson_mixed`
  and `shadow_policy=contact` with real-crop foreground stats. Dataset check
  passes. The focused composite-edge audit shows the intended mechanism moved:
  old realfgstyle boundary/outside gradient ratio mean `1.5383` versus smoke
  `1.2093`, and edge color-step mean `0.2384 -> 0.0755`. This is visual/edge
  evidence only, not model-transfer proof; next step is larger visual QA and a
  bounded fair transfer probe only if full-size samples look physically sane.
  The Poisson path is ROI-optimized; the 26-image smoke rebuild dropped from
  about `89s` full-frame to about `35s`.
- Geometry scale smoke diagnostics: `box_scale=1.55` barely moved quad means
  (`340.7x224.5 -> 382.4x242.1`) and worsened edge ratio (`1.2093 -> 1.3230`);
  `box_scale=2.4` still only reached about `409.3x234.9`, failed visual QA on
  soft crops (`0.154 > 0.150`), and worsened edge ratio to `1.3905`. Brute
  scaling is not the geometry repair. The next geometry fix needs targeted
  close-up/pose sampling or caps/aspect policy, not a global multiplier.
- AABB aspect-repair pose smoke:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_pose_smoke_puresynth_realval.yaml`
  uses `pose_policy=aabb_aspect_repair`, `min_render_short_px=80`, and
  Poisson/contact. Dataset and visual QA pass after adding the rendered
  short-side guard. It moves the shape toward taller AABBs (`quad mean
  345.3x309.5`, area `70.1k`; visual QA `p50_short_px=166.1`) while keeping
  edge metrics better than old realfgstyle (`boundary ratio 1.2895`, color step
  `0.0670`). Treat as a useful geometry mechanism clue, not trainable proof:
  area/width are still far below real-test crop means, so the next repair needs
  close-up/pose sampling targeted to real crop-size obligations.
- Close-up pose-selection smoke:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_poseclose_smoke_puresynth_realval.yaml`
  adds `min_render_short_px=120` and `min_render_area_px=90000` to force
  larger real-geometry samples while using Poisson/contact and AABB aspect
  repair. This is the first smoke that attacks both P0 edge and geometry
  obligations without failing QA: dataset check passes, visual QA passes
  (`p50_short_px=283.1`, no tiny/small), composite edge ratio stays low
  (`1.2128`, color step `0.0863`), and quad AABB means are near real-test crop
  width/height (`500.9x464.7`). It is still smoke-only and needs full-size
  visual review plus a larger bounded candidate before any model proof.
- Close-up pose-selection 8/class probe:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_poseclose_probe_puresynth_realval.yaml`
  has 104 images, 8/class, all 26 latest-design assets, all 25 strict
  no-note train backgrounds, and 102 unique geometry sources. Dataset check
  passes and visual QA passes (`p50_short_px=270.7`, no tiny/small,
  soft fraction `0.106`). Composite-edge audit improves over old realfgstyle
  and slightly over the 26-image close-up smoke (`boundary ratio 1.1979`, color
  step `0.0784`). Quad AABB means are stable near real width/height
  (`494.1x451.4`, polygon area `132.3k`). This is the current best bounded
  image-formation+geometry probe, still not trainable proof.
- Close-up pose-selection full candidate:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_poseclose_puresynth_realval_v1.yaml`
  has 1,248 images, 96/class, all 26 latest-design assets, all 25 strict
  no-note train backgrounds, and 1,083 unique geometry sources. Build time was
  about 23.4 minutes with ROI Poisson. Dataset check passes. Visual QA passes
  (`p50_short_px=261.3`, `p05_short_px=158.18`, no tiny/small, soft fraction
  `0.100`). Composite-edge audit stays in the improved band (`boundary ratio
  1.2040`, color step `0.0771`) while preserving larger real-like crop geometry
  (`quad mean 484.2x435.8`, polygon area `127.9k`). This is the current best
  visual/edge candidate, but not promoted: first model-side probes say it is
  not a drop-in replacement for target-anchor latest.
- Poisson/contact close-up model probe:
  b64/AMP 150-step proof reused the old target-anchor leader but the candidate
  stopped under host-RAM pressure in epoch 2; b32 accidentally became a
  300-update run because batch changes the epoch-derived step count, then
  stopped around candidate step 128; b16/AMP 150-step stopped before baseline
  training while LongRun/Codex RAM was high. The completed quiet b8/AMP
  150-update A/B is a low-batch diagnostic, not promotion parity:
  target-anchor latest `0.028350` vs poseclose `0.027729`
  (`delta -0.000622`), failing aggregate and `KHR_10000`
  (`-0.0546`). Box behavior split hard: baseline precision/recall
  `0.405/0.0107`, poseclose `0.00188/0.970`. Real-empty background FPs
  improved strongly at `conf=0.05` (`val detections 88->21`, images
  `59->19`; `test detections 61->16`, images `38->13`), but the transfer
  guardrail still fails full-test transfer and notes a `USD_20` FP class
  increase. Read: Poisson/contact+close-up likely suppresses empty-frame
  confidence shortcuts but hurts or distorts positive class transfer,
  especially `KHR_10000`. Next isolate compositor from close-up pose/scale
  before trying another full replacement.
- Isolated Poisson/contact full candidate:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_puresynth_realval_v1.yaml`
  keeps the original target-anchor geometry policy (`pose_policy=current`,
  no close-up guards) while adding `poisson_mixed`, contact shadow, and
  real-crop foreground stats. It has 1,248 images, 96/class, all 26
  latest-design assets, all 25 strict no-note backgrounds, and 1,134 unique
  geometry sources. Dataset and visual QA pass (`p50_short_px=150.8`,
  `p05_short_px=83.2`, tiny `0.0008`, small `0.0393`, soft `0.1002`).
  Edge audit improves over old realfgstyle and is close to the close-up
  candidate on color step (`boundary ratio 1.2549`, color step `0.0761`,
  quad mean `365.6x245.9`, area `73.4k`).
- Isolated Poisson/contact model probe:
  quiet b8/AMP 150-update A/B against target-anchor latest passes the
  positive-transfer comparison: `0.028350 -> 0.042401`
  (`delta +0.014050`), no per-class guard failures, worst class `KHR_1000`
  at `-0.0094`, and test box precision/recall improves to `0.556/0.0496`.
  This is low-batch diagnostic evidence, not b64 promotion parity, but it
  cleanly isolates the compositor as helpful. The background-FP guard fails:
  real-empty val detections/images worsen `88/59 -> 168/100`, and test worsens
  `61/38 -> 72/48`, driven by `KHR_2000`, `USD_100`, `KHR_500`, `USD_1`, and
  `USD_10`. Read: keep Poisson/contact; add realistic near-negative/empty-frame
  pressure before any promotion attempt.
- Poisson/contact hardnegdiv8 dose:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_hardnegdiv8_puresynth_realval_v1.yaml`
  appends 8 zero-label rows from
  `data/synthetic/cashsnap_webgl_hard_negative_diversity_catalog_gate_v1`
  to the isolated Poisson/contact base. Dataset check passes (`1256` train
  images, `8` backgrounds, `1248` boxes) and hard-negative diversity gate
  passes, but the model probe rejects it. Quiet b8/AMP 150-update A/B loses
  the isolated compositor gain and falls slightly below target-anchor latest:
  `0.028350 -> 0.026388` (`delta -0.001962`), with no per-class drop over
  `0.05` but worst `KHR_10000 -0.0288`. Empty-frame FPs worsen hard
  (`val detections/images 88/59 -> 206/107`; `test 61/38 -> 106/59`), mainly
  `USD_100` and `KHR_500`. Read: the existing stylized WebGL hard-negative
  diversity rows are not the right negative repair for Poisson/contact. Do not
  mine real val/test empty frames into training; that would leak validation
  backgrounds. The next negative branch needs reviewed external/realistic
  synthetic near-negatives or a synthetic-only FP-mined pool.
- Pause/reassess after hardnegdiv8:
  recent work produced a mechanism clue, not a finish-line trajectory. Isolated
  Poisson/contact is the useful part; close-up geometry and stylized WebGL hard
  negatives are not. Repeating small local doses will not bridge the gap from
  the current low/medium transfer probes to product-grade performance. The
  single biggest blocker remains realistic target-domain negative pressure plus
  real-transfer proof, not another visual proxy or a prettier contact edge.
- Poisson/contact realbgneg25 dose:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_realbgneg25_puresynth_realval_v1.yaml`
  appends the 25 train-only no-note background patches from
  `data/backgrounds/cashsnap_v1_no_note_patches_strict_v1` to the isolated
  Poisson/contact base. Empty label files were added only for the `_train`
  background images so the trainable config has explicit zero-label rows and
  avoids val/test leakage. Dataset check passes (`1273` train images, `25`
  backgrounds, `1248` boxes). Quiet b8/AMP 150-update A/B preserves a small
  positive-transfer gain over target-anchor latest (`0.028350 -> 0.030660`,
  `delta +0.002309`) with no per-class guard failures, but it gives back most
  of the isolated compositor gain (`0.042401`) and fails empty-frame guardrails:
  val worsens `88/59 -> 190/106`, test worsens `61/38 -> 110/59`, driven
  mainly by `USD_100` and `KHR_50000`. Read: simply adding the same train
  background patches as empty rows is not enough and may amplify note-like
  false positives; the next serious negative branch needs richer, reviewed
  near-negatives or synthetic-only FP-mined negatives that match the failing
  real-empty modes without touching val/test images.
- Synthetic-only FP-mined near-negative slice:
  Existing synthetic negative pools are too easy at the real guard threshold:
  Poisson/contact b8 has `0` FPs at `conf=0.05` on
  `unknown_currency_soft_negative_smoke_v1`, `hard_negative_candidate_v1`,
  `hard_negative_diversity_catalog_gate_v1`, and `negative_batch_smoke`.
  At `conf=0.005`, unknown-soft produces mineable low-confidence detections.
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_unknownsoftfp8lowconf_puresynth_realval_v1.yaml`
  appends the 8 hardest unknown-soft rows mined from
  `runs/cashsnap/background_fp_poisson_contact_b8_unknownsoft_conf0005.json`.
  Dataset check passes (`1256` train images, `8` backgrounds, `1248` boxes).
  Quiet b8/AMP 150-update A/B passes positive transfer versus target-anchor
  latest (`0.028350 -> 0.031460`, `delta +0.003109`) with no per-class guard
  failures, but full guardrails still fail: real-empty test improves
  (`61/38 -> 57/35`), while val worsens (`88/59 -> 113/63`), mostly
  `USD_100` and `KHR_500`. Read: this is the least-bad negative dose tried so
  far and supports the big-step hypothesis direction, but current synthetic
  negative pools still do not match the real-empty failure distribution closely
  enough. A real step-change probably requires a new realistic negative bank,
  not more mining from the existing stylized roots.
- Naive repair fusion:
  `configs/webgl_ablation/cashsnap_target_anchor_latest_plus_repairmix50_puresynth_realval_v1.yaml`
  repeats 210 WebGL repair rows 6x beside the 1,248 target-anchor rows. The
  fair batch-32 150-step probe drops `0.071116 -> 0.042483` and fails class
  guards. It is a rejected diagnostic, not a recipe.

Domain gap:

- Earlier aggregate geometry/aspect reads are partly stale: before 2026-06-07,
  `scripts/audit_yolo_domain_gap.py` computed `box_aspect` from normalized YOLO
  width/height, so square renders versus non-square real photos created a
  mathematical aspect artifact. The audit now uses pixel-space aspect; recompute
  before making geometry-selection decisions. Width/height/area deltas were
  still useful, but old aspect deltas were not.
- Image/crop stats are strongly wrong: `luma_p05 +0.222`, `luma_p95 -0.146`,
  `luma_std -0.132`, `saturation_std -0.107`, `sharpness_grad_var -0.018`.
  Sharpness is also now measured after fixed-size resize in the audit tools, so
  old raw-resolution sharpness deltas should be treated as directional only.
- Domain separator on filtered185 current geometry:
  - image stats AUC `0.998`;
  - box geometry AUC `0.718`;
  - top image shortcuts: `luma_std`, `aspect`, `saturation_std`,
    `saturation_mean`, `luma_p05`.
- Read: synthetic is too visually separable, especially camera/ISP/dynamic
  range. Attack that before more row-dose tuning.

Sim-to-real research anchors:

- Tobin domain randomization and Tremblay synthetic detection: randomization can
  work when variability makes real another simulator variant, but blind chaos is
  not enough for CashSnap; randomize around real capture causes.
- Carlson camera-effects: exposure, blur, noise, color cast, compression, and
  phone artifacts are likely label-preserving levers.
- Dvornik context and Structured Domain Randomization: plausible counter/table
  context beats arbitrary pasted or rendered clutter.
- Shrivastava SimGAN: unlabeled real images could become appearance anchors
  while synthetic labels stay preserved.
- Domain separators are warning lights. Inspect what they use; if luma,
  saturation, sharpness, aspect, or background simplicity separates domains,
  the detector can learn those shortcuts too.

Taxonomy and positive bridge:

- Official current target is 21 classes: 7 USD plus 14 KHR. Current model schema
  is only 13 classes. Do not say "all KHR/USD" when referring to the active
  detector.
- Roboflow Khmer-US-currency v10 (`dataset/10`) has 2,539 images, CC BY 4.0,
  no augmentation, stretch-to-640, but only 11 classes. It includes `USD_5`,
  `KHR_20000`, `KHR_50000`, and `KHR_100`, but misses `USD_1`, `USD_20`,
  `USD_50`.
- Roboflow v3 has 1,782 images and complements v10 with `USD_1`, `USD_20`,
  `USD_50`, but misses `USD_5`, `KHR_20000`, `KHR_50000`.
- The Roboflow positive bridge is stretched to `640x640`; it is useful as a
  labeled positive transfer bridge, not as a physical aspect-ratio authority.
  Re-export/replace with letterboxed or original-resolution labels if exact
  geometry calibration becomes the decision hinge.
- `scripts/build_roboflow_khmer_us_currency_bridge.py` materializes the
  deduped v10+v3 operational bridge. It excludes `KHR_100` images for core-13
  eval instead of silently treating them as background. It also converts the
  few polygon labels to enclosing detector boxes.
- Empty/after-mapping Roboflow frames are currently skipped from the positive
  bridge. That keeps positive eval clean, but it means the bridge has no true
  background pressure. Build a separate reviewed bridge-negative diagnostic
  instead of contaminating the core-13 positive bridge with unsupported
  denomination positives.
- Current bridge summary: images by split `train=1854`, `val=238`, `test=241`;
  skipped `unsupported_image=412`, `empty_after_mapping=207`,
  `exact_duplicate_image=1369`; label formats `bbox=4117`,
  `polygon_to_bbox=37`.
- Official-scope partial bridge:
  `data/processed/roboflow_khmer_us_currency_official21_partial_bridge_v1`
  has 2,539 images and preserves `KHR_100`. Missing official classes from the
  Roboflow sources are `USD_2`, `KHR_50`, `KHR_200`, `KHR_15000`,
  `KHR_30000`, `KHR_100000`, `KHR_200000`.
- Official taxonomy gap plan:
  `runs/cashsnap/currency_taxonomy_gap_plan_official_latest.md` says 13/21
  classes are complete. `USD_2`, `KHR_100`, `KHR_200`, `KHR_15000`,
  `KHR_30000`, `KHR_100000`, and `KHR_200000` have candidate cutouts needing
  review/promotion; `KHR_50` needs status review because raw current front/back
  is not ready.
- Official texture candidates:
  `cashsnap_webgl_texture_bank_official21_current_partial_candidate_v1.json`
  covers 40 class-sides and misses only `KHR_50/front` and `KHR_50/back`.
  `cashsnap_webgl_texture_bank_official21_any_status_candidate_v1.json` covers
  all 42 class-sides but its `KHR_50` rows are out-of-circulation
  `1993-1999`; do not make that trainable without status resolution.
- `KHR_50` source gap:
  `configs/synthetic_recipes/cashsnap_official_source_gap_registry_v1.json`
  records NBC's current listing (issued `2002-08-29`) and official image URLs,
  but NBC rights are not cleared for training/rendering and practical 2026
  retail circulation is unconfirmed/likely very low. The available Numista
  candidate is out-of-circulation and watermarked; keep it blocked. Do not add
  `KHR_50` to v1 operational training unless real retail/bank capture evidence
  or an explicit product requirement justifies it.
- User-provided Numista Cambodia catalogue source index:
  `https://en.numista.com/catalogue/index.php?e=cambodia_section&r=&st=148&cat=y&im1=&im2=&ru=&ie=&no=&v=&cu=&a=&dg=&i=&b=&m=&f=&t=&t2=&w=&mt=&u=&p=1`.
  Numista remains source/intake material; individual scans still need status,
  rights, current-design, and texture-review promotion before trainable use.

Current candidates:

- Clean reference:
  `configs/webgl_ablation/cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_puresynth_realval_v1.yaml`.
  Fixed-step full real test: `0.075907`.
- Current fixed-step aggregate leader:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml`.
  Fixed-step full real test: `0.144740`; passes aggregate and per-class guard
  versus fast filtered185. It also edges the older slow filtered185 result
  (`0.142098`) but is still nowhere near the clean-base target. Promote it as a
  mechanism clue and starting point for foreground-style work, not as a solved
  synthetic package.
- Rejected target-anchor all-manifest MVP:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_mvp_puresynth_realval_v1.yaml`.
  Fixed-step full real test: `0.119088`; fails `KHR_5000` and `USD_10`
  class guards. Lesson: latest-design asset policy is trainable-critical.
- Rejected naive fusion:
  `configs/webgl_ablation/cashsnap_target_anchor_latest_plus_repairmix50_puresynth_realval_v1.yaml`.
  Batch-32 fixed-step test drops below its b32 filtered185 control
  (`0.042483` vs `0.071116`) and collapses precision. Do not repeat duplicated
  50/50 repair mixing without a new mechanism.
- Unproven foreground-style candidate:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_realfgstyle_puresynth_realval_v1.yaml`.
  Dataset check passes and appearance deltas are mixed/modestly improved.
  Transfer is not known: b64, b32, and b16 fixed-step attempts were stopped by
  host RAM pressure before a fair comparison completed. Do not promote or reject
  on partial weights.
- Strong donor `hardnegold8+topdownsupport10`: fixed-step full real test
  `0.111497`; aggregate improves but per-class and background FP guards fail.
- Best repair `unknownsoftfp3_repair_khr500_usd10_topdown`: fixed-step full
  real test `0.116104`; positive-slice aggregates beat fixed-step filtered185,
  protected riel passes, and test empty-frame FPs improve. Still fails
  `KHR_5000`, `KHR_2000`, some USD classes, and val empty-frame FPs
  (`414 -> 426` detections, `270 -> 281` images). It trails the older slow
  filtered185 baseline on every main slice. Useful clue, not promoted.
- Bill-AE context train260: crop visual gate passes and fixed-step full real
  test improves `0.075907 -> 0.093702`, but this is still under `0.10`, full val
  is only `0.0999`, and `USD_10` drops `-0.112`. Real-empty FPs are mixed:
  val worsens `414 -> 439` detections and `270 -> 288` images, while test
  improves `266 -> 225` detections and `173 -> 153` images. Useful clue, not a
  path. On the Roboflow core-13 bridge it improves test
  `0.225793 -> 0.254125` but worsens val `0.223646 -> 0.220701` and fails
  per-class guards (`KHR_5000`, `USD_50`, `USD_10`, `USD_20/USD_1` depending
  split). Diagnostic only.
- Bridge leader among existing fixed-step candidates is
  `unknownsoftfp3_repair_khr500_usd10_khr50000_topdown`: Roboflow core-13
  bridge val `0.223646 -> 0.287568`, test `0.225793 -> 0.272985`. It still
  fails per-class guards: val drops `USD_10`, `USD_100`, `KHR_5000`,
  `KHR_50000`; test drops `USD_50`, `KHR_2000`, `KHR_5000`. This is a stronger
  donor/clue than billae for positive transfer, not a promotion.
- Naive bridgeprotect support dosing from the phone-auto pool is rejected.
  `bridgeprotect_v1` improves over filtered185 on Roboflow core-13 aggregate
  (val `0.223646 -> 0.254688`, test `0.225793 -> 0.242882`) but fails
  per-class guards hard (`KHR_50000 -0.105` val, `KHR_50000 -0.146` test).
  Against the bridge leader it regresses val `0.287568 -> 0.254688` and test
  `0.272985 -> 0.242882` with large class drops (`USD_50 -0.309` val,
  `KHR_500 -0.191` test). It also worsens CashSnap full real test
  `0.109467 -> 0.096412` with `KHR_500 -0.119`. Lesson: the missing bridge
  mechanism is not "add a few same-pool support rows"; inspect class/domain
  mismatch and errors before dosing more.
- Bridge error review:
  `runs/cashsnap/bridge_core13_error_review_filtered185_vs_bridge_leader_conf005/`
  shows the bridge leader roughly doubles matched true positives versus
  filtered185, but many bridge objects are still missed at `conf=0.05`,
  `iou=0.50`. Confusions cluster around `KHR_50000/KHR_5000/KHR_500 ->
  KHR_10000` and `USD_10/USD_20/USD_50` swaps. This is a visual/geometry/class
  discrimination problem, not a support-count problem.
- Cross-dataset bridge visual gap audit:
  `scripts/audit_yolo_cross_dataset_visual_gap.py` compares arbitrary YOLO
  datasets instead of relying on the old CashSnap-vs-synthetic-only grouping.
  Older bridge-vs-synth `box_aspect` deltas such as `-0.673` were produced
  before the pixel-aspect audit fix and are contaminated. Recompute corrected
  gap reports before building another geometry selector.
- Bridge-calibrated renderer policies were added:
  `real_bridge_dynamic_range_v1`, `real_bridge_print_contrast_v1`,
  `phone_bridge_square_topdown_v1`, and `real_aspect_bridge_v1`. The first
  appearance-only probe
  `data/synthetic/cashsnap_webgl_bridge_calibrated_clean_probe_v1/` cut crop
  `luma_std` gap to `-0.020` and `luma_p05` to `+0.074`, but left boxes too
  large (`area +0.209`) and aspect wrong (`-0.647`).
- Geometry-calibrated probe v3
  `data/synthetic/cashsnap_webgl_bridge_calibrated_clean_geometry_probe_v3/`
  moved area/scale into a new regime, but its aspect evidence is stale until
  corrected pixel-aspect audits are rerun. It preserves some appearance gain
  but still overshoots luma mean/saturation. Diagnostic only; do not train it
  as if solved.
- Bridge-calibrated selected-v1 model probe:
  `runs/cashsnap/fixed_step_bridge_calibrated_selected_v1_summary.json` rejects
  the selected subset on CashSnap full real test (`0.048918` vs filtered185
  `0.075907`) and on Roboflow bridge val/test (`0.126261/0.121071` vs
  filtered185 `0.223646/0.225793`). Error review:
  `runs/cashsnap/bridge_core13_error_review_bridge_calibrated_selected_v1_conf005/`
  shows zero bridge recall for all USD classes. Do not revive this as a row dose.

## External Advice Audit 2026-06-07

Accepted after repo checks:

- Audit bugs: `audit_yolo_crop_visual_domain_gap.py` had a `limits` NameError
  in the missing-family gate path; `audit_yolo_domain_gap.py` used normalized
  YOLO `width/height` for `box_aspect`; crop/image sharpness was raw-resolution
  dependent. These are fixed; regenerate gap reports before trusting old
  aspect/sharpness conclusions.
- Roboflow bridge caveats: raw exports are `Resize to 640x640 (Stretch)`, so
  they are not geometry truth. Current positive bridge skips empty-after-mapping
  frames, so it has no background/near-negative pressure.
- Renderer risks: current visual output stacks print-tone texture adjustment,
  Three.js lighting/material response, CSS canvas filters, and alpha-blended
  gray grain. This can create an easy camera/ISP shortcut. Future appearance
  work should move toward one pixel-space ISP pass with physical-ish noise/tone
  and real reviewed backgrounds via the existing `--background-dir` path.
- Validation strategy: train a Roboflow bridge-only real baseline and a
  separate reviewed bridge-negative diagnostic. These reframe what synthetic
  must beat and expose whether the positive bridge is itself learnable.
- ID mask color round-trip was spot-checked on 80 existing v3 WebGL variants:
  expected visible-box colors were present exactly and no extra off-by-one ID
  colors appeared. Not a current blocker, but keep an exact-ID-color check in
  future renderer gates.

Qualified:

- Polygon-to-AABB and fragmented visible masks are real geometry concerns, but
  YOLO detect eval is still AABB. Use OBB/fragment metadata for audits and
  future fusion, not as a direct replacement for detect labels without a model
  change.
- Neural ISP / SimGAN / DANN / self-supervised real pretraining are regime
  candidates. They need label-preservation and metric harnesses before becoming
  trainable data, but they are more plausible step changes than more WebGL row
  dosing.

Rejected or lower priority:

- Fixed-step LR/default criticism: true of parser defaults, false for the
  actual probes, which are pinned to 50 epochs, `batch=64`, `lr0=0.01`, AMP.
- `workers=0` is an intentional laptop-memory/comparability posture, not a
  transfer explanation.
- The 8 instance-ID colors are a real ceiling for future crowded scenes, but
  not the current clean-single blocker.

Corrected audit refresh:

- After the pixel-aspect/sharpness fixes, Roboflow core-13 versus filtered185
  all-class deltas are: box `width +0.102`, `height +0.143`, `area +0.188`,
  `aspect -0.331`; crop `luma_mean +0.040`, `luma_std -0.071`,
  `luma_p05 +0.112`, `luma_p95 -0.060`, `saturation_mean +0.068`,
  `saturation_std -0.032`, `sharpness -0.004`.
- Selected-v1 still matches area (`-0.009`) and luma contrast (`luma_std
  -0.003`, `luma_p05 -0.015`) better, but keeps aspect debt (`-0.317`),
  overshoots brightness/saturation, has low USD support (`USD_1=6`,
  `USD_20=7`, `USD_50=7`), and fails transfer badly. Do not optimize summary
  visual distance alone.
- Bridge-only baseline result: `runs/cashsnap/roboflow_core13_realonly_yolo26n_e50_i416_b64_w0_auto_lr1e2_warmup3_amp_cachefalse_seed0/`
  stopped at epoch 31 from RAM pressure but saved `best.pt`; evals are
  `runs/cashsnap/bridge_core13_roboflow_realonly_epoch31_best_test_i416_metrics.json`,
  `runs/cashsnap/cashsnap_val_roboflow_core13_realonly_epoch31_best_i416_metrics.json`,
  and `runs/cashsnap/cashsnap_test_roboflow_core13_realonly_epoch31_best_i416_metrics.json`.
  Do not rerun blindly; if continuing it, use a memory-safer resume/no-val plan.
- Background-FP review pack
  `runs/cashsnap/background_fp_review_filtered185_vs_camera_isp_context_billae_train260_conf005/`
  shows top empty-label false positives are full non-target banknotes (Thai,
  Chinese, Korean, deferred USD_2-like), often on the same plain paper surface
  as target positives. This is now a primary synthetic-data gap.
- Low-LR finetune from donor on repair is rejected: it clears the soft synthetic
  negative root but trails donor real test and worsens real-empty test FPs.

## Multi-Agent STOP Audit 2026-06-07

Progress-rate verdict:

- The current rate is not a finish-line trajectory. Best pure-synth fixed-step
  progress moved from `0.075907` to `0.144740`, while selected-v1 stat matching
  and 50/50 repair fusion both regressed. Ten more convenient `+0.01/+0.02`
  local wins would still leave the project far below `0.82-0.85`, and the large
  target-anchor jump is unlikely to repeat by scaling the same distribution.
- The actual blocker is target-domain image formation plus real-error closure:
  foreground paper/camera pixels, pasted-edge/contact/focus consistency,
  realistic target-vs-non-target banknote pressure, rare-class/capture gaps, and
  validation obligations tied to real failures.
- The next idea must survive this attack: if it only improves a summary stat,
  only makes contact sheets prettier, only adds rows, or only fixes one visible
  artifact without proving real failure clusters improve, it is proxy work.

Advisor claims accepted after repo checks:

- Shortcut learning / simulator signature is the right diagnosis. Reverse
  transfer is asymmetric, and the domain separator reached image-stat AUC
  `0.998`; YOLO can learn those shortcuts too.
- Target-anchor helped because real backgrounds and real geometry removed some
  easy cheats, but the foreground still has a pasted/composited image-formation
  signature. The latest realfgstyle screenshots support this directly.
- Global stat matching is already disproven as a promotion path. Selected-v1
  looked better on summary audits but dropped real transfer and showed zero USD
  bridge recall.
- Naive fusion is already disproven as a promotion path. The target-anchor plus
  repair 50/50 mix dropped `0.071116 -> 0.042483` under the fair batch-32 probe.
- Realistic near-negatives matter now, not only later. Background-FP review and
  the synthetic scorecard show full foreign/unknown banknotes in empty-label
  frames; stylized line-prop negatives are not enough.
- More WebGL knobs, row-dose tuning, or target-anchor scale are not credible
  next moves while real transfer is under `0.25`.

Advisor claims qualified:

- "Foreground is the main problem" is directionally right for the clean-positive
  transfer gap, but not sufficient. The scorecard still blocks taxonomy,
  capture, hard-negative, geometry, and diagnostic-real-utility axes.
- Physical ISP, Poisson/contact-shadow blending, and local histogram/frequency
  matching are plausible bounded experiments, but only if judged by real-error
  transfer and visual QA, not by aggregate luma/saturation deltas alone.
- SimGAN/CUT/CycleGAN-lite is a plausible regime bet, not a default answer. It
  needs label-preservation gates: class-detail retention, box/mask consistency,
  low hallucination, real-trained model consistency, and real transfer proof.
- DANN/domain-adversarial training is mathematically plausible but belongs to a
  model-strategy branch because it modifies the training loop. It should not be
  confused with making the synthetic data itself perfect.
- Few-shot real injection is the fastest likely score lift and a useful teacher
  or ceiling check, but it is not synthetic-only proof. Use it deliberately if
  the product goal outranks the synthetic-only constraint.

Advisor claims rejected or deprioritized:

- Hard negatives are not a `0.75+`-only problem. They cannot fix positive recall
  by themselves, but false target-banknote behavior is already a primary gap.
- ControlNet/Stable Diffusion refinement is too risky as a first move because
  denomination text and security details can mutate. Consider only after simpler
  label-preserving refiners/ISP experiments have clear gates.
- Domain-separator AUC reduction alone is not success. Selected-v1 proved a
  prettier/closer domain audit can still train a worse detector.
- Do not claim the repo has the right labels/classes/scenes fully covered:
  the scorecard still blocks 14 axes and the current detector is 13/21 official
  classes.

Deferred neural/AI refiner plan:

- Keep this as a serious future regime bet, not a casual image-generation
  shortcut. The right first version is a label-preserving SimGAN/CUT/CycleGAN-lite
  refiner or a mask-aware neural ISP/refiner trained on train-only real
  CashSnap imagery. Diffusion/ControlNet is higher mutation risk and should be
  later unless denoise is low and label-preservation gates are hard.
- Corrected model ranking after advisor/source check: the best future learned
  refiner bet is **CycleGAN-Turbo/img2img-turbo** on top of the
  mask-preserving Poisson/contact/ISP compositor. It is closer to our problem
  than prompt-based image generation because it is built for paired/unpaired
  image-to-image translation and retaining input structure. **FastCUT/CUT** is
  the lower-VRAM backup. **SD1.5 ControlNet/img2img/inpaint** is only a quick
  low-denoise baseline. **FLUX.2 Klein 4B** is a side experiment because its
  model card says about `13GB` VRAM, above this laptop's 8GB. **Qwen-Image-Edit**
  is not a first path because it is built on a `20B` editor with strong text
  editing ability, exactly the kind of mutation risk currency labels cannot
  tolerate.
- Verified source anchors to remember: `https://github.com/GaParmar/img2img-turbo`,
  `https://github.com/taesungp/contrastive-unpaired-translation`,
  `https://huggingface.co/black-forest-labs/FLUX.2-klein-4B`,
  `https://huggingface.co/Qwen/Qwen-Image-Edit`,
  `https://bfl.ai/flux-1-tools/`, and Carlson ECCVW 2018 camera-effects
  paper. BFL marks FLUX.1 Canny/Depth deprecated; Carlson supports
  chromatic-aberration, blur, exposure, Poisson-Gaussian noise, and color-cast
  style camera effects as synthetic-to-real object-detection levers.
- Preserve labels by design: boxes/quad geometry remain unchanged; the refiner
  must not rewrite denomination text, portraits, numerals, color identity, or
  visible class evidence. Prefer mask-aware losses: strong self-regularization
  on note interiors, stronger freedom only on contact edges, shadows,
  background interaction, tone, noise, compression, and paper/camera texture.
- Use train-only real images/crops for the target domain. Do not train any
  refiner, discriminator, or prompt-tuning loop on CashSnap val/test. Roboflow
  bridge data can be a positive-domain aid, but it is stretched and not geometry
  truth.
- Required gates before a refined dataset is trainable: exact label/metadata
  preservation, full-size visual QA, composite-edge audit, crop/geometry audit,
  real-trained detector consistency on synthetic labels, class-detail spot
  review for weak classes, domain separator as a warning light, fixed-step
  `yolo26n` transfer, background-FP guardrail, per-class guard, and at least one
  seed repeat before promotion.
- Failure modes to actively attack: hallucinated or erased denomination
  evidence, class-color drift, over-smoothed engraving texture, synthetic
  sticker edges surviving under prettier tone, discriminator overfitting to
  backgrounds, and proxy success where AUC/visual stats improve but real AP
  does not.
- Hardware/harness posture for this laptop: design for RTX 4060 Laptop 8GB VRAM
  and about 16GB RAM. Start with crop/ROI refiners at `256-384px`, AMP/fp16,
  tiny batches, gradient accumulation if needed, repo-local caches/checkpoints,
  resumable jobs, and `run_with_headroom.py`-style caps. Avoid full-resolution
  diffusion sweeps until the smoke harness proves memory and label safety. For
  CycleGAN-Turbo/img2img-turbo, start with batch `1`, fp16/AMP, gradient
  checkpointing/offload only if needed, and explicit resume/checkpoint cadence;
  for CUT/FastCUT, use it as the first low-VRAM training sanity check.
- Staging: 20-50 image smoke with visual/label gates; 200-500 image probe with
  edge/crop/domain audits; only then a 1k+ trainable candidate and fixed-step
  `yolo26n` probe. Every stage must name the obligation IDs it attacks from
  `runs/cashsnap/synthetic_obligation_ledger_latest.json`.
- Concrete future probes: A) physical compositor only, which is already started
  with Poisson/contact smoke; B) CUT/FastCUT refiner trained synthetic-to-real
  as the low-VRAM learned translator; C) CycleGAN-Turbo/img2img-turbo
  `CashSnap-Refiner-v1` once label-preservation gates and hardware smoke pass.

Agent-by-agent verdict:

- Agent 1 was mostly right on mechanism: shortcut learning, pasted-edge
  signature, weak realistic negatives, and physical camera/ISP gap are supported
  by reverse transfer, background-FP review, and the realfgstyle screenshots.
  Keep Poisson/contact shadow and realistic foreign-currency curriculum as
  bounded bets. Qualify the exact implementation claims and any speed claims;
  prove with transfer, not visual appeal.
- Agent 2 was right that target-anchor worked by removing background/geometry
  shortcuts and that stat matching is a trap. Physical ISP plus foreign
  banknote negatives is a strong data-centric pair. DANN is plausible but is a
  separate model-training branch, not proof of a perfect synthetic pipeline.
- Agent 3 was right to deprioritize WebGL knob tuning and to raise SimGAN/CUT
  as a plausible regime-change refiner. However, "hard negatives matter only at
  `0.75+`" is rejected, and "we have the right labels/classes/scenes" is too
  strong given the scorecard and 13/21 official-class scope. Treat learned
  refinement as high-upside but gated by label preservation.

## Current Bet

The current high-level bet is **real-error-conditioned image-formation
curriculum**.

Target-anchor latest remains the seed because it is the strongest recent
mechanism clue, but the next move is not "make the same transplant batch a bit
prettier." Realfgstyle v1 showed that first-order crop stats still leave a
sticker-like foreground. The generator must be driven by obligations mined from
real failures and validation blockers: positive misses/confusions, real
empty-frame false positives, rare-class weakness, visible geometry gap,
foreground/background edge-contact/focus mismatch, and capture/taxonomy holes.

Builder case: this attacks the bottleneck directly. It turns each real failure
cluster into a generation obligation and promotes synthetic data only when that
specific real cluster improves without breaking protected classes or background
FPs.

Skeptic case: this can become paperwork, or a new proxy loop, if obligations
are not tied to held-out real improvements. Poisson blending and ISP can still
make fake stickers; SimGAN can hallucinate class details; hard negatives can
improve precision while recall stays broken. Therefore every candidate must
ship with visual QA, real-transfer guardrails, background-FP checks, and at
least one concrete real-failure cluster it is expected to fix.

Near-term bold-but-bounded ideas:

- Build a real-error obligation ledger from existing artifacts:
  positive-error reviews, background-FP reviews, transfer scorecards, real
  geometry/crop audits, and the synthetic dataset scorecard. Output generation
  obligations, not just observations. Current ledger:
  `runs/cashsnap/synthetic_obligation_ledger_latest.json` has `74` open
  obligations, `56` at P0, after fixing long-ID dedupe and excluding the
  rejected selected-v1 error review from default obligation scope.
- Add a target-anchor visual QA/edge-contact/focus audit for full-size samples.
  Contact sheets are insufficient. The latest failing examples are
  `cashsnap_target_anchor_000029_usd_20.jpg` and
  `cashsnap_target_anchor_000018_khr_50000.jpg` from realfgstyle v1.
- Build a bounded compositor ablation that changes the actual failure mechanism:
  Poisson or gradient-domain blending, local contact shadows, coupled
  foreground/background blur, physical-ish Poisson-Gaussian noise, JPEG/ISP, and
  edge alpha treatment. Judge it against target-anchor v1 with real transfer and
  hard-negative guardrails, not crop stats alone. First smoke exists with
  `poisson_mixed` plus `contact`; edge metrics improved, but visual and model
  proof are still open.
- Build a governed realistic near-negative bank: foreign/unknown currencies,
  target-lookalike partial notes, receipts, cards, patterned paper, and retail
  clutter composited/rendered through the same camera-domain policy.
- Try SimGAN/CUT-style refinement only as a bounded label-preserving experiment
  with explicit class/detail/box consistency gates. Do not start with diffusion
  unless mutation risk is controlled.
- Keep DANN and small real-data injection as separate model/teacher strategies.
  They may accelerate the product path, but they do not prove the synthetic
  data pipeline is "perfect/done."
- Keep domain separators as warning lights, not judges. If real-vs-synth image
  stats remain nearly separable, expect the detector to learn the shortcut too;
  if AUC falls but real AP does not improve, the proxy failed.
- Larger calibrated synth is allowed only after transfer scorecards say the
  distribution is less wrong. Scaling target-anchor latest or realfgstyle v1
  as-is is not enough.
- In parallel, start an official-taxonomy asset/model expansion plan for
  denominations that matter in practice. `KHR_100` is not wrong, it is just
  outside the current 13-class detector. `KHR_50` is official-listed but
  operationally unconfirmed/low-frequency; keep it parked for v1 unless real
  use evidence appears.

## Promotion Rules

A synthetic axis is credible only with:

- full real val/test improvement or preservation;
- clean-visible val/test preservation;
- labeled-positive and geometry-stress preservation;
- protected-class preservation, especially riel;
- real empty-frame FP detections and images-with-FP not worse at `conf=0.05`,
  `imgsz=416`, `batch=1`, `device=0`;
- max per-class mAP50-95 drop `<=0.05` unless explicitly waived;
- at least one seed repeat for serious promotion, more for large claims.

Synthetic package gates are necessary filters, not promotion authority.

Clean-base can move toward overlap/fan/hand only when synthetic-only `yolo26n`
is near the target line, clean-visible and labeled-positive test are `>=0.75`,
protected riel passes, real-empty FPs are no worse than control, and the result
survives at least two seeds or a slow-promotion run.

## Validation Slices

- Full real val/test includes many empty-label images; always pair aggregate AP
  with empty-frame FP probes.
- Roboflow core-13 bridge is the cleaner positive KHR/USD judge for the current
  detector. Pair it with CashSnap empty-frame FP diagnostics; neither replaces
  the other.
- Roboflow official21 partial bridge preserves official classes present in the
  source, including `KHR_100`, but is not evaluable with current 13-class
  weights.
- `runs/cashsnap/real_geometry_stress_slices_v1` is the current provisional
  real geometry slice set.
- `labeled_all`: positive-only val/test (`988/814` images).
- `geometry_stress`: broad non-clean labeled val/test (`791/659` images).
- True `multi_note` is only `2/3` images and is not fan/overlap proof.
- Protected-riel slices are small; use scoped checks and do not over-read one
  noisy class row.

## Active Assets

Key configs:

- Filtered clean reference:
  `configs/webgl_ablation/cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_puresynth_realval_v1.yaml`
- Previous transfer-proven target-anchor leader:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml`
- Poisson/contact close-up diagnostic candidate:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_poseclose_puresynth_realval_v1.yaml`
- Isolated Poisson/contact diagnostic candidate:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_puresynth_realval_v1.yaml`
- Rejected Poisson/contact plus hardnegdiv8 diagnostic:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_hardnegdiv8_puresynth_realval_v1.yaml`
- Rejected Poisson/contact plus train-background negative diagnostic:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_realbgneg25_puresynth_realval_v1.yaml`
- Mixed Poisson/contact plus low-confidence unknown-soft FP-mined diagnostic:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_unknownsoftfp8lowconf_puresynth_realval_v1.yaml`
- Unproven target-anchor foreground-style candidate:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_realfgstyle_puresynth_realval_v1.yaml`
- Poisson/contact target-anchor smoke:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_smoke_puresynth_realval.yaml`
- Rejected/weak geometry-scale target-anchor diagnostics:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_geoscale_smoke_puresynth_realval.yaml`
  and
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_geoscale24_smoke_puresynth_realval.yaml`
- Pose-repair target-anchor smoke:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_pose_smoke_puresynth_realval.yaml`
- Close-up pose-selection target-anchor smoke:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_poseclose_smoke_puresynth_realval.yaml`
- Close-up pose-selection target-anchor 8/class probe:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_poseclose_probe_puresynth_realval.yaml`
- Rejected all-manifest target-anchor diagnostic:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_mvp_puresynth_realval_v1.yaml`
- Rejected duplicated repair fusion diagnostic:
  `configs/webgl_ablation/cashsnap_target_anchor_latest_plus_repairmix50_puresynth_realval_v1.yaml`
- Topdownsupport donor:
  `configs/webgl_ablation/cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_hardnegold8_topdownsupport10_puresynth_realval_v1.yaml`
- Rejected repair:
  `configs/webgl_ablation/cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_hardnegold8_topdownsupport10_unknownsoftfp3_repair_khr500_usd10_topdown_puresynth_realval_v1.yaml`
- Rejected bridgeprotect support dose:
  `configs/webgl_ablation/cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_hardnegold8_topdownsupport10_unknownsoftfp3_repair_bridgeprotect_v1_puresynth_realval.yaml`
- Recipe catalog:
  `configs/synthetic_recipes/cashsnap_webgl_recipe_catalog_v1.json`
- Synthetic governance:
  `configs/synthetic_recipes/cashsnap_synthetic_governance_v1.json`
- Data lifecycle registry:
  `configs/synthetic_recipes/cashsnap_data_lifecycle_registry_v1.json`
- Approved texture bank:
  `configs/synthetic_recipes/cashsnap_webgl_approved_texture_bank_v1.json`
- Official texture-bank candidates:
  `configs/synthetic_recipes/cashsnap_webgl_texture_bank_official21_current_partial_candidate_v1.json`
  and
  `configs/synthetic_recipes/cashsnap_webgl_texture_bank_official21_any_status_candidate_v1.json`
- Official source-gap registry:
  `configs/synthetic_recipes/cashsnap_official_source_gap_registry_v1.json`
- External negative registry:
  `configs/synthetic_recipes/cashsnap_external_negative_banks_v1.json`
- Bridge-calibrated clean diagnostic recipe:
  `webgl_bridge_calibrated_clean_v1` in
  `configs/synthetic_recipes/cashsnap_webgl_recipe_catalog_v1.json`

Key roots:

- Active cutout bank:
  `data/asset_candidates/numista_current_cutout_bank_v1/`
- Official-scope candidate cutout banks:
  `data/asset_candidates/numista_current_fullscope_cutout_bank_probe_v1/` and
  `data/asset_candidates/numista_official_fullscope_any_status_cutout_bank_probe_v1/`
- Roboflow v10+v3 core-13 positive bridge:
  `data/processed/roboflow_khmer_us_currency_core13_bridge_v1/`
- Roboflow official21 partial positive bridge:
  `data/processed/roboflow_khmer_us_currency_official21_partial_bridge_v1/`
- Official taxonomy gap plan:
  `runs/cashsnap/currency_taxonomy_gap_plan_official_latest.md`
- Raw Roboflow intake roots:
  `data/raw_datasets/roboflow_khmer_us_currency_v10/` and
  `data/raw_datasets/roboflow_khmer_us_currency/`
- Hard unknown-currency diagnostic root:
  `data/synthetic/cashsnap_webgl_unknown_currency_negative_smoke_v1/`
- Soft unknown-currency diagnostic root:
  `data/synthetic/cashsnap_webgl_unknown_currency_soft_negative_smoke_v1/`
- Target-anchor latest train root:
  `data/synthetic/cashsnap_target_anchor_transplant_latest_v1/`
- Target-anchor real-foreground-style train root:
  `data/synthetic/cashsnap_target_anchor_transplant_realfgstyle_v1/`
- Target-anchor Poisson/contact smoke root:
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_smoke_v1/`
- Target-anchor Poisson/contact geometry-scale diagnostic roots:
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_geoscale_smoke_v1/`
  and
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_geoscale24_smoke_v1/`
- Target-anchor Poisson/contact pose-repair smoke root:
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_pose_smoke_v1/`
- Target-anchor Poisson/contact close-up pose-selection smoke root:
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_poseclose_smoke_v1/`
- Target-anchor Poisson/contact close-up pose-selection 8/class probe root:
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_poseclose_probe_v1/`
- Target-anchor Poisson/contact close-up pose-selection full candidate root:
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_poseclose_v1/`
- Target-anchor isolated Poisson/contact full candidate root:
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_v1/`
- WebGL hard-negative diversity root used by rejected hardnegdiv8 dose:
  `data/synthetic/cashsnap_webgl_hard_negative_diversity_catalog_gate_v1/`
- Target-anchor all-manifest diagnostic root:
  `data/synthetic/cashsnap_target_anchor_transplant_mvp_v1/`
- Strict no-note positive background patches:
  `data/backgrounds/cashsnap_v1_no_note_patches_strict_v1/`
- Latest synthetic dataset scorecard probe:
  `runs/cashsnap/synthetic_dataset_scorecard_agent4_probe.json`
- Real-error synthetic obligation ledger:
  `runs/cashsnap/synthetic_obligation_ledger_latest.json` and
  `runs/cashsnap/synthetic_obligation_ledger_latest.md`
- Poisson/contact close-up b8 diagnostic transfer artifacts:
  `runs/cashsnap/fixed_step_target_anchor_latest_vs_poisson_contact_poseclose_b8_e50_s150_seed0_summary.json`,
  `runs/cashsnap/background_fp_target_anchor_latest_vs_poseclose_b8_conf005.json`,
  and `runs/cashsnap/transfer_guardrails_target_anchor_latest_vs_poseclose_b8.json`
- Isolated Poisson/contact b8 diagnostic transfer artifacts:
  `runs/cashsnap/fixed_step_target_anchor_latest_vs_poisson_contact_b8_e50_s150_seed0_summary.json`,
  `runs/cashsnap/background_fp_target_anchor_latest_vs_poisson_contact_b8_conf005.json`,
  and `runs/cashsnap/transfer_guardrails_target_anchor_latest_vs_poisson_contact_b8.json`
- Rejected Poisson/contact hardnegdiv8 b8 diagnostic transfer artifacts:
  `runs/cashsnap/fixed_step_target_anchor_latest_vs_poisson_contact_hardnegdiv8_b8_e50_s150_seed0_summary.json`,
  `runs/cashsnap/background_fp_target_anchor_latest_vs_poisson_contact_hardnegdiv8_b8_conf005.json`,
  and `runs/cashsnap/transfer_guardrails_target_anchor_latest_vs_poisson_contact_hardnegdiv8_b8.json`
- Rejected Poisson/contact realbgneg25 b8 diagnostic transfer artifacts:
  `runs/cashsnap/fixed_step_target_anchor_latest_vs_poisson_contact_realbgneg25_b8_e50_s150_seed0_summary.json`,
  `runs/cashsnap/background_fp_target_anchor_latest_vs_poisson_contact_realbgneg25_b8_conf005.json`,
  and `runs/cashsnap/transfer_guardrails_target_anchor_latest_vs_poisson_contact_realbgneg25_b8.json`
- Mixed Poisson/contact unknownsoftfp8lowconf b8 diagnostic transfer artifacts:
  `runs/cashsnap/background_fp_poisson_contact_b8_unknownsoft_conf0005.json`,
  `runs/cashsnap/fixed_step_target_anchor_latest_vs_poisson_contact_unknownsoftfp8lowconf_b8_e50_s150_seed0_summary.json`,
  `runs/cashsnap/background_fp_target_anchor_latest_vs_poisson_contact_unknownsoftfp8lowconf_b8_conf005.json`,
  and `runs/cashsnap/transfer_guardrails_target_anchor_latest_vs_poisson_contact_unknownsoftfp8lowconf_b8.json`
- Composite-edge audits:
  `runs/cashsnap/composite_edge_audit_target_anchor_realfgstyle_v1.json` and
  `runs/cashsnap/composite_edge_audit_target_anchor_poisson_contact_smoke_v1.json`,
  plus pose/scale/full-candidate variants under the same filename pattern.

Key scripts:

- `scripts/run_webgl_recipe.py`
- `scripts/render_webgl_variant_batch.py`
- `scripts/check_webgl_trainable_candidate_gate.py`
- `scripts/check_webgl_trainable_candidate_suite.py`
- `scripts/check_webgl_appearance_diversity.py`
- `scripts/check_synthetic_pipeline_readiness.py`
- `scripts/check_data_lifecycle_registry.py`
- `scripts/check_webgl_texture_asset_policy.py`
- `scripts/probe_yolo_background_false_positives.py`
- `scripts/run_yolo_fixed_step_probe.py`
- `scripts/compare_yolo_metrics.py`
- `scripts/check_yolo_transfer_guardrails.py`
- `scripts/build_yolo_transfer_scorecard.py`
- `scripts/download_roboflow_datasets.py`
- `scripts/build_roboflow_khmer_us_currency_bridge.py`
- `scripts/build_yolo_positive_error_review.py`
- `scripts/audit_yolo_cross_dataset_visual_gap.py`
- `scripts/audit_yolo_domain_separator.py`
- `scripts/audit_yolo_domain_gap.py`
- `scripts/audit_yolo_crop_visual_domain_gap.py`
- `scripts/build_real_geometry_stress_slices.py`
- `scripts/build_cashsnap_target_anchor_transplant.py`
- `scripts/build_synthetic_obligation_ledger.py`
- `scripts/audit_synthetic_composite_edges.py`
- `scripts/build_webgl_hard_negative_dose_config.py` accepts directory train
  splits as well as `.txt` lists, so target-anchor transplant configs can be
  used directly as hard-negative dose bases; it also supports
  `--filename-contains` for split-safe flat roots like train-only no-note
  background patches.
- `scripts/build_fp_mined_negative_dose_config.py` accepts directory train
  splits as well as `.txt` lists, so target-anchor transplant configs can be
  used directly as FP-mined negative dose bases.

## Label Policy

- Detector labels are visible-instance AABBs derived from the renderer/CV
  visible mask or source annotation, one box per visible class instance.
- OBB/quadrilateral metadata preserves side/pose information for audits and
  future oriented/fusion work; it is not a direct YOLO detect label today.
- Fragment/evidence labels are for disconnected visible evidence and future
  count fusion. They are not direct physical-note count labels.
- Zero-label hard-negative roots must remain zero-label. Do not silently turn
  unknown/foreign note props into target classes.
- Raw Roboflow exports are intake. Use the bridge builder to convert them into
  a schema-aware processed root; do not point YOLO eval/training directly at raw
  Roboflow `data.yaml` files.
- `KHR_100` is official KHR, not garbage. It is excluded only from the current
  core-13 bridge because the active detector cannot predict it yet.
- Do not train/render from raw or mixed data containers just because files
  exist. A root must be working/diagnostic in
  `cashsnap_data_lifecycle_registry_v1.json` and must satisfy its checks.
- Trainable WebGL target-note renders must pass
  `check_webgl_texture_asset_policy.py` against the approved texture bank.
  USD_20 must remain the reviewed `2004-2021` current design; old/watermarked
  or out-of-circulation scans are archive/reject material.

## Machine And Repo Rules

- RunLong mode: prefix terminal commands with `rl`.
- Work directly on `master` unless asked for a branch.
- Repo-local runtime storage is enforced through `scripts/local_runtime.py`.
- YOLO train posture: `batch=64`, `workers=0`, `device=0`, `cache=false`.
- YOLO eval posture: `batch=64`, `workers=2`; background-FP guardrail uses
  `batch=1`.
- WebGL posture: `--render-jobs 2 --renderer-batch-size 32 --check-jobs 4`.
- `cache=disk` is rejected for YOLO probes; it created many `.npy` files, used
  about 12 GB, and slowed throughput.
- Active docs are `README.md`, `AGENTS.md`, and this file.
- Old working memory belongs in `docs/archive/`.
- Generated synthetic roots stay under ignored `data/synthetic/`.
- External negatives stay under ignored `data/external_negatives/`.
- Training/eval outputs stay under ignored `runs/`.
- Browser/temp/cache state stays under ignored `.cache_runtime/` or `tmp/`.

## Useful Commands

```powershell
rl python scripts\check_currency_taxonomy_coverage.py
rl python scripts\download_roboflow_datasets.py --dataset roboflow_khmer_us_currency --version 10 --output-name roboflow_khmer_us_currency_v10
rl python scripts\build_roboflow_khmer_us_currency_bridge.py --raw-root data\raw_datasets\roboflow_khmer_us_currency_v10 --raw-root data\raw_datasets\roboflow_khmer_us_currency --out-root data\processed\roboflow_khmer_us_currency_core13_bridge_v1 --scope operational --unsupported-policy exclude_image --clean --require-all-target-classes
rl python scripts\build_roboflow_khmer_us_currency_bridge.py --raw-root data\raw_datasets\roboflow_khmer_us_currency_v10 --raw-root data\raw_datasets\roboflow_khmer_us_currency --out-root data\processed\roboflow_khmer_us_currency_official21_partial_bridge_v1 --scope official --unsupported-policy exclude_image --clean --summary-json runs\cashsnap\roboflow_khmer_us_currency_official21_partial_bridge_v1_summary.json
rl python scripts\build_currency_taxonomy_gap_plan.py --class-scope official --json-out runs\cashsnap\currency_taxonomy_gap_plan_official_latest.json --md-out runs\cashsnap\currency_taxonomy_gap_plan_official_latest.md
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
rl python scripts\check_yolo_dataset.py --data <config.yaml>
rl python scripts\build_cashsnap_target_anchor_transplant.py --background-root data\backgrounds\cashsnap_v1_no_note_patches_strict_v1 --background-split train --canvas-size "640,640" --asset-quality-policy latest_design --out-root data\synthetic\cashsnap_target_anchor_transplant_latest_v1 --out-config configs\webgl_ablation\cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml --per-class 96 --seed 20260607 --clean
rl python scripts\build_cashsnap_target_anchor_transplant.py --background-root data\backgrounds\cashsnap_v1_no_note_patches_strict_v1 --background-split train --canvas-size "640,640" --asset-quality-policy latest_design --foreground-style-policy real_crop_stats --foreground-style-max-class-samples 128 --out-root data\synthetic\cashsnap_target_anchor_transplant_realfgstyle_v1 --out-config configs\webgl_ablation\cashsnap_target_anchor_transplant_realfgstyle_puresynth_realval_v1.yaml --per-class 96 --seed 1 --clean
rl python scripts\check_yolo_dataset.py --data configs\webgl_ablation\cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml --min-train-class-images 96 --min-train-class-boxes 96 --fail-on-problems
rl python scripts\check_yolo_dataset.py --data configs\webgl_ablation\cashsnap_target_anchor_transplant_realfgstyle_puresynth_realval_v1.yaml --min-train-class-images 96 --min-train-class-boxes 96 --fail-on-problems
rl python scripts\run_yolo_fixed_step_probe.py --baseline-data configs\webgl_ablation\cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_puresynth_realval_v1.yaml --candidate-data configs\webgl_ablation\cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml --baseline-label filtered185 --candidate-label target_anchor_transplant_latest_v1 --model yolo26n.pt --epochs 50 --imgsz 416 --batch 64 --workers 0 --eval-batch 64 --eval-workers 2 --optimizer auto --lr0 0.01 --lrf 0.01 --warmup-epochs 3 --warmup-bias-lr 0.1 --warmup-momentum 0.8 --seed 0 --cache false --device 0 --max-per-class-drop 0.05 --reuse-existing --amp --summary-json runs\cashsnap\fixed_step_target_anchor_transplant_latest_v1_summary.json
rl python scripts\check_yolo_transfer_guardrails.py --help
rl python scripts\build_yolo_transfer_scorecard.py --help
rl python scripts\audit_yolo_domain_separator.py --help
rl python scripts\probe_yolo_background_false_positives.py --model label=path\to\best.pt --data <config.yaml> --split val --split test --conf 0.05 --imgsz 416 --batch 1 --device 0 --json-out runs\cashsnap\<name>.json
```
