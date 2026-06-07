# CashSnap Model Brain

This is the active desk for model and synthetic-data work. Keep it compact and useful.
Move old detail to archive instead of letting this file become a wall of text.

Full pre-cleanup history snapshot:
`docs/archive/model_brain_full_history_2026-06-06.md`.

## Mission

Build a small phone/browser-deployable model that counts mixed USD and Khmer
Riel from one casual retail photo.

The hard requirement is partial-note recognition: half notes, quarters, thin
edges, overlaps, fans, stacks, and hand occlusions must be identified when a
human can see enough denomination evidence.

Counterfeit detection and authenticity classification are out of scope.

## Current Read

- WebGL rendering and package QA are strong enough for bounded model probes.
  They are not a "perfect/done" pipeline yet.
- Approved texture fidelity is no longer the obvious renderer blocker.
  `cashsnap_webgl_approved_texture_bank_v1` selects 26 latest-design/current
  front/back textures and records manual raw-vs-render `texture_qa` pass status.
  The texture gate can now require `latest_design`, `in_circulation`, and
  reviewed source textures before smoke/trainable promotion.
- `texture_qa` baseline passed all front/back class sides. Targeted stress
  probes then passed for standard lit material, backing plane, postprocess, and
  handled-clean condition on KHR_1000/KHR_2000/USD_20/USD_50/USD_100 front/back.
  Continue later knob QA with targeted/random cards, not full 26-card review
  unless a new stage visibly breaks.
- User-corrected condition ladder: `handled_clean` is texture-faithful but too
  subtle for handled-bill stress. `handled_3d` now provides physical mesh bends,
  multiple ridge/valley creases, stronger crinkle/curl, and no painted crease
  strokes; keep it as a targeted stress knob until model-side utility is proven.
- The old black texture-QA outline was a renderer QA artifact: the table used a
  Standard material while texture-QA lighting was zero. Texture-QA now uses a
  flat basic table, and lit-material probes use calibrated neutral light.
- The current strategic ladder is clean-visible-note first, not overlap first:
  build pure-synthetic training data good enough to train `yolo26n` from
  scratch on normal, clearly visible notes; only after synthetic-only clean
  training is comparable to or better than the current non-synth baseline should
  the project mix real+synth or move into overlap/fan/hand curricula.
- Real overlap/fan/hand labels are not needed to render overlap. They are
  needed later to prove overlap transfer. Do not let missing overlap labels
  block the clean-base synthetic phase.
- The main current blocker is whether clean synthetic teaches the base detector
  the real task, not whether more stress scenes can be rendered.
- Production direction check: the synth pipeline is the right engine, but not
  the judge. "Perfect" synthetic data means controllable, dense, exact-label
  coverage whose knobs survive real transfer guardrails; it does not remove the
  need for real validation slices. More renderer polish or one-seed row unions
  will not reach production unless every curriculum step is promoted by clean
  positives, labeled stress positives, real empty-frame FPs, and near-negative
  foreign/unknown-cash slices together.
- The current clean-base model-side reference is the filtered 185-image
  pure-synth curriculum blend, not a single renderer root:
  `cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_puresynth_realval_v1.yaml`
  trained as
  `yolo26n_puresynth_clean_768topdown_squareclassdiverse_no_square_khr20000_blend185_v1_e50_i416_bauto_seed0`.
  It beats the 768x640 component on full real val/test (`0.126892/0.142098`
  mAP50-95 vs `0.054845/0.079471`) and clean-visible val/test
  (`0.217311/0.220057` vs `0.107864/0.121662`), with synthetic self-fit
  `0.914675`.
- The unfiltered 192-image blend remains a useful comparator: it is slightly
  weaker on full real val/test (`0.119674/0.135021`) but stronger on
  clean-visible val/test (`0.273372/0.230763`). This says clean-base promotion
  should use both full-real and clean-visible guardrails, not one scalar.
- Hard-negative rows are a useful curriculum axis, but not a solved clean-base
  promotion. Adding the 32-image old WebGL no-note root to filtered185 reduced
  real empty-frame FP detections at `conf=0.05` (`1556 -> 1373` val,
  `1025 -> 859` test) and improved clean-visible val/test
  (`0.217311/0.220057 -> 0.307545/0.262289`), but full real val/test fell
  (`0.126892/0.142098 -> 0.110287/0.121439`) with large `KHR_500`,
  `KHR_5000`, and `KHR_10000` regressions. Adding only the 8 prop-diverse
  no-note rows gives stronger real empty-frame restraint (`1556 -> 1077` val,
  `1025 -> 618` test) and clears its synthetic no-note root, but full val/test
  are worse (`0.107929/0.116607`) and clean-visible test drops to `0.194990`.
  Do not promote or scale background negatives alone; next use is a
  class-balanced positive curriculum search that keeps the no-note restraint
  signal without sacrificing strong riel classes.
- Targeted `KHR_5000`/`KHR_10000` positive dose8 from the square mixed-condition
  postprocess pool is not that class-balanced repair. It barely passes full val
  (`0.126892 -> 0.127136`) and improves clean-visible val
  (`0.217311 -> 0.261930`), but full test drops to `0.128172` and
  clean-visible test drops to `0.188710`; full-test `KHR_5000` falls
  `0.3107 -> 0.1382`. Treat val-only gains from small staged positive doses as
  suspect until the test slice and protected classes agree.
- Square-only roots are diagnostic, not promotable. The class-aware square v2
  root passed geometry, appearance, note-condition, texture, and trainable gates,
  and improved full real transfer over square v1, but stayed far below 768x640
  (`0.031727/0.036792` full val/test). Geometry and appearance gates are
  necessary, not sufficient.
- Strict geometry selection is also diagnostic, not a promotion rule. A
  185-image mixed-root selection with an 80-image 768x640 floor passed strict
  real-vs-synth geometry, but full real val fell to `0.102368`, below the
  filtered185 reference `0.126892`.
- Refreshed filtered185 geometry targets now show aggregate box geometry is
  close (`area +0.061`, `width +0.094`, `height -0.016`, `aspect -0.002`).
  The remaining accepted-blend geometry gate is class-aspect only:
  `USD_5 -0.369`, `USD_10 +0.279`, `USD_20 +0.265`, `KHR_500 +0.263`.
  Do not use broad geometry replacement; class-aspect work must be low-churn
  and model-gated.
- Class-aspect repair is now a split lesson, not a promotion. Full replacement
  of the four failing classes passes geometry but collapses real transfer:
  full val/test `0.072321/0.075329` and clean-visible val/test
  `0.139659/0.140548`, with `KHR_500` worst. A capped cap2 repair changes only
  8 rows and also passes geometry; it improves clean-visible val/test
  (`0.217311/0.220057 -> 0.243640/0.226025`) but fails full val/test
  (`0.121003/0.119412`) and raises real empty-frame FPs at `conf=0.05`
  (`1556 -> 1996` val, `1025 -> 1317` test), especially `KHR_50000`.
  Next geometry-positive work needs an FP/stress-positive counterbalance and
  full/clean/background guardrails together.
- Simple cap2-plus-hard-negative union is also rejected. Adding the 8
  prop-diverse no-note rows to cap2 gives 193 train images but full real
  val/test fall to `0.089727/0.109779`; `KHR_10000` is the worst protected
  regression. Do not assume geometry-positive rows and no-note rows are
  additive; the missing piece is slice-aware curriculum composition, likely
  stress-positive coverage/selection, not more small row unions.
- `scripts/build_real_geometry_stress_slices.py` now builds provisional real
  label-geometry guardrails under `runs/cashsnap/real_geometry_stress_slices_v1`.
  `labeled_all` is the positive-only half of full val/test
  (`988/814` images) and should be paired with the empty-frame FP probe to
  explain full-score movement. The broad `geometry_stress` slice is non-clean
  labeled real data (`791/659` val/test images); true `multi_note` is only
  `2/3` images and is not fan/overlap proof. Filtered185 gets labeled-all test
  `0.211886` and geometry-stress val/test
  `0.207427/0.211413`; cap2 ties val but fails test
  `0.207099/0.183828`, with `KHR_10000` worst. The protected-riel slice is
  small (`56/26` val/test images) and should use aggregate/scoped checks:
  cap2 improves val (`0.165208 -> 0.269127`) but fails test
  (`0.310953 -> 0.229968`).
- Existing hard-negative probes show why slice-aware gates matter. Hardneg32
  improves labeled-all test (`0.211886 -> 0.232245`) and geometry-stress test
  (`0.211413 -> 0.220407`) while reducing empty-frame FPs, but it still fails
  per-class guardrails, especially `KHR_10000`. Propdiverse8 fails positive
  transfer (`labeled_all 0.185877`, geometry-stress `0.186745`) despite strong
  empty-frame restraint. Future hard-negative use needs protected-class
  positive counterexamples, not just more or fewer no-note rows.
- A small old-root hard-negative dose is a useful ingredient, not a promotion.
  Hardnegold8 improves full real val/test (`0.126892/0.142098 ->
  0.127069/0.149065`), labeled-all test (`0.211886 -> 0.222479`),
  geometry-stress test (`0.211413 -> 0.213747`), protected-riel val/test
  (`0.165208/0.310953 -> 0.288961/0.334709`), and real empty-frame FPs at
  `conf=0.05` (`1556 -> 1446` val, `1025 -> 864` test). It still fails strict
  class guardrails: `KHR_500` full-val, `KHR_2000` full/labeled/stress test,
  clean-visible `USD_5/USD_20/KHR_500`, and borderline `KHR_50000` protected.
  Keep old8 as a restraint donor, not the new baseline.
- Hardnegold8 plus 10 targeted single-note support rows is the strongest
  stress/background donor so far, but still not promotable. The support rows
  target `KHR_500,KHR_2000,USD_5,USD_20,KHR_50000` from the mixed-condition
  phone-auto pool and raise full real val/test to `0.136993/0.151828`,
  labeled-all test to `0.232250`, and geometry-stress test to `0.258421`
  (strict geometry-stress guard passes). Real empty-frame FPs improve further
  (`1556 -> 1331` val, `1025 -> 721` test). The cost is clean-visible
  val/test regression (`0.200242/0.211852`) and protected-riel test
  `0.307626`, just below filtered185 `0.310953`; `KHR_500` and `KHR_50000`
  are the main damaged classes. Next support work should keep the old8
  background/stress gains while replacing mixed-condition support with cleaner
  class-balanced positives or a smaller per-class dose.
- Halving that mixed-condition support to one row per damaged class is not the
  compromise. Hardnegold8+targetsupport5 gets full val `0.132863` and recovers
  clean-visible val to `0.219413`, but full test drops to `0.138938` and
  clean-visible test to `0.194457`; `KHR_500`/`KHR_50000` are still damaged.
  The next probe should change support source or weighting, not merely reduce
  the same phone-auto mixed-condition rows.
- Cleaner topdown support is the current best positive-transfer donor, but not
  promoted. Hardnegold8+topdownsupport10 uses the same 10 support count as
  targetsupport10, replacing phone-auto mixed-condition rows with
  `cashsnap_webgl_clean_base_topdown_768x640_handled_probe_v1` rows. It raises
  full val/test to `0.156479/0.163250`, clean-visible val/test to
  `0.339813/0.255199`, labeled-all test to `0.243480`, geometry-stress test to
  `0.253148`, and protected-riel val/test to `0.244036/0.343944` (protected
  scoped guard passes). Strict promotion still fails: background FP val/test at
  `conf=0.05` move `1556 -> 1692` and `1025 -> 1017` detections with
  images-with-FP `735 -> 741` and `474 -> 483`; `KHR_500` drops in full/clean
  comparisons, `KHR_50000` drops in clean/labeled-all, and `KHR_2000` drops in
  geometry-stress. Next repair should start from this donor and target only
  those class/background failures, not re-search broad clean support. The
  combined guardrail blocker summary is: class tripwires `KHR_500` x4,
  `KHR_50000` x2, `KHR_2000` x1; background FP positive deltas are dominated
  by `KHR_10000` +227, `KHR_20000` +88, `KHR_500` +73, `KHR_5000` +55, and
  `USD_10` +54.
- The topdownsupport10 background blocker is mostly unknown-currency rejection,
  not plain blank-background failure. Review pack
  `data/review/background_fp_topdownsupport10_candidate_conf005_v1/` was built
  from candidate real-empty val/test rows at `conf=0.05`, targeting
  `KHR_10000,KHR_20000,KHR_500,KHR_5000,USD_10`; 50/60 top reviewed
  detections are `asian_currency` foreign/unknown-note frames and most of the
  rest are coin/table `cashcountingxl` frames. Next background repair should
  synthesize explicit foreign/unknown banknote negatives first, plus a smaller
  metallic clutter/table dose, instead of broad empty-scene scaling.
- Unknown-currency procedural negatives work as a false-positive signal, but
  the naive dose is not promotable yet. `webgl_unknown_currency_negative_v1`
  rendered `data/synthetic/cashsnap_webgl_unknown_currency_negative_smoke_v1/`
  (16 zero-label images, required `unknown_banknote` and `coin_cluster`,
  diagnostic hard-negative gate passed). Fixed-step models show strong FP
  transfer: topdownsupport10 vs unknownneg2 real-empty detections drop
  `722 -> 347` val and `457 -> 168` test; unknownneg4 drops to `535/257`;
  unknownneg16 drops to `480/227`. But every naive dose trips a protected
  positive: unknownneg2 fails donor-relative `KHR_50000` (`-0.1209`) and
  filtered185-relative `KHR_2000` (`-0.0814`), unknownneg4 fails
  donor-relative `KHR_500` (`-0.0909`) and filtered185-relative `USD_10`
  (`-0.0625`), unknownneg16 fails donor-relative `KHR_500/KHR_50000/USD_1`,
  and unknownneg1 loses aggregate versus filtered185. Keep this as a repair
  donor; next attempt needs either a smaller/curated external negative bank or
  a paired positive counterweight for the damaged class, not more of the same
  procedural dose.
- The unknown-currency hardness ladder is now explicit. `unknown_currency_soft_v1`
  rendered `data/synthetic/cashsnap_webgl_unknown_currency_soft_negative_smoke_v1/`
  with 21 soft unknown notes and 9 coin clusters. The topdownsupport10 donor
  fires less on this soft root than the hard root (`3/16`, 8 detections vs
  `9/16`, 14 detections), and the FP-mined soft dose clears its own synthetic
  root (`8 -> 0`) while reducing real-empty FPs (`722 -> 524` val,
  `457 -> 288` test). It is still rejected: fixed-step aggregate vs
  filtered185 improves only to `0.096569` and fails `USD_10` (`-0.079858`);
  donor-relative aggregate drops (`0.111497 -> 0.096569`) with `KHR_500`
  worst (`-0.220164`). `scripts/build_fp_mined_negative_dose_config.py` is
  useful, but mined negatives still need paired protected-class positives or
  a different weighting/loss strategy before promotion.
- Pairing mined soft negatives with a tiny clean positive counterweight is the
  best current negative-repair shape, but still not promotion-ready.
  `unknownsoftfp3_repair_khr500_usd10_topdown` adds 2 clean topdown `KHR_500`
  and 2 `USD_10` rows to the 3 FP-mined soft negatives. It passes the
  filtered185 fixed-step screen (`0.075907 -> 0.116104`, worst `KHR_5000`
  `-0.048328`) and improves real-empty FPs (`722 -> 426` val,
  `457 -> 207` test), but donor-relative per-class still fails `KHR_500`
  (`-0.057820`) and `KHR_50000` (`-0.146133`). Adding 2 more `KHR_500` and
  2 `KHR_50000` rows is worse: it fails filtered185 on `KHR_2000`
  (`-0.113131`) and still fails donor-relative `KHR_50000`. Next repair should
  change weighting/selection or add a targeted `KHR_50000` strategy without
  broad extra topdown dose, not keep stacking the same support rows.
- Low-LR finetuning from the topdownsupport10 donor on
  `unknownsoftfp3_repair_khr500_usd10_topdown` is also rejected. It clears the
  soft synthetic negative root (`8 -> 0` FPs), but real test mAP50-95 trails the
  donor (`0.111497 -> 0.104718`), still fails donor-relative `KHR_50000`
  (`-0.149111`), fails filtered185-relative `KHR_2000` (`-0.096979`), and makes
  real-empty test FPs worse (`457 -> 476` detections, `276 -> 307` images).
  The negative signal is learnable; the next production-oriented move is a
  transfer scorecard/selection loop or better counterexamples, not finetuning
  the same small negative dose harder.
- External/stock negative images are now a planned source lane, not an
  untracked scrape path. `cashsnap_external_negative_banks_v1` plus
  `scripts/check_external_negative_banks.py` require local files, source and
  license metadata, intended categories, target USD/KHR absence review, and
  accepted license status before any external negative bank can allow
  trainable-candidate use. Once a bank is accepted,
  `scripts/build_external_negative_dose_config.py` can insert a capped
  external-negative dose into a list-backed YOLO curriculum; the planned bank
  is intentionally blocked from materialization today.
- A fast hardware-optimized fixed-step screen confirms
  hardnegold8+topdownsupport10 is still a donor, not a promotion. With
  train `batch=64, workers=0`, eval `batch=64, workers=2`, cache off, and
  `150` train batches from the filtered185 row-count reference, the candidate
  improves test mAP50-95 (`0.075907 -> 0.111497`) but fails strict per-class
  tripwires on `KHR_2000` (`-0.0526`) and `KHR_5000` (`-0.0924`). Treat this
  b64 run as a fast screen only; it is not directly comparable to earlier
  bauto/b16-style 50-epoch metrics.
- Hardnegold12+topdownsupport10 is not the missing background compromise. The
  same b64 fixed-step screen keeps aggregate test above filtered185
  (`0.075907 -> 0.088440`) but is much weaker than hardnegold8+topdownsupport10
  and still fails strict per-class tripwires on `KHR_2000` (`-0.0647`) and
  `USD_10` (`-0.0579`). Do not spend slow guardrails on old12; background
  repair needs better negative selection/calibration, not just interpolating
  old-root dose count.
- Hardnegold8+topdownsupport10+repair5_topdownv1 is not a cleaner support
  improvement. The same b64 fixed-step screen improves aggregate test
  (`0.075907 -> 0.099315`) but trails plain hardnegold8+topdownsupport10 and
  fails strict per-class on `KHR_5000` (`-0.0874`). Do not spend slow guardrails
  on this repair branch.
- Hardnegold16+topdownsupport10 over-corrects background pressure before it has
  proven useful: full val/test fall to `0.131507/0.137755`, below
  topdownsupport10 and with full-test below filtered185. Do not spend slow
  background/slice gates on that branch; future background repair should be
  narrower than old16 or use better negative selection/calibration.
- Small-probe exit criteria before a serious scaled run: a candidate must beat
  or preserve filtered185 on full val/test, clean-visible val/test, labeled-all
  test, geometry-stress test, protected-riel val/test, and real empty-frame FP
  detections/images-with-FP at `conf=0.05`; strict class-tripwire budget remains
  max per-class drop `0.05`. Then rerun the winner with
  `scripts/run_yolo_fixed_step_probe.py` so row-count changes do not buy extra
  optimizer steps. Only after that should the project scale the winning axes to
  a 1k-2k synthetic run with learning curves/early stopping and at least one
  seed repeat.
- Crop visual audits show a real gap: filtered185 note crops have near-correct
  mean luma, but low dynamic range and sharpness (`luma_std -0.0971`,
  `luma_p05 +0.1426`, `luma_p95 -0.1249`, `sharpness_grad_var -0.0206`).
  However, current `local_dynamic_range_v1` print tone is rejected as a
  trainable clean-base knob: it halves much of that crop gap but drops matched
  768x640 full-val transfer (`0.054845 -> 0.028250` default aug, `0.015666`
  low aug). Crop-stat matching is useful QA, not sufficient model evidence.
- `scripts/select_webgl_geometry_subset.py` now supports per-class camera and
  note-condition diversity (`--min-class-camera-profiles`,
  `--min-class-condition-profiles`) plus selected postprocess range floors
  (`--min-postprocess-range field=value`). Use these when selecting mixed-root
  square candidates so global diversity cannot hide a fragile class monoculture.
- `scripts/build_webgl_staged_rare_dose_configs.py --dose-selection spread`
  handles large single-class dose pools, and `scripts/compare_yolo_metrics.py`
  now accepts dotted metrics such as `box.map50_95`.
- `scripts/audit_yolo_crop_visual_domain_gap.py` now reports aggregate crop
  deltas and supports aggregate `--max-abs-crop-delta` gates. Use it to catch
  note-crop dynamic-range collapse; `--gate-preset clean_dynamic_range_v1`
  fails filtered185 and passes the print-tone diagnostic. Keep model transfer
  as the promotion authority because print-tone passes the visual crop gate
  while failing TSTR.
- `scripts/probe_yolo_background_false_positives.py` now streams predictions
  and can read YOLO dataset YAMLs via `--data ... --split ...`, filtering to
  empty/missing-label rows. It also has opt-in `--review-out-dir` artifacts
  with crop/overlay manifests. Use `--batch 1` for exact promotion/guardrail
  parity; larger batches are okay for fast visual review but can change
  aggregate counts through batched inference details.
- `webgl_unknown_currency_negative_v1` adds explicit zero-label
  `unknown_banknote` and `coin_cluster` hard negatives behind
  `--negative-prop-policy unknown_currency_v1`. Existing `negative` recipes
  remain `classic`; do not silently replace old hard-negative roots.
- `scripts/build_webgl_targeted_support_config.py` builds capped additive
  positive-support probes for damaged classes by appending single-note rows
  from approved clean candidate roots to a list-backed base config. Use it for
  low-churn support-dose tests; do not treat support-dose aggregate gains as
  promotion without clean-visible/protected/background guardrails.
- `scripts/build_webgl_hard_negative_dose_config.py` builds capped additive or
  replacement hard-negative doses from empty-label WebGL roots. Use it for
  narrow background-FP repairs after a positive donor exists; broad old16
  replacement already failed on topdownsupport10.
- The first large clean-base transfer jump came from curriculum composition:
  keep the 768x640 topdown root that transfers some classes, then add
  class-aware square diversity for complementary classes. Do not blindly scale
  one root just because it passes gates.
- KHR_20000 is still a protected weak class. The 768x640 component gets strong
  KHR_20000 full-test mAP50-95 (`0.3608`), but both blends collapse that class
  (`0.0177` unfiltered, `0.0111` filtered) while improving aggregate transfer.
  A 64-image 768x640 KHR_20000 boost only partly recovered full-test KHR_20000
  (`0.0862`) and reduced aggregate transfer (`0.142098 -> 0.135966`) by
  damaging KHR_500/KHR_5000/KHR_10000. Smaller repairs confirm the tradeoff:
  fresh dose8 reached only full test `0.120823`, and row-control dose8 reached
  `0.129718`. Next repair should be a curriculum-composition search that
  protects all strong riel classes, not KHR_20000-only weighting or oversampling.
- Keep the clean-base sequence easy first: top-down/near-top-down visible notes
  before fan, overlap, hand, HDRI, or broad lens distortion. Those axes are
  valuable later only as controlled ablations once the clean-base curriculum has
  stronger protected-class transfer.
- Local real images inspected so far expose no easy EXIF camera/lens metadata;
  any iPhone/DSLR/35mm-style camera or lens family registry should be
  source-backed and tested as a separate ablation, not invented as random lens
  noise.
- The Poly Haven HDRI bank is registered proof-only/diagnostic. One HDRI+lens
  smoke rendered correctly but showed washout risk, so HDRI lighting should not
  enter trainable-candidate clean data until visual review and transfer ablation
  prove benefit.
- The current scorecard is blocked: `4` pass / `13` blocked axes after adding
  the project-level real-capture requirement axis.
- Taxonomy and asset coverage must be read by layer, not assumed. Official
  current USD/KHR scope is 21 classes. The active model schema and active WebGL
  cutout bank are still 13-class operational subsets; raw Numista current-status
  cache has front/back for 20/21 classes; raw Numista any-status cache has
  front/back for all 21. `KHR_50` is the only official class missing from the
  current-status raw cache, while any-status raw has 6 front/back pairs.
- `scripts/build_currency_taxonomy_gap_plan.py` turns that scope gap into a
  promotion queue. Current full-scope Numista candidate cutouts cover the seven
  active/model-missing current-status classes (`USD_2`, `KHR_100`, `KHR_200`,
  `KHR_15000`, `KHR_30000`, `KHR_100000`, `KHR_200000`); `KHR_50` is any-status
  only and needs explicit status review before trainable current-currency use.
- Browser failures are mostly detector-proposal/curriculum limited. Fragment
  or final-count heuristics alone are not enough.
- The latest base-first clean probe failed hard. A fresh `yolo26n.pt` trained
  for 20 epochs on p24 real + 32 WebGL clean-base images reached only
  `mAP50-95=0.17524`. Do not stage overlap/fan from that checkpoint.
- The 512-image clean WebGL curriculum root now passes the trainable gate after
  making note-condition diversity scene-aware. It is accepted for a bounded
  clean-checkpoint ablation, not for blind scale or overlap/fan staging.
- The seed0 clean-checkpoint ablation is not promotable: p24 real + 512 clean
  WebGL beats the matched p24 real-only fine-tune (`0.841867` vs `0.835861`,
  `+0.006006` mAP50-95), but still fails the stronger clean checkpoint
  (`0.883801`, delta `-0.041934`). Worst clean-checkpoint drops are
  `KHR_50000=-0.246982` and `KHR_20000=-0.074754`.
- Older stack selected-geometry roots were confounded by synthetic finger
  occluders. Treat stack overlap and hand/finger occlusion as separate recipe
  variables; no-hand stack probes now require explicit occluder-policy gates.
- No-hand stack dose1 has a tiny positive detector signal but fails deploy
  usefulness: seed0 exact class-mix control probe improved test mAP50-95 by
  `+0.002033` and `KHR_50000` by `+0.022303`, but browser mined-real stress
  regressed on dense overlap, thin edge, and weak-class value behavior.
- Prop-diverse hard negatives are more realistic than the old weak root, but
  not a solved path: diversity8 loses `-0.003090` mAP50-95 versus old8, and
  browser rejection/threshold tweaks clear hard negatives by killing positives.
- Background false-positive probing explains the detector/browser mismatch:
  no-hand dose1 and both hard-negative fine-tunes hallucinate far more than the
  default detector on zero-label synthetic prop scenes. Do not scale those
  recipes until proposal false positives improve under a real/deploy guardrail.
- Existing model families split the problem: the old overlap-stage detector
  improves mined-real overlap/fan recall but is noisy on prop-diverse negatives;
  the full-real-only detector is background-saner but loses mined-real and
  synthetic stress recall. The next synthetic/curriculum probe must preserve
  overlap recall while borrowing the full-real/background restraint.
- Current WebGL overlap-stack training does not transfer like the old staged
  overlap detector: it is worse than default on mined-real dense overlap and
  thin-edge browser stress, and it is worse than default on synthetic prop
  false positives. The useful old-overlap signal is curriculum/data-shape
  evidence, not proof that the current WebGL overlap recipe is right.
- Tiny prop-diverse hard-negative repair of the old-overlap detector reduces
  raw prop false positives, but browser behavior shifts into duplicate proposal
  overcount. The next repair needs duplicate/proposal calibration, not simply
  more no-note hard negatives.
- Browser NMS is a real deploy lever for the old-overlap recall donor. NMS
  `0.65` keeps mined-real perfect cases at `12/17` while reducing total
  predicted count `44 -> 39` and absolute count error `11 -> 8`; adding the
  unclassified/disagreement guard clears synthetic hard negatives but drops
  mined-real perfect cases to `11/17`. Lowering the guard threshold from
  `0.14` to `0.10` does not recover those real matches.
- The existing broad 14-class real-fragment classifier is not the missing
  hard-negative gate. With old-overlap + NMS `0.65`, no rejection still fails
  synthetic card/phone and wallet/sticky negatives; disagreement rejection
  clears negatives by deleting almost all synthetic positives.
- Broad real-fragment `background` predictions are useful evidence but not a
  solved gate. Old-overlap + NMS `0.65` + broad background-only rejection after
  NMS clears all three synthetic hard negatives, but mined-real does not improve
  over plain NMS (`12/17` perfect still, `same 31/35` vs `32/35`, abs KHR
  `152000` vs `92000`) and synthetic hand evidence is deleted.
- A binary old-overlap proposal gate is the strongest calibration signal so
  far. V1 trained on detector proposal crops improves mined-real to `13/17`
  perfect and clears synthetic hard negatives; v2 adds WebGL partial-positive
  proposal crops and improves aggregate mined-real further (`pred=35`,
  `same=31/35`, `any=32/35`, abs count `4`, abs KHR `21000`) while still
  clearing the three synthetic hard negatives. This is diagnostic only until
  real no-note/non-banknote and protected real partial validation exist.
- Proposal-gate promotion is now explicitly blocked by capture buckets for
  `khr_50000_hard_positive_partials` and `usd_hard_positive_partials`; these
  target high-confidence banknote->background errors from the v2 failure pack.
- Browser-count comparison is now a first-class detector-probe artifact. Use
  `scripts/run_browser_detector_probe.py` after bounded synth detector probes
  when an ONNX export is available; it runs/reuses synthetic and mined-real
  browser reports and compares them with `scripts/compare_browser_reports.py`.
  The comparator caught no-hand stack dose1 as blocked despite its tiny detector
  mAP gain: synthetic stress worsened abs count by `+3`, abs KHR by `+26000`,
  and hard-negative predictions by `+1`; mined-real stress lost `3` perfect
  cases, `5` same-class matches, and added `+166000` abs KHR error.
- Strict browser comparison says proposal-gate v2 is the best diagnostic stack
  but still not promotable: mined-real passes versus default (`+4` perfect,
  `+8` same-class, `-6` abs count, `-62500` abs KHR, `-80` abs USD), while
  synthetic stress still blocks on positive evidence (`-1` same/any,
  `+2` abs count, `+66000` abs KHR) even though hard-negative predictions drop
  by `5` and failures drop by `3`.
- Proposal-gate v2 case effects are now reviewable with
  `scripts/summarize_browser_gate_effects.py`. The browser smoke summary now
  exposes proposal, classified, and clustered diagnostic sources, so the
  analyzer no longer infers gate effects from counters alone. At NMS `0.65`,
  synthetic stress has `2` hard-negative helps, `1` hard-negative same,
  `1` positive NMS harm, `1` positive reject harm, and `4` positive same.
  Mined-real has `14` positive same, `1` positive reject safe, and `2`
  positive reject harms. The concrete next target is protected thin/weak
  positives such as `cashsnapv1_weak_khr_50000_03_khmer_scan_img_9189_jpeg_jpg`,
  not generic hard-negative scale.
- Browser gate review targets are now generated with
  `scripts/build_browser_gate_review_targets.py` and embedded in detector-probe
  summaries as `gate_review`. For proposal-gate v2, the P1 review/capture
  targets are mined KHR_50000 hard-positive partials, mined KHR_20000 thin
  slices, synthetic hand-fan NMS/fusion harm, and synthetic same-denomination
  fan reject harm.
- `scripts/check_capture_requirements.py --gate-review-targets <targets.csv>`
  annotates the real shot list with those browser failure examples; use it when
  refreshing `runs/cashsnap/real_capture_shot_list_latest.md` so capture work is
  tied to concrete deploy failures.
- Raising proposal-gate v2 browser NMS from `0.65` to `0.80` is not the fix.
  It keeps mined-real strict comparison passing but worsens the v2 aggregate
  (`pred 35->38`, abs count `4->5`, abs KHR `21000->26000`) and synthetic KHR
  error (`474500->531500`) despite reducing synthetic abs count (`19->17`).
  The corrected gate-effect categories still point at the same protected-positive
  reject harms, so this needs proposal-gate calibration/validation, not a
  simple NMS threshold bump.
- Strict accepted-blend visible-note geometry is now diagnosable with
  `scripts/summarize_domain_gap_geometry_targets.py`. Current accepted-nowarmup
  synthetic boxes are much smaller than real (`area -0.372`, `width -0.417`,
  `height -0.381`). Thin-edge partials should remain a protected partial slice,
  not evidence that general visible-note scale is correct; clean/base and
  back-side confusion rows need larger visible-note framing or selected-geometry
  filtering. Worst class aspect gaps are `USD_10`, `USD_20`, and `KHR_10000`.
- Selection alone does not repair accepted general-visible geometry. Bounded
  selectors over clean-base + back-side roots still fail strict geometry at
  32 images (`area -0.352`) and 16 images (`area -0.341`). Clean-closeup v2
  proves aggregate scale is fixable (`area -0.009`, `width +0.027`,
  `height -0.039`) but still fails five per-class aspect checks and already
  regressed the real-only ablation, so a v3 needs class-conditioned pose/aspect
  calibration plus model proof, not just larger notes.
- The research note
  `docs/research/What Makes a Dataset Perfect for Synthetic Data Pipelines.pdf`
  supports an operational definition of "perfect": coverage-rich, label-trusty,
  task-useful data that improves real deployment metrics. Its most relevant
  lessons here are: coverage beats raw size, quality beats realism theater,
  content/scene structure gaps can matter more than pixel polish, TSTR/TRTS
  style utility is essential, and bold synthetic generation should still be
  judged by real/deploy holdouts.

## North Star

Synthetic data should be a controlled experiment generator, not just an image
generator.

A future failure should map to one of these causes:

- missing real condition or label bridge
- bad label semantics
- domain/geometry/appearance gap
- curriculum or row-order problem
- model/fusion/deploy bug
- governance or taxonomy gap

Scale only after a synthetic recipe improves a real stress scoreboard under
matched controls and seed stability.

Clean-base exception: before overlap/fan/hand work, the key proof is
synthetic-only utility on normal visible notes. A pure-synth `yolo26n` run that
approaches or beats the current non-synth clean baseline is a meaningful
milestone even before real overlap stress labels exist.

## Direction Discipline

- Take a step back before starting another narrow repair. Ask whether the next
  action could change the project direction, not just make one metric look tidy.
- Prefer bold, bounded experiments over guaranteed tiny improvements when the
  result can teach a real lesson. Risk is acceptable when the hypothesis,
  control, failure signal, and cleanup path are clear.
- Do research when the path is fuzzy, but only preserve conclusions in
  `model.md` when they change decisions.
- Batch commits into coherent engineering memory. Do not commit every tiny
  probe, and do not clutter git history with half-steps unless the half-step is
  itself an important durable decision.

## Local Machine Posture

- Laptop: AMD Ryzen 5 7640HS, 6 cores / 12 threads, about 16 GB RAM, and an
  NVIDIA GeForce RTX 4060 Laptop GPU with about 8 GB VRAM; PyTorch sees CUDA
  device `0`.
- Prefer GPU-backed PyTorch/Ultralytics work with `--device 0` when VRAM has
  room. Heavy jobs should go through `scripts/run_with_headroom.py` when
  practical. User rule: prefer GPU work while keeping CPU/RAM usable for
  browsing/video. The wrapper now throttles CPU load and memory by default;
  GPU utilization is allowed to saturate unless `--throttle-gpu-util` is
  passed, while GPU memory remains capped.
- Do not let the headroom wrapper wait forever if it cannot find launch
  headroom. If it stalls, shrink the job first: lower batch/workers, lower RAM
  pressure, adjust preflight floors, or split the run.
- RAM and CPU contention are the main interactive bottlenecks. Keep training
  workers low and push throughput into CUDA batch size; browser smoke
  validation is ONNX Runtime Web/WASM and remains CPU-side.
- WebGL recipe rendering now uses Node batch chunks by default:
  `--renderer-batch-size 32 --render-jobs 2 --check-jobs 4`. A 24-image
  768x640 clean-topdown benchmark measured `0.886 images/s` full render+package
  throughput with that posture; batch-size 32 beat 8/16 at that count.
  Shared-browser reuse is exposed but remains opt-in because it did not beat
  normal batched rendering. Subprocess smoke checks remain the measured faster
  default than in-process checks at `check_jobs=4`. Keep 95/88 headroom caps
  and reduce jobs if the laptop is under active pressure.
- Hardware defaults are centralized in `scripts/hardware_profile.py`. On the
  Lenovo 82Y9 / Ryzen 5 7640HS / RTX 4060 Laptop profile it currently selects
  CUDA device `0`, train batch/workers `64/0` at 416, val batch/workers `64/2`
  at 416, WebGL `render_jobs=2`, `renderer_batch_size=32`, `check_jobs=4`,
  and browser smoke `jobs=1` for stability.
- Repo-local runtime storage is enforced through `scripts/local_runtime.py`.
  Active YOLO train/eval/probe/export scripts and long-run wrappers configure
  `.cache_runtime/` for Ultralytics, Torch, Matplotlib, pip/Numba caches, and
  `TMP`/`TEMP` before heavy imports or child launches. C-drive paths in logs
  should be executable locations only, not model/data/cache outputs.
- Training benchmark: at 416, workers `0` beat workers `2` for
  `batch=8/16/24`, and larger GPU batches are feasible on the RTX 4060 laptop.
  `batch=64, workers=0` uses about `3.9 GB` VRAM on `yolo26n` and is the fast
  exposure-equivalent probe default; keep `cache=false`.
- Validation benchmark: on the full CashSnap val split, `batch=32, workers=2`
  completed the progress loop in about `18.0s` (`44s` wall), `batch=64,
  workers=2` in about `18.6s` (`35s` wall), while `batch=64, workers=0`
  regressed (`39.4s` loop, `52s` wall). Use `64/2` for eval speed; use
  `workers=0` only when interactive CPU pressure matters more than eval time.
- `cache=disk` is rejected for this repo's YOLO probes: it cached train plus
  the real val split, created `2288` `.npy` files / `11.88 GB`, and slowed b64
  throughput (`0.674 -> 0.255` batches/s). Those cache artifacts were removed.
- `train_yolo.py --no-val` now uses a no-validation trainer shim, because
  Ultralytics still validates on final epoch and `final_eval()` even when
  `val=False`. One-batch smoke dropped from `73.5s` to `41.9s`; the remaining
  cost is startup, AMP, dataloaders, and checkpoint work.
- Wrappers that train first and evaluate separately now pass `--no-val` during
  training (`run_yolo_fixed_step_probe.py`, `run_webgl_recipe_ablation.py`,
  trainable-candidate smoke, and P1 smoke). This avoids duplicate full-val
  passes in harness probes.
- Browser smoke parallelism is opt-in only. `--jobs 2` exposed Edge CDP startup
  flakiness on this Windows laptop, so default stays `1`; parallel runs retry
  no-summary infrastructure failures sequentially before reporting.
- Browser smoke now has a Python-side subprocess watchdog:
  `--subprocess-grace-seconds` kills stuck Node/Edge cases after
  `--timeout-ms` plus a small grace window. Avoid long opaque harness commands;
  prefer bounded smokes with progress prints or explicit timeouts.
- For long YOLO jobs, poll progress from `results.csv` with
  `scripts/summarize_yolo_progress.py --run <run-dir> --expected-epochs N`.
  It reports epoch, elapsed time, train losses, loss deltas, and mAP when
  validation rows exist; use this instead of silently waiting on 30+ minute
  runs.
- Dataset blend needs an orientation distribution, not just visual realism.
  Current clean synthetic roots are mostly upright; before calling a blend good,
  audit real app rotation/orientation and add bounded upside-down/rotated
  coverage only to the degree the product will actually see it.
- A good synthetic/real blend is a matched distribution, not a pile of nice
  images: class, side, orientation, scale, crop, aspect, luma, focus,
  background, hard negatives, and stress frequency all need controls plus real
  validation deltas.

## Immediate Plan

1. Keep the desk clean:
   `model.md` stays current and concise; old details go to `docs/archive/`.
2. Use the reviewed texture bank and texture-policy gate for any new WebGL root
   intended to inform promotion. Later appearance knobs only need targeted
   stress-card QA unless they introduce a visible failure; use `handled_3d` for
   physical crease/bend stress instead of texture-painted folds.
3. Work the clean-synth ladder first. Keep the filtered 185-image blend as the
   current model-side reference; new blends must beat both full-real and
   clean-visible controls without protected-class collapse.
4. Do not aim for overlap/fan/hand yet. Keep those old results as diagnostic
   memory, but do not let them steer the next phase unless clean-base synth has
   reached a credible baseline.
5. Make the clean-base data actually complete: promote/review the seven
   current-status missing official classes, resolve `KHR_50` status, and avoid
   calling a 13-class subset "perfect" for the 21-class product scope.
6. Repair clean visible-note distribution before scaling: compare top-down
   handled synth against real clean note scale, luma, crop sharpness, texture,
   aspect/pose, and class-specific failures. Run geometry/domain-gap checks, but
   require model utility too.
7. Use TSTR-style thinking for the clean phase: synthetic-train -> real-test is
   the decisive question. Synthetic-only gates and visual quality are necessary
   filters, not the final proof.
8. Pause tiny row-dose hill-climbing unless it is tied to a transfer scorecard.
   Prefer experiments that can fail loudly and teach direction across clean
   positives, labeled positives, protected classes, real empty frames, and
   near-negative clutter together.
9. Keep browser/deploy comparison in the loop once a detector is credible, but
   do not overfit early clean-synth work to overlap-era proposal-gate diagnostics.
10. Use the chunked WebGL renderer defaults before more 500+ image runs:
   `--renderer-batch-size 32 --render-jobs 2`, with 95/88 headroom caps.
   Throughput is no longer the main blocker; real-transfer proof is.
11. For every detector-side synth probe that gets trained, export/reuse ONNX and
   run `scripts/run_browser_detector_probe.py` before treating mAP as useful.
   The pass/fail comparison and, when artifacts exist, gate-effect counts should
   travel with the training summary.
12. Real bridge work still matters, but for the current phase it should focus
   on clean visible-note holdout confidence and no-note/background sanity.
   Promoted overlap/fan/hand labels become P0 only after clean synth is useful.

## Promotion Rules

For CashSnap, "perfect synth data" means operational fit, not raw realism.

Promote synthetic data only when all relevant checks pass:

- real P1 exact-count accuracy improves by about `+3pp`, or total-value MAE
  drops by about `10%`
- no high-value or rare class drops by more than about `2pp` / `2 AP`
- hard-negative, unknown, and non-identifiable sliver hallucination does not
  materially increase
- at least 3 seeds are positive; use 5 seeds for large promotions
- matched row-count and class-exposure controls do not explain the gain
- browser/mobile model size, latency, and quantized accuracy stay within budget
- qualitative error review shows real fixes, not shortcut learning

## Label Policy

Visible evidence is authoritative.

- `visible_polygon` / visible mask is the detection truth.
- `visible_bbox` is a compatibility label for detect-only probes.
- `oriented_bbox` is allowed only when the visible region is compact and
  rectangle-like.
- `physical_note_id` is the count target. One note split into several visible
  islands is still one physical note.
- `fragment_id` labels local visible evidence and must be fused back to the
  parent before counting.
- `human_identifiable=false` or `uncertain` should train/evaluate abstention or
  ignore behavior, not forced denomination guesses.
- Preserve side/design, visibility fraction, cue types, occlusion order, and
  ignore reason whenever labels support it.

## Current Assets

Active backbone:

- `data/cashsnap_v1/` is the canonical clean/base YOLO dataset.
  It has the current 13 bill classes and 9,048 labeled boxes.
- `data/asset_candidates/numista_current_cutout_bank_v1/` is the current clean
  scan/cutout bank for WebGL. It covers front/back for the 13 operational
  classes and still has usage/release limits.
- Raw Numista coverage is broader than the active bank. Diagnostic probe banks
  under ignored `data/asset_candidates/` show current-status full-scope cutouts
  cover 20/21 official classes, and any-status full-scope cutouts cover 21/21.
  Do not use the any-status bank for trainable current-currency recipes without
  design/status review.

Active diagnostic real bridge:

- `runs/cashsnap/mined_real_scoreable_dataset_latest/`
- `runs/cashsnap/mined_real_scoreable_dataset_holdout_latest/`
- `runs/cashsnap/mined_real_browser_cases_latest.csv`

Active WebGL trainable-candidate roots:

- `data/synthetic/cashsnap_webgl_clean_base_candidate_v1`
- `data/synthetic/cashsnap_webgl_overlap_stack_candidate_v1`
- `data/synthetic/cashsnap_webgl_fan_fullschema_candidate_v1`
- `data/synthetic/cashsnap_webgl_hand_occlusion_candidate_v1`
- `data/synthetic/cashsnap_webgl_thin_edge_partial_candidate_v1`
- `data/synthetic/cashsnap_webgl_hard_negative_diversity_catalog_gate_v1`
- `data/synthetic/cashsnap_webgl_back_side_confusion_candidate_v1`

New clean-curriculum probe root:

- `data/synthetic/cashsnap_webgl_clean_base_curriculum_probe_v1`
- Status: rendered, label-checked, and trainable-gate accepted.
- Evidence: 512 images, 1,023 physical count targets, roughly 78-79 boxes per
  current class, appearance diversity passed, OBB trainable for all images.
- Note-condition diversity: clean-scene mixed policy has `pristine=197`,
  `circulated=826`, `dirtiness_range=0.45`, `crinkle_range=0.42`,
  `wetness_range=0.28`. Heavy/wet stress belongs in dedicated stress recipes.
- Model result: one-epoch clean-checkpoint ablation with seed0 passes the
  matched p24 real-only comparison but rejects versus the clean checkpoint.

Earlier top-down handled clean-base probe:

- `data/synthetic/cashsnap_webgl_clean_base_topdown_handled_probe_v1`
- Status: diagnostic reference for the clean-first ladder. It passes the
  trainable-candidate gate with required visual note quality under the relaxed
  readable top-down appearance thresholds.
- Evidence: 96 single-note images, balanced 13-class operational coverage, no
  tiny/small note failures at `imgsz=416`, p05 short side `118.52 px`, p50
  `189.51 px`, soft crops `10/96`, and handled-clean condition diversity
  (`lightly_handled=15`, `circulated=48`, `well_handled=33`).
- Model result: pure-synth `yolo26n` seed0 at `imgsz=416` reaches real val
  `mAP50-95=0.0215`, real test `mAP50-95=0.0350`, and synthetic self-eval
  `mAP50-95=0.745`. This beats the older 512-image clean pure-synth real test
  (`0.02369`) with far fewer images, but it is still transfer-limited and not
  promotable.

No-hand stack diagnostic:

- `data/synthetic/cashsnap_webgl_no_hand_real_scale_stack_print_tone_selected_geometry_v1`
- Status: diagnostic only. It isolates note-on-note stack overlap from
  hand/finger occlusion and passes strict geometry after selecting 20 images
  from a 40-image no-hand pool.
- Evidence: selected20 has 83 YOLO boxes, occluder policy `no_hand`, zero
  occluders, local dynamic-range print tone passing, and strict geometry pass.
- Model result: dose1 selected variant 7 and beat an exact class-mix exposure
  control by `+0.002033` mAP50-95 at seed0; `KHR_50000` improved `+0.022303`,
  worst class drop was `USD_20=-0.009805`.
- Deploy result: the same detector is not promotable. Browser synthetic stress
  still fails hard negatives and overcounts overlap; mined-real browser stress
  worsens dense-overlap recall/count, thin-edge count/value, and weak-class
  value behavior versus the default detector.
- Remaining blockers: class balance is loose, trainable-candidate appearance
  diversity is too narrow for promotion, exact real class-mix controls are
  unavailable from current repo data, and deploy behavior does not transfer.

## Current Models

Default browser detector:

- `runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt`
- ONNX is wired through `configs/cashsnap_two_stage_oldcommon_browser_stack.json`.
- Useful for current browser diagnostics, not a solved final model.

Useful curriculum signal:

- `runs/cashsnap/yolo26n_pristine_overlap_e2_i416_b8/weights/best.pt`
- Improves mined-real browser same-class/fan/overlap recall versus default, but
  overcounts and has value noise. Treat as evidence for staged curriculum, not
  as a promotion candidate.

Rejected latest base clean probe:

- `runs/cashsnap/yolo26n_base_clean_webgl_clean_base_v1_e20_i416_b8_seed0/`
- Best/last val: `mAP50-95=0.17524`, `mAP50=0.19789`, precision `0.20579`,
  recall `0.31653`, train class loss `3.4868`.

Current pure-synth top-down handled probe:

- `runs/cashsnap/yolo26n_puresynth_topdown_handled96_v1_e50_i416_b8_seed0/`
- Real val: `mAP50-95=0.0215`, `mAP50=0.0344`, precision `0.085`, recall
  `0.0457`.
- Real test: `mAP50-95=0.0350`, `mAP50=0.0589`, precision `0.0929`, recall
  `0.0479`.
- Synthetic self-eval: `mAP50-95=0.745`, `mAP50=0.808`, precision `0.642`,
  recall `0.78`. The root is learnable; the unresolved problem is
  real-domain transfer.

Current pure-synth 768x640 handled probe:

- `runs/cashsnap/yolo26n_puresynth_topdown_768x640_handled96_v2_e50_i416_b8_seed0/`
- Full real val/test: `mAP50-95=0.0548` / `0.0795`.
- Clean-visible real val/test: `mAP50-95=0.1079` / `0.1217`.
- Synthetic self-eval: `mAP50-95=0.7681`, `mAP50=0.868`.
- Read: clear improvement over topdown handled96 v1, but still far below
  real-trained clean-visible test `0.8975`; keep working transfer, class
  aspect, crop/scale, orientation blend, and real validation bridge before
  promotion.

Current pure-synth clean curriculum reference:

- `runs/cashsnap/yolo26n_puresynth_clean_768topdown_squareclassdiverse_no_square_khr20000_blend185_v1_e50_i416_bauto_seed0/`
- Config:
  `configs/webgl_ablation/cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_puresynth_realval_v1.yaml`.
- Full real val/test: `mAP50-95=0.126892` / `0.142098`.
- Clean-visible real val/test: `mAP50-95=0.217311` / `0.220057`.
- Synthetic self-eval: `mAP50-95=0.914675`.
- Read: strongest clean-base pure-synth reference so far, but not promotable
  because protected `KHR_20000` collapses on full test (`0.0111`).
- Rejected repair:
  `yolo26n_puresynth_clean_768topdown_squareclassdiverse_khr20000boost249_v1_e50_i416_bauto_seed0`
  raises full-test `KHR_20000` to `0.0862` but drops aggregate full test to
  `0.135966` and harms KHR_500/KHR_5000/KHR_10000, so brute class oversampling
  is diagnostic only.
- Rejected small-dose repairs: fresh KHR_20000 dose8 improves full-test
  `KHR_20000` to `0.0361` but drops aggregate to `0.120823`; row-control dose8
  duplicates the original 768x640 KHR_20000 rows and improves full-test
  `KHR_20000` to `0.0951`, but aggregate still drops to `0.129718`, led by
  KHR_500/KHR_10000/KHR_2000 regressions. KHR_20000-only weighting is not the
  clean-base repair.
- Rejected geometry-only curriculum: strict mixed-root geometry-selected185 v2
  passes the accepted geometry gate, including aggregate deltas
  `area +0.008`, `width +0.053`, `height -0.058`, `aspect +0.043`, but full
  real val is only `0.102368`. It improves `KHR_20000` val slightly
  (`0.003868 -> 0.015439`) while cutting strong riel classes such as
  `KHR_500` and `KHR_5000`.
- Rejected augmentation/staging repairs: filtered185 with 768-style low aug
  fell to full-val `0.033325`, and 768-weights staged for 20 epochs into
  filtered185 reached only `0.089816`. The blend's current aggregate utility
  needs its existing broader training posture; KHR_20000 collapse is not fixed
  by simply removing default mosaic/flip/erasing or staging from the 768 root.

## Known Results

- Accepted WebGL blend is not stable across seeds and rejects on mined held-out
  rare/edge diagnostic utility. Worst issue remains `KHR_50000`.
- Negative model probes from older selected stack roots are not pure no-hand
  overlap evidence because every stack image had synthetic finger capsules.
- No-hand stack dose1 is a weak positive detector result, not a promotion:
  exact class-mix mAP improved, but browser mined-real stress regressed.
- Hard-negative diversity8 is not promotable by itself: it improves some KHR
  value noise on synthetic stress, but loses detector AP versus old8 and
  threshold/disagreement rejection trades false positives for missed bills.
- Background FP comparison (`runs/cashsnap/background_fp_default_nohand_hardneg_comparison.json`):
  at `conf=0.05`, default has `13/32` detections on the old hard-negative root
  and `9/8` on the prop-diverse root; no-hand dose1 jumps to `56/32` and
  `22/8`, while hardneg old8/diversity8 remain high (`48/32`/`18/8` and
  `46/32`/`16/8`). The no-hand AP lift is likely proposal hallucination, not a
  deploy-useful overlap fix.
- The 512 clean WebGL root is a useful scarcity-control signal but is not
  promotable. It improves over matched p24 real-only by `+0.006006` mAP50-95
  and fails clean-checkpoint guardrails by `-0.041934`, led by `KHR_50000`.
- The 96-image top-down handled clean root is a better direction than the old
  512-image pure-synth clean root: real test transfer improves to `0.0350`
  mAP50-95 versus `0.02369`, while synthetic self-eval reaches `0.745`. This
  says the corrected mechanics are learnable and less wrong, but the remaining
  transfer gap is still the main bottleneck.
- The square mixcam orientation probe closes the visible-note geometry gap that
  blocked 768x640 v2. Its lesson is data-shape calibration, not promotion:
  model-side TSTR must decide whether the geometry repair helps transfer.

Current 768x640 repaired clean-base diagnostic:

- `data/synthetic/cashsnap_webgl_clean_base_topdown_768x640_handled_probe_v2`
- Status: diagnostic only. It rerenders the same 7600-7695 variant span as the
  pre-fix 768x640 v1 root; do not use v1 for model claims because it was
  contaminated by bill-face shadow acne/striping.
- Renderer lesson: individual ablation traced the bad bill-face lines to the
  Standard-material path only when note/backing shadow receiving was active.
  The durable fix is `receiveShadow=false` for note and backing meshes, with a
  bounded 4096 VSM shadow map for cast table shadows; the note may cast a table
  shadow but must not receive engine shadows.
- Evidence: package render/check completed for 96 images and 96 boxes; visual
  spot-checks on variants `7600`, `7624`, and `7634` are clean at native size;
  texture asset policy passes against the approved latest-design bank; clean
  handled condition diversity passes with `lightly_handled=96`,
  `dirtiness_range=0.17`, and `crinkle_range=0.14`.
- Geometry result: aggregate real-vs-synth box aspect is repaired
  (`box_aspect +0.000`, `box_area +0.075`, `box_width +0.111`,
  `box_height -0.007`), but per-class aspect still fails for `USD_5`,
  `USD_10`, `USD_20`, `KHR_500`, `KHR_5000`, `KHR_10000`, and `USD_100`.
- Model result: matched pure-synth `yolo26n` seed0 at `imgsz=416` validates
  the direction but not promotion. Full real val/test mAP50-95 are `0.0548` /
  `0.0795`; clean-visible val/test are `0.1079` / `0.1217`; synthetic self-eval
  is `0.7681`. Compared with topdown handled96 v1, deltas are `+0.0445` full
  test, `+0.0620` clean-visible test, `+0.0379` clean-visible val, and
  `+0.0232` synthetic self-eval. Remaining class regressions matter:
  clean-visible test drops `USD_20=-0.1045`, `KHR_2000=-0.0436`, and
  `KHR_10000=-0.0053`; full test drops `KHR_500=-0.0254` and
  `KHR_10000=-0.0108`.
- Next use: treat v2 as the clean visual/aggregate-geometry baseline; next
  work should preserve its transfer-positive classes while changing one
  appearance/curriculum axis at a time. Direct print-tone and geometry-only
  replacements have now failed, so do not scale either without a new control.

Current 768x640 print-tone diagnostic:

- `data/synthetic/cashsnap_webgl_clean_topdown_768x640_printtone_probe_v1`
- Config:
  `configs/webgl_ablation/cashsnap_webgl_clean_topdown_768x640_printtone_puresynth_realval_v1.yaml`.
- Status: rejected transfer diagnostic. It proves per-note local dynamic range
  can reduce the crop visual gap but can still harm model transfer.
- Geometry result: aggregate geometry is acceptable
  (`box_area +0.089`, `box_width +0.095`, `box_height +0.028`,
  `box_aspect -0.075`), but `KHR_10000.box_aspect -0.371` fails the strict
  class gate.
- Crop result: versus filtered185, aggregate crop gaps improve from
  `luma_std -0.0971` to `-0.0497`, `luma_p05 +0.1426` to `+0.0661`,
  `luma_p95 -0.1249` to `-0.0460`, and `sharpness_grad_var -0.0206` to
  `-0.0098`; saturation spread is still low.
- Model result: default-aug full real val is `0.028250`; matched 768-style
  low-aug full real val is `0.015666`. Both are below the 768x640 handled v2
  control `0.054845`, so `local_dynamic_range_v1` should stay diagnostic until
  a gentler/tunable policy is proven by model-side ablation.

Current square mixcam clean-base geometry probe:

- `data/synthetic/cashsnap_webgl_clean_base_square_mixcam_sqorient_probe_v1`
- Status: rejected transfer diagnostic. It proves strict visible-note geometry
  can be matched, but model utility regresses versus the 768x640 v2 control.
- Renderer/data lesson: the real CashSnap audit images are square 640x640;
  evaluating 768x640 synthetic against square real images depresses normalized
  box aspect by about 17% and can make good bill pose look class-aspect-bad.
- Evidence: 96 square 640x640 single-note images,
  `phone_clean_base_readable_mix_v1` camera mix, `real_aspect_square_v1`
  class-conditioned orientation, approved texture policy pass, handled-clean
  diversity pass, and visual contact sheet spot-check without the old
  self-shadow artifacts.
- Geometry result: strict real-vs-synth geometry passes. Aggregate deltas are
  `box_area +0.124`, `box_width +0.159`, `box_height +0.017`, and
  `box_aspect +0.038`; all class-level area/width/height/aspect checks are
  inside the gate after square-orientation calibration.
- Model result: matched pure-synth `yolo26n` seed0 at `imgsz=416` fails TSTR
  despite strong synthetic self-eval. Full real val/test mAP50-95 are
  `0.0171` / `0.0208`; clean-visible real val/test are `0.0410` / `0.0637`;
  synthetic self-eval is `0.7917`. Versus 768x640 v2, deltas are `-0.0377`
  full val, `-0.0587` full test, `-0.0669` clean-visible val, `-0.0580`
  clean-visible test, and `+0.0236` synthetic self-eval. Clean-visible test
  has 9 classes beyond the `0.02` per-class drop guard, worst `USD_10=-0.1436`.
- Next use: do not scale this root. Use it as a contrast case for domain-gap
  debugging: compare real-vs-synth luma, texture/background, camera distance,
  crop sharpness, and pose/orientation effects that strict box geometry did not
  capture.
- Real-only p48/bg48 is the best expanded real-only control on the mined
  held-out rare/edge slice, but it is not promotable and still misses the p24
  clean gate.
- Duplicate-only rare oversampling, p96 scaling, and background removal are not
  sufficient repairs.
- Mined-real browser default stack gets `9/17` strict perfect cases on the
  scoreable stress set.
- The old staged overlap detector gets `12/17` perfect on the same mined-real
  browser set and improves fan/dense-overlap recall, but shifts errors into
  overcount/value noise.
- Current WebGL overlap-stack ablation is rejected in deploy terms. Exported
  `webgl_ablation_overlap_stack_from_clean` gets mined-real dense-overlap
  final same recall `3/9` versus default `5/9` and old-overlap `8/9`, and
  synthetic hard-negative browser cases still all fail. Raw prop FP is also
  worse than default (`43/32` and `12/8` detections at `conf=0.05`).
- Old-overlap prop-diverse hardneg8 repair is diagnostic only. Full one-epoch
  repair cuts raw prop FPs (`11 -> 6` detections on the 8 prop-diverse frames)
  but overcounts badly in browser (`synthetic_overlap_stack pred=21/6`;
  mined-real fan `pred=13/8`, thin-edge `pred=18/10`). A gentler 32-batch
  repair still overcounts synthetic overlap (`pred=11/6`) while only partly
  reducing hard-negative FPs.
- Old-overlap browser NMS `0.65` is a useful calibration result. It keeps the
  old-overlap mined-real strict score at `12/17` but improves aggregate
  stability (`pred 44 -> 39`, abs count error `11 -> 8`, abs KHR error
  `107000 -> 92000`) versus default NMS `0.85`. NMS `0.65` plus
  disagreement/unclassified rejection clears all three synthetic hard-negative
  browser cases, but mined-real drops to `11/17` and loses fan/thin matches;
  unclassified-only leaves wallet/sticky false positives, disagreement-only
  leaves card/phone false positives.
- Existing broad real-fragment classifier diagnostic:
  `configs/cashsnap_two_stage_broad_realfrag_browser_stack.json` plus
  old-overlap/NMS `0.65` is rejected. Without disagreement rejection, synthetic
  card/phone and wallet/sticky still false-positive; with rejection, synthetic
  hard negatives clear but positive stress collapses to near zero recall.
- Background-only broad real-fragment gate is also diagnostic only. The browser
  app can reject configured fragment classes and optionally run NMS before
  rejection; `configs/cashsnap_two_stage_oldoverlap_broad_realfrag_bg_reject_browser_stack.json`
  uses this to reject `background` while preserving detector denominations.
  It clears synthetic hard negatives (`failures 2 -> 0`, perfect `1/9 -> 3/9`)
  but does not beat plain old-overlap NMS on mined-real (`12/17` perfect,
  `same 31/35`, `any 32/35`, abs count `7`, abs KHR `152000`, one rejected
  real proposal). Use it as proof that proposal-level background evidence
  exists, not as a default deploy stack.
- Binary proposal-gate diagnostics are promising. `scripts/build_proposal_gate_dataset.py`
  auto-labels old-overlap detector proposal crops as `banknote`/`background`
  by YOLO-label IoU, adds random safe background crops, and writes ImageFolder
  data. V1 (`mobilenet_v3_proposal_gate_oldoverlap_v1_e6`) clears synthetic
  hard negatives and improves mined-real to `13/17` perfect, abs count `5`,
  abs KHR `26000`, but lowers recall (`same 30/35`). V2
  (`mobilenet_v3_proposal_gate_oldoverlap_synpartial_v2_e6`) adds WebGL
  clean/overlap/fan/hand/thin positive proposals; it keeps synthetic
  hard-negative failures at zero and gets the best mined-real aggregate so far:
  `13/17` perfect, `pred=35`, `same 31/35`, `any 32/35`, abs count `4`, abs KHR
  `21000`. Still diagnostic: the validation set is tiny, and real no-note plus
  reviewed real partial hard positives are missing. V2 validation errors are
  concentrated in real proposal crops (`cashsnap_v1` USD banknotes and mined
  weak `KHR_50000`) being rejected as background, so the next repair is hard
  real positives plus reviewed real no-note props, not a simple threshold sweep.
  Failure review pack: `data/review/proposal_gate_synpartial_v2_failures/`
  (`9` background->banknote, `11` banknote->background high-confidence errors).
- Full-real-only is a useful negative/proposal reference, not a browser base.
  It is cleanest on synthetic prop false positives (`4/32` and `1/8`
  detections at `conf=0.05`, and zero detections by `conf=0.18`), but browser
  replay regresses mined-real same-class recall versus default/old-overlap and
  collapses positive synthetic stress. Use it to shape restraint, not to replace
  the current detector.
- Disagreement/unclassified browser veto flags are diagnostic only. They clear
  some synthetic hard negatives but do not improve mined-real stress.

## Hard Blockers

- No protected/promoted real P1 fan/overlap/hand stress benchmark yet.
- No real repeated-denomination fan labels for same-class counting/fusion.
- No reviewed real no-note/non-banknote paper capture inventory.
- No real mixed USD+KHR rare/common stack captures, especially with
  `KHR_50000`.
- Current model schema and active WebGL cutout bank are incomplete relative to
  full USD/KHR. Verify raw/current/active/model coverage with the taxonomy
  coverage check before saying a class exists or is missing.
- Clean-real train split still has thin rare anchors:
  `KHR_20000=14` and `KHR_50000=15` unique train images.
- Square mixcam strict geometry is now a negative control: it passed geometry
  and synthetic self-eval but failed real transfer versus 768x640 v2. The next
  clean-base repair is protected class-balanced curriculum search plus non-box
  domain features, not larger square scale or brute weak-class oversampling.

## Commands

Keep this list short. Use script help or archive history for older workflows.

Scorecard:

```powershell
rl python scripts\check_synthetic_dataset_scorecard.py
```

Currency coverage:

```powershell
rl python scripts\check_currency_taxonomy_coverage.py
```

Readiness:

```powershell
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
```

Crop visual gap:

```powershell
rl python scripts\audit_yolo_crop_visual_domain_gap.py --data <mixed-real-synth-data.yaml> --split train --pad-frac 0.02 --gate-preset clean_dynamic_range_v1
```

Real geometry stress slices:

```powershell
rl python scripts\build_real_geometry_stress_slices.py --data configs\webgl_ablation\cashsnap_webgl_clean_768topdown_squareclassdiverse_no_square_khr20000_blend_puresynth_realval_v1.yaml --out-dir runs\cashsnap\real_geometry_stress_slices_v1
```

Background FP guardrail:

```powershell
rl python scripts\probe_yolo_background_false_positives.py --model base=<weights.pt> --data <real-eval-data.yaml> --split val --split test --conf 0.05 --imgsz 416 --batch 1 --json-out runs\cashsnap\background_fp_<run>.json
```

Transfer guardrail summary:

```powershell
rl python scripts\check_yolo_transfer_guardrails.py --compare full_val=<compare.json> --compare full_test=<compare.json> --compare clean_visible_val=<compare.json> --compare clean_visible_test=<compare.json> --background-fp-json <background_fp.json> --baseline-label <base> --candidate-label <candidate> --json-out runs\cashsnap\transfer_guardrails_<run>.json
```

Low-churn class-aspect repair:

```powershell
rl python scripts\build_webgl_class_aspect_repair_config.py --selection-mode current_plus_candidates --max-new-rows-per-class 2 --out-list configs\generated_lists\webgl_ablation\<name>_train.txt --out-config configs\webgl_ablation\<name>.yaml
```

Trainable-candidate suite:

```powershell
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
```

Occluder policy:

```powershell
rl python scripts\check_webgl_occluder_policy.py --root <root> --expected-policy no_hand --forbid-hand-occluders
```

Current clean-curriculum root checks:

```powershell
rl python scripts\check_webgl_note_condition_diversity.py --root data\synthetic\cashsnap_webgl_clean_base_curriculum_probe_v1 --allow-missing
rl python scripts\check_webgl_trainable_candidate_gate.py --root data\synthetic\cashsnap_webgl_clean_base_curriculum_probe_v1 --require-recipe webgl_clean_base_v1 --train-views detect,fragment,obb --require-scene-mode clean --require-asset-side-policy any --require-camera-profile phone_auto
```

Mined real bridge:

```powershell
rl python scripts\check_mined_real_review_quality.py
rl python scripts\build_mined_real_scoreable_dataset.py --min-images 17 --min-stress-images 17 --min-boxes 35
```

Capture gaps:

```powershell
rl python scripts\check_capture_requirements.py --json-out runs\cashsnap\capture_requirements_latest.json --shot-list-out runs\cashsnap\real_capture_shot_list_latest.md
```

YOLO progress:

```powershell
rl python scripts\summarize_yolo_progress.py --run runs\cashsnap\<run-name> --expected-epochs 50
```

## Laptop Runtime

Machine profile:

- Lenovo 82Y9 laptop
- Ryzen 5 7640HS, 6 cores / 12 logical processors
- 16 GB RAM
- RTX 4060 Laptop GPU, 8 GB VRAM

Current operating posture:

- Heavy training/rendering should use `scripts/run_with_headroom.py` when
  practical. The wrapper refuses caps above 95%, lowers child priority, waits
  for initial headroom, and treats free RAM as preflight/runtime pressure.
- `scripts/profile_system.py --out runs\cashsnap\local_hardware_profile_latest.json`
  prints the current laptop profile and selected train/val/render defaults.
- Current cap posture: using up to 95% max CPU/RAM/GPU/VRAM is acceptable when
  needed, but never set any cap above 95%; resume around 88%. GPU utilization
  is no longer a pause signal by default; pass `--throttle-gpu-util` only for
  jobs where GPU contention hurts interactive use.
- If the wrapper cannot find initial headroom, do not wait indefinitely. Reduce
  batch/workers, lower RAM pressure, relax only safe preflight floors, or split
  the run into smaller chunks.
- Prefer GPU for PyTorch/Ultralytics training/inference when VRAM is available;
  pass `--device 0` instead of letting long jobs drift onto CPU.
- RAM/CPU pressure is still the main laptop bottleneck. Current 416 YOLO probe
  posture is train `batch=64`, `workers=0`, `device=0`, `cache=false`; eval
  posture is `batch=64`, `workers=2` unless interactive CPU pressure says to
  trade speed for `workers=0`.
- Browser smoke/render jobs can run in parallel only through the renderer's
  chunked paths, which isolate browser/cache state per worker process.
- WebGL scale should use Node batch rendering (`--renderer-batch-size 32`),
  `--render-jobs 2`, and `--check-jobs 4` first. Older per-variant Edge/Node
  launch loops are for debugging only, and shared-browser reuse is opt-in
  because it did not beat normal batched rendering on clean-topdown benchmarks.
  Keep smoke checks in subprocess mode by default; the in-process validator is
  available for debugging/API reuse but was slightly slower on the laptop.
- Browser/deploy smoke remains a separate CPU/browser harness bottleneck. Keep
  `run_browser_smoke_cases.py` sequential by default; use `--jobs 2` only for
  timed experiments and expect sequential retry on CDP/no-summary failures.

## Repo Hygiene

- Work on `master` unless the user asks for a branch.
- Keep generated training and browser outputs under ignored `runs/`.
- Keep generated synthetic roots under ignored `data/synthetic/`.
- Keep active planning in this `model.md`; archive old model history under
  `docs/archive/`.
- Do not add new active model-planning docs under `docs/`.
- Leave unrelated dirty/untracked files alone unless they support the current
  task.

## Research Sources

- `docs/research/CashSnap_Fact_Checked_Dataset_Strategy.pdf`
- `docs/research/What Makes a Dataset Perfect for Synthetic Data Pipelines.pdf`
- USD denominations: https://www.uscurrency.gov/denominations
- KHR banknotes: https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php
