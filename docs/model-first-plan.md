# CashSnap Model-First Plan

## Goal

Build a banknote detector that can count visible USD and Cambodian Riel notes from one photo, including overlapped and partially visible notes like a hand-held fan of bills.

The first model should detect denominations only. It should not attempt counterfeit detection, exchange-rate conversion, or note-series identification.

## Recommended Training Strategy

Use a two-stage lightweight stack:

1. A small detector proposes visible note regions.
2. A tiny denomination/side classifier re-reads each crop when detector confidence is low or classes are known-confusing.
3. A calibrated fusion/NMS layer combines detector and classifier evidence before counting.

Default detector:
- YOLO26n

Default classifier:
- MobileNetV3-small or similarly small image classifier exported to ONNX/NCNN.

Fallback detector models:
- Next-smallest viable YOLO26 detect variant if YOLO26n is too weak.
- YOLO26n OBB for rotated-box probes once matching labels are available.

Why:
- Training from scratch needs far more data than this project is likely to have.
- A pretrained detector already knows useful visual features such as edges, paper shapes, texture, color contrast, and object boundaries.
- Cambodia's real use case mixes USD and KHR in the same scene, so one combined detector is simpler and less fragile than separate USD/KHR models.
- YOLO26 is the current model family for this project because Ultralytics positions it for edge and low-power deployment, with NMS-free inference and export support that should help browser/mobile experiments.
- The local environment has loaded `yolo26n.pt` as a detect model and `yolo26n-obb.pt` as an OBB model, so OBB is not blocked by weights availability.
- The partial/occluded-note target is not ordinary OCR. It is denomination recognition from visible visual evidence. A tiny crop classifier is a better fit than trying to run text OCR on worn Khmer/English numerals that may be absent, blurred, or hidden.
- A 2026-05-27 Khmer OCR cue probe with `mer` on the six-box shop-overlap draft found some Khmer-like text fragments but no reliable denomination reads from full-note or strip crops. Treat OCR as an auxiliary cue to test after detection/crop localization, not as a replacement for visible-region detection and denomination classification.

Avoid fine-tuning on KHR only after starting from USD data unless USD samples remain in the training set. Otherwise the model can weaken on USD classes.

Do not continue first-pass training from the old e2/e4 CashSnap checkpoints. Treat them as contaminated comparison baselines because their synthetic/reference KHR data likely mixed current notes with older, rare, specimen-heavy, collector-like, or low-priority variants.

## V1 Classes

Use denomination-only classes:

- USD_1
- USD_5
- USD_10
- USD_20
- USD_50
- USD_100
- KHR_500
- KHR_1000
- KHR_2000
- KHR_5000
- KHR_10000
- KHR_20000
- KHR_50000

Defer KHR_100 and KHR_100000 unless there is enough real local data. KHR_100 is low value, and KHR_100000 may be less common in everyday small-shop counting.

## Partial And Overlapped Notes

The target is visible note instance detection.

For overlapped notes, annotate the visible footprint of each identifiable note, not the imagined full hidden bill. If a note is mostly hidden but still has enough denomination-specific color, pattern, numeral, portrait, or layout cues to identify it confidently, label it.

Recommended rule:
- Label if the denomination is confidently identifiable and roughly 30 percent or more of useful note area is visible.
- Do not label tiny edge strips, plain corners, or hidden parts where the denomination is guessed from context.
- If a human annotator cannot identify the denomination without looking at neighboring notes, put it in an uncertain bucket instead of training.

For fan-style images, this means many boxes will be narrow and vertical because only a slice of each note is visible. That is acceptable if the visible slice contains enough distinctive detail.

## Data Mix

The training set should intentionally include:

- Single-note images for every class
- Mixed USD + KHR images
- Fan-style overlapped notes
- Table/counter spreads with touching notes
- Front and back sides
- Current first-pass KHR note designs under denomination labels
- Worn, folded, wrinkled, and partly blurred notes
- Different backgrounds and lighting
- Negative images with receipts, cards, paper, wallets, phones, and empty counters

Legacy, low-priority, specimen-heavy, watermarked, unclear, or collector-like KHR designs should stay out of first-pass synthetic generation unless the experiment is explicitly testing legacy support.

## Current KHR Synthetic Source

Run `scripts/curate_reference_images.py` to refresh NBC issue buckets, then run `scripts/build_current_khr_cutout_bank.py --clean` to create `data/asset_candidates/khr_nbc_current_cutout_bank_v1/`. That bank is the default source for `scripts/generate_synthetic_fan_dataset.py` so fresh probes do not silently mix older NBC reference images into current-KHR training.

For normal YOLO26n probes, keep the default detect labels. For YOLO26n OBB probes, generate a separate dataset with `--label-format obb --save-visible-masks` so rotated boxes and visible masks stay paired with the exact same synthetic geometry.

## Dataset Sources

Use public datasets only as seeds:

- Khmer-US-currency: useful mixed USD/KHR seed, but missing some USD denominations.
- KHMER SCAN: useful small KHR supplement, but needs cleanup and likely removal/remapping of the generic objects class.
- Cambodia Currency Project: useful KHR detection seed with 552 images, 7 classes, YOLO11/YOLO11n tags, and CC BY 4.0 license. Classes are 100_Riel, 500_Riel, 1000_Riel, 5000_Riel, 10000_Riel, 20000_Riel, and 50000_Riel.
- Hugging Face USD Side Detection Dataset: useful USD seed, but labels must be collapsed from front/back and authentic/counterfeit variants into denomination-only USD classes.
- Asian Currency Detection/Thesis Roboflow sources include some Cambodian classes, but local audit shows they only cover `KHR_100`, `KHR_1000`, and `KHR_5000`; they cannot solve `KHR_10000`, `KHR_20000`, or `KHR_50000`.
- National Bank of Cambodia Banknotes in Circulation: official reference for KHR front/back images, note sizes, and issue dates. Use it for class/version planning and visual reference, not as the main training dataset unless usage rights and image quality are checked.

Custom KHR phone photos are still expected because public KHR data is incomplete and may not cover old/new designs or fan-style counting scenes.

## Current Direction Check

The current best shop-overlap diagnostic is not a larger synthetic detector. It is:

- real-cutout YOLO proposal detector: `runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.pt`
- focused old/common KHR real-box classifier: `runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.pt`
- calibrated fusion: detector overrides at about `det_conf >= 0.17`, then class-agnostic NMS ranked by detector confidence

On the draft shop overlap label set, this gives 6 predictions for 6 draft notes, 6/6 any-class region coverage, and 5/6 same-class recall. Treat this as a diagnostic direction, not a production result, because it is calibrated on one draft image.

Raw detector refresh on 2026-05-27 reinforces the same conclusion. Against the six-box `real_overlap_0003` draft labels, `yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8` reaches 5/6 same-class and 6/6 any-class at `416/conf=0.05`, but with 14 predictions and heavy class duplication; at `416/conf=0.03` it reaches 6/6 same-class only by overpredicting 23 boxes. The cleaner alpha `yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8` tops out at 3/6 same-class in the same sweep. Keep using detector proposals plus calibration/classifier evidence instead of treating raw detector thresholding as solved.
