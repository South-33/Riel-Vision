# Synthetic Harness Runbook

This is the current short path for CashSnap partial-banknote synthetic work. Keep generated data under ignored `data/` paths and train/evaluate through the existing headroom scripts.

## 1. Build A Clean Scan Cutout Bank

```powershell
python scripts/build_numista_cutout_bank.py --out data/asset_candidates/numista_current_cutout_bank_v1 --clean
python scripts/audit_cutout_bank.py --bank data/asset_candidates/numista_current_cutout_bank_v1
```

Review:

- `data/asset_candidates/numista_current_cutout_bank_v1/contact_sheet.jpg`
- `data/asset_candidates/numista_current_cutout_bank_v1/audit/suspect_contact.jpg`
- `data/asset_candidates/numista_current_cutout_bank_v1/audit/suspects.csv`

Do not treat a red/pink note as a specimen automatically. KHR 500 designs naturally contain large red areas; visual audit is the authority.

## 2. Mine Real Background Patches

Use labeled real images to extract square patches that avoid every YOLO note label, then optionally reject candidates that the current detector still recognizes as banknotes. These patches can reduce the synthetic-to-real gap, but they are not trustworthy until the contact sheet is visually clean.

```powershell
python scripts/extract_yolo_background_patches.py `
  --data data/cashsnap_v1/data.yaml `
  --out data/backgrounds/cashsnap_v1_no_note_patches `
  --count 500 `
  --output-size 640 `
  --seed 20260530 `
  --reject-model runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt `
  --reject-conf 0.02 `
  --reject-imgsz 640 `
  --clean
```

Review:

- `data/backgrounds/cashsnap_v1_no_note_patches/contact_sheet.jpg`
- `data/backgrounds/cashsnap_v1_no_note_patches/manifest.csv`

If the contact sheet includes accidental note fragments, do not use that bank for training. Lower `--reject-conf`, lower `--max-label-overlap-frac`, raise `--box-pad-frac`, or switch to a true note-free background source.

## 3. Generate A Small 2.5D Smoke Dataset

```powershell
python scripts/generate_synthetic_fan_dataset.py `
  --out data/synthetic/smoke_default_scan_v1 `
  --count 32 `
  --image-size 416 `
  --seed 532 `
  --clean `
  --min-notes 3 `
  --max-notes 8 `
  --layout-modes "tight_fan,fan" `
  --drop-unknown-denom-labels `
  --note-shadow-prob 0.8 `
  --hand-prob 0.8 `
  --balance-classes `
  --perspective-prob 0.65 `
  --save-visible-crops `
  --crop-include-unknown
```

Add `--background-dir data/backgrounds/cashsnap_v1_no_note_patches` only after the background contact sheet is clean.

Check:

```powershell
python scripts/summarize_synthetic_metadata.py data/synthetic/smoke_default_scan_v1
python scripts/check_yolo_dataset.py data/synthetic/smoke_default_scan_v1/data.yaml
```

Inspect a few images under `images/train/` and verifier crops under `crops/train/`.

## 4. Scale Only After QA Passes

For the first serious scan-based probe, keep it modest:

```powershell
python scripts/generate_synthetic_fan_dataset.py `
  --out data/synthetic/cashsnap_scan_2p5d_fan_v1 `
  --count 2000 `
  --image-size 416 `
  --seed 20260530 `
  --clean `
  --min-notes 3 `
  --max-notes 10 `
  --layout-modes "tight_fan,fan,crossed,row" `
  --drop-unknown-denom-labels `
  --note-shadow-prob 0.65 `
  --hand-prob 0.55 `
  --balance-classes `
  --perspective-prob 0.55 `
  --save-visible-crops `
  --crop-include-unknown
```

Add the reviewed `--background-dir ...` argument when the background bank passes contact-sheet QA.

Use the metadata summary to verify:

- `banknote_unknown` exists but does not dominate denomination labels.
- Class balance is not collapsing toward USD or easy KHR classes.
- Real backgrounds are being sampled from multiple source patches, not one repeated image.
- Visible-area quantiles are closer to partial/fan reality than broad full-note boxes.
- Crops include both identifiable fragments and `banknote_unknown` examples for verifier calibration.

Build the mixed real + scan-synthetic probe lists without silently reusing older synthetic roots:

```powershell
python scripts/build_mixed_cashsnap_probe_lists.py `
  --out data/sampled/cashsnap_scan_2p5d_probe_v1 `
  --cashsnap-train-count 2400 `
  --cashsnap-val-count 600 `
  --extra-synthetic-root data/synthetic/cashsnap_scan_2p5d_fan_v1 `
  --no-default-synthetic `
  --clean
python scripts/check_yolo_dataset.py --data configs/cashsnap_v1_scan_2p5d_probe.yaml
```

## 5. Training Direction

Train detector probes from fresh YOLO26n-family weights, not from contaminated historical e2/e4 checkpoints. Use `scripts/bench_train_with_headroom.py` for long runs so CPU, RAM, and GPU stay under 90%.

Evaluate in this order:

1. Clean validation.
2. Synthetic fan/overlap validation.
3. Reviewed real fan/overlap draft labels.
4. Detector-plus-fragment-verifier fusion.
5. Browser/mobile export smoke only after PyTorch quality improves.

Do not train on `data/real_fan_benchmark/`.
