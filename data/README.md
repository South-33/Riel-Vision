# Data

Use root `model.md` for the live data ranking.

Current rule of thumb:

- Canonical assets: `numista_raw/`, `asset_candidates/numista_current_cutout_bank_v1/`.
- Clean validation: `cashsnap_v1/`.
- Evaluation only: `real_fan_benchmark/`.
- Human review and captures: `review/`, `inbox/`.
- Conditional/reference: `raw_datasets/`, `backgrounds/`, `curated/`, `reference/`, `picwish_upload_batches_cashsnap_khr_v1/`.

Generated scratch datasets should not live here unless they are tied to the current plan in `model.md`.
