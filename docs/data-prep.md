# CashSnap Data Prep

## Downloaded Buckets

- `data/raw_datasets/hf_usd_side_coco_annotations/`: Hugging Face USD Side Detection Dataset. This is the main downloadable USD detection seed and still needs label remapping from front/back/authentic/counterfeit variants into denomination-only classes.
- `data/reference/usd_wikimedia/`: clean USD front/back reference scans from Wikimedia Commons, intended for synthetic generation experiments. Public release must follow U.S. currency reproduction rules.
- `data/reference/khr_nbc/`: official National Bank of Cambodia reference images from the Banknotes in Circulation page. Use for visual reference and synthetic experiments only after checking reuse rights; some notes may contain SPECIMEN watermarks.

## Manual Or Blocked Sources

See `manifests/blocked_or_manual_sources.csv`.

Roboflow datasets are useful but need manual export or an API key:

- Khmer-US-currency
- Cambodia Currency Project
- KHMER SCAN

Recommended export format is YOLOv8/YOLO26 TXT or COCO JSON. After export, place each source under `data/raw_datasets/roboflow_<source_name>/`.

Numista is not bulk-downloaded because licensing appears mixed/unclear. Use it for research and source discovery until rights are verified.

## Prep Rules

- Keep raw downloads unchanged.
- Put derived files under `data/processed/`.
- Keep synthetic generated scenes separate from real validation/test photos.
- Do not train directly on USD raw reference scans if a public dataset release is planned; generate size-compliant, one-sided derivative images and document compliance.
- Do not train heavily on KHR images with SPECIMEN watermark unless the watermark is removed or strongly augmented away.
