This is the project's AGENTS.md

## Notes
- Phase 1 scope is documentation for CashSnap: a USD + KHR banknote denomination counter using computer vision and a Hugging Face API flow; counterfeit detection is intentionally out of scope.
- Current technical direction is a lightweight object detector, currently YOLO26n first with YOLO11n/YOLOv8n fallbacks, using denomination-only labels and custom KHR phone-photo data expected because public KHR datasets are incomplete.
- Data prep has a verified local YOLO dataset at `data/cashsnap_v1/` with 13 classes and 9,048 boxes; `KHR_20000` and `KHR_50000` are the weakest classes and need more real/synthetic examples before strong claims.
- YOLO26n baseline `yolo26n_baseline_e20_i4162` scores high on curated splits but fails the real fan photo at normal confidence; prioritize fan/overlap validation data and targeted synthetic augmentation before product claims.
