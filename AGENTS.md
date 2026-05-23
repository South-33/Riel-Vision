This is the project's AGENTS.md

## Notes
- Phase 1 scope is documentation for CashSnap: a USD + KHR banknote denomination counter using computer vision and a Hugging Face API flow; counterfeit detection is intentionally out of scope.
- Current technical direction is a lightweight object detector, currently YOLO26n first with YOLO11n/YOLOv8n fallbacks, using denomination-only labels and custom KHR phone-photo data expected because public KHR datasets are incomplete.
- Data prep has a verified local YOLO dataset at `data/cashsnap_v1/` with 13 classes and 9,048 boxes; `KHR_20000` and `KHR_50000` are the weakest classes and need more real/synthetic examples before strong claims.
- YOLO runs must be written under the repo's ignored `runs/` directory on D; avoid `C:\Users\Venom\runs` because the user profile drive is space-constrained.
- Best fan-photo checkpoint so far is dense synthetic v3 (`runs/ultralytics_migrated/detect/runs/cashsnap/yolo26n_messy_synth_v3_e8_i416/weights/best.pt`): still weak, but beats baseline/v2/v4 on the real fanned KHR photo.
- Use `ideas.md` as the short living board for high-value CashSnap experiment ideas and results; keep it curated, not append-only.
- Background removal can be automated for free using `scripts/process_picwish_batches.py` with the `picwish` PyPI library; keep concurrency under 15 (using `asyncio.Semaphore(10)`) with short sleeps to avoid Cloudflare rate blocks.
- Current best cutout set comes from scoring PicWish and BEN2 outputs, then selecting via `scripts/select_best_cutouts.py`; use the stricter `data/asset_candidates/rare_pristine_asset_bank_v1/` subset for first synthetic probes.
- First rare-cutout synthetic stage should be clean curriculum (`data/synthetic/khr_rare_pristine_clean_v1/`): 1-3 notes, no synthetic fingers, no strip/fan chaos; only scale up to harder fan/overlap after this probe is evaluated.
