# CashSnap YOLO26n Baseline

## Run

- Model: `yolo26n.pt`
- Dataset: `data/cashsnap_v1/data.yaml`
- Command: `python scripts/train_yolo.py --model yolo26n.pt --data data/cashsnap_v1/data.yaml --epochs 20 --imgsz 416 --batch 16 --name yolo26n_baseline_e20_i416`
- Saved run: `C:\Users\Venom\runs\detect\runs\cashsnap\yolo26n_baseline_e20_i4162`
- Best weights: `C:\Users\Venom\runs\detect\runs\cashsnap\yolo26n_baseline_e20_i4162\weights\best.pt`

## Validation

Final validation on the curated validation split:

- Precision: `0.930`
- Recall: `0.913`
- mAP50: `0.962`
- mAP50-95: `0.920`

Weak validation classes:

- `KHR_20000`: recall `0.795`, mAP50 `0.868`
- `KHR_50000`: recall `0.591`, mAP50 `0.841`

## Test Split

Held-out test split:

- Precision: `0.958`
- Recall: `0.903`
- mAP50: `0.972`
- mAP50-95: `0.935`

Weak test classes:

- `KHR_20000`: recall `0.567`, mAP50 `0.884`
- `KHR_50000`: recall `0.524`, mAP50 `0.912`

## Fan Photo Check

Image: `D:\Download\Banknotes_of_Cambodian_Khmer_Riel.jpg`

- At `imgsz=416`, `conf=0.25`: no detections.
- At `imgsz=640`, `conf=0.05`: 3 low-confidence detections, not enough for counting.

Conclusion: the baseline is good for isolated/normal banknote photos in the curated datasets, but it is not ready for the real target scene: fanned, overlapping, hand-occluded KHR notes. The next dataset step should be a small real fan/overlap validation set plus targeted synthetic fan augmentation from clean note crops.

## Roboflow Model Comparison

Checked the relevant Roboflow Universe projects:

- `Khmer-US-currency`: public page shows `Model 1`, but direct hosted endpoints tested at versions `1`, `2`, `3`, and `10` returned `403 Forbidden` with the local key.
- `Cambodia Currency Project`: public page shows `Model 1`; direct hosted endpoint `cambodia-currency-project/2` was accessible.
- `KHMER SCAN`: public page shows `Model 0`; direct hosted endpoints tested at versions `1` and `2` returned `403 Forbidden`.

The accessible Roboflow endpoint, `cambodia-currency-project/2`, returned zero predictions on:

- the real fan photo, even at `confidence=5`
- 155 held-out KHR test images at `confidence=25`
- several images originating from the Cambodia Currency Project subset, even with confidence values from `0` through `25`

Because the endpoint returns no predictions even on its own-source images, treat it as not operational or not comparable through the direct hosted API path. It should not be interpreted as a reliable quality benchmark for the underlying Roboflow training run.

On the same 155-image KHR test subset for the six classes shared with the accessible Roboflow endpoint (`KHR_500`, `KHR_1000`, `KHR_5000`, `KHR_10000`, `KHR_20000`, `KHR_50000`), the local YOLO26n baseline at `conf=0.25` produced:

- True positives: `148`
- False positives: `8`
- False negatives: `9`
- Precision: `0.949`
- Recall: `0.943`
- F1: `0.946`
