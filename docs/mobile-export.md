# CashSnap Mobile Export

Current balanced checkpoint:

```powershell
runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt
```

Export commands:

```powershell
lr python scripts/export_yolo.py --model runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt --format onnx --imgsz 416 --simplify --opset 12
lr python scripts/export_yolo.py --model runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt --format ncnn --imgsz 416
```

Observed outputs:

- ONNX: `best.onnx`, 9.3 MB, `(1, 300, 6)` output. Ultralytics ONNX Runtime normal-val smoke reached mAP50 0.976 and mAP50-95 0.921.
- NCNN: `best_ncnn_model/`, 9.2 MB. Ultralytics disables the YOLO26 end-to-end branch for NCNN; rare-overlap smoke reached mAP50 0.631 and mAP50-95 0.406.

TFLite is blocked in the current Python 3.14 environment because Ultralytics requires `tensorflow>=2.0.0,<=2.19.0`, and no compatible TensorFlow wheel is available for this interpreter. Use a separate Python 3.11 or 3.12 environment for TFLite export.
