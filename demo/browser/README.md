# CashSnap Browser Probe

Static ONNX Runtime Web demo for the current two-stage old/common KHR diagnostic stack.

Run from the repository root so the page can fetch the model artifact:

```powershell
lr python -m http.server 8787
```

Open:

```text
http://localhost:8787/demo/browser/
```

The demo reads `configs/cashsnap_two_stage_oldcommon_browser_stack.json`, then loads:

- Detector: `runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.onnx`
- Fragment classifier: `runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.onnx`

It fuses low-confidence detector proposals with the KHR 1k/5k/10k/20k crop classifier and uses detector-confidence NMS. It is suitable for browser plumbing and review hints, not reliable denomination totals yet.
