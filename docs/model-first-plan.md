# CashSnap Model-First Plan

## Goal

Build a banknote detector that can count visible USD and Cambodian Riel notes from one photo, including overlapped and partially visible notes like a hand-held fan of bills.

The first model should detect denominations only. It should not attempt counterfeit detection, exchange-rate conversion, or note-series identification.

## Recommended Training Strategy

Fine-tune one pretrained lightweight object detector on a combined USD + KHR dataset.

Default model:
- YOLO26n

Fallback models:
- YOLO11n or YOLOv8n

Why:
- Training from scratch needs far more data than this project is likely to have.
- A pretrained detector already knows useful visual features such as edges, paper shapes, texture, color contrast, and object boundaries.
- Cambodia's real use case mixes USD and KHR in the same scene, so one combined detector is simpler and less fragile than separate USD/KHR models.
- YOLO26 is the best current first candidate because Ultralytics positions it for edge and low-power deployment, with NMS-free inference and export support that should help browser/mobile experiments.

Avoid fine-tuning on KHR only after starting from USD data unless USD samples remain in the training set. Otherwise the model can weaken on USD classes.

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
- Old and new KHR note designs under the same denomination label
- Worn, folded, wrinkled, and partly blurred notes
- Different backgrounds and lighting
- Negative images with receipts, cards, paper, wallets, phones, and empty counters

## Dataset Sources

Use public datasets only as seeds:

- Khmer-US-currency: useful mixed USD/KHR seed, but missing some USD denominations.
- KHMER SCAN: useful small KHR supplement, but needs cleanup and likely removal/remapping of the generic objects class.
- Cambodia Currency Project: useful KHR detection seed with 552 images, 7 classes, YOLO11/YOLO11n tags, and CC BY 4.0 license. Classes are 100_Riel, 500_Riel, 1000_Riel, 5000_Riel, 10000_Riel, 20000_Riel, and 50000_Riel.
- Hugging Face USD Side Detection Dataset: useful USD seed, but labels must be collapsed from front/back and authentic/counterfeit variants into denomination-only USD classes.
- National Bank of Cambodia Banknotes in Circulation: official reference for KHR front/back images, note sizes, and issue dates. Use it for class/version planning and visual reference, not as the main training dataset unless usage rights and image quality are checked.

Custom KHR phone photos are still expected because public KHR data is incomplete and may not cover old/new designs or fan-style counting scenes.
