# CashSnap: Khmer/USD Cash Counting Assistant

CashSnap is an AI-assisted computer vision application designed to help retail vendors and cashiers count mixed US Dollars (USD) and Khmer Riel (KHR) from a single casual photo. The app is designed for retail counter layouts, handling fanning, overlapping bills, hand occlusions, and circulating wear (folded, crinkled, or dirty banknotes).

## Project Overview

In everyday retail contexts, banknotes are rarely isolated. They overlap and form fans. If a standard object detector is used, one bill is often split visually into multiple fragments, leading to double-counting. 

To overcome this, CashSnap implements a two-stage classification and detection pipeline along with a **Fragment Fusion NMS algorithm** to merge fragment proposals and recover physical note counts.

## How to Run the App

### 1. Prerequisites
- Python 3.8 or higher.
- A webcam or camera (optional, for camera input mode).

### 2. Installation
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 3. Running Streamlit
Start the local development server:
```bash
streamlit run app.py
```
This will launch the app in your default web browser (usually at `http://localhost:8501`).

---

## Live Inference vs. Demo Mode

### Live Inference (Hugging Face API)
To run live inference on new images using your custom models:
1. Copy `.env.example` to a new file named `.env`.
2. Fill in your Hugging Face developer token and your target model ID:
   ```text
   HF_TOKEN=your_huggingface_token_here
   HF_MODEL_ID=your_username/your_model_name
   ```
3. Check the **"Use Hugging Face API Live Inference"** box in the app sidebar.

### Interactive Demo Mode (Offline fallback)
If you don't have API keys or a trained model deployed on Hugging Face yet:
1. Select **"Interactive Demo Cases"** in the main panel.
2. Choose from the preloaded retail scenarios (Mixed Overlaps, Circulated Conditions, Environment Lighting Reviews).
3. Adjust the confidence and IoU overlap sliders in the sidebar. The bounding boxes and total values will update dynamically!

---

## System Pipeline & Logic

1. **Camera/Webcam Capture**: User captures cash layout.
2. **Hugging Face API Request**: Image is resized, normalized, and sent to the cloud model.
3. **banknote Detection (YOLO)**: Bounding boxes are proposed for banknotes and visible segments.
4. **Denomination classification (MobileNetV3)**: Region of Interest (RoI) crops are classified to verify denomination on low-confidence boxes.
5. **Fragment Fusion NMS**: Bounding boxes overlapping above the IoU threshold (default `0.85`) are merged into singular physical bill counts.
6. **Result Summary**: The app aggregates unit values by denomination and displays total counts and values separately by currency.

---

## Limitations & Project Disclaimer

- **Counterfeit Detection**: CashSnap is for cash counting assistance only. Counterfeit detection is outside the project scope.
- **Visual Validation Gaps**: Clean, isolated-note detection is highly mature. However, dense overlap and fanned notes remain a complex research challenge.
- **Visual Pipeline Proof**: The WebGL synthetic renderer and label pipelines are strong, but real-world fan/overlap/hand stress validation is still the next major proof step to achieve production-grade counting.
