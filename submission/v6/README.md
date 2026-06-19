# Riel Vision: Presentation & Live Demo

**GitHub Repository:** [https://github.com/South-33/Riel-Vision](https://github.com/South-33/Riel-Vision)

This folder contains the complete submission package for **Riel Vision**, a client-side AI banknote counter for mixed Cambodian Riel (KHR) and US Dollar (USD) cash transactions.

---

## 🚀 How to Run

1. **Double-click the batch file in this folder:**
   `autostart.bat`
2. It will start a local server and automatically open the interactive demo in your default browser at:
   **[http://127.0.0.1:8000/Presentation/RielVision.html](http://127.0.0.1:8000/Presentation/RielVision.html)**

> [!IMPORTANT]
> **Why is a local server required?**
> Modern browsers block loading WebGL 3D bill textures and local ONNX model files directly from the filesystem (`file://`) due to CORS security policies. The script starts a lightweight local server to bypass these restrictions.

---

## 📁 Folder Contents

This package is fully self-contained:
*   **`autostart.bat`**: The launcher script.
*   **`Presentation/RielVision.html`**: Interactive presentation slides and the live in-browser prototype.
*   **`PRD/RielVisionPRD.html`**: Product Requirement Document with project requirements, audited datasets, and performance benchmarks.
*   **`models/rielvision.onnx`**: The compiled YOLOv8 object detection model (9.8 MB). Bounding box inference runs 100% locally in your browser.
*   **`images/`**: Audited real photos, synthetic renders, and illustrations.
*   **`js/`**: Script logic (`demo.js` for model loading/inference and `threejs_scene.js` for the 3D WebGL arrangement).
