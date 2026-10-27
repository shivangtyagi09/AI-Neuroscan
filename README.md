# NeuroScan PACS — Clinical Brain MRI Analysis & AI Review Workbench

A clinical-style web workbench for reviewing brain MRI scans, calculating real pixel radiomics, running K-means unsupervised tissue segmentation, executing forward-chaining expert reasoning, and performing preliminary AI-assisted radiological reviews using Google's multimodal **Gemini Flash** models (`gemini-3.8-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`).

---

## 🚀 Quick Start (Zero Setup Required)

You have two ways to run this project:

### Option 1: Standalone Single File (Easiest)
- Double-click **`NeuroScan-AI.html`**.
- It opens immediately in Chrome, Edge, Safari, or Firefox with **zero installation, zero dependencies, and no build tools**.

### Option 2: Modular Project
- Open **`index.html`** directly in any browser (it loads `app.js` and `ai-lab.js`).
- Or serve via a local web server:
  ```bash
  npx serve .
  # or
  python -m http.server 8000
  ```

---

## 🌟 Clinical PACS & Analysis Features

1. **Direct Brain MRI Ingestion**:
   - Supports user data in `.png`, `.jpg`, `.jpeg`, `.jfif`, `.webp`, `.bmp`, `.avif`, and `.pdf` formats.
   - **One-Click Pre-Calibrated Benchmarks**: Instant 1-click loading of realistic anatomical axial MRI benchmark studies:
     - **Normal Axial T1-SE**: Clear anatomical cortical gyri, dark ventricles, bright white matter.
     - **Axial T2 Hyperintense**: Bright CSF, focal hyperintense lesion in right periventricular white matter.
     - **Axial FLAIR Plaque**: Attenuated CSF with distinct focal hyperintense plaque.
   - Batch upload support with interactive filmstrip carousel and thumbnail switcher.

2. **PACS-Grade DICOM Viewport**:
   - **4-Corner Telemetry Overlay**: Matrix dimensions, study metadata, dynamic zoom percentage, and calibrated Rec.601 FOV indicator.
   - **Real-Time Cursor Pixel Inspector**: Live tracking of mouse coordinates `(X, Y)` and exact signal luminance in HU (`0–255 HU` scale) under the cursor.
   - **Interactive Window/Level Colormaps**: Cycle between Standard Grayscale, Inverted Film (Bone/Negative), Thermal False-Color Heatmap, and High Contrast.
   - **DICOM Crosshair HUD**: High-precision alignment crosshair overlay.
   - **Smooth Pan & Zoom Engine**: Drag-to-pan when zoomed in, mouse-wheel zoom, and one-click zoom reset.

3. **True Raster Pixel Radiomics (Client-Side HTML5 Canvas Engine)**:
   - **Mean Signal Luminance**: Grayscale intensity (0–255 HU equivalent).
   - **Contrast Dispersion ($\sigma$)**: Population standard deviation.
   - **Mean/$\sigma$ Intensity Ratio**: Normalized dispersion ratio across raster pixels.
   - **24-Bin Grayscale Histogram**: Frequency distribution color-coded across anatomical tissue bands (Air/Bone, CSF, Gray Matter, White Matter, Hyperintensities).
   - **One-Click Data Exports**: Export quantitative radiomic profiles to CSV and JSON.

4. **🧠 AI Lab — Syllabus Concepts (Artificial Intelligence, 702CO0C032)**:
   - Runs 100% locally in the browser with zero external APIs:
     - **Unit 5 (Learning)**: Unsupervised **K-means intensity clustering** ($k=2,3,4,5$) segmenting brain tissue into distinct clusters (CSF, Cortical Gray Matter, Subcortical White Matter, Hyperintensities). Includes interactive cluster isolation.
     - **Units 3 & 6 (Knowledge Representation & Expert Systems)**: **Forward-chaining inference engine** evaluating facts against clinical heuristic rules ($R_1 \dots R_7$) to output a transparent audit trace and actionable clinical directives.

5. **Live Gemini Multimodal AI Consultation**:
   - Integrates directly with Google's latest **`gemini-3.8-flash`**, **`gemini-2.0-flash`**, and **`gemini-1.5-flash`** via Google AI Studio.
   - Resilient multi-tier fallback ensuring connection across any valid API key.
   - Generates structured radiological consultation impressions.
   - **Clinical Consultation Report**: Formatted hospital-grade print view (`Ctrl + P` or Print Report button) with embedded scan slice and quantitative metrics.

6. **Interactive Co-Pilot Chat Assistant**:
   - Grounded in uploaded MRI data, calculated radiomic metrics, and pulse sequence physics (T1, T2, FLAIR).
   - Reliable quick prompt pills and natural language diagnostic Q&A.

---

## 🔑 Setting Up Your Gemini API Key

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Open NeuroScan and click **`Gemini API: Offline`** in the top navigation bar.
3. Paste your key and click **"Test Key"** to verify connection.
4. Click **"Save Key"**. The key is stored securely in your private browser storage.

---

## 📁 Project Structure

```text
nueroscan/
├── NeuroScan-AI.html       # Standalone all-in-one edition (complete embedded app)
├── index.html              # Modular application markup & PACS workstation layout
├── app.js                  # Radiomics engine, DICOM viewport, chat, & Gemini vision
├── ai-lab.js               # K-means tissue segmentation & forward-chaining expert system
└── README.md               # System documentation & user guide
```

---

## ⚖️ Medical Disclaimer
*NeuroScan PACS is engineered for educational, academic, and clinical research workflow demonstration only. It is not an FDA-cleared diagnostic medical device. It should never substitute for licensed physician evaluation or board-certified clinical radiology interpretation.*
