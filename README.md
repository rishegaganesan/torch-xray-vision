# TorchXRayVision Flask UI + REST API

A minimal clinical-style Flask web app that serves pretrained models from **mlmed/torchxrayvision**.  
It lets users upload chest X-ray images (JPG/PNG/DICOM), returns predicted pathologies with latency metrics, and generates a Grad-CAM heatmap overlay.

## Features
- Warm-loads a TorchXRayVision pretrained model (default: `densenet121-res224-all`) at app start and reuses it.
- Accepts **JPG/PNG/DICOM** (≤ 10 MB). DICOM is **de-identified** before storing/processing.
- Preprocesses to grayscale, normalizes per TorchXRayVision, resizes to 224×224.
- Returns **class probabilities** for known labels.
- Includes **latency metrics**: preprocess and inference times.
- Provides **Grad-CAM** heatmap for the top predicted class.
- **Browser UI** (upload form, results table, side-by-side heatmap) and **JSON REST API**.

## Quickstart (CPU)
```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

#  Run the server
export FLASK_ENV=production
export MODEL_NAME=densenet121-res224-all  
python app.py

# Open: http://127.0.0.1:5000
