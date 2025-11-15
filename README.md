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
```


## 🧠 REST API — Programmatic Access

In addition to the web-based upload UI, this app provides a simple JSON REST API for programmatic use.

### Endpoint
- POST /api/predict

### Description

Send a chest X-ray (JPG, PNG, or DICOM) and receive:
-	Predicted pathologies with probabilities
- Preprocessing and inference latency metrics
- A Grad-CAM heatmap overlay (as base64 PNG)

### Example Request (curl)

```bash

curl -X POST http://127.0.0.1:5000/api/predict \
  -F "file=@/path/to/chest_xray.png"

```

### Example Response

```bash

{
  "labels": ["Atelectasis", "Cardiomegaly", "Effusion"],
  "probabilities": {"Atelectasis": 0.14, "Cardiomegaly": 0.03, "Effusion": 0.21},
  "top_class": "Effusion",
  "timings_ms": {"preprocess": 25.7, "inference": 63.2},
  "heatmap_png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}

```
