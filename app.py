import io
import os
import time
import base64
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import torch

from preprocessing import is_allowed, load_image_bytes_safe, read_image, to_model_tensor
from models import ModelSingleton, get_last_conv_layer
from explain import GradCAM

app = Flask(__name__)

# Warm-load model
_model = ModelSingleton.instance()
_last_conv = get_last_conv_layer(_model.model)
_gradcam = GradCAM(_model.model, _last_conv)

def to_png_b64(pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def heatmap_overlay(input_gray: np.ndarray, cam: np.ndarray, alpha=0.35) -> Image.Image:
    import cv2
    h, w = input_gray.shape
    heat = (cam * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    input_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(input_rgb, 1.0 - alpha, heat_color, alpha, 0)
    overlay = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    return overlay

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html", error=None)

@app.route("/predict", methods=["POST"])
def predict_form():
    file = request.files.get("file")
    if not file or not file.filename:
        return render_template("upload.html", error="Please choose a file.")
    if not is_allowed(file.filename):
        return render_template("upload.html", error="Unsupported file type. Upload JPG, PNG, or DICOM (.dcm).")
    try:
        raw = load_image_bytes_safe(file)
        t0 = time.time()
        pil = read_image(raw, file.filename)
        display_img = pil.copy()
        img_np = to_model_tensor(pil)
        t1 = time.time()
        probs, logits = _model.predict(img_np)
        t2 = time.time()
        labels = _model.labels
        prob_map = dict(zip(labels, probs.tolist()))
        top_idx = int(np.argmax(probs))
        top_class = labels[top_idx]

        x = torch.from_numpy(img_np).float().unsqueeze(0).to(_model.device)
        cam = _gradcam(x, top_idx)
        disp = display_img.resize((img_np.shape[-1], img_np.shape[-2]))
        disp_np = np.array(disp, dtype=np.uint8)
        heat_overlay = heatmap_overlay(disp_np, cam, alpha=0.38)
        sorted_items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)

        return render_template(
            "result.html",
            probs=sorted_items,
            timings={"preprocess": (t1-t0)*1000.0, "inference": (t2-t1)*1000.0},
            top_class=top_class,
            input_png_b64=to_png_b64(disp.convert("L")),
            heatmap_png_b64=to_png_b64(heat_overlay),
        )
    except Exception as e:
        return render_template("upload.html", error=f"Processing error: {str(e)}")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "file is required"}), 400
    if not is_allowed(file.filename):
        return jsonify({"error": "Unsupported file type. Upload JPG, PNG, or DICOM (.dcm)."}), 400
    try:
        raw = load_image_bytes_safe(file)
        t0 = time.time()
        pil = read_image(raw, file.filename)
        img_np = to_model_tensor(pil)
        t1 = time.time()
        probs, logits = _model.predict(img_np)
        t2 = time.time()
        labels = _model.labels
        prob_map = {k: float(v) for k, v in zip(labels, probs.tolist())}
        top_idx = int(np.argmax(probs))
        top_class = labels[top_idx]
        x = torch.from_numpy(img_np).float().unsqueeze(0).to(_model.device)
        cam = _gradcam(x, top_idx)
        disp = pil.resize((img_np.shape[-1], img_np.shape[-2])).convert("L")
        disp_np = np.array(disp, dtype=np.uint8)
        heat_overlay = heatmap_overlay(disp_np, cam, alpha=0.38)
        buf = io.BytesIO()
        heat_overlay.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({
            "labels": labels,
            "probabilities": prob_map,
            "top_class": top_class,
            "timings_ms": {"preprocess": (t1-t0)*1000.0, "inference": (t2-t1)*1000.0},
            "heatmap_png_base64": heatmap_b64
        })
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)