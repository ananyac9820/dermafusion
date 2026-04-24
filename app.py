"""
app.py  —  DermaFusion Flask Backend
Replaces Streamlit entirely. Serves HTML frontend + REST API.

Run with:
    python app.py

Then open: http://localhost:5000
"""

import os
import sys
import io
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory

from models.fusion_model import DermaFusionModel, GradCAM
from data.dataset import encode_metadata
from preprocessing.preprocess import preprocess_pil

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH   = 'models/fusion_model.pth'
FRONTEND_DIR = 'frontend'
IMG_SIZE     = 224
CLASS_NAMES  = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)

# ── Load model once at startup ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[DermaFusion] Loading model on {device}...')

model = DermaFusionModel(pretrained=False).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f'[DermaFusion] Model loaded from {MODEL_PATH}')
else:
    print(f'[DermaFusion] WARNING: No model found at {MODEL_PATH} — predictions will be random')
model.eval()

# ── Helper: PIL image to base64 JPEG string ─────────────────────────────────
def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ── Helper: overlay Grad-CAM heatmap on image ───────────────────────────────
def overlay_heatmap(pil_img, heatmap):
    orig  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    h_u8  = (heatmap * 255).astype(np.uint8)
    h_rgb = cv2.applyColorMap(h_u8, cv2.COLORMAP_INFERNO)
    h_rgb = cv2.cvtColor(h_rgb, cv2.COLOR_BGR2RGB)
    h_rgb = cv2.resize(h_rgb, (IMG_SIZE, IMG_SIZE))
    blend = cv2.addWeighted(orig, 0.5, h_rgb, 0.5, 0)
    return Image.fromarray(blend)

# ── Routes: serve frontend pages ────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/analyse')
@app.route('/analyse.html')
def analyse():
    return send_from_directory(FRONTEND_DIR, 'analyse.html')

@app.route('/about')
@app.route('/about.html')
def about():
    return send_from_directory(FRONTEND_DIR, 'about.html')

# Serve any other static files (CSS, JS, images if needed)
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ── API: run inference ───────────────────────────────────────────────────────
@app.route('/api/analyse', methods=['POST'])
def api_analyse():
    # ── Validate inputs ──
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    age  = float(request.form.get('age',  45))
    sex  = request.form.get('sex',  'unknown')
    site = request.form.get('site', 'unknown')

    # ── Load and preprocess image ──
    try:
        pil_img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Could not read image: {str(e)}'}), 400

    try:
        preprocessed = preprocess_pil(pil_img)
    except Exception:
        preprocessed = pil_img  # fallback if preprocessing fails

    # ── Prepare tensors ──
    img_tensor  = INFER_TRANSFORM(preprocessed).unsqueeze(0).to(device)
    meta_tensor = encode_metadata(age, sex, site).unsqueeze(0).to(device)

    # ── Inference ──
    with torch.no_grad():
        logits = model(img_tensor, meta_tensor)
    probs    = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_key = CLASS_NAMES[pred_idx]

    # ── Grad-CAM ──
    try:
        cam     = GradCAM(model)
        heatmap = cam(img_tensor, meta_tensor)
        cam.remove_hooks()
        overlay = overlay_heatmap(preprocessed, heatmap)
    except Exception:
        overlay = preprocessed  # fallback if Grad-CAM fails

    # ── Build response ──
    response = {
        'prediction':     pred_key,
        'confidence':     float(probs[pred_idx]),
        'probabilities':  {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        'processed_image': pil_to_b64(preprocessed),
        'heatmap_image':   pil_to_b64(overlay),
    }

    return jsonify(response)

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('[DermaFusion] Starting server at http://localhost:5000')
    app.run(debug=False, host='0.0.0.0', port=5000)
