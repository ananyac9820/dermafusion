"""
app.py  —  DermaFusion Streamlit Web Application
─────────────────────────────────────────────────
Run with:
    streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import cv2
import streamlit as st
from PIL import Image

from models.fusion_model import DermaFusionModel, GradCAM
from data.dataset import encode_metadata
from preprocessing.preprocess import preprocess_pil

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'models/fusion_model.pth'
NUM_CLASSES  = 7
IMG_SIZE     = 380

CLASS_NAMES  = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_LABELS = {
    'nv'    : 'Melanocytic Nevi',
    'mel'   : 'Melanoma',
    'bkl'   : 'Benign Keratosis',
    'bcc'   : 'Basal Cell Carcinoma',
    'akiec' : 'Actinic Keratoses',
    'vasc'  : 'Vascular Lesions',
    'df'    : 'Dermatofibroma',
}
CLASS_INFO   = {
    'nv'    : 'Common mole — usually benign.',
    'mel'   : 'Malignant melanoma — requires urgent specialist review.',
    'bkl'   : 'Benign skin growth, typically harmless.',
    'bcc'   : 'Most common skin cancer — slow-growing, rarely spreads.',
    'akiec' : 'Pre-cancerous lesion caused by UV damage.',
    'vasc'  : 'Lesion of blood vessel origin — often benign.',
    'df'    : 'Benign fibrous nodule — usually harmless.',
}
SITE_OPTIONS = [
    'scalp', 'ear', 'face', 'back', 'trunk', 'chest',
    'upper extremity', 'abdomen', 'lower extremity',
    'genital', 'neck', 'hand', 'foot', 'acral', 'unknown'
]

from torchvision import transforms
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

# ── Load model (cached so it loads only once) ─────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DermaFusionModel(pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        st.sidebar.success('Model loaded ✓')
    else:
        st.sidebar.warning('No trained model found — predictions will be random.')
    model.eval()
    return model, device

# ── Overlay heatmap on image ──────────────────────────────────────────────────
def overlay_heatmap(original_pil: Image.Image, heatmap: np.ndarray) -> Image.Image:
    orig_np  = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    heat_u8  = (heatmap * 255).astype(np.uint8)
    heat_rgb = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    heat_rgb = cv2.resize(heat_rgb, (IMG_SIZE, IMG_SIZE))
    overlay  = cv2.addWeighted(orig_np, 0.55, heat_rgb, 0.45, 0)
    return Image.fromarray(overlay)

# ── Page layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='DermaFusion', page_icon='🔬', layout='wide')

st.title('🔬 DermaFusion')
st.caption('Multimodal · Explainable AI · Dermatological Second Opinion')
st.markdown(
    '> ⚠️ **Disclaimer**: This tool is a decision-support aid only and does **not** '
    'replace diagnosis by a qualified dermatologist.'
)
st.divider()

model, device = load_model()

col_input, col_result = st.columns([1, 1], gap='large')

# ── Left column: inputs ───────────────────────────────────────────────────────
with col_input:
    st.subheader('1 · Upload skin lesion image')
    uploaded_file = st.file_uploader('JPG or PNG only', type=['jpg', 'jpeg', 'png'])

    st.subheader('2 · Patient details')
    age  = st.slider('Age', min_value=0, max_value=90, value=45)
    sex  = st.selectbox('Sex', ['male', 'female', 'unknown'])
    site = st.selectbox('Anatomical site', SITE_OPTIONS)

    run_btn = st.button('Analyse', type='primary', use_container_width=True)

# ── Right column: results ─────────────────────────────────────────────────────
with col_result:
    st.subheader('3 · Results')

    if not uploaded_file:
        st.info('Upload an image and fill in patient details, then click **Analyse**.')

    elif run_btn or uploaded_file:
        original_pil = Image.open(uploaded_file).convert('RGB')

        with st.spinner('Preprocessing…'):
            preprocessed_pil = preprocess_pil(original_pil)

        # Prepare tensors
        img_tensor  = INFER_TRANSFORM(preprocessed_pil).unsqueeze(0).to(device)
        meta_tensor = encode_metadata(float(age), sex, site).unsqueeze(0).to(device)

        with st.spinner('Running inference…'):
            with torch.no_grad():
                logits = model(img_tensor, meta_tensor)
            probs      = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_idx   = int(probs.argmax())
            pred_key   = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx]) * 100

        # Grad-CAM
        with st.spinner('Generating heatmap…'):
            cam     = GradCAM(model)
            heatmap = cam(img_tensor, meta_tensor)
            cam.remove_hooks()
            overlay = overlay_heatmap(preprocessed_pil, heatmap)

        # ── Display ──
        st.markdown(f'### Prediction: **{CLASS_LABELS[pred_key]}** (`{pred_key}`)')
        st.markdown(f'Confidence: **{confidence:.1f}%**')
        st.info(CLASS_INFO[pred_key])

        img_col, heat_col = st.columns(2)
        with img_col:
            st.image(preprocessed_pil, caption='Preprocessed image', use_container_width=True)
        with heat_col:
            st.image(overlay, caption='Grad-CAM heatmap', use_container_width=True)

        st.divider()
        st.markdown('#### Confidence across all classes')
        for i, (key, prob) in enumerate(zip(CLASS_NAMES, probs)):
            label = CLASS_LABELS[key]
            st.progress(float(prob), text=f'{label}: {prob*100:.1f}%')
