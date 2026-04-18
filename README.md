# DermaFusion

### A Multimodal, Explainable AI System for Dermatological Diagnosis

> ⚠️ **Disclaimer**: This system is a decision-support tool only and does **not** replace diagnosis by a qualified dermatologist.

---

## Overview

DermaFusion is an AI-powered web-based diagnostic support system for early detection of skin diseases. Unlike standard image classifiers, it combines visual feature extraction from dermoscopic images with structured patient metadata (age, sex, anatomical site) — replicating how a clinician reasons.

Integrated Explainable AI (XAI) via Grad-CAM generates visual heatmaps that highlight the exact regions of the lesion that influenced the prediction, making the system transparent and clinically trustworthy.

---

## Key Features

- Upload dermoscopic skin lesion images (JPG / PNG)
- Input patient metadata: age, sex, anatomical site
- Multimodal AI model — CNN (EfficientNet-B4) + MLP fusion
- 7-class skin disease classification
- Prediction confidence score per class
- Grad-CAM heatmap overlaid on the original image
- User-friendly Streamlit web interface

---

## System Architecture

```
Image + Metadata
       │
       ▼
 Preprocessing
 (hair removal + colour normalisation)
       │
       ├─────────────────────┐
       ▼                     ▼
 Visual stream          Clinical stream
 EfficientNet-B4        MLP (19 → 64 → 32)
 → 1792-dim vector      → 32-dim vector
       │                     │
       └──────────┬──────────┘
                  ▼
          Feature fusion
          concat → 1824-dim
                  ▼
          FC classifier
          → 7-class output
                  ▼
       Prediction + Grad-CAM heatmap
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Deep learning framework | PyTorch + TorchVision |
| Image backbone | EfficientNet-B4 (pretrained on ImageNet) |
| Metadata model | Custom MLP (19 → 64 → 32) |
| Explainability | Grad-CAM (built-in, no extra library) |
| Preprocessing | OpenCV — hair removal + colour normalisation |
| Web application | Streamlit |
| Dataset | HAM10000 / ISIC Archive (10,015 images, 7 classes) |
| Evaluation | scikit-learn (accuracy, precision, recall, F1) |

---

## Project Structure

```
dermafusion_phase1_2/
│
├── data/
│   ├── dataset.py              ← REPLACE with new version
│   ├── create_labels.py        ← keep
│   ├── labels.csv              ← keep
│   └── images/                 ← put all HAM10000 JPEGs here
│
├── models/
│   ├── fusion_model.py         ← NEW file
│   ├── train_fusion.py         ← NEW file
│   ├── train_cnn.py            ← replaced, can delete
│   └── train_mlp.py            ← replaced, can delete
│
├── preprocessing/
│   └── preprocess.py           ← REPLACE with new version
│
├── utils/
│   └── helpers.py              ← keep
│
├── app/                        ← create this folder
│   └── app.py                  ← NEW file
│
├── HAM10000_metadata.csv       ← keep in root
├── hmnist_8_8_RGB.csv          ← keep
├── hmnist_28_28_L.csv          ← keep
├── hmnist_28_28_RGB.csv        ← keep
├── requirements.txt            ← REPLACE with new version
└── README.md
```

---

## Setup and Run Order

> All commands must be run from the `dermafusion_phase1_2/` root folder.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

Run once. Installs torch, torchvision, streamlit, opencv-python, pandas, scikit-learn, Pillow.

---

### Step 2 — Copy HAM10000 images

Copy all `.jpg` files from `HAM10000_images_part_1/` and `HAM10000_images_part_2/` into `data/images/`.

```
data/images/ISIC_0027419.jpg
data/images/ISIC_0025030.jpg
... (10,000+ files)
```

Filenames must stay exactly as-is — they match the `image_id` column in `labels.csv`.

---

### Step 3 — Sanity check (no GPU needed)

```bash
python models/fusion_model.py
```

Expected output:
```
Output shape : torch.Size([2, 7])
Heatmap shape: (12, 12)
Fusion model + Grad-CAM OK
```

If this passes, training will work.

---

### Step 4 — Train the fusion model

```bash
python models/train_fusion.py
```

- Trains for 20 epochs with a 75 / 15 / 10 train / val / test split
- Saves best checkpoint to `models/fusion_model.pth`
- Saves per-epoch log to `models/training_log.csv`
- Prints a full classification report on the test set at the end
- ~30–90 min on GPU, several hours on CPU

---

### Step 5 — Launch the web app

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Upload a dermoscopic image, fill in age / sex / anatomical site, click **Analyse**. Displays the predicted disease, confidence scores for all 7 classes, and the Grad-CAM heatmap overlay.

---

## Disease Classes

| Label | Code | Full Name | Notes |
|---|---|---|---|
| 0 | `nv` | Melanocytic Nevi | Common mole — most frequent class |
| 1 | `mel` | Melanoma | Malignant — highest clinical priority |
| 2 | `bkl` | Benign Keratosis | Harmless skin growth |
| 3 | `bcc` | Basal Cell Carcinoma | Most common skin cancer, slow-growing |
| 4 | `akiec` | Actinic Keratoses | Pre-cancerous UV-induced lesion |
| 5 | `vasc` | Vascular Lesions | Blood vessel origin, often benign |
| 6 | `df` | Dermatofibroma | Benign fibrous nodule |

---

## Metadata Vector (19 dimensions)

The clinical stream takes a 19-element `float32` tensor built from `HAM10000_metadata.csv`:

| Dimensions | Field | Encoding |
|---|---|---|
| 0 | Age | Normalised to [0, 1] — max age in dataset = 85 |
| 1–3 | Sex | One-hot: male / female / unknown |
| 4–18 | Anatomical site | One-hot across 15 sites |

Missing values default to age = 45/85, sex = `unknown`, site = `unknown`.

---

## Evaluation

- Train / Val / Test split: **75% / 15% / 10%** (seed = 42)
- Best model checkpoint saved by highest validation accuracy
- Final test metrics: accuracy, precision, recall, F1 per class
- Training log: `models/training_log.csv`

---

## Future Scope

- Federated learning for privacy-preserving hospital collaboration
- Mobile application for offline use in remote areas
- LLM chatbot integration for patient follow-up questions
- Real-time dermatoscope device integration
- Class imbalance handling (weighted loss / oversampling)
