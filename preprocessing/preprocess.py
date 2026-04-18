"""
preprocess.py
─────────────
Image preprocessing pipeline for DermaFusion:
  1. Hair removal   — black-hat morphology + inpainting (FR4)
  2. Color normalization — Macenko stain / channel standardisation (FR5)
  3. Resize + basic transforms handled by dataset.py / torchvision

Can be used standalone or imported by dataset.py for on-the-fly preprocessing.
"""

import cv2
import numpy as np


# ── Hair removal ─────────────────────────────────────────────────────────────
def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Removes dark hair artifacts from a dermoscopic image using
    black-hat morphological transform + inpainting.

    Args:
        image : BGR uint8 numpy array (H, W, 3)
    Returns:
        BGR uint8 numpy array with hair inpainted
    """
    # Convert to grayscale for morphological operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Black-hat: highlights dark structures (hairs) on bright background
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to binary hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint removes the hair pixels, replacing with local neighbourhood
    result = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return result


# ── Color normalization (Macenko-inspired channel standardisation) ────────────
def normalize_color(image: np.ndarray) -> np.ndarray:
    """
    Normalises colour so that images from different devices / skin tones
    have comparable channel statistics.

    Strategy: standardise each channel to a target mean and std
    (derived from the ISIC/HAM10000 dataset distribution).

    Args:
        image : BGR uint8 numpy array
    Returns:
        BGR uint8 numpy array with normalised colour
    """
    # Target statistics estimated from HAM10000 training set
    TARGET_MEAN = np.array([163.0, 130.0, 128.0], dtype=np.float32)  # B G R
    TARGET_STD  = np.array([ 37.0,  40.0,  42.0], dtype=np.float32)

    img_float = image.astype(np.float32)

    src_mean = img_float.mean(axis=(0, 1))
    src_std  = img_float.std(axis=(0, 1)) + 1e-6   # avoid division by zero

    # Shift to zero mean, scale to target std, shift to target mean
    normalised = (img_float - src_mean) / src_std * TARGET_STD + TARGET_MEAN
    normalised = np.clip(normalised, 0, 255).astype(np.uint8)
    return normalised


# ── Full preprocessing pipeline ───────────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline: load → hair removal → color normalization.

    Args:
        image_path : path to a .jpg / .png dermoscopic image
    Returns:
        BGR uint8 numpy array ready for further transforms
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    img = remove_hair(img)
    img = normalize_color(img)
    return img


def preprocess_pil(pil_image):
    """
    Accepts a PIL Image (RGB), applies hair removal + color normalization,
    and returns a PIL Image (RGB).
    Used inside dataset.py or the Streamlit app.
    """
    from PIL import Image

    # PIL is RGB, OpenCV is BGR
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    bgr = remove_hair(bgr)
    bgr = normalize_color(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python preprocess.py <image_path>')
        sys.exit(0)

    result = preprocess_image(sys.argv[1])
    out_path = 'preprocessed_output.jpg'
    cv2.imwrite(out_path, result)
    print(f'Saved preprocessed image → {out_path}')
