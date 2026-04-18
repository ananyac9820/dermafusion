import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# ── Encoding maps ────────────────────────────────────────────────────────────
SEX_CATEGORIES = ['male', 'female', 'unknown']          # 3 dims
SITE_CATEGORIES = [                                      # 15 dims
    'scalp', 'ear', 'face', 'back', 'trunk', 'chest',
    'upper extremity', 'abdomen', 'unknown',
    'lower extremity', 'genital', 'neck', 'hand', 'foot', 'acral'
]
# Total metadata vector: 1 (age) + 3 (sex) + 15 (site) = 19
METADATA_DIM = 19


def encode_metadata(age, sex, localization):
    """
    Returns a float32 tensor of size 19.
      [0]      : age normalised to [0, 1]  (max age in HAM10000 = 85)
      [1..3]   : sex  one-hot (male / female / unknown)
      [4..18]  : localization one-hot (15 sites)
    """
    vec = torch.zeros(METADATA_DIM, dtype=torch.float32)

    # Age — fill with median (45/85 ≈ 0.53) when missing
    if pd.isna(age):
        vec[0] = 45.0 / 85.0
    else:
        vec[0] = float(age) / 85.0

    # Sex one-hot
    sex_str = str(sex).lower().strip() if not pd.isna(sex) else 'unknown'
    if sex_str not in SEX_CATEGORIES:
        sex_str = 'unknown'
    vec[1 + SEX_CATEGORIES.index(sex_str)] = 1.0

    # Localization one-hot
    site_str = str(localization).lower().strip() if not pd.isna(localization) else 'unknown'
    if site_str not in SITE_CATEGORIES:
        site_str = 'unknown'
    vec[4 + SITE_CATEGORIES.index(site_str)] = 1.0

    return vec


class SkinDataset(Dataset):
    """
    Loads HAM10000 images + metadata + labels.

    Returns (image_tensor, metadata_tensor, label_tensor) per sample.

    Args:
        csv_file  : path to labels.csv          (image_id, label)
        meta_file : path to HAM10000_metadata.csv
        image_dir : folder containing <image_id>.jpg files
    """

    def __init__(self, csv_file, meta_file, image_dir):
        self.labels_df = pd.read_csv(csv_file)
        meta_df        = pd.read_csv(meta_file)

        # Keep only the columns we need and index by image_id
        self.meta_df = meta_df[['image_id', 'age', 'sex', 'localization']].set_index('image_id')

        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # ImageNet mean/std — suitable for EfficientNet pretrained weights
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row      = self.labels_df.iloc[idx]
        image_id = row['image_id']

        # ── Image ─────────────────────────────────────────────────────────────
        img_path = os.path.join(self.image_dir, image_id + '.jpg')
        image    = Image.open(img_path).convert('RGB')
        image    = self.transform(image)

        # ── Metadata ──────────────────────────────────────────────────────────
        if image_id in self.meta_df.index:
            meta_row = self.meta_df.loc[image_id]
            # .loc can return a DataFrame when image_id has duplicates — take first row
            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]
            metadata = encode_metadata(meta_row['age'], meta_row['sex'], meta_row['localization'])
        else:
            # Fallback: unknown values for missing entries
            metadata = encode_metadata(None, 'unknown', 'unknown')

        # ── Label ─────────────────────────────────────────────────────────────
        label = torch.tensor(int(row['label']), dtype=torch.long)

        return image, metadata, label
