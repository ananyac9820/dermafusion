import pandas as pd

# Load original metadata
df = pd.read_csv("HAM10000_metadata.csv")

# Map disease labels to numbers
label_map = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

# Keep only required columns
df = df[['image_id', 'dx']]

# Convert dx → numeric label
df['label'] = df['dx'].map(label_map)

# Drop old dx column
df = df[['image_id', 'label']]

# Save new CSV
df.to_csv("data/labels.csv", index=False)

print("labels.csv created successfully!")