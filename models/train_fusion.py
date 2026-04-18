"""
train_fusion.py
Trains the full DermaFusion multimodal model.

Usage:
    python models/train_fusion.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

from data.dataset import SkinDataset
from models.fusion_model import DermaFusionModel

# ── Config ────────────────────────────────────────────────────────────────────
CSV_FILE   = 'data/labels.csv'
META_FILE  = 'HAM10000_metadata.csv'
IMAGE_DIR  = 'data/images'
SAVE_PATH  = 'models/fusion_model.pth'
LOG_PATH   = 'models/training_log.csv'

BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.10
SEED       = 42

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    full_dataset = SkinDataset(
        csv_file  = CSV_FILE,
        meta_file = META_FILE,
        image_dir = IMAGE_DIR
    )
    print(f'Total samples: {len(full_dataset)}')

    n_total = len(full_dataset)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = int(n_total * TEST_SPLIT)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f'Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model     = DermaFusionModel(pretrained=False, dropout=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs('models', exist_ok=True)

    best_val_acc = 0.0
    log_rows     = []

    for epoch in range(1, EPOCHS + 1):

        print(f'\n--- Epoch {epoch}/{EPOCHS} ---')

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, metadata, labels) in enumerate(train_loader):
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            logits = model(images, metadata)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

            if batch_idx % 10 == 0:
                print(f'  [Train] Batch {batch_idx}/{len(train_loader)}  Loss: {loss.item():.4f}')

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        print('  Running validation...')
        with torch.no_grad():
            for batch_idx, (images, metadata, labels) in enumerate(val_loader):
                images   = images.to(device)
                metadata = metadata.to(device)
                labels   = labels.to(device)

                logits      = model(images, metadata)
                loss        = criterion(logits, labels)
                val_loss   += loss.item() * images.size(0)
                preds       = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

                if batch_idx % 10 == 0:
                    print(f'  [Val]   Batch {batch_idx}/{len(val_loader)}')

        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step()

        print(f'\nEpoch {epoch:02d}/{EPOCHS}  '
              f'Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  '
              f'Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}')

        log_rows.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  Saved best model (val_acc={val_acc:.4f})')

    with open(LOG_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f'\nTraining log saved -> {LOG_PATH}')

    print('\nLoading best model for test evaluation...')
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, metadata, labels in test_loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            logits   = model(images, metadata)
            preds    = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print('\n-- Test Set Classification Report --')
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))