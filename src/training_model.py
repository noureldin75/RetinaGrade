"""
End-to-end training script for RetinaGrade (APTOS 2019 diabetic retinopathy grading).

Pipeline:
  1. Load the already-preprocessed images (see preprocessor.py / the notebook's
     "Add processed images" step) and split into train/val/test.
  2. Handle class imbalance via OVERSAMPLING of the training indices only
     (over_sample from augumentation.py) + augmentation transforms that also
     apply ImageNet normalization.
  3. Fine-tune EfficientNet-B3 (frozen backbone, trainable classifier head).
  4. Train with per-epoch train/val loss tracking, evaluate with evalution.py,
     and save the best checkpoint by macro F1 on the validation set.
  5. Final evaluation on the held-out test set.


"""

import os
import sys
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

# --- Make local modules importable (matches the notebook's sys.path setup) ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, data_split
from augumentation import over_sample, train_transform, val_transform
from augumented_set import AugmentedDataset
from evalution import evaluate_model, plot_training_curve

# CONFIG
DATA_CSV = "./data/train.csv"
PROCESSED_IMAGE_DIR = "./data/processed_train_images"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42
OVERSAMPLE_TARGETS = {1: 2, 3: 5, 4: 3}  # multipliers for Mild/Severe/Proliferative
CHECKPOINT_PATH = "best_model.pt"

grade_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 1. DATA LOADING + SPLIT
train_df = pd.read_csv(DATA_CSV)

x, y = load_data(df=train_df, image_dir=PROCESSED_IMAGE_DIR)
x_train, x_val, x_test, y_train, y_val, y_test = data_split(
    x, y, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
del x, y
gc.collect()

print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")


# 2. OVERSAMPLING (train split only — never oversample val/test)
final_indices_train = over_sample(y_train, targets=OVERSAMPLE_TARGETS)
final_indices_val = np.arange(len(y_val))  # identity: use val set as-is
final_indices_test = np.arange(len(y_test))

train_dataset = AugmentedDataset(x_train, y_train, final_indices_train, train_transform)
val_dataset = AugmentedDataset(x_val, y_val, final_indices_val, val_transform)
test_dataset = AugmentedDataset(x_test, y_test, final_indices_test, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples after oversampling: {len(final_indices_train)}")


# 3. MODEL SETUP — EfficientNet-B3, frozen backbone, new classifier head
weights = models.EfficientNet_B3_Weights.DEFAULT
model = models.efficientnet_b3(weights=weights)

in_features = model.classifier[1].in_features
out_features = len(grade_labels)
model.classifier[1] = nn.Linear(in_features=in_features, out_features=out_features)

for param in model.features.parameters():
    param.requires_grad = False

model.to(device)


# 4. LOSS + OPTIMIZER SETUP
# Class weights computed on the ORIGINAL (pre-oversampling) y_train

class_counts = np.bincount(y_train, minlength=len(grade_labels))
class_weights = 1.0 / (class_counts / class_counts.sum())
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights.cpu().numpy()}")

loss_function = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.classifier.parameters(), lr=LR)


# 5. TRAINING LOOP with per-epoch validation
train_losses = []
val_losses = []
best_val_f1 = -1.0

for epoch in range(EPOCHS):
    # --- train ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- validate ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f}")

    # --- checkpoint on best macro F1 ---
    val_metrics = evaluate_model(model, val_loader, device, average="macro", plot=False)
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> New best model saved (val macro F1: {best_val_f1:.4f})")

plot_training_curve(train_losses, title="Training Loss")
plot_training_curve(val_losses, title="Validation Loss")


# 6. FINAL EVALUATION ON THE HELD-OUT TEST SET
print("\nLoading best checkpoint for final test evaluation...")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

print("\n=== Test set performance ===")
test_metrics = evaluate_model(model, test_loader, device, average="macro", plot=True)