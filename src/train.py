import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models import CustomCNN


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "results" / "train.csv"
VAL_PATH = PROJECT_ROOT / "results" / "validation.csv"

MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DIR = PROJECT_ROOT / "results" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32

LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 5

RANDOM_SEED = 42

TARGETS = [
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed",
]


# =========================================================
# REPRODUCIBILITY
# =========================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("MODEL 1 - CUSTOM CNN TRAINING")
print("=" * 80)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# =========================================================
# DATASET
# =========================================================

class GalaxyDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [
                float(row[target])
                for target in TARGETS
            ],
            dtype=torch.float32,
        )

        return image, labels


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"Training samples   : {len(train_df):,}")
print(f"Validation samples : {len(val_df):,}")


# =========================================================
# DATASETS
# =========================================================

train_dataset = GalaxyDataset(
    train_df,
    transform=train_transform,
)

val_dataset = GalaxyDataset(
    val_df,
    transform=val_transform,
)


# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# =========================================================
# MODEL
# =========================================================

model = CustomCNN(
    num_classes=len(TARGETS)
).to(device)

print("\nModel:")
print(model)


# =========================================================
# CLASS WEIGHTS
# =========================================================

pos_weights = []

print("\n")
print("=" * 80)
print("CLASS WEIGHTS")
print("=" * 80)

for target in TARGETS:

    positive = (train_df[target] == 1).sum()
    negative = (train_df[target] == 0).sum()

    weight = negative / positive

    pos_weights.append(weight)

    print(
        f"{target:20} "
        f"positive={positive:6,} "
        f"negative={negative:6,} "
        f"pos_weight={weight:.4f}"
    )


pos_weight_tensor = torch.tensor(
    pos_weights,
    dtype=torch.float32,
    device=device,
)


# =========================================================
# LOSS
# =========================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight_tensor
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# =========================================================
# LEARNING RATE SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# =========================================================
# METRIC FUNCTION
# =========================================================

def calculate_metrics(y_true, y_prob):

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    metrics["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = average_precision_score(
            y_true,
            y_prob,
        )
    except ValueError:
        metrics["pr_auc"] = None

    return metrics


# =========================================================
# TRAINING
# =========================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

history = []

total_start_time = time.time()

print("\n")
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)


for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start_time = time.time()

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_loss = 0.0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            val_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(
                outputs
            )

            all_labels.append(
                labels.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    val_loss /= len(val_loader.dataset)

    y_true = np.concatenate(
        all_labels,
        axis=0,
    )

    y_prob = np.concatenate(
        all_probabilities,
        axis=0,
    )


    # -----------------------------------------------------
    # PER-TARGET METRICS
    # -----------------------------------------------------

    target_metrics = {}

    for i, target in enumerate(TARGETS):

        target_metrics[target] = calculate_metrics(
            y_true[:, i],
            y_prob[:, i],
        )


    # -----------------------------------------------------
    # AVERAGE METRICS
    # -----------------------------------------------------

    avg_f1 = np.mean([
        target_metrics[target]["f1"]
        for target in TARGETS
    ])

    avg_roc_auc = np.mean([
        target_metrics[target]["roc_auc"]
        for target in TARGETS
        if target_metrics[target]["roc_auc"] is not None
    ])


    # -----------------------------------------------------
    # SCHEDULER
    # -----------------------------------------------------

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = time.time() - epoch_start_time


    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    epoch_record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "avg_f1": float(avg_f1),
        "avg_roc_auc": float(avg_roc_auc),
        "learning_rate": current_lr,
        "epoch_time_seconds": epoch_time,
    }

    history.append(epoch_record)


    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print(
        f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        f" | Train Loss: {train_loss:.4f}"
        f" | Val Loss: {val_loss:.4f}"
        f" | Avg F1: {avg_f1:.4f}"
        f" | Avg ROC-AUC: {avg_roc_auc:.4f}"
        f" | LR: {current_lr:.6f}"
        f" | Time: {epoch_time:.1f}s"
    )


    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_DIR / "custom_cnn_best.pth",
        )

        print("  ✓ Best model saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"  No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

total_training_time = time.time() - total_start_time

history_path = HISTORY_DIR / "custom_cnn_history.json"

with open(history_path, "w") as f:

    json.dump(
        history,
        f,
        indent=4,
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 80)
print("CUSTOM CNN TRAINING COMPLETE")
print("=" * 80)

print(f"\nBest validation loss : {best_val_loss:.4f}")
print(f"Total training time  : {total_training_time / 60:.2f} minutes")

print(f"\nBest model saved to:")
print(MODEL_DIR / "custom_cnn_best.pth")

print(f"\nTraining history saved to:")
print(history_path)

print("\n")
print("Next step: evaluate the saved model on the TEST set.")
print("=" * 80)