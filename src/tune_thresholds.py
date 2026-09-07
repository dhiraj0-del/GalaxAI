import os
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VALIDATION_CSV = os.path.join(
    BASE_DIR,
    "results",
    "validation.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "custom_cnn_best.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "evaluation"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32

TARGETS = [
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed",
]

DISPLAY_NAMES = {
    "spiral_arms": "Spiral Arms",
    "bar": "Bar",
    "smooth_featured": "Smooth vs Featured",
    "disturbed": "Disturbed",
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("CUSTOM CNN - VALIDATION THRESHOLD TUNING")
print("=" * 80)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATASET
# ============================================================

class GalaxyDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [
                float(row[target])
                for target in TARGETS
            ],
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# VALIDATION TRANSFORMATION
# IMPORTANT: NO DATA AUGMENTATION
# ============================================================

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

validation_df = pd.read_csv(
    VALIDATION_CSV
)

print(
    f"Validation samples: {len(validation_df):,}"
)

required_columns = TARGETS + ["image_path"]

missing_columns = [
    column
    for column in required_columns
    if column not in validation_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


validation_dataset = GalaxyDataset(
    validation_df,
    transform=validation_transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Custom CNN...")

# Make src importable
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from models import CustomCNN


model = CustomCNN(num_classes=4)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support either a plain state_dict or checkpoint dictionary
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# GET VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
        )

        all_labels.append(
            labels.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )


y_true = np.concatenate(
    all_labels,
    axis=0
)

y_prob = np.concatenate(
    all_probabilities,
    axis=0
)

print(
    f"Predictions generated: {len(y_true):,}"
)


# ============================================================
# FIND BEST THRESHOLD FOR EACH ATTRIBUTE
# ============================================================

print("\n" + "=" * 80)
print("SEARCHING FOR OPTIMAL THRESHOLDS")
print("=" * 80)

threshold_results = {}

# Search from 0.05 to 0.95
thresholds = np.arange(
    0.05,
    0.951,
    0.01
)

for i, target in enumerate(TARGETS):

    true = y_true[:, i]
    probabilities = y_prob[:, i]

    best_threshold = 0.50
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            true,
            predictions,
            zero_division=0
        )

        precision = precision_score(
            true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(
                threshold
            )
            best_precision = precision
            best_recall = recall

    # Calculate ranking metrics independently
    roc_auc = roc_auc_score(
        true,
        probabilities
    )

    pr_auc = average_precision_score(
        true,
        probabilities
    )

    threshold_results[target] = {
        "threshold": best_threshold,
        "f1": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    print(
        f"\n{DISPLAY_NAMES[target]}"
    )

    print("-" * 50)

    print(
        f"Best threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"F1-score       : "
        f"{best_f1:.4f}"
    )

    print(
        f"Precision      : "
        f"{best_precision:.4f}"
    )

    print(
        f"Recall         : "
        f"{best_recall:.4f}"
    )

    print(
        f"ROC-AUC        : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{pr_auc:.4f}"
    )


# ============================================================
# SAVE THRESHOLDS
# ============================================================

threshold_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_optimal_thresholds.json"
)

with open(
    threshold_path,
    "w"
) as file:

    json.dump(
        threshold_results,
        file,
        indent=4
    )


print("\n" + "=" * 80)
print("THRESHOLD TUNING COMPLETE")
print("=" * 80)

print("\nOptimal thresholds:")

for target in TARGETS:

    print(
        f"{DISPLAY_NAMES[target]:22s} : "
        f"{threshold_results[target]['threshold']:.2f}"
    )

print("\nSaved to:")
print(threshold_path)

print("\nIMPORTANT:")
print(
    "These thresholds were determined ONLY "
    "using the validation set."
)

print(
    "They can now be applied to the untouched "
    "test set for the final Custom CNN evaluation."
)

print("=" * 80)