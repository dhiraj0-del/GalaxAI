import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(BASE_DIR, "results", "test.csv")
MODEL_PATH = os.path.join(
    BASE_DIR, "results", "models", "custom_cnn_best.pth"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results", "evaluation")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("MODEL 1 - CUSTOM CNN TEST EVALUATION")
print("=" * 80)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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
            [float(row[target]) for target in TARGETS],
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# TEST TRANSFORMATION
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_CSV)

print(f"Test samples: {len(test_df):,}")

missing_columns = [
    column for column in TARGETS + ["image_path"]
    if column not in test_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in test.csv: {missing_columns}"
    )


test_dataset = GalaxyDataset(
    test_df,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Custom CNN...")

# Import the exact CustomCNN architecture
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CustomCNN

model = CustomCNN(num_classes=4)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Handle both normal state_dict and checkpoint dictionaries
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nRunning predictions on TEST set...")

all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device, non_blocking=True)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())


y_true = np.concatenate(all_labels, axis=0)
y_prob = np.concatenate(all_probabilities, axis=0)

# Default threshold for initial evaluation
y_pred = (y_prob >= 0.5).astype(int)


# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)


results = []

for i, target in enumerate(TARGETS):

    true = y_true[:, i]
    prob = y_prob[:, i]
    pred = y_pred[:, i]

    accuracy = accuracy_score(true, pred)

    precision = precision_score(
        true,
        pred,
        zero_division=0
    )

    recall = recall_score(
        true,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        true,
        pred,
        zero_division=0
    )

    # ROC-AUC requires both classes to be present
    try:
        roc_auc = roc_auc_score(true, prob)
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC
    try:
        pr_auc = average_precision_score(true, prob)
    except ValueError:
        pr_auc = float("nan")

    cm = confusion_matrix(true, pred)

    print(f"\n{DISPLAY_NAMES[target]}")
    print("-" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "target": target,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    })


# ============================================================
# AVERAGE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("AVERAGE METRICS")
print("=" * 80)

print(
    f"\nAverage Accuracy  : {results_df['accuracy'].mean():.4f}"
)

print(
    f"Average Precision : {results_df['precision'].mean():.4f}"
)

print(
    f"Average Recall    : {results_df['recall'].mean():.4f}"
)

print(
    f"Average F1        : {results_df['f1'].mean():.4f}"
)

print(
    f"Average ROC-AUC   : {results_df['roc_auc'].mean():.4f}"
)

print(
    f"Average PR-AUC    : {results_df['pr_auc'].mean():.4f}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_test_metrics.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)

print(f"\nMetrics saved to:")
print(metrics_path)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

confusion_matrices = {}

for i, target in enumerate(TARGETS):

    cm = confusion_matrix(
        y_true[:, i],
        y_pred[:, i]
    )

    confusion_matrices[target] = cm.tolist()

cm_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_confusion_matrices.json"
)

with open(cm_path, "w") as f:
    json.dump(confusion_matrices, f, indent=4)

print("\nConfusion matrices saved to:")
print(cm_path)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

for i, target in enumerate(TARGETS):

    predictions_df[f"{target}_probability"] = y_prob[:, i]

    predictions_df[f"{target}_prediction"] = y_pred[:, i]

predictions_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print("\nPredictions saved to:")
print(predictions_path)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("CUSTOM CNN TEST EVALUATION COMPLETE")
print("=" * 80)