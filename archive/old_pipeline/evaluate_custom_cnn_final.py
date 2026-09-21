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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "results",
    "test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "custom_cnn_best.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "results",
    "evaluation",
    "custom_cnn_optimal_thresholds.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "evaluation"
)

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
print("MODEL 1 - CUSTOM CNN FINAL TEST EVALUATION")
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
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# TEST PREPROCESSING
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
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

print(
    f"Test samples: {len(test_df):,}"
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
# LOAD THRESHOLDS
# ============================================================

print("\nLoading validation-derived thresholds...")

with open(
    THRESHOLD_PATH,
    "r"
) as file:

    threshold_data = json.load(file)


thresholds = np.array([
    threshold_data[target]["threshold"]
    for target in TARGETS
])

for target in TARGETS:

    print(
        f"{DISPLAY_NAMES[target]:22s}: "
        f"{threshold_data[target]['threshold']:.2f}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Custom CNN...")

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from models import CustomCNN


model = CustomCNN(num_classes=4)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

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
# GENERATE TEST PREDICTIONS
# ============================================================

print("\nRunning predictions on TEST set...")

all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

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


# ============================================================
# APPLY VALIDATION-DERIVED THRESHOLDS
# ============================================================

y_pred = (
    y_prob >= thresholds
).astype(int)


# ============================================================
# FINAL TEST RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

results = []

confusion_matrices = {}

for i, target in enumerate(TARGETS):

    true = y_true[:, i]
    prob = y_prob[:, i]
    pred = y_pred[:, i]

    accuracy = accuracy_score(
        true,
        pred
    )

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

    roc_auc = roc_auc_score(
        true,
        prob
    )

    pr_auc = average_precision_score(
        true,
        prob
    )

    cm = confusion_matrix(
        true,
        pred
    )

    confusion_matrices[target] = cm.tolist()

    print(
        f"\n{DISPLAY_NAMES[target]}"
    )

    print("-" * 50)

    print(
        f"Threshold : {thresholds[i]:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    results.append({
        "target": target,
        "display_name": DISPLAY_NAMES[target],
        "threshold": thresholds[i],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    })


# ============================================================
# AVERAGE METRICS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("FINAL AVERAGE METRICS")
print("=" * 80)

print(
    f"\nAverage Accuracy  : "
    f"{results_df['accuracy'].mean():.4f}"
)

print(
    f"Average Precision : "
    f"{results_df['precision'].mean():.4f}"
)

print(
    f"Average Recall    : "
    f"{results_df['recall'].mean():.4f}"
)

print(
    f"Average F1        : "
    f"{results_df['f1'].mean():.4f}"
)

print(
    f"Average ROC-AUC   : "
    f"{results_df['roc_auc'].mean():.4f}"
)

print(
    f"Average PR-AUC    : "
    f"{results_df['pr_auc'].mean():.4f}"
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_final_test_metrics.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

cm_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_final_confusion_matrices.json"
)

with open(
    cm_path,
    "w"
) as file:

    json.dump(
        confusion_matrices,
        file,
        indent=4
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

for i, target in enumerate(TARGETS):

    predictions_df[
        f"{target}_probability"
    ] = y_prob[:, i]

    predictions_df[
        f"{target}_prediction"
    ] = y_pred[:, i]

    predictions_df[
        f"{target}_threshold"
    ] = thresholds[i]


predictions_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_final_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("CUSTOM CNN FINAL TEST EVALUATION COMPLETE")
print("=" * 80)

print("\nFinal metrics saved to:")
print(metrics_path)

print("\nConfusion matrices saved to:")
print(cm_path)

print("\nPredictions saved to:")
print(predictions_path)

print("\nIMPORTANT:")
print(
    "Thresholds were selected using ONLY the validation set."
)

print(
    "The test set was evaluated using those fixed thresholds."
)

print("=" * 80)