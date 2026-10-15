"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask_training import GalaxyMultiTaskDataset
from models import GalaxyCNNV2


# ============================================================
# CONFIG
# ============================================================

TEST_CSV = Path("results/multitask_test.csv")
CHECKPOINT = Path("checkpoints/galaxai_model1_v2_best.pth")

RESULTS_DIR = Path("results/model1_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RESULTS_DIR / "test_metrics.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_WORKERS = 2

BINARY_TASKS = [
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",
]

MULTICLASS_TASKS = {
    "bulge": 4,
    "roundedness": 3,
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GALAXAI — MODEL 1 V2 TEST EVALUATION")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_df = pd.read_csv(TEST_CSV)

    print(
        f"\nTest images: {len(test_df)}"
    )

    test_dataset = GalaxyMultiTaskDataset(
        test_df,
        train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = GalaxyCNNV2().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"\nLoaded checkpoint from epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint.get('best_val_loss', 'unknown')}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    binary_targets = {
        task: []
        for task in BINARY_TASKS
    }

    binary_probs = {
        task: []
        for task in BINARY_TASKS
    }

    multiclass_targets = {
        task: []
        for task in MULTICLASS_TASKS
    }

    multiclass_predictions = {
        task: []
        for task in MULTICLASS_TASKS
    }

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning test inference...")

    with torch.no_grad():

        for images, targets, masks in tqdm(
            test_loader,
            desc="Testing",
        ):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            targets = {
                task: value.to(DEVICE)
                for task, value in targets.items()
            }

            masks = {
                task: value.to(DEVICE)
                for task, value in masks.items()
            }

            outputs = model(images)

            # ------------------------------------------------
            # Binary
            # ------------------------------------------------

            for task in BINARY_TASKS:

                mask = masks[task].bool()

                if mask.any():

                    probabilities = torch.sigmoid(
                        outputs[task]
                    )

                    binary_targets[task].extend(
                        targets[task][mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    binary_probs[task].extend(
                        probabilities[mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

            # ------------------------------------------------
            # Multiclass
            # ------------------------------------------------

            for task in MULTICLASS_TASKS:

                mask = masks[task].bool()

                if mask.any():

                    predictions = torch.argmax(
                        outputs[task],
                        dim=1,
                    )

                    multiclass_targets[task].extend(
                        targets[task][mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    multiclass_predictions[task].extend(
                        predictions[mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    results = {
        "checkpoint": str(CHECKPOINT),
        "test_samples": len(test_df),
        "binary": {},
        "multiclass": {},
    }

    # --------------------------------------------------------
    # Binary metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BINARY TASK RESULTS")
    print("=" * 80)

    binary_f1s = []

    for task in BINARY_TASKS:

        y_true = np.asarray(
            binary_targets[task]
        )

        y_prob = np.asarray(
            binary_probs[task]
        )

        y_pred = (
            y_prob >= 0.5
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        if len(np.unique(y_true)) == 2:

            roc_auc = roc_auc_score(
                y_true,
                y_prob,
            )

            pr_auc = average_precision_score(
                y_true,
                y_prob,
            )

        else:

            roc_auc = None
            pr_auc = None

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        result = {
            "samples": int(len(y_true)),
            "positive_samples": int(y_true.sum()),
            "negative_samples": int(
                len(y_true) - y_true.sum()
            ),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(
                balanced_acc
            ),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": (
                float(roc_auc)
                if roc_auc is not None
                else None
            ),
            "pr_auc": (
                float(pr_auc)
                if pr_auc is not None
                else None
            ),
            "confusion_matrix": cm.tolist(),
        }

        results["binary"][task] = result

        binary_f1s.append(f1)

        print(
            f"\n{task.upper()}"
        )

        print(
            f"  Samples:             {len(y_true)}"
        )

        print(
            f"  Precision:           {precision:.4f}"
        )

        print(
            f"  Recall:              {recall:.4f}"
        )

        print(
            f"  F1:                  {f1:.4f}"
        )

        print(
            f"  Balanced Accuracy:   {balanced_acc:.4f}"
        )

        print(
            f"  ROC-AUC:             "
            f"{roc_auc:.4f}"
            if roc_auc is not None
            else "  ROC-AUC:             N/A"
        )

        print(
            f"  PR-AUC:              "
            f"{pr_auc:.4f}"
            if pr_auc is not None
            else "  PR-AUC:              N/A"
        )

        print(
            f"  Confusion Matrix:\n{cm}"
        )

    # --------------------------------------------------------
    # Multiclass metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("MULTICLASS TASK RESULTS")
    print("=" * 80)

    multiclass_f1s = []

    for task, num_classes in MULTICLASS_TASKS.items():

        y_true = np.asarray(
            multiclass_targets[task]
        )

        y_pred = np.asarray(
            multiclass_predictions[task]
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
        )

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            average=None,
            labels=list(range(num_classes)),
            zero_division=0,
        )

        result = {
            "samples": int(len(y_true)),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(
                balanced_acc
            ),
            "macro_precision": float(
                precision
            ),
            "macro_recall": float(
                recall
            ),
            "macro_f1": float(f1),
            "per_class_f1": [
                float(x)
                for x in per_class_f1
            ],
            "confusion_matrix": cm.tolist(),
        }

        results["multiclass"][task] = result

        multiclass_f1s.append(f1)

        print(
            f"\n{task.upper()}"
        )

        print(
            f"  Samples:             {len(y_true)}"
        )

        print(
            f"  Accuracy:            {accuracy:.4f}"
        )

        print(
            f"  Balanced Accuracy:   {balanced_acc:.4f}"
        )

        print(
            f"  Macro Precision:     {precision:.4f}"
        )

        print(
            f"  Macro Recall:        {recall:.4f}"
        )

        print(
            f"  Macro F1:            {f1:.4f}"
        )

        print(
            f"  Per-class F1:        "
            f"{[round(x, 4) for x in per_class_f1]}"
        )

        print(
            f"  Confusion Matrix:\n{cm}"
        )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    all_task_f1s = (
        binary_f1s +
        multiclass_f1s
    )

    macro_task_f1 = float(
        np.mean(all_task_f1s)
    )

    results["summary"] = {
        "binary_macro_f1": float(
            np.mean(binary_f1s)
        ),
        "multiclass_macro_f1": float(
            np.mean(multiclass_f1s)
        ),
        "overall_macro_task_f1": macro_task_f1,
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MODEL 1 V2 — FINAL TEST SUMMARY")
    print("=" * 80)

    print(
        f"\nBinary Macro F1:       "
        f"{results['summary']['binary_macro_f1']:.4f}"
    )

    print(
        f"Multiclass Macro F1:   "
        f"{results['summary']['multiclass_macro_f1']:.4f}"
    )

    print(
        f"Overall Macro Task F1:  "
        f"{results['summary']['overall_macro_task_f1']:.4f}"
    )

    print(
        f"\nSaved results to:"
    )

    print(
        f"{METRICS_FILE}"
    )

    print("\n" + "=" * 80)
    print("MODEL 1 V2 TEST EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""


import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask_training import GalaxyMultiTaskDataset
from models import GalaxyCNNV2


# ============================================================
# CONFIG
# ============================================================

TEST_CSV = Path("results/multitask_test.csv")
CHECKPOINT = Path("checkpoints/galaxai_model1_v2_best.pth")

RESULTS_DIR = Path("results/model1_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RESULTS_DIR / "test_metrics.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_WORKERS = 2

BINARY_TASKS = [
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",
]

MULTICLASS_TASKS = {
    "bulge": 4,
    "roundedness": 3,
}

# Human-readable class names for confusion-matrix plots.
MULTICLASS_LABELS = {
    "bulge": [
        "None",
        "Just noticeable",
        "Obvious",
        "Dominant",
    ],
    "roundedness": [
        "Completely rounded",
        "In-between",
        "Cigar-shaped",
    ],
}


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def save_confusion_matrix(
    cm,
    task_name,
    labels,
    filename,
):
    """Save a presentation-ready Seaborn confusion-matrix heatmap."""

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{task_name} - Confusion Matrix")
    plt.tight_layout()

    output_path = RESULTS_DIR / filename
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GALAXAI — MODEL 1 V2 TEST EVALUATION")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_df = pd.read_csv(TEST_CSV)

    print(
        f"\nTest images: {len(test_df)}"
    )

    test_dataset = GalaxyMultiTaskDataset(
        test_df,
        train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = GalaxyCNNV2().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"\nLoaded checkpoint from epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint.get('best_val_loss', 'unknown')}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    binary_targets = {
        task: []
        for task in BINARY_TASKS
    }

    binary_probs = {
        task: []
        for task in BINARY_TASKS
    }

    multiclass_targets = {
        task: []
        for task in MULTICLASS_TASKS
    }

    multiclass_predictions = {
        task: []
        for task in MULTICLASS_TASKS
    }

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning test inference...")

    with torch.no_grad():

        for images, targets, masks in tqdm(
            test_loader,
            desc="Testing",
        ):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            targets = {
                task: value.to(DEVICE)
                for task, value in targets.items()
            }

            masks = {
                task: value.to(DEVICE)
                for task, value in masks.items()
            }

            outputs = model(images)

            # ------------------------------------------------
            # Binary
            # ------------------------------------------------

            for task in BINARY_TASKS:

                mask = masks[task].bool()

                if mask.any():

                    probabilities = torch.sigmoid(
                        outputs[task]
                    )

                    binary_targets[task].extend(
                        targets[task][mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    binary_probs[task].extend(
                        probabilities[mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

            # ------------------------------------------------
            # Multiclass
            # ------------------------------------------------

            for task in MULTICLASS_TASKS:

                mask = masks[task].bool()

                if mask.any():

                    predictions = torch.argmax(
                        outputs[task],
                        dim=1,
                    )

                    multiclass_targets[task].extend(
                        targets[task][mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    multiclass_predictions[task].extend(
                        predictions[mask]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    results = {
        "checkpoint": str(CHECKPOINT),
        "test_samples": len(test_df),
        "binary": {},
        "multiclass": {},
    }

    # These counters are used to calculate one pooled accuracy
    # across every valid task prediction.
    overall_correct = 0
    overall_valid_predictions = 0

    # --------------------------------------------------------
    # Binary metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BINARY TASK RESULTS")
    print("=" * 80)

    binary_f1s = []

    for task in BINARY_TASKS:

        y_true = np.asarray(
            binary_targets[task]
        )

        y_prob = np.asarray(
            binary_probs[task]
        )

        y_pred = (
            y_prob >= 0.5
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        overall_correct += int(np.sum(y_true == y_pred))
        overall_valid_predictions += int(len(y_true))

        if len(np.unique(y_true)) == 2:

            roc_auc = roc_auc_score(
                y_true,
                y_prob,
            )

            pr_auc = average_precision_score(
                y_true,
                y_prob,
            )

        else:

            roc_auc = None
            pr_auc = None

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        # Save Seaborn heatmap.
        cm_path = save_confusion_matrix(
            cm,
            task.replace("_", " ").title(),
            ["Negative", "Positive"],
            f"{task}_confusion_matrix.png",
        )

        result = {
            "samples": int(len(y_true)),
            "positive_samples": int(y_true.sum()),
            "negative_samples": int(
                len(y_true) - y_true.sum()
            ),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(
                balanced_acc
            ),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": (
                float(roc_auc)
                if roc_auc is not None
                else None
            ),
            "pr_auc": (
                float(pr_auc)
                if pr_auc is not None
                else None
            ),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_plot": str(cm_path),
        }

        results["binary"][task] = result

        binary_f1s.append(f1)

        print(
            f"\n{task.upper()}"
        )

        print(
            f"  Samples:             {len(y_true)}"
        )

        print(
            f"  Accuracy:            {accuracy:.4f}"
        )

        print(
            f"  Precision:           {precision:.4f}"
        )

        print(
            f"  Recall:              {recall:.4f}"
        )

        print(
            f"  F1:                  {f1:.4f}"
        )

        print(
            f"  Balanced Accuracy:   {balanced_acc:.4f}"
        )

        print(
            f"  ROC-AUC:             "
            f"{roc_auc:.4f}"
            if roc_auc is not None
            else "  ROC-AUC:             N/A"
        )

        print(
            f"  PR-AUC:              "
            f"{pr_auc:.4f}"
            if pr_auc is not None
            else "  PR-AUC:              N/A"
        )

        print(
            f"  Confusion Matrix:\n{cm}"
        )

        print(
            f"  Heatmap saved to:    {cm_path}"
        )

    # --------------------------------------------------------
    # Multiclass metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("MULTICLASS TASK RESULTS")
    print("=" * 80)

    multiclass_f1s = []

    for task, num_classes in MULTICLASS_TASKS.items():

        y_true = np.asarray(
            multiclass_targets[task]
        )

        y_pred = np.asarray(
            multiclass_predictions[task]
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        overall_correct += int(np.sum(y_true == y_pred))
        overall_valid_predictions += int(len(y_true))

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
        )

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            average=None,
            labels=list(range(num_classes)),
            zero_division=0,
        )

        class_labels = MULTICLASS_LABELS.get(
            task,
            [str(i) for i in range(num_classes)],
        )

        # Save Seaborn heatmap.
        cm_path = save_confusion_matrix(
            cm,
            task.replace("_", " ").title(),
            class_labels,
            f"{task}_confusion_matrix.png",
        )

        result = {
            "samples": int(len(y_true)),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(
                balanced_acc
            ),
            "macro_precision": float(
                precision
            ),
            "macro_recall": float(
                recall
            ),
            "macro_f1": float(f1),
            "per_class_f1": [
                float(x)
                for x in per_class_f1
            ],
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_plot": str(cm_path),
        }

        results["multiclass"][task] = result

        multiclass_f1s.append(f1)

        print(
            f"\n{task.upper()}"
        )

        print(
            f"  Samples:             {len(y_true)}"
        )

        print(
            f"  Accuracy:            {accuracy:.4f}"
        )

        print(
            f"  Balanced Accuracy:   {balanced_acc:.4f}"
        )

        print(
            f"  Macro Precision:     {precision:.4f}"
        )

        print(
            f"  Macro Recall:        {recall:.4f}"
        )

        print(
            f"  Macro F1:            {f1:.4f}"
        )

        print(
            f"  Per-class F1:        "
            f"{[round(x, 4) for x in per_class_f1]}"
        )

        print(
            f"  Confusion Matrix:\n{cm}"
        )

        print(
            f"  Heatmap saved to:    {cm_path}"
        )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    all_task_f1s = (
        binary_f1s +
        multiclass_f1s
    )

    macro_task_f1 = float(
        np.mean(all_task_f1s)
    )

    binary_macro_f1 = float(
        np.mean(binary_f1s)
    )

    multiclass_macro_f1 = float(
        np.mean(multiclass_f1s)
    )

    # Pooled accuracy across ALL valid task predictions.
    # This is different from Macro Task F1.
    overall_accuracy = (
        float(overall_correct / overall_valid_predictions)
        if overall_valid_predictions > 0
        else 0.0
    )

    results["summary"] = {
        "binary_macro_f1": binary_macro_f1,
        "multiclass_macro_f1": multiclass_macro_f1,
        "overall_macro_task_f1": macro_task_f1,
        "overall_accuracy": overall_accuracy,
        "overall_correct_predictions": int(
            overall_correct
        ),
        "overall_valid_predictions": int(
            overall_valid_predictions
        ),
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MODEL 1 V2 — FINAL TEST SUMMARY")
    print("=" * 80)

    print(
        f"\nOverall Accuracy:      "
        f"{overall_accuracy:.4f} "
        f"({overall_accuracy * 100:.2f}%)"
    )

    print(
        f"Binary Macro F1:       "
        f"{binary_macro_f1:.4f}"
    )

    print(
        f"Multiclass Macro F1:   "
        f"{multiclass_macro_f1:.4f}"
    )

    print(
        f"Overall Macro Task F1: "
        f"{macro_task_f1:.4f}"
    )

    print(
        f"\nCorrect task predictions: "
        f"{overall_correct}"
    )

    print(
        f"Valid task predictions:   "
        f"{overall_valid_predictions}"
    )

    print(
        "\nNOTE: Overall Accuracy is pooled across all "
        "valid task predictions. It is NOT the same as "
        "Overall Macro Task F1."
    )

    print(
        f"\nSaved results to:"
    )

    print(
        f"{METRICS_FILE}"
    )

    print("\nConfusion-matrix heatmaps saved in:")

    print(
        f"{RESULTS_DIR}"
    )

    print("\n" + "=" * 80)
    print("MODEL 1 V2 TEST EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
