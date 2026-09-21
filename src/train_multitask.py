import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask_training import (
    GalaxyMultiTaskDataset,
    MaskedMultiTaskLoss,
)
from models import GalaxyCNNV2


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = Path("results/multitask_train.csv")
VAL_CSV = Path("results/multitask_validation.csv")
WEIGHTS_JSON = Path("results/class_weights.json")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = CHECKPOINT_DIR / "galaxai_model1_v2_best.pth"
HISTORY_FILE = Path("results/model1_v2_history.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 7

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
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# WEIGHTED LOSS
# ============================================================

class WeightedMaskedMultiTaskLoss(nn.Module):

    def __init__(self, weights):

        super().__init__()

        self.binary_pos_weights = {}

        for task in BINARY_TASKS:

            value = weights["binary"][task]["pos_weight"]

            self.binary_pos_weights[task] = torch.tensor(
                value,
                dtype=torch.float32,
                device=DEVICE,
            )

        self.multiclass_weights = {}

        for task in MULTICLASS_TASKS:

            values = weights["multiclass"][task]["class_weights"]

            self.multiclass_weights[task] = torch.tensor(
                values,
                dtype=torch.float32,
                device=DEVICE,
            )

    def forward(self, outputs, targets, masks):

        task_losses = {}

        # ----------------------------------------------------
        # BINARY TASKS
        # ----------------------------------------------------

        for task in BINARY_TASKS:

            logits = outputs[task]
            target = targets[task].float()
            mask = masks[task].float()

            safe_target = target.clamp(0, 1)

            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                safe_target,
                pos_weight=self.binary_pos_weights[task],
                reduction="none",
            )

            valid = mask.sum()

            if valid.item() == 0:

                task_loss = logits.sum() * 0.0

            else:

                task_loss = (
                    loss * mask
                ).sum() / valid

            task_losses[task] = task_loss

        # ----------------------------------------------------
        # MULTICLASS TASKS
        # ----------------------------------------------------

        for task, num_classes in MULTICLASS_TASKS.items():

            logits = outputs[task]

            target = targets[task].long()

            mask = masks[task].float()

            loss = nn.functional.cross_entropy(
                logits,
                target,
                weight=self.multiclass_weights[task],
                reduction="none",
                ignore_index=-1,
            )

            valid = mask.sum()

            if valid.item() == 0:

                task_loss = logits.sum() * 0.0

            else:

                task_loss = (
                    loss * mask
                ).sum() / valid

            task_losses[task] = task_loss

        total_loss = sum(task_losses.values())

        return total_loss, task_losses


# ============================================================
# METRICS
# ============================================================

def calculate_binary_metrics(
    targets,
    probabilities,
):

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {}

    metrics["precision"] = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    metrics["f1"] = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    metrics["balanced_accuracy"] = balanced_accuracy_score(
        targets,
        predictions,
    )

    if len(np.unique(targets)) == 2:

        metrics["roc_auc"] = roc_auc_score(
            targets,
            probabilities,
        )

        metrics["pr_auc"] = average_precision_score(
            targets,
            probabilities,
        )

    else:

        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def calculate_multiclass_metrics(
    targets,
    predictions,
):

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    return {
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "macro_precision": precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            targets,
            predictions,
        ),
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    batches = 0

    task_loss_sum = {
        task: 0.0
        for task in BINARY_TASKS + list(MULTICLASS_TASKS)
    }

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

    for images, targets, masks in tqdm(
        loader,
        desc="Validation",
        leave=False,
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

        loss, task_losses = criterion(
            outputs,
            targets,
            masks,
        )

        total_loss += loss.item()
        batches += 1

        for task in task_loss_sum:

            task_loss_sum[task] += (
                task_losses[task].item()
            )

        # ----------------------------------------------------
        # Binary predictions
        # ----------------------------------------------------

        for task in BINARY_TASKS:

            mask = masks[task].bool()

            if mask.any():

                probs = torch.sigmoid(
                    outputs[task]
                )

                binary_targets[task].extend(
                    targets[task][mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                binary_probs[task].extend(
                    probs[mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

        # ----------------------------------------------------
        # Multiclass predictions
        # ----------------------------------------------------

        for task in MULTICLASS_TASKS:

            mask = masks[task].bool()

            if mask.any():

                predictions = torch.argmax(
                    outputs[task],
                    dim=1,
                )

                multiclass_targets[task].extend(
                    targets[task][mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                multiclass_predictions[task].extend(
                    predictions[mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

    # ========================================================
    # BUILD METRICS
    # ========================================================

    metrics = {}

    metrics["val_loss"] = total_loss / batches

    metrics["task_losses"] = {
        task: task_loss_sum[task] / batches
        for task in task_loss_sum
    }

    metrics["binary"] = {}

    for task in BINARY_TASKS:

        metrics["binary"][task] = calculate_binary_metrics(
            binary_targets[task],
            binary_probs[task],
        )

    metrics["multiclass"] = {}

    for task in MULTICLASS_TASKS:

        metrics["multiclass"][task] = calculate_multiclass_metrics(
            multiclass_targets[task],
            multiclass_predictions[task],
        )

    # Overall macro F1 across tasks
    binary_f1s = [
        metrics["binary"][task]["f1"]
        for task in BINARY_TASKS
    ]

    multiclass_f1s = [
        metrics["multiclass"][task]["macro_f1"]
        for task in MULTICLASS_TASKS
    ]

    metrics["macro_task_f1"] = float(
        np.mean(
            binary_f1s + multiclass_f1s
        )
    )

    return metrics


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 80)
    print("GALAXAI — MODEL 1 V2 TRAINING")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    print(
        f"\nTraining images: {len(train_df)}"
    )

    print(
        f"Validation images: {len(val_df)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = GalaxyMultiTaskDataset(
        train_df,
        train=True,
    )

    val_dataset = GalaxyMultiTaskDataset(
        val_df,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GalaxyCNNV2().to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: {total_params:,}"
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    with open(WEIGHTS_JSON, "r") as f:

        weights = json.load(f)

    criterion = WeightedMaskedMultiTaskLoss(
        weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for images, targets, masks in progress:

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

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)

            loss, _ = criterion(
                outputs,
                targets,
                masks,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_loss += loss.item()
            batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss / batches
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = validate(
            model,
            val_loader,
            criterion,
        )

        val_loss = val_metrics["val_loss"]

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = (
            time.time() - epoch_start
        )

        print("\n" + "-" * 80)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Val Loss:   {val_loss:.6f}"
        )

        print(
            f"Macro Task F1: "
            f"{val_metrics['macro_task_f1']:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        print(
            f"Time: {epoch_time:.1f}s"
        )

        print("\nBinary tasks:")

        for task in BINARY_TASKS:

            m = val_metrics["binary"][task]

            print(
                f"  {task:15s} "
                f"F1={m['f1']:.4f} "
                f"Precision={m['precision']:.4f} "
                f"Recall={m['recall']:.4f} "
                f"ROC-AUC={m['roc_auc']}"
            )

        print("\nMulticlass tasks:")

        for task in MULTICLASS_TASKS:

            m = val_metrics["multiclass"][task]

            print(
                f"  {task:15s} "
                f"Macro-F1={m['macro_f1']:.4f} "
                f"Precision={m['macro_precision']:.4f} "
                f"Recall={m['macro_recall']:.4f}"
            )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": val_metrics,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        })

        with open(HISTORY_FILE, "w") as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "model_parameters": total_params,
                },
                BEST_CHECKPOINT,
            )

            print(
                f"\n✓ BEST MODEL SAVED"
            )

            print(
                f"  {BEST_CHECKPOINT}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 80)
    print("MODEL 1 V2 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{BEST_CHECKPOINT}"
    )

    print(
        f"Training history: "
        f"{HISTORY_FILE}"
    )


if __name__ == "__main__":
    main()