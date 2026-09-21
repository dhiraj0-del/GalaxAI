import pandas as pd
import torch
from torch.utils.data import DataLoader

from multitask_training import (
    GalaxyMultiTaskDataset,
    MaskedMultiTaskLoss,
    BINARY_TASKS,
    MULTICLASS_TASKS,
)

from models import GalaxyCNNV2


def main():

    print("=" * 75)
    print("GALAXAI — MULTI-TASK DATASET + LOSS TEST")
    print("=" * 75)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Load a small subset
    # ---------------------------------------------------------

    df = pd.read_csv(
        "results/multitask_train.csv"
    ).head(8)

    dataset = GalaxyMultiTaskDataset(
        df,
        train=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, targets, masks = next(iter(loader))

    images = images.to(device)

    print(f"\nImages: {tuple(images.shape)}")

    print("\nTargets:")
    for task, value in targets.items():
        print(f"  {task:15s}: {tuple(value.shape)}")

    print("\nMasks:")
    for task, value in masks.items():
        print(f"  {task:15s}: {tuple(value.shape)}")

    # ---------------------------------------------------------
    # Move targets/masks to GPU
    # ---------------------------------------------------------

    targets = {
        k: v.to(device)
        for k, v in targets.items()
    }

    masks = {
        k: v.to(device)
        for k, v in masks.items()
    }

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = GalaxyCNNV2().to(device)

    outputs = model(images)

    print("\nModel outputs:")
    for task, value in outputs.items():
        print(f"  {task:15s}: {tuple(value.shape)}")

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    pos_weights = {
        task: 1.0
        for task in BINARY_TASKS
    }

    criterion = MaskedMultiTaskLoss(
        binary_pos_weights=pos_weights,
    ).to(device)

    total_loss, individual_losses = criterion(
        outputs,
        targets,
        masks,
    )

    print("\nLosses:")
    for task, loss in individual_losses.items():
        print(
            f"  {task:15s}: "
            f"{loss.item():.6f}"
        )

    print(
        f"\nTotal loss: {total_loss.item():.6f}"
    )

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    total_loss.backward()

    print("\nBackward pass: PASS")

    print("=" * 75)
    print("MULTI-TASK TRAINING TEST: PASS")
    print("=" * 75)


if __name__ == "__main__":
    main()