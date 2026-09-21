import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


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


class GalaxyMultiTaskDataset(Dataset):

    def __init__(self, dataframe, train=False):

        self.df = dataframe.reset_index(drop=True)
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        targets = {}
        masks = {}

        # -------------------------------
        # Binary targets
        # -------------------------------

        for task in BINARY_TASKS:

            targets[task] = torch.tensor(
                float(row[task]),
                dtype=torch.float32,
            )

            masks[task] = torch.tensor(
                float(row[f"{task}_mask"]),
                dtype=torch.float32,
            )

        # -------------------------------
        # Multi-class targets
        # -------------------------------

        for task in MULTICLASS_TASKS:

            targets[task] = torch.tensor(
                int(row[task]),
                dtype=torch.long,
            )

            masks[task] = torch.tensor(
                float(row[f"{task}_mask"]),
                dtype=torch.float32,
            )

        return image, targets, masks


class MaskedMultiTaskLoss(nn.Module):
    """
    Masked multi-task loss.

    Binary tasks:
        BCEWithLogitsLoss

    Multi-class tasks:
        CrossEntropyLoss

    Samples with mask == 0 are ignored.
    """

    def __init__(self, binary_pos_weights=None):

        super().__init__()

        if binary_pos_weights is None:
            binary_pos_weights = {
                task: 1.0
                for task in BINARY_TASKS
            }

        self.binary_losses = {}

        for task in BINARY_TASKS:

            weight = torch.tensor(
                binary_pos_weights[task],
                dtype=torch.float32,
            )

            self.register_buffer(
                f"{task}_pos_weight",
                weight,
            )

    def masked_binary_loss(
        self,
        logits,
        targets,
        mask,
        pos_weight,
    ):

        safe_targets = targets.clamp(0, 1)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            pos_weight=pos_weight,
            reduction="none",
        )

        valid = mask.sum()

        if valid == 0:
            return logits.sum() * 0.0

        return (loss * mask).sum() / valid

    def masked_multiclass_loss(
        self,
        logits,
        targets,
        mask,
    ):

        loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=-1,
        )

        valid = mask.sum()

        if valid == 0:
            return logits.sum() * 0.0

        return (loss * mask).sum() / valid

    def forward(self, outputs, targets, masks):

        losses = {}

        # Binary tasks
        for task in BINARY_TASKS:

            pos_weight = getattr(
                self,
                f"{task}_pos_weight",
            )

            losses[task] = self.masked_binary_loss(
                outputs[task],
                targets[task],
                masks[task],
                pos_weight,
            )

        # Multi-class tasks
        for task in MULTICLASS_TASKS:

            losses[task] = self.masked_multiclass_loss(
                outputs[task],
                targets[task],
                masks[task],
            )

        total_loss = sum(losses.values())

        return total_loss, losses