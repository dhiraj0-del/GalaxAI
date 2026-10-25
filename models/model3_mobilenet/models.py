
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


# ============================================================
# GALAXAI MODEL 3
# MobileNetV3-Large Multi-Task Galaxy Morphology Classifier
# ============================================================


BINARY_TASKS = (
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",
)

MULTICLASS_TASKS = {
    "bulge": 4,
    "roundedness": 3,
}


class GalaxyMobileNetV3(nn.Module):

    def __init__(self, pretrained=True, dropout=0.40):

        super().__init__()

        # ----------------------------------------------------
        # 1. Load MobileNetV3-Large
        # ----------------------------------------------------

        weights = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = mobilenet_v3_large(
            weights=weights
        )

        # ----------------------------------------------------
        # 2. Extract the backbone feature dimension
        # ----------------------------------------------------

        feature_count = backbone.classifier[0].in_features

        # ----------------------------------------------------
        # 3. Remove the original ImageNet classification head
        # ----------------------------------------------------

        backbone.classifier = nn.Identity()

        self.backbone = backbone

        # ----------------------------------------------------
        # 4. Shared feature layer
        # ----------------------------------------------------

        self.shared = nn.Sequential(

            nn.Linear(
                feature_count,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=dropout
            ),

        )

        # ----------------------------------------------------
        # 5. Binary classification heads
        # ----------------------------------------------------

        self.binary_heads = nn.ModuleDict({

            task: nn.Linear(
                256,
                1
            )

            for task in BINARY_TASKS

        })

        # ----------------------------------------------------
        # 6. Multiclass classification heads
        # ----------------------------------------------------

        self.multiclass_heads = nn.ModuleDict({

            task: nn.Linear(
                256,
                classes
            )

            for task, classes
            in MULTICLASS_TASKS.items()

        })

    # ========================================================
    # Forward pass
    # ========================================================

    def forward(self, x):

        # Extract image features
        features = self.backbone(x)

        # Shared representation
        features = self.shared(features)

        outputs = OrderedDict()

        # Binary tasks
        for task, head in self.binary_heads.items():

            outputs[task] = head(
                features
            ).squeeze(1)

        # Multiclass tasks
        for task, head in self.multiclass_heads.items():

            outputs[task] = head(
                features
            )

        return outputs


# ============================================================
# MODEL SUMMARY
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = GalaxyMobileNetV3(
        pretrained=False
    ).to(device)

    model.eval()

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    with torch.no_grad():

        outputs = model(
            dummy_input
        )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_logits = sum(
        output.shape[-1]
        if output.ndim > 1
        else 1
        for output in outputs.values()
    )

    print("\n" + "=" * 60)

    print("GALAXAI MODEL 3")
    print("MobileNetV3-Large")

    print("=" * 60)

    print("Device:", device)

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Total output logits:",
        total_logits
    )

    print("\nOutput shapes:")

    for task, output in outputs.items():

        print(
            f"{task:15s}: {tuple(output.shape)}"
        )

    print("\nModel smoke test completed.")
