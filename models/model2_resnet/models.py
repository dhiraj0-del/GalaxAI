from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


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


class GalaxyResNet18(nn.Module):
    """
    Multi-task galaxy morphology classifier using ResNet-18.

    Input:
        RGB images shaped (batch_size, 3, 224, 224).

    Output:
        Eight binary logits shaped (batch_size,)
        Four bulge-class logits shaped (batch_size, 4)
        Three roundedness-class logits shaped (batch_size, 3)
    """

    def __init__(self, pretrained=True, dropout=0.40):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        feature_count = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.shared = nn.Sequential(
            nn.Linear(feature_count, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.binary_heads = nn.ModuleDict({
            task: nn.Linear(256, 1)
            for task in BINARY_TASKS
        })

        self.multiclass_heads = nn.ModuleDict({
            task: nn.Linear(256, classes)
            for task, classes in MULTICLASS_TASKS.items()
        })

    def forward(self, x):
        features = self.backbone(x)
        features = self.shared(features)

        outputs = OrderedDict()

        for task, head in self.binary_heads.items():
            outputs[task] = head(features).squeeze(1)

        for task, head in self.multiclass_heads.items():
            outputs[task] = head(features)

        return outputs


if __name__ == "__main__":
    model = GalaxyResNet18(pretrained=False)
    model.eval()

    sample = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        predictions = model(sample)

    print("Architecture: GalaxyResNet18")
    print("Trainable parameters:", sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ))

    for task, logits in predictions.items():
        print(f"{task:12s} {tuple(logits.shape)}")

    print("Total output logits:", sum(
        logits.shape[-1] if logits.ndim == 2 else 1
        for logits in predictions.values()
    ))
    print("Architecture smoke test passed.")
