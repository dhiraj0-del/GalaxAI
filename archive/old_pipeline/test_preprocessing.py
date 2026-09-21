import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "results" / "train.csv"


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32


# =========================================================
# TRANSFORMS
# =========================================================

# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Validation/test transform
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =========================================================
# DATASET CLASS
# =========================================================

class GalaxyDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        self.targets = [
            "spiral_arms",
            "bar",
            "smooth_featured",
            "disturbed"
        ]

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
                for target in self.targets
            ],
            dtype=torch.float32
        )

        return image, labels


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("=" * 70)
print("GALAXY ZOO: HUBBLE - PREPROCESSING TEST")
print("=" * 70)

print("\nLoading training metadata...")

train_df = pd.read_csv(TRAIN_PATH)

print(f"Training samples: {len(train_df):,}")


# =========================================================
# CREATE DATASET
# =========================================================

dataset = GalaxyDataset(
    train_df,
    transform=train_transform
)

print(f"Dataset size: {len(dataset):,}")


# =========================================================
# TEST ONE IMAGE
# =========================================================

print("\nTesting single image...")

image, labels = dataset[0]

print(f"Image tensor shape : {image.shape}")
print(f"Image tensor dtype : {image.dtype}")
print(f"Label shape        : {labels.shape}")
print(f"Labels             : {labels.tolist()}")


# =========================================================
# CREATE DATALOADER
# =========================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# =========================================================
# TEST ONE BATCH
# =========================================================

print("\nTesting DataLoader...")

images, labels = next(iter(loader))

print(f"Batch image shape  : {images.shape}")
print(f"Batch label shape  : {labels.shape}")


# =========================================================
# PIXEL RANGE
# =========================================================

print("\nChecking normalized pixel values...")

print(f"Minimum pixel value: {images.min().item():.4f}")
print(f"Maximum pixel value: {images.max().item():.4f}")


# =========================================================
# CLASS WEIGHTS
# =========================================================

print("\n")
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

targets = [
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed"
]

for target in targets:

    positive = (train_df[target] == 1).sum()
    negative = (train_df[target] == 0).sum()

    pos_weight = negative / positive

    print(f"\n{target}")
    print(f"  Positive samples : {positive:,}")
    print(f"  Negative samples : {negative:,}")
    print(f"  pos_weight       : {pos_weight:.4f}")


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 70)
print("PREPROCESSING TEST COMPLETE")
print("=" * 70)

print("\nIf the tensor shapes and class weights look correct,")
print("the dataset is ready for model implementation.")