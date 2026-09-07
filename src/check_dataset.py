import pandas as pd
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "results" / "final_dataset.csv"


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 80)
print("FINAL DATASET QUALITY CHECK")
print("=" * 80)

df = pd.read_csv(DATASET_PATH)

print(f"\nTotal samples: {len(df):,}")


# =========================================================
# CHECK 1 — DUPLICATE FILENAMES
# =========================================================

print("\n")
print("=" * 80)
print("CHECK 1: DUPLICATE IMAGES")
print("=" * 80)

duplicate_filenames = df["filename"].duplicated().sum()

print(f"Duplicate filenames: {duplicate_filenames:,}")

if duplicate_filenames == 0:
    print("PASS: No duplicate images found.")
else:
    print("WARNING: Duplicate images found.")


# =========================================================
# CHECK 2 — DUPLICATE IMAGE PATHS
# =========================================================

duplicate_paths = df["image_path"].duplicated().sum()

print(f"Duplicate image paths: {duplicate_paths:,}")

if duplicate_paths == 0:
    print("PASS: No duplicate image paths found.")
else:
    print("WARNING: Duplicate image paths found.")


# =========================================================
# CHECK 3 — MISSING IMAGE FILES
# =========================================================

print("\n")
print("=" * 80)
print("CHECK 2: IMAGE FILE EXISTENCE")
print("=" * 80)

missing_images = 0

for path in df["image_path"]:
    if not Path(path).exists():
        missing_images += 1

print(f"Missing images: {missing_images:,}")

if missing_images == 0:
    print("PASS: All image files exist.")
else:
    print("WARNING: Some image files are missing.")


# =========================================================
# CHECK 4 — SPLIT DISTRIBUTION
# =========================================================

print("\n")
print("=" * 80)
print("CHECK 3: DATA SPLITS")
print("=" * 80)

split_counts = df["split"].value_counts()

print(split_counts)

print("\nPercentages:")

for split, count in split_counts.items():
    percentage = count / len(df) * 100
    print(f"{split:12}: {count:,} ({percentage:.2f}%)")


# =========================================================
# CHECK 5 — LABEL DISTRIBUTION PER SPLIT
# =========================================================

targets = [
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed",
]

print("\n")
print("=" * 80)
print("CHECK 4: LABEL DISTRIBUTION ACROSS SPLITS")
print("=" * 80)

for target in targets:

    print(f"\n--- {target} ---")

    for split in ["train", "validation", "test"]:

        subset = df[df["split"] == split]

        total = len(subset)
        positive = (subset[target] == 1).sum()
        negative = (subset[target] == 0).sum()

        positive_percentage = positive / total * 100
        negative_percentage = negative / total * 100

        print(
            f"{split:12}: "
            f"Positive = {positive:,} ({positive_percentage:.2f}%), "
            f"Negative = {negative:,} ({negative_percentage:.2f}%)"
        )


# =========================================================
# CHECK 6 — FINAL SAMPLE COUNTS
# =========================================================

print("\n")
print("=" * 80)
print("CHECK 5: FINAL SAMPLE COUNTS")
print("=" * 80)

expected_splits = {
    "train": 12648,
    "validation": 2710,
    "test": 2711,
}

all_correct = True

for split, expected in expected_splits.items():

    actual = (df["split"] == split).sum()

    if actual == expected:
        print(f"PASS: {split} = {actual:,}")
    else:
        print(
            f"WARNING: {split} = {actual:,} "
            f"(expected {expected:,})"
        )
        all_correct = False


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 80)
print("FINAL QUALITY CHECK SUMMARY")
print("=" * 80)

if (
    duplicate_filenames == 0
    and duplicate_paths == 0
    and missing_images == 0
    and all_correct
):
    print("\nALL BASIC DATASET CHECKS PASSED.")
    print("\nDataset is ready for preprocessing.")
else:
    print("\nSOME CHECKS REQUIRE ATTENTION.")
    print("Do NOT start model training yet.")

print("\n")
print("=" * 80)