import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

CATALOG_PATH = "data/gzh/hubble_ortho_train_catalog.parquet"
IMAGE_ROOT = "data/gzh/images"

OUTPUT_DIR = "results"

THRESHOLD = 0.70
LOW_THRESHOLD = 0.30

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CATALOG
# ============================================================

print("=" * 75)
print("GALAXAI — MULTI-TASK DATASET PREPARATION")
print("=" * 75)

df = pd.read_parquet(CATALOG_PATH)

print(f"Total catalogue rows: {len(df):,}")
print(f"Confidence threshold: {THRESHOLD:.0%}")
print()


# ============================================================
# IMAGE PATH
# ============================================================

df["image_path"] = df["filename"].apply(
    lambda x: os.path.join(IMAGE_ROOT, x)
)

missing_images = ~df["image_path"].apply(os.path.exists)

print(f"Missing image files: {missing_images.sum():,}")

df = df[~missing_images].copy()

print(f"Rows with valid images: {len(df):,}")
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def binary_target(df, yes_col, no_col):
    """
    Creates:
        target = 1 for confident YES
        target = 0 for confident NO
        target = -1 for ambiguous/unavailable

        mask = 1 when target is valid
        mask = 0 otherwise
    """

    yes = df[yes_col].fillna(0)
    no = df[no_col].fillna(0)

    target = pd.Series(-1, index=df.index, dtype="int64")
    mask = pd.Series(0, index=df.index, dtype="int64")

    yes_confident = yes >= THRESHOLD
    no_confident = no >= THRESHOLD

    target.loc[yes_confident] = 1
    mask.loc[yes_confident] = 1

    target.loc[no_confident] = 0
    mask.loc[no_confident] = 1

    # If both somehow pass the threshold, invalidate.
    both = yes_confident & no_confident
    target.loc[both] = -1
    mask.loc[both] = 0

    return target, mask


def multiclass_target(df, columns):
    """
    Creates:
        target = class index
        mask = 1 when one category reaches threshold
        target = -1 when ambiguous/unavailable
    """

    values = df[columns].fillna(0)

    max_values = values.max(axis=1)
    max_indices = values.idxmax(axis=1)

    target = pd.Series(-1, index=df.index, dtype="int64")
    mask = pd.Series(0, index=df.index, dtype="int64")

    confident = max_values >= THRESHOLD

    mapping = {col: i for i, col in enumerate(columns)}

    target.loc[confident] = (
        max_indices.loc[confident].map(mapping).astype(int)
    )

    mask.loc[confident] = 1

    return target, mask


# ============================================================
# 1. SMOOTH / FEATURED
# ============================================================

df["featured"], df["featured_mask"] = binary_target(
    df,
    "smooth-or-featured-hubble_features_fraction",
    "smooth-or-featured-hubble_smooth_fraction",
)


# ============================================================
# 2. EDGE-ON
# ============================================================

df["edge_on"], df["edge_on_mask"] = binary_target(
    df,
    "disk-edge-on-hubble_yes_fraction",
    "disk-edge-on-hubble_no_fraction",
)


# ============================================================
# 3. BAR
# ============================================================

df["bar"], df["bar_mask"] = binary_target(
    df,
    "bar-hubble_yes_fraction",
    "bar-hubble_no_fraction",
)


# ============================================================
# 4. SPIRAL ARMS
# ============================================================

df["spiral_arms"], df["spiral_arms_mask"] = binary_target(
    df,
    "has-spiral-arms-hubble_yes_fraction",
    "has-spiral-arms-hubble_no_fraction",
)


# ============================================================
# 5. BULGE PROMINENCE — 4 CLASS
# ============================================================

bulge_columns = [
    "bulge-size-hubble_none_fraction",
    "bulge-size-hubble_just-noticeable_fraction",
    "bulge-size-hubble_obvious_fraction",
    "bulge-size-hubble_dominant_fraction",
]

df["bulge"], df["bulge_mask"] = multiclass_target(
    df,
    bulge_columns,
)


# ============================================================
# 6. ROUNDEDNESS — 3 CLASS
# ============================================================

round_columns = [
    "how-rounded-hubble_completely_fraction",
    "how-rounded-hubble_in-between_fraction",
    "how-rounded-hubble_cigar-shaped_fraction",
]

df["roundedness"], df["roundedness_mask"] = multiclass_target(
    df,
    round_columns,
)


# ============================================================
# 7 & 8. DISTURBED / MERGER
#
# These belong to the GZH odd-feature branch.
#
# We use:
#   - positive when the corresponding fraction >= 70%
#   - negative only when GZH explicitly indicated NO odd feature
#   - otherwise ambiguous
# ============================================================

disturbed_fraction = df[
    "t08_odd_feature_a03_disturbed_fraction"
].fillna(0)

merger_fraction = df[
    "t08_odd_feature_a06_merger_fraction"
].fillna(0)

odd_yes = df[
    "t06_odd_a01_yes_fraction"
].fillna(0)

odd_no = df[
    "t06_odd_a02_no_fraction"
].fillna(0)


# DISTURBED

df["disturbed"] = -1
df["disturbed_mask"] = 0

disturbed_positive = disturbed_fraction >= THRESHOLD
no_odd_feature = odd_no >= THRESHOLD

df.loc[disturbed_positive, "disturbed"] = 1
df.loc[disturbed_positive, "disturbed_mask"] = 1

df.loc[
    no_odd_feature & ~disturbed_positive,
    "disturbed"
] = 0

df.loc[
    no_odd_feature & ~disturbed_positive,
    "disturbed_mask"
] = 1


# MERGER

df["merger"] = -1
df["merger_mask"] = 0

merger_positive = merger_fraction >= THRESHOLD

df.loc[merger_positive, "merger"] = 1
df.loc[merger_positive, "merger_mask"] = 1

df.loc[
    no_odd_feature & ~merger_positive,
    "merger"
] = 0

df.loc[
    no_odd_feature & ~merger_positive,
    "merger_mask"
] = 1


# ============================================================
# 9. CLUMPY APPEARANCE
# ============================================================

df["clumpy"], df["clumpy_mask"] = binary_target(
    df,
    "clumpy-appearance-hubble_yes_fraction",
    "clumpy-appearance-hubble_no_fraction",
)


# ============================================================
# 10. GALAXY SYMMETRY
#
# This is the GZH galaxy symmetry question.
# ============================================================

df["symmetry"], df["symmetry_mask"] = binary_target(
    df,
    "galaxy-symmetrical-hubble_yes_fraction",
    "galaxy-symmetrical-hubble_no_fraction",
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    "filename",
    "image_path",

    # Targets
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "bulge",
    "roundedness",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",

    # Masks
    "featured_mask",
    "edge_on_mask",
    "bar_mask",
    "spiral_arms_mask",
    "bulge_mask",
    "roundedness_mask",
    "disturbed_mask",
    "merger_mask",
    "clumpy_mask",
    "symmetry_mask",
]

dataset = df[output_columns].copy()


# ============================================================
# DATASET SUMMARY
# ============================================================

targets = [
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "bulge",
    "roundedness",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",
]

print("=" * 75)
print("TARGET SUMMARY")
print("=" * 75)

for target in targets:

    mask_col = target + "_mask"

    valid = dataset[mask_col] == 1
    valid_count = valid.sum()

    if target in ["bulge", "roundedness"]:
        counts = dataset.loc[valid, target].value_counts().sort_index()

        print(f"\n{target.upper()}")
        print(f"  Valid: {valid_count:,}")

        for cls, count in counts.items():
            print(f"  Class {cls}: {count:,}")

    else:
        positives = (
            (dataset[target] == 1) &
            valid
        ).sum()

        negatives = (
            (dataset[target] == 0) &
            valid
        ).sum()

        print(f"\n{target.upper()}")
        print(f"  Valid     : {valid_count:,}")
        print(f"  Positive  : {positives:,}")
        print(f"  Negative  : {negatives:,}")


# ============================================================
# SAVE COMPLETE DATASET
# ============================================================

full_path = os.path.join(
    OUTPUT_DIR,
    "multitask_dataset.csv"
)

dataset.to_csv(full_path, index=False)

print()
print(f"Saved complete dataset:")
print(f"  {full_path}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# We split images, not individual task labels.
# Random seed guarantees reproducibility.
# ============================================================

train_df, temp_df = train_test_split(
    dataset,
    test_size=0.30,
    random_state=RANDOM_STATE,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_path = os.path.join(
    OUTPUT_DIR,
    "multitask_train.csv"
)

val_path = os.path.join(
    OUTPUT_DIR,
    "multitask_validation.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "multitask_test.csv"
)

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("DATASET SPLIT")
print("=" * 75)

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(val_df):,}")
print(f"Test       : {len(test_df):,}")

print()
print("Files created:")
print(f"  {full_path}")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"  {test_path}")

print()
print("=" * 75)
print("DATASET PREPARATION COMPLETE")
print("=" * 75)