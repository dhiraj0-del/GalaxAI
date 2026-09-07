import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "gzh"
    / "hubble_ortho_train_catalog.parquet"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "gzh"
    / "images"
)

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

THRESHOLD = 0.70
RANDOM_STATE = 42


# =========================================================
# LOAD CATALOGUE
# =========================================================

print("=" * 80)
print("GALAXY ZOO: HUBBLE - DATASET PREPARATION")
print("=" * 80)

print("\nLoading catalogue...")

df = pd.read_parquet(CATALOG_PATH)

print(f"Total catalogue entries: {len(df):,}")


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "has-spiral-arms-hubble_yes_fraction",
    "bar-hubble_yes_fraction",
    "smooth-or-featured-hubble_smooth_fraction",
    "smooth-or-featured-hubble_features_fraction",
    "t08_odd_feature_a03_disturbed_fraction",
    "filename",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")
    for column in missing_columns:
        print(" -", column)
    raise SystemExit(1)


# =========================================================
# CREATE MORPHOLOGY TARGETS
# =========================================================

print("\nCreating morphology labels...")

spiral = pd.to_numeric(
    df["has-spiral-arms-hubble_yes_fraction"],
    errors="coerce"
)

bar = pd.to_numeric(
    df["bar-hubble_yes_fraction"],
    errors="coerce"
)

smooth = pd.to_numeric(
    df["smooth-or-featured-hubble_smooth_fraction"],
    errors="coerce"
)

featured = pd.to_numeric(
    df["smooth-or-featured-hubble_features_fraction"],
    errors="coerce"
)

disturbed = pd.to_numeric(
    df["t08_odd_feature_a03_disturbed_fraction"],
    errors="coerce"
)


# =========================================================
# TARGET 1: SPIRAL ARMS
# =========================================================

df["spiral_arms"] = pd.NA

df.loc[spiral >= THRESHOLD, "spiral_arms"] = 1
df.loc[spiral <= (1 - THRESHOLD), "spiral_arms"] = 0


# =========================================================
# TARGET 2: BAR
# =========================================================

df["bar"] = pd.NA

df.loc[bar >= THRESHOLD, "bar"] = 1
df.loc[bar <= (1 - THRESHOLD), "bar"] = 0


# =========================================================
# TARGET 3: SMOOTH / FEATURED
#
# 1 = Smooth
# 0 = Featured
#
# Only keep galaxies where one side clearly dominates.
# =========================================================

df["smooth_featured"] = pd.NA

df.loc[
    (smooth >= THRESHOLD) &
    (smooth > featured),
    "smooth_featured"
] = 1

df.loc[
    (featured >= THRESHOLD) &
    (featured > smooth),
    "smooth_featured"
] = 0


# =========================================================
# TARGET 4: DISTURBED
# =========================================================

df["disturbed"] = pd.NA

df.loc[disturbed >= THRESHOLD, "disturbed"] = 1
df.loc[disturbed <= (1 - THRESHOLD), "disturbed"] = 0


# =========================================================
# IMAGE PATH
# =========================================================

print("\nChecking image paths...")

df["image_path"] = df["filename"].apply(
    lambda x: str(IMAGE_DIR / str(x))
)


# =========================================================
# CHECK IMAGE EXISTENCE
# =========================================================

image_exists = df["image_path"].apply(
    lambda x: Path(x).exists()
)

print(f"Images found: {image_exists.sum():,}")
print(f"Images missing: {(~image_exists).sum():,}")

df = df[image_exists].copy()


# =========================================================
# DISPLAY LABEL COUNTS
# =========================================================

targets = [
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed",
]

print("\n")
print("=" * 80)
print("LABEL COUNTS BEFORE FINAL FILTERING")
print("=" * 80)

for target in targets:

    yes = (df[target] == 1).sum()
    no = (df[target] == 0).sum()
    ambiguous = df[target].isna().sum()

    print(f"\n{target}")
    print(f"  YES       : {yes:,}")
    print(f"  NO        : {no:,}")
    print(f"  Ambiguous : {ambiguous:,}")


# =========================================================
# KEEP GALAXIES WITH ALL FOUR CONFIDENT LABELS
# =========================================================

print("\n")
print("=" * 80)
print("FILTERING")
print("=" * 80)

before = len(df)

df_clean = df.dropna(
    subset=targets
).copy()

after = len(df_clean)

print(f"Before filtering : {before:,}")
print(f"After filtering  : {after:,}")
print(f"Removed          : {before - after:,}")


# =========================================================
# CONVERT TARGETS TO INTEGER
# =========================================================

for target in targets:
    df_clean[target] = df_clean[target].astype(int)


# =========================================================
# CHECK FINAL DISTRIBUTION
# =========================================================

print("\n")
print("=" * 80)
print("FINAL LABEL DISTRIBUTION")
print("=" * 80)

for target in targets:

    counts = df_clean[target].value_counts().sort_index()

    no_count = counts.get(0, 0)
    yes_count = counts.get(1, 0)

    print(f"\n{target}")
    print(f"  0 : {no_count:,}")
    print(f"  1 : {yes_count:,}")


# =========================================================
# CREATE TRAIN / VALIDATION / TEST SPLITS
# =========================================================

print("\n")
print("=" * 80)
print("CREATING DATA SPLITS")
print("=" * 80)

# First: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=RANDOM_STATE,
    shuffle=True,
)

# Then: split temporary into 15% validation and 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    shuffle=True,
)


# =========================================================
# ADD SPLIT COLUMN
# =========================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# =========================================================
# SPLIT SUMMARY
# =========================================================

print(f"\nTraining samples   : {len(train_df):,}")
print(f"Validation samples : {len(val_df):,}")
print(f"Test samples       : {len(test_df):,}")
print(f"Total samples      : {len(final_df):,}")


# =========================================================
# SAVE DATASET METADATA
# =========================================================

output_columns = [
    "filename",
    "image_path",
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed",
    "split",
]

final_df[output_columns].to_csv(
    OUTPUT_DIR / "final_dataset.csv",
    index=False
)

train_df[output_columns].to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df[output_columns].to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df[output_columns].to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 80)
print("DATASET PREPARATION COMPLETE")
print("=" * 80)

print("\nFiles created:")
print(" - results/final_dataset.csv")
print(" - results/train.csv")
print(" - results/validation.csv")
print(" - results/test.csv")

print("\nThese files contain metadata and labels only.")
print("The original Hubble images remain in data/gzh/.")


print("\n")
print("=" * 80)