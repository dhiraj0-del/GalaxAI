import pandas as pd
import os

# ============================================================
# GALAXY ZOO HUBBLE DATASET INSPECTION
# ============================================================

BASE = r"C:\Users\Charan Deep Reddy\OneDrive\Desktop\GalaxAI\data\gzh"

TRAIN_PATH = os.path.join(
    BASE,
    "hubble_ortho_train_catalog.parquet"
)

TEST_PATH = os.path.join(
    BASE,
    "hubble_ortho_test_catalog.parquet"
)

print("=" * 70)
print("GALAXY ZOO HUBBLE DATASET INSPECTION")
print("=" * 70)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training catalogue...")

train_df = pd.read_parquet(TRAIN_PATH)

print("Training catalogue loaded successfully!")

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test catalogue...")

test_df = pd.read_parquet(TEST_PATH)

print("Test catalogue loaded successfully!")

# ============================================================
# TRAINING INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DATA")
print("=" * 70)

print("Shape:", train_df.shape)
print("Number of images:", len(train_df))
print("Number of columns:", len(train_df.columns))

# ============================================================
# TEST INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TEST DATA")
print("=" * 70)

print("Shape:", test_df.shape)
print("Number of images:", len(test_df))
print("Number of columns:", len(test_df.columns))

# ============================================================
# COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COLUMNS")
print("=" * 70)

for i, column in enumerate(train_df.columns):
    print(f"{i:3} -> {column}")

# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 TRAINING ROWS")
print("=" * 70)

print(train_df.head())

# ============================================================
# IMAGE FILENAMES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE IMAGE FILENAMES")
print("=" * 70)

print(train_df["filename"].head(20).to_string(index=False))

# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(train_df.dtypes)

# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = train_df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found!")
else:
    print(missing)

# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)