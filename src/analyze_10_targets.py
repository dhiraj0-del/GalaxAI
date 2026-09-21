import pandas as pd

PATH = "data/gzh/hubble_ortho_train_catalog.parquet"
THRESHOLD = 0.70

df = pd.read_parquet(PATH)

print("=" * 70)
print("GALAXAI — 10 TARGET ANALYSIS")
print("=" * 70)
print(f"Total galaxies: {len(df):,}")
print(f"Confidence threshold: {THRESHOLD:.0%}\n")


def binary(name, yes_col, no_col):
    yes = (df[yes_col] >= THRESHOLD).sum()
    no = (df[no_col] >= THRESHOLD).sum()
    ambiguous = len(df) - yes - no

    print(f"{name}")
    print(f"  YES       : {yes:,}")
    print(f"  NO        : {no:,}")
    print(f"  AMBIGUOUS : {ambiguous:,}")
    print()


binary(
    "1. FEATURED",
    "smooth-or-featured-hubble_features_fraction",
    "smooth-or-featured-hubble_smooth_fraction",
)

binary(
    "2. EDGE-ON",
    "disk-edge-on-hubble_yes_fraction",
    "disk-edge-on-hubble_no_fraction",
)

binary(
    "3. BAR",
    "bar-hubble_yes_fraction",
    "bar-hubble_no_fraction",
)

binary(
    "4. SPIRAL ARMS",
    "has-spiral-arms-hubble_yes_fraction",
    "has-spiral-arms-hubble_no_fraction",
)

binary(
    "7. DISTURBED",
    "t08_odd_feature_a03_disturbed_fraction",
    "t06_odd_a02_no_fraction",
)

binary(
    "8. MERGER",
    "t08_odd_feature_a06_merger_fraction",
    "t06_odd_a02_no_fraction",
)

binary(
    "9. CLUMPY",
    "clumpy-appearance-hubble_yes_fraction",
    "clumpy-appearance-hubble_no_fraction",
)

binary(
    "10. CLUMP SYMMETRY",
    "galaxy-symmetrical-hubble_yes_fraction",
    "galaxy-symmetrical-hubble_no_fraction",
)


def multiclass(name, columns):
    print(name)

    confident = df[columns].max(axis=1) >= THRESHOLD
    labels = df.loc[confident, columns].idxmax(axis=1)

    for col in columns:
        print(f"  {col.split('_')[-2]:20s}: {(labels == col).sum():,}")

    print(f"  CONFIDENT TOTAL     : {confident.sum():,}")
    print(f"  AMBIGUOUS           : {(~confident).sum():,}")
    print()


multiclass(
    "5. BULGE PROMINENCE",
    [
        "bulge-size-hubble_none_fraction",
        "bulge-size-hubble_just-noticeable_fraction",
        "bulge-size-hubble_obvious_fraction",
        "bulge-size-hubble_dominant_fraction",
    ],
)

multiclass(
    "6. ROUNDEDNESS",
    [
        "how-rounded-hubble_completely_fraction",
        "how-rounded-hubble_in-between_fraction",
        "how-rounded-hubble_cigar-shaped_fraction",
    ],
)