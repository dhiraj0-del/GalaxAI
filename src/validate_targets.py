import pandas as pd
from pathlib import Path


# =========================================================
# PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "gzh"
    / "hubble_ortho_train_catalog.parquet"
)


# =========================================================
# SETTINGS
# =========================================================

THRESHOLD = 0.70


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 80)
print("FINAL MORPHOLOGY TARGET VALIDATION")
print("=" * 80)

df = pd.read_parquet(CATALOG_PATH)

print(f"\nTotal galaxies: {len(df):,}")


# =========================================================
# LOAD VOTE FRACTIONS
# =========================================================

smooth = pd.to_numeric(
    df["smooth-or-featured-hubble_smooth_fraction"],
    errors="coerce"
)

featured = pd.to_numeric(
    df["smooth-or-featured-hubble_features_fraction"],
    errors="coerce"
)


# =========================================================
# CHECK 1 — SMOOTH + FEATURED RELATIONSHIP
# =========================================================

print("\n")
print("=" * 80)
print("CHECK 1: SMOOTH VS FEATURED")
print("=" * 80)

print(
    f"\nSmooth >= {THRESHOLD:.0%}: "
    f"{(smooth >= THRESHOLD).sum():,}"
)

print(
    f"Featured >= {THRESHOLD:.0%}: "
    f"{(featured >= THRESHOLD).sum():,}"
)

print(
    f"\nSmooth >= {THRESHOLD:.0%} AND "
    f"Featured >= {THRESHOLD:.0%}: "
    f"{((smooth >= THRESHOLD) & (featured >= THRESHOLD)).sum():,}"
)

print(
    f"\nSmooth <= {(1-THRESHOLD):.0%} AND "
    f"Featured <= {(1-THRESHOLD):.0%}: "
    f"{((smooth <= (1-THRESHOLD)) & (featured <= (1-THRESHOLD))).sum():,}"
)


# =========================================================
# CHECK 2 — SMOOTH/FEATURED TARGET
# =========================================================

print("\n")
print("=" * 80)
print("CHECK 2: FINAL SMOOTH/FEATURED TARGET")
print("=" * 80)

smooth_target = (
    (smooth >= THRESHOLD) &
    (smooth > featured)
)

featured_target = (
    (featured >= THRESHOLD) &
    (featured > smooth)
)

ambiguous_target = ~(smooth_target | featured_target)

print(f"\nSmooth target      : {smooth_target.sum():,}")
print(f"Featured target    : {featured_target.sum():,}")
print(f"Ambiguous          : {ambiguous_target.sum():,}")

print(
    f"\nTotal: "
    f"{smooth_target.sum() + featured_target.sum() + ambiguous_target.sum():,}"
)


# =========================================================
# CHECK 3 — OTHER FOUR TARGETS
# =========================================================

print("\n")
print("=" * 80)
print("CHECK 3: OTHER TARGETS")
print("=" * 80)

other_targets = {
    "Spiral Arms": "has-spiral-arms-hubble_yes_fraction",
    "Bar": "bar-hubble_yes_fraction",
    "Disturbed": "t08_odd_feature_a03_disturbed_fraction",
}


for name, column in other_targets.items():

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    yes = (values >= THRESHOLD).sum()
    no = (values <= (1 - THRESHOLD)).sum()
    ambiguous = (
        (values > (1 - THRESHOLD)) &
        (values < THRESHOLD)
    ).sum()

    print(f"\n{name}")
    print(f"  YES       : {yes:,}")
    print(f"  NO        : {no:,}")
    print(f"  Ambiguous : {ambiguous:,}")


# =========================================================
# CHECK 4 — FINAL TARGET SUMMARY
# =========================================================

print("\n")
print("=" * 80)
print("FINAL TARGET SUMMARY")
print("=" * 80)

print("""
Target 1: Spiral Arms
Target 2: Bar
Target 3: Smooth vs Featured
Target 4: Disturbed
""")


# =========================================================
# FINAL MESSAGE
# =========================================================

print("=" * 80)
print("TARGET VALIDATION COMPLETE")
print("=" * 80)