import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "gzh" / "hubble_ortho_train_catalog.parquet"


# ---------------------------------------------------------
# Load catalogue
# ---------------------------------------------------------
print("=" * 70)
print("GALAXY ZOO: HUBBLE - LABEL DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\nLoading catalogue...")

df = pd.read_parquet(CATALOG_PATH)

print(f"Total galaxies: {len(df):,}")


# ---------------------------------------------------------
# Attributes to analyze
# ---------------------------------------------------------
attributes = {
    "Spiral Arms": "has-spiral-arms-hubble_yes_fraction",
    "Bar": "bar-hubble_yes_fraction",
    "Smooth": "smooth-or-featured-hubble_smooth_fraction",
    "Featured": "smooth-or-featured-hubble_features_fraction",
    "Disturbed": "t08_odd_feature_a03_disturbed_fraction",
    "Merger": "t08_odd_feature_a06_merger_fraction",
    "Ring": "t08_odd_feature_a01_ring_fraction",
}


# ---------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------
THRESHOLD = 0.70


print(f"\nConfidence threshold: {THRESHOLD:.0%}")
print("YES  = vote fraction >= 70%")
print("NO   = vote fraction <= 30%")
print("AMB  = between 30% and 70%")


# ---------------------------------------------------------
# Analyze each attribute
# ---------------------------------------------------------
results = []

for attribute, column in attributes.items():

    if column not in df.columns:
        print(f"\nWARNING: Column not found: {column}")
        continue

    values = pd.to_numeric(df[column], errors="coerce")

    yes_count = (values >= THRESHOLD).sum()
    no_count = (values <= (1 - THRESHOLD)).sum()
    ambiguous_count = (
        (values > (1 - THRESHOLD)) &
        (values < THRESHOLD)
    ).sum()

    missing_count = values.isna().sum()

    total_valid = yes_count + no_count + ambiguous_count

    results.append({
        "Attribute": attribute,
        "YES (>=70%)": yes_count,
        "NO (<=30%)": no_count,
        "Ambiguous": ambiguous_count,
        "Missing": missing_count,
        "Total": total_valid
    })


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n")
print("=" * 100)
print("LABEL DISTRIBUTION")
print("=" * 100)

print(results_df.to_string(index=False))


# ---------------------------------------------------------
# Positive / negative percentages
# ---------------------------------------------------------
print("\n")
print("=" * 100)
print("HIGH-CONFIDENCE SAMPLE SUMMARY")
print("=" * 100)

for _, row in results_df.iterrows():

    yes = row["YES (>=70%)"]
    no = row["NO (<=30%)"]
    confident_total = yes + no

    if confident_total > 0:
        yes_percentage = yes / confident_total * 100
        no_percentage = no / confident_total * 100
    else:
        yes_percentage = 0
        no_percentage = 0

    print(
        f"\n{row['Attribute']}"
        f"\n  High-confidence YES : {yes:,}"
        f"\n  High-confidence NO  : {no:,}"
        f"\n  YES percentage      : {yes_percentage:.2f}%"
        f"\n  NO percentage       : {no_percentage:.2f}%"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "label_distribution.csv"

results_df.to_csv(output_file, index=False)

print("\n")
print("=" * 70)
print(f"Results saved to: {output_file}")
print("=" * 70)