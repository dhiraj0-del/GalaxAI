import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "gzh" / "hubble_ortho_train_catalog.parquet"

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
THRESHOLD = 0.70

print("=" * 80)
print("GALAXY ZOO: HUBBLE - MULTI-LABEL OVERLAP ANALYSIS")
print("=" * 80)

print("\nLoading catalogue...")

df = pd.read_parquet(CATALOG_PATH)

print(f"Total galaxies: {len(df):,}")
print(f"Confidence threshold: {THRESHOLD:.0%}")


# ---------------------------------------------------------
# Four selected morphological attributes
# ---------------------------------------------------------
attributes = {
    "Spiral Arms": "has-spiral-arms-hubble_yes_fraction",
    "Bar": "bar-hubble_yes_fraction",
    "Smooth": "smooth-or-featured-hubble_smooth_fraction",
    "Featured": "smooth-or-featured-hubble_features_fraction",
}


# ---------------------------------------------------------
# Create YES / NO / AMBIGUOUS labels
# ---------------------------------------------------------
labels = pd.DataFrame(index=df.index)

for attribute, column in attributes.items():

    values = pd.to_numeric(df[column], errors="coerce")

    labels[attribute] = "Ambiguous"

    labels.loc[values >= THRESHOLD, attribute] = "YES"
    labels.loc[values <= (1 - THRESHOLD), attribute] = "NO"


# ---------------------------------------------------------
# Individual label counts
# ---------------------------------------------------------
print("\n")
print("=" * 80)
print("INDIVIDUAL ATTRIBUTE COUNTS")
print("=" * 80)

for attribute in attributes:

    yes = (labels[attribute] == "YES").sum()
    no = (labels[attribute] == "NO").sum()
    ambiguous = (labels[attribute] == "Ambiguous").sum()

    print(f"\n{attribute}")
    print(f"  YES       : {yes:,}")
    print(f"  NO        : {no:,}")
    print(f"  Ambiguous : {ambiguous:,}")


# ---------------------------------------------------------
# Pairwise YES overlap
# ---------------------------------------------------------
print("\n")
print("=" * 80)
print("PAIRWISE YES OVERLAP")
print("=" * 80)

attribute_names = list(attributes.keys())

for i in range(len(attribute_names)):

    for j in range(i + 1, len(attribute_names)):

        attr1 = attribute_names[i]
        attr2 = attribute_names[j]

        overlap = (
            (labels[attr1] == "YES") &
            (labels[attr2] == "YES")
        ).sum()

        print(
            f"\n{attr1} + {attr2}"
            f"\n  Both YES: {overlap:,}"
        )


# ---------------------------------------------------------
# Number of positive attributes per galaxy
# ---------------------------------------------------------
print("\n")
print("=" * 80)
print("NUMBER OF POSITIVE ATTRIBUTES PER GALAXY")
print("=" * 80)

yes_matrix = labels[attribute_names] == "YES"

positive_count = yes_matrix.sum(axis=1)

for count in range(5):

    if count == 0:
        description = "No positive attributes"
    elif count == 1:
        description = "Exactly 1 positive attribute"
    elif count == 2:
        description = "Exactly 2 positive attributes"
    elif count == 3:
        description = "Exactly 3 positive attributes"
    else:
        description = "All 4 attributes positive"

    number = (positive_count == count).sum()

    print(f"{description}: {number:,}")


# ---------------------------------------------------------
# Four-way YES overlap
# ---------------------------------------------------------
all_four = (
    (labels["Spiral Arms"] == "YES") &
    (labels["Bar"] == "YES") &
    (labels["Smooth"] == "YES") &
    (labels["Featured"] == "YES")
).sum()

print("\n")
print("=" * 80)
print("ALL FOUR ATTRIBUTES")
print("=" * 80)

print(f"All four YES: {all_four:,}")


# ---------------------------------------------------------
# Fully confident samples
# ---------------------------------------------------------
fully_confident = (
    (labels != "Ambiguous").all(axis=1)
)

fully_confident_count = fully_confident.sum()

print("\n")
print("=" * 80)
print("FULLY CONFIDENT SAMPLES")
print("=" * 80)

print(
    f"Galaxies with a confident YES/NO label "
    f"for ALL FOUR attributes: {fully_confident_count:,}"
)


# ---------------------------------------------------------
# Save labelled data
# ---------------------------------------------------------
output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "multilabel_labels.csv"

labels.to_csv(output_file, index=False)

print("\n")
print("=" * 80)
print(f"Label matrix saved to: {output_file}")
print("=" * 80)