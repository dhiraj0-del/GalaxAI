import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\Charan Deep Reddy\OneDrive\Desktop\GalaxAI"

CATALOG = os.path.join(
    BASE,
    "data",
    "gzh",
    "hubble_ortho_train_catalog.parquet"
)

IMAGE_DIR = os.path.join(
    BASE,
    "data",
    "gzh",
    "images"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "galaxy_morphology_samples.png"
)

# ============================================================
# START
# ============================================================

print("=" * 70)
print("GALAXY ZOO HUBBLE - GALAXY SAMPLE SELECTOR")
print("=" * 70)

print("\nLoading catalogue...")

df = pd.read_parquet(CATALOG)

print("Catalogue loaded!")
print("Training galaxies:", len(df))

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "filename",
    "smooth-or-featured-hubble_smooth_fraction",
    "smooth-or-featured-hubble_features_fraction",
    "smooth-or-featured-hubble_artifact_fraction",
    "has-spiral-arms-hubble_yes_fraction",
    "bar-hubble_yes_fraction",
    "t08_odd_feature_a03_disturbed_fraction"
]

print("\nChecking required columns...")

for column in required_columns:

    if column in df.columns:
        print("OK:", column)
    else:
        print("MISSING:", column)

# ============================================================
# REMOVE HIGH ARTIFACT PROBABILITY
# ============================================================

print("\nFiltering obvious artifacts...")

df = df[
    df["smooth-or-featured-hubble_artifact_fraction"].fillna(0) < 0.20
].copy()

print("Remaining galaxies:", len(df))

# ============================================================
# IMAGE LOOKUP
# ============================================================

print("\nBuilding image index...")

image_index = {}

for root, dirs, files in os.walk(IMAGE_DIR):

    for file in files:

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_index[file] = os.path.join(
                root,
                file
            )

print("Images indexed:", len(image_index))

# ============================================================
# FUNCTION TO GET IMAGES
# ============================================================

def get_samples(dataframe, number=5):

    results = []

    for _, row in dataframe.iterrows():

        filename = str(row["filename"])

        if filename in image_index:

            results.append(row)

        if len(results) >= number:
            break

    return results


# ============================================================
# SELECT SMOOTH GALAXIES
# ============================================================

print("\nSelecting smooth galaxies...")

smooth_df = df.sort_values(
    "smooth-or-featured-hubble_smooth_fraction",
    ascending=False
)

smooth_samples = get_samples(smooth_df, 5)

print("Smooth samples:", len(smooth_samples))

# ============================================================
# SELECT SPIRAL GALAXIES
# ============================================================

print("\nSelecting spiral galaxies...")

spiral_df = df[
    df["has-spiral-arms-hubble_yes_fraction"] >= 0.80
].sort_values(
    "has-spiral-arms-hubble_yes_fraction",
    ascending=False
)

spiral_samples = get_samples(spiral_df, 5)

print("Spiral samples:", len(spiral_samples))

# ============================================================
# SELECT BARRED GALAXIES
# ============================================================

print("\nSelecting barred galaxies...")

bar_df = df[
    df["bar-hubble_yes_fraction"] >= 0.80
].sort_values(
    "bar-hubble_yes_fraction",
    ascending=False
)

bar_samples = get_samples(bar_df, 5)

print("Barred samples:", len(bar_samples))

# ============================================================
# SELECT DISTURBED GALAXIES
# ============================================================

print("\nSelecting disturbed galaxies...")

disturbed_df = df[
    df["t08_odd_feature_a03_disturbed_fraction"] >= 0.60
].sort_values(
    "t08_odd_feature_a03_disturbed_fraction",
    ascending=False
)

disturbed_samples = get_samples(
    disturbed_df,
    5
)

print("Disturbed samples:", len(disturbed_samples))

# ============================================================
# COMBINE
# ============================================================

categories = [
    ("Smooth galaxies", smooth_samples),
    ("Spiral galaxies", spiral_samples),
    ("Barred galaxies", bar_samples),
    ("Disturbed galaxies", disturbed_samples)
]

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# CREATE GRID
# ============================================================

print("\nCreating image grid...")

fig, axes = plt.subplots(
    4,
    5,
    figsize=(15, 12)
)

for row_number, (category, samples) in enumerate(categories):

    print("\n" + "-" * 60)
    print(category)
    print("-" * 60)

    for column_number in range(5):

        ax = axes[row_number, column_number]

        if column_number >= len(samples):

            ax.axis("off")
            continue

        row = samples[column_number]

        filename = str(row["filename"])

        image_path = image_index.get(filename)

        print(filename)

        if image_path is None:

            ax.text(
                0.5,
                0.5,
                "Image not found",
                ha="center",
                va="center"
            )

            ax.axis("off")
            continue

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            ax.imshow(image)

            ax.axis("off")

            # Determine confidence score

            if category == "Smooth galaxies":

                score = row[
                    "smooth-or-featured-hubble_smooth_fraction"
                ]

            elif category == "Spiral galaxies":

                score = row[
                    "has-spiral-arms-hubble_yes_fraction"
                ]

            elif category == "Barred galaxies":

                score = row[
                    "bar-hubble_yes_fraction"
                ]

            else:

                score = row[
                    "t08_odd_feature_a03_disturbed_fraction"
                ]

            ax.set_title(
                f"{filename}\nConfidence: {score:.2f}",
                fontsize=7
            )

        except Exception as error:

            print(
                "Error loading",
                filename,
                ":",
                error
            )

            ax.axis("off")

# ============================================================
# TITLE
# ============================================================

plt.suptitle(
    "Galaxy Zoo Hubble - Label-Based Galaxy Samples",
    fontsize=18
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE SELECTION COMPLETE")
print("=" * 70)

print("\nOutput saved at:")

print(OUTPUT_FILE)

print("\nTotal selected samples:")

print(
    len(smooth_samples)
    + len(spiral_samples)
    + len(bar_samples)
    + len(disturbed_samples)
)