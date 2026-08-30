import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_DIR = r"C:\Users\Charan Deep Reddy\OneDrive\Desktop\GalaxAI\data\gzh\images"

OUTPUT_FILE = r"C:\Users\Charan Deep Reddy\OneDrive\Desktop\GalaxAI\results\best_hubble_images.png"

NUMBER_OF_IMAGES = 25

# --------------------------------------------------
# FIND IMAGE FILES
# --------------------------------------------------

print("Scanning images...")

image_files = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(os.path.join(root, file))

print("Total images found:", len(image_files))

# --------------------------------------------------
# QUALITY SCORE
# --------------------------------------------------

scores = []

for i, path in enumerate(image_files):

    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32)

        # Overall brightness
        brightness = np.mean(arr)

        # Contrast
        contrast = np.std(arr)

        # Fraction of very bright pixels
        bright_fraction = np.mean(arr > 230)

        # Fraction of very dark pixels
        dark_fraction = np.mean(arr < 15)

        # Prefer reasonable brightness and contrast
        brightness_score = 1 - abs(brightness - 80) / 80
        brightness_score = max(0, brightness_score)

        contrast_score = min(contrast / 60, 1)

        # Penalize images dominated by extreme pixels
        saturation_penalty = max(0, bright_fraction - 0.15)
        darkness_penalty = max(0, dark_fraction - 0.75)

        score = (
            0.45 * brightness_score
            + 0.55 * contrast_score
            - 0.50 * saturation_penalty
            - 0.30 * darkness_penalty
        )

        scores.append((score, path))

    except Exception as e:
        print("Could not read:", path)

    if (i + 1) % 5000 == 0:
        print("Processed:", i + 1)

# --------------------------------------------------
# SELECT BEST IMAGES
# --------------------------------------------------

scores.sort(reverse=True)

best_images = scores[:NUMBER_OF_IMAGES]

print("\nBest images selected:")
for rank, (score, path) in enumerate(best_images, 1):
    print(rank, "Score:", round(score, 4), "File:", path)

# --------------------------------------------------
# CREATE GRID
# --------------------------------------------------

fig, axes = plt.subplots(5, 5, figsize=(15, 15))

for ax, (score, path) in zip(axes.flat, best_images):

    img = Image.open(path).convert("RGB")

    ax.imshow(img)
    ax.axis("off")

    ax.set_title(
        os.path.basename(path),
        fontsize=7
    )

plt.suptitle(
    "Galaxy Zoo Hubble - Best Quality Sample",
    fontsize=18
)

plt.tight_layout()

# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved grid to:")
print(OUTPUT_FILE)