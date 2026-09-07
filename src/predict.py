import os
import sys
import json

import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "custom_cnn_best.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "results",
    "evaluation",
    "custom_cnn_optimal_thresholds.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

TARGETS = [
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed",
]

DISPLAY_NAMES = {
    "spiral_arms": "Spiral Arms",
    "bar": "Bar",
    "smooth_featured": "Smooth vs Featured",
    "disturbed": "Disturbed",
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GALAXAI - GALAXY MORPHOLOGY PREDICTOR")
print("=" * 70)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from models import CustomCNN


print("\nLoading Custom CNN...")

model = CustomCNN(num_classes=4)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD THRESHOLDS
# ============================================================

with open(
    THRESHOLD_PATH,
    "r"
) as file:

    threshold_data = json.load(file)


thresholds = {
    target: threshold_data[target]["threshold"]
    for target in TARGETS
}


# ============================================================
# PREPROCESSING
# SAME AS VALIDATION / TEST
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# GET IMAGE PATH
# ============================================================

print("\nEnter the path to a galaxy image.")

image_path = input(
    "\nImage path: "
).strip().strip('"')


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.isfile(image_path):

    print("\nERROR: Image file not found.")

    sys.exit(1)


try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print(
        f"\nERROR: Could not open image: {e}"
    )

    sys.exit(1)


print(
    f"\nImage loaded: "
    f"{image.width} x {image.height}"
)


# ============================================================
# PREPROCESS
# ============================================================

input_tensor = transform(
    image
).unsqueeze(0)

input_tensor = input_tensor.to(
    device
)


# ============================================================
# PREDICTION
# ============================================================

print("\nAnalyzing galaxy...")

with torch.no_grad():

    outputs = model(
        input_tensor
    )

    probabilities = torch.sigmoid(
        outputs
    )[0].cpu().numpy()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GALAXY MORPHOLOGY ANALYSIS")
print("=" * 70)

for i, target in enumerate(TARGETS):

    probability = float(
        probabilities[i]
    )

    threshold = thresholds[target]

    prediction = (
        probability >= threshold
    )

    if target == "smooth_featured":

        if prediction:
            label = "FEATURED"
        else:
            label = "SMOOTH"

    else:

        if prediction:
            label = "YES"
        else:
            label = "NO"

    print(
        f"\n{DISPLAY_NAMES[target]}"
    )

    print(
        f"  Prediction : {label}"
    )

    print(
        f"  Confidence : {probability * 100:.2f}%"
    )

    print(
        f"  Threshold  : {threshold:.2f}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for i, target in enumerate(TARGETS):

    probability = float(
        probabilities[i]
    )

    threshold = thresholds[target]

    prediction = (
        probability >= threshold
    )

    if target == "smooth_featured":

        label = (
            "FEATURED"
            if prediction
            else "SMOOTH"
        )

    else:

        label = (
            "YES"
            if prediction
            else "NO"
        )

    print(
        f"{DISPLAY_NAMES[target]:22s}: "
        f"{label:8s} "
        f"({probability * 100:.1f}%)"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)