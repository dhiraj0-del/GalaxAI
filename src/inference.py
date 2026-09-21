from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models import GalaxyCNNV2

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "checkpoints/galaxai_model1_v2_best.pth"
)

IMAGE_SIZE = 224


# ============================================================
# TASK DEFINITIONS
# ============================================================

BINARY_TASKS = [
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",
]

BINARY_LABELS = {
    "featured": "Smooth / Featured",
    "edge_on": "Edge-on",
    "bar": "Bar",
    "spiral_arms": "Spiral Arms",
    "disturbed": "Disturbed",
    "merger": "Merger",
    "clumpy": "Clumpy Appearance",
    "symmetry": "Galaxy Symmetry",
}

BINARY_THRESHOLDS = {
    task: 0.5
    for task in BINARY_TASKS
}


BULGE_LABELS = [
    "None",
    "Just noticeable",
    "Obvious",
    "Dominant",
]

ROUNDEDNESS_LABELS = [
    "Completely rounded",
    "In-between",
    "Cigar-shaped",
]


# ============================================================
# PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ),
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = GalaxyCNNV2().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(image, model=None):

    if model is None:
        model = load_model()

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = TRANSFORM(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

    results = {}

    # --------------------------------------------------------
    # Binary tasks
    # --------------------------------------------------------

    for task in BINARY_TASKS:

        score = torch.sigmoid(
            outputs[task]
        ).item()

        threshold = BINARY_THRESHOLDS[task]

        results[task] = {
            "label": BINARY_LABELS[task],
            "score": float(score),
            "detected": bool(
                score >= threshold
            ),
        }

    # --------------------------------------------------------
    # Bulge
    # --------------------------------------------------------

    bulge_probs = torch.softmax(
        outputs["bulge"],
        dim=1,
    )[0]

    bulge_class = torch.argmax(
        bulge_probs
    ).item()

    results["bulge"] = {
        "label": "Bulge Prominence",
        "prediction": BULGE_LABELS[bulge_class],
        "class_index": int(bulge_class),
        "score": float(
            bulge_probs[bulge_class].item()
        ),
    }

    # --------------------------------------------------------
    # Roundedness
    # --------------------------------------------------------

    rounded_probs = torch.softmax(
        outputs["roundedness"],
        dim=1,
    )[0]

    rounded_class = torch.argmax(
        rounded_probs
    ).item()

    results["roundedness"] = {
        "label": "Roundedness",
        "prediction": ROUNDEDNESS_LABELS[
            rounded_class
        ],
        "class_index": int(rounded_class),
        "score": float(
            rounded_probs[rounded_class].item()
        ),
    }

    return results


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_info():

    return {
        "model": "GALAXAI Custom CNN V2",
        "parameters": 1_242_863,
        "input_size": "224 × 224 RGB",
        "tasks": 10,
        "output_logits": 15,
        "device": str(DEVICE),
    }