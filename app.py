
from pathlib import Path
import importlib.util
import json
import sys

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# ============================================================
# GALAXAI — Galaxy Morphology Research Laboratory
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="GALAXAI | Galaxy Morphology",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SPACE-THEMED DESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --cyan: #72efff;
    --violet: #a18aff;
    --text: #eef3ff;
    --muted: #b4c2e3;
    --line: rgba(130,164,255,.25);
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(ellipse at 15% 5%, rgba(49,69,155,.38), transparent 35%),
        radial-gradient(ellipse at 85% 20%, rgba(87,39,139,.26), transparent 32%),
        radial-gradient(ellipse at 50% 95%, rgba(14,77,110,.22), transparent 42%),
        #050711;
    background-attachment: fixed;
}

.stApp:before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .35;
    background-image:
        radial-gradient(1px 1px at 7% 18%, #fff 99%, transparent),
        radial-gradient(1px 1px at 24% 78%, #9eeaff 99%, transparent),
        radial-gradient(2px 2px at 44% 32%, #fff 99%, transparent),
        radial-gradient(1px 1px at 66% 14%, #fff 99%, transparent),
        radial-gradient(1px 1px at 83% 61%, #b7b3ff 99%, transparent),
        radial-gradient(1px 1px at 92% 28%, #fff 99%, transparent);
    background-size: 310px 240px;
    animation: star-drift 85s linear infinite;
}

@keyframes star-drift {
    from { background-position: 0 0; }
    to { background-position: 260px 180px; }
}

@keyframes rise-in {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes planet-pulse {
    0%, 100% { transform: scale(1); opacity: .48; }
    50% { transform: scale(1.045); opacity: .72; }
}

.block-container {
    max-width: 1500px;
    padding-top: 1.3rem;
    padding-bottom: 3rem;
    position: relative;
    z-index: 1;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,15,34,.98), rgba(7,9,21,.98));
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] * {
    color: #e7edff;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 300px;
    padding: 2.4rem 2.5rem;
    border: 1px solid rgba(123,170,255,.4);
    border-radius: 28px;
    background:
        radial-gradient(ellipse at 78% 50%, rgba(72,107,218,.23), transparent 35%),
        linear-gradient(118deg, rgba(12,22,51,.96), rgba(10,13,32,.94) 54%, rgba(36,19,67,.89));
    box-shadow:
        0 25px 65px rgba(0,0,0,.38),
        inset 0 1px 0 rgba(255,255,255,.11),
        0 0 38px rgba(83,119,255,.13);
    display: flex;
    flex-direction: column;
    justify-content: center;
    animation: rise-in .75s ease-out both;
    margin-bottom: 1.5rem;
}

.hero:before {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    border-radius: 50%;
    right: 9%;
    top: 12%;
    background: radial-gradient(
        circle at 32% 28%,
        rgba(202,248,255,.9),
        rgba(82,149,255,.58) 12%,
        rgba(71,70,190,.4) 34%,
        rgba(16,21,63,.08) 68%,
        transparent 72%
    );
    box-shadow: 0 0 55px rgba(91,132,255,.25);
    animation: planet-pulse 9s ease-in-out infinite;
    pointer-events: none;
}

.eyebrow {
    color: var(--cyan);
    letter-spacing: .22em;
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Orbitron', sans-serif;
    color: #f7f8ff;
    font-size: clamp(2.4rem, 5vw, 4.1rem);
    letter-spacing: .04em;
    margin: .45rem 0;
    text-shadow: 0 0 26px rgba(114,239,255,.28);
}

.hero p {
    color: #c9d4f5;
    max-width: 950px;
    font-size: 1.03rem;
    line-height: 1.7;
}

h1, h2, h3 {
    font-family: 'Orbitron', 'Inter', sans-serif !important;
    letter-spacing: .015em;
}

h2 { color: #e6eaff; }
h3 { color: #d9e5ff; }
p, li { line-height: 1.65; }

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(25,38,75,.9), rgba(10,15,34,.9));
    border: 1px solid var(--line);
    padding: 15px 17px;
    border-radius: 18px;
    box-shadow: 0 12px 28px rgba(0,0,0,.22);
    transition: transform .22s ease, box-shadow .22s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 32px rgba(0,0,0,.3), 0 0 22px rgba(102,151,255,.13);
}

div[data-testid="stMetricLabel"] { color: #b6c6ee !important; }
div[data-testid="stMetricValue"] { color: #f2f6ff !important; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(130,164,255,.26) !important;
    border-radius: 20px !important;
    background: linear-gradient(145deg, rgba(20,29,60,.72), rgba(9,13,29,.72)) !important;
    box-shadow: 0 14px 32px rgba(0,0,0,.22);
    transition: transform .22s ease, border-color .22s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);
    border-color: rgba(114,239,255,.5) !important;
}

.stButton > button {
    border: 1px solid rgba(117,211,255,.45);
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(43,72,133,.95), rgba(76,49,132,.95));
    color: white;
    font-weight: 650;
    transition: all .2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: var(--cyan);
    box-shadow: 0 0 18px rgba(114,239,255,.18);
    color: white;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #5de4ff, #9b8cff);
}

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(120,191,255,.48);
    border-radius: 18px;
    padding: .55rem;
    background: rgba(13,21,46,.45);
}

.stAlert {
    border-radius: 14px;
}

.stMarkdown code {
    color: #bcefff !important;
    background: rgba(76,108,168,.18) !important;
    border-radius: 5px;
}

.arch-flow {
    display: flex;
    align-items: stretch;
    gap: 10px;
    margin: 1rem 0;
}

.arch-node {
    flex: 1;
    min-width: 0;
    padding: 15px 13px;
    border-radius: 15px;
    border: 1px solid rgba(115,181,255,.32);
    background: linear-gradient(145deg, rgba(36,53,101,.72), rgba(11,18,40,.86));
    transition: transform .24s ease, border-color .24s ease;
}

.arch-node:hover {
    transform: translateY(-4px);
    border-color: var(--cyan);
}

.arch-node span {
    color: var(--cyan);
    font-size: .68rem;
    letter-spacing: .14em;
    font-weight: 800;
}

.arch-node b {
    color: #f2f6ff;
    display: block;
    margin: 7px 0;
}

.arch-node small {
    color: #afc0e5;
    line-height: 1.45;
}

@media (max-width: 850px) {
    .hero { padding: 1.6rem; min-height: 240px; }
    .hero:before { width: 125px; height: 125px; right: 6%; opacity: .3; }
    .arch-flow { flex-direction: column; }
}

@media (prefers-reduced-motion: reduce) {
    *, *:before, *:after {
        animation: none !important;
        transition: none !important;
    }
}
</style>

<div class="hero">
  <div class="eyebrow">HUBBLE SPACE TELESCOPE · MULTI-TASK AI RESEARCH</div>
  <h1>✦ GALAXAI</h1>
  <p>
    <b>Explore the hidden morphology of galaxies.</b>
    A deep-learning research laboratory comparing a custom CNN,
    ResNet-18 and MobileNetV3-Large on ten galaxy morphology attributes.
    Upload an image, inspect model predictions, explore the architectures,
    and compare the held-out test evaluation.
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TASKS, LABELS AND MODEL FILES
# ============================================================

BINARY = [
    "featured", "edge_on", "bar", "spiral_arms",
    "disturbed", "merger", "clumpy", "symmetry",
]

MULTI = ["bulge", "roundedness"]
TASKS = BINARY + MULTI

LABELS = {
    "featured": "Featured",
    "edge_on": "Edge-on",
    "bar": "Bar",
    "spiral_arms": "Spiral arms",
    "disturbed": "Disturbed",
    "merger": "Merger",
    "clumpy": "Clumpy",
    "symmetry": "Symmetry",
    "bulge": "Bulge prominence",
    "roundedness": "Roundedness",
}

CLASSES = {
    "bulge": ["None", "Just noticeable", "Obvious", "Dominant"],
    "roundedness": [
        "Completely rounded",
        "In-between",
        "Cigar-shaped",
    ],
}

MODELS = {
    "Model 1 — Custom CNN V2": {
        "architecture": "Custom CNN V2",
        "description": (
            "Four convolutional blocks with 32, 64, 128 and 256 channels, "
            "adaptive average pooling, and a shared 256-unit head."
        ),
        "metrics": [
            BASE_DIR / "models/model1_custom_cnn/results/test_metrics.json",
            BASE_DIR / "results/model1_v2/test_metrics.json",
        ],
        "heatmaps": BASE_DIR / "models/model1_custom_cnn/results",
        "best_epoch": 27,
        "parameters": "1,242,863",
        "checkpoint": BASE_DIR / "checkpoints/galaxai_model1_v2_best.pth",
        "module_path": BASE_DIR / "src/models.py",
        "class_name": "GalaxyCNNV2",
    },
    "Model 2 — ResNet-18": {
        "architecture": "ImageNet-pretrained ResNet-18",
        "description": (
            "ResNet-18 backbone producing 512 features, a shared 256-unit "
            "head, and ten task-specific prediction heads."
        ),
        "metrics": [
            BASE_DIR / "models/model2_resnet/results/test_metrics.json",
            BASE_DIR / "results/model2_resnet/test_metrics.json",
        ],
        "heatmaps": BASE_DIR / "models/model2_resnet/results",
        "best_epoch": 2,
        "parameters": "See model definition",
        "checkpoint": (
            BASE_DIR / "models/model2_resnet/checkpoints/"
            "galaxai_model2_resnet18_best.pth"
        ),
        "module_path": BASE_DIR / "models/model2_resnet/models.py",
        "class_name": "GalaxyResNet18",
    },
    "Model 3 — MobileNetV3-Large": {
        "architecture": "ImageNet-pretrained MobileNetV3-Large",
        "description": (
            "MobileNetV3-Large backbone producing 960 features, a shared "
            "256-unit head, and ten task-specific prediction heads."
        ),
        "metrics": [
            BASE_DIR / "models/model3_mobilenet/results/test_metrics.json",
            BASE_DIR / "results/model3_mobilenet/results/test_metrics.json",
        ],
        "heatmaps": BASE_DIR / "models/model3_mobilenet/results",
        "best_epoch": 3,
        "parameters": "3,221,823",
        "checkpoint": (
            BASE_DIR / "models/model3_mobilenet/checkpoints/"
            "galaxai_model3_mobilenetv3_large_best.pth"
        ),
        "module_path": BASE_DIR / "models/model3_mobilenet/models.py",
        "class_name": "GalaxyMobileNetV3",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


@st.cache_data(show_spinner=False)
def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as file:
            result = json.load(file)
        return result if isinstance(result, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_metrics(model_name):
    for path in MODELS[model_name]["metrics"]:
        if path.is_file():
            return read_json(path), path
    return None, None


def fmt(value, percent=False):
    try:
        number = float(value)
        return f"{number * 100:.2f}%" if percent else f"{number:.4f}"
    except (TypeError, ValueError):
        return "—"


def get_summary(data):
    return data.get("summary", {}) if data else {}

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading neural network...")
def load_model(model_name):
    config = MODELS[model_name]
    checkpoint_path = config["checkpoint"]
    module_path = config["module_path"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not module_path.is_file():
        raise FileNotFoundError(
            f"Model definition not found: {module_path}"
        )

    # Load the correct architecture module independently.
    module_name = {
        "Model 1 — Custom CNN V2": "galaxai_model1_module",
        "Model 2 — ResNet-18": "galaxai_model2_module",
        "Model 3 — MobileNetV3-Large": "galaxai_model3_module",
    }[model_name]

    spec = importlib.util.spec_from_file_location(
        module_name, str(module_path)
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to import model definition: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    model_class = getattr(module, config["class_name"], None)

    if model_class is None:
        raise AttributeError(
            f"Class {config['class_name']} was not found in "
            f"{module_path.name}. Check the class name in models.py."
        )

    # All three architectures support pretrained=False.
    # For inference, the trained checkpoint supplies the weights.
    try:
        model = model_class(pretrained=False)
    except TypeError:
        model = model_class()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
        )

        if state_dict is None:
            # Some checkpoints contain only the state dictionary.
            if checkpoint and all(
                isinstance(value, torch.Tensor)
                for value in checkpoint.values()
            ):
                state_dict = checkpoint
            else:
                raise ValueError(
                    f"Could not locate model_state_dict in "
                    f"{checkpoint_path.name}."
                )
    else:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint_path.name}"
        )

    # Strict loading catches architecture/checkpoint mismatches.
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# OUTPUT DECODING
# ============================================================

def unpack_outputs(outputs):
    """
    Supports the task-output dictionary used by GalaxAI models.
    """
    if not isinstance(outputs, dict):
        raise TypeError(
            "The model must return a dictionary of task logits."
        )

    missing = [task for task in TASKS if task not in outputs]

    if missing:
        raise ValueError(
            "Missing model output(s): " + ", ".join(missing)
        )

    return outputs


def predict_image(model, image):
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = unpack_outputs(model(tensor))

    predictions = {}

    for task in BINARY:
        logits = outputs[task].reshape(-1)
        probability = torch.sigmoid(logits)[0].item()

        # Match the 0.70 decision threshold used in evaluation.
        predictions[task] = {
            "type": "binary",
            "probability": probability,
            "label": (
                "Positive" if probability >= 0.70
                else "Negative"
            ),
            "confidence": (
                probability if probability >= 0.70
                else 1.0 - probability
            ),
        }

    for task in MULTI:
        logits = outputs[task].reshape(1, -1)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(probabilities.argmax().item())

        predictions[task] = {
            "type": "multiclass",
            "label": CLASSES[task][index],
            "class_index": index,
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                name: float(probabilities[i].item())
                for i, name in enumerate(CLASSES[task])
            },
        }

    return predictions


# ============================================================
# VISUAL ARCHITECTURE
# ============================================================

def architecture_flow(model_name):
    if model_name == "Model 1 — Custom CNN V2":
        nodes = [
            ("01", "INPUT", "224 × 224 × 3 RGB galaxy image"),
            ("02", "CNN BLOCKS", "3→32→64→128→256 channels; convolution, BN, ReLU and pooling"),
            ("03", "POOLING", "Adaptive average pooling and flatten"),
            ("04", "SHARED HEAD", "256 features, ReLU, dropout 0.40"),
            ("05", "TASK HEADS", "8 binary outputs + bulge (4) + roundedness (3)"),
        ]
    elif model_name == "Model 2 — ResNet-18":
        nodes = [
            ("01", "INPUT", "224 × 224 × 3 RGB galaxy image"),
            ("02", "RESNET-18", "Convolutional stem and residual layers 1–4"),
            ("03", "FEATURES", "Global average pooling; 512 features"),
            ("04", "SHARED HEAD", "Linear 512→256, ReLU, dropout 0.40"),
            ("05", "TASK HEADS", "8 binary outputs + bulge (4) + roundedness (3)"),
        ]
    else:
        nodes = [
            ("01", "INPUT", "224 × 224 × 3 RGB galaxy image"),
            ("02", "MOBILENETV3", "Large backbone with inverted residual blocks and squeeze-excitation"),
            ("03", "FEATURES", "Backbone feature representation; 960 features"),
            ("04", "SHARED HEAD", "Linear 960→256, ReLU, dropout 0.40"),
            ("05", "TASK HEADS", "8 binary outputs + bulge (4) + roundedness (3)"),
        ]

    html = '<div class="arch-flow">'

    for number, title, detail in nodes:
        html += (
            '<div class="arch-node">'
            f'<span>{number} / PIPELINE</span>'
            f'<b>{title}</b>'
            f'<small>{detail}</small>'
            '</div>'
        )

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

    st.caption(
        "All models receive the same 224×224 RGB input and produce "
        "15 logits across ten morphology tasks. Arrows are represented "
        "by the left-to-right pipeline order."
    )


def show_architecture_details(model_name):
    config = MODELS[model_name]

    st.subheader("Architecture blueprint")
    architecture_flow(model_name)

    st.markdown("### Design specification")
    st.write(config["description"])

    if model_name == "Model 1 — Custom CNN V2":
        st.markdown("""
        - Four convolutional blocks.
        - Each block uses two 3×3 convolutions, batch normalization,
          ReLU activation and max pooling.
        - Channel progression: 3 → 32 → 64 → 128 → 256.
        - Adaptive average pooling produces a compact feature vector.
        - Shared fully connected layer: 256 → 256.
        - Dropout probability: 0.40.
        """)
    elif model_name == "Model 2 — ResNet-18":
        st.markdown("""
        - ImageNet-pretrained ResNet-18 backbone.
        - Residual blocks help preserve gradient flow through the network.
        - Original 1000-class classifier is replaced by a 512-feature
          representation and a custom shared head.
        - Shared head: 512 → 256, ReLU, dropout 0.40.
        - Separate heads for each binary and multiclass attribute.
        """)
    else:
        st.markdown("""
        - ImageNet-pretrained MobileNetV3-Large backbone.
        - Efficient inverted residual blocks and lightweight convolutions.
        - Backbone representation: 960 features.
        - Shared head: 960 → 256, ReLU, dropout 0.40.
        - Separate heads for each binary and multiclass attribute.
        """)

    st.markdown("### Prediction head dimensions")

    head_data = [
        {
            "Task": LABELS[task],
            "Output type": "Binary",
            "Logits": 1,
            "Decision": "Sigmoid probability; threshold 0.70",
        }
        for task in BINARY
    ]

    head_data.extend([
        {
            "Task": LABELS["bulge"],
            "Output type": "Multiclass",
            "Logits": 4,
            "Decision": "Softmax across four classes",
        },
        {
            "Task": LABELS["roundedness"],
            "Output type": "Multiclass",
            "Logits": 3,
            "Decision": "Softmax across three classes",
        },
    ])

    st.dataframe(
        pd.DataFrame(head_data),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Training and checkpoint")

    a, b, c = st.columns(3)

    a.metric("Best validation epoch", config["best_epoch"])
    b.metric("Input resolution", "224 × 224")
    c.metric("Compute device", str(DEVICE).upper())

    if config["checkpoint"].is_file():
        st.success(
            "Checkpoint file found: " + config["checkpoint"].name
        )
    else:
        st.error(
            "Checkpoint is missing: " + config["checkpoint"].name
        )

    st.caption(
        "Architecture descriptions are based on the project configuration. "
        "Check the corresponding models.py file for the exact implementation."
    )


# ============================================================
# LIVE IMAGE PREDICTION PAGE
# ============================================================

def prediction_page(model_name):
    config = MODELS[model_name]

    st.title(model_name)
    st.write(config["description"])

    tab_predict, tab_architecture = st.tabs([
        "✦ Live prediction",
        "⌘ Architecture",
    ])

    with tab_predict:
        st.markdown("### Upload a galaxy image")

        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
            key="upload_" + model_name,
            help="Upload a galaxy image. The model uses 224×224 RGB input.",
        )

        if uploaded is None:
            st.info(
                "Upload a galaxy image to generate predictions for "
                "all ten morphology attributes."
            )

            st.markdown("""
            **What this model predicts**

            - Featured
            - Edge-on
            - Bar
            - Spiral arms
            - Disturbed
            - Merger
            - Clumpy
            - Symmetry
            - Bulge prominence
            - Roundedness
            """)

        else:
            try:
                image = Image.open(uploaded).convert("RGB")

                left, right = st.columns([1, 1.25], gap="large")

                with left:
                    st.image(
                        image,
                        caption="Uploaded galaxy image",
                        use_container_width=True,
                    )

                    st.caption(
                        f"Original image: {image.width} × {image.height}"
                    )

                with right:
                    with st.spinner(
                        f"Loading {config['architecture']} and predicting..."
                    ):
                        model = load_model(model_name)
                        predictions = predict_image(model, image)

                    st.success("Inference completed.")

                    st.markdown("### Morphology results")

                    for task in BINARY:
                        result = predictions[task]
                        probability = result["probability"]

                        with st.container(border=True):
                            st.markdown(f"**{LABELS[task]}**")

                            x, y = st.columns([1, 1])

                            x.metric(
                                "Predicted label",
                                result["label"],
                            )
                            y.metric(
                                "Positive probability",
                                f"{probability:.1%}",
                            )

                            st.progress(
                                max(0.0, min(1.0, probability)),
                                text="Positive-class probability",
                            )

                    st.markdown("### Multiclass attributes")

                    for task in MULTI:
                        result = predictions[task]

                        with st.container(border=True):
                            st.markdown(f"**{LABELS[task]}**")

                            st.metric(
                                "Predicted class",
                                result["label"],
                                f"Confidence {result['confidence']:.1%}",
                            )

                            for name, probability in result[
                                "probabilities"
                            ].items():
                                st.progress(
                                    max(0.0, min(1.0, probability)),
                                    text=f"{name}: {probability:.1%}",
                                )

                    export = {
                        "model": model_name,
                        "device": str(DEVICE),
                        "image": uploaded.name,
                        "predictions": predictions,
                    }

                    st.download_button(
                        "Download prediction results (JSON)",
                        data=json.dumps(export, indent=2),
                        file_name="galaxai_prediction.json",
                        mime="application/json",
                        key="download_" + model_name,
                    )

                    st.caption(
                        "Probabilities are model outputs, not calibrated "
                        "scientific certainty. Binary labels use the "
                        "0.70 threshold; validate threshold behaviour "
                        "before using predictions in research conclusions."
                    )

            except Exception as exc:
                st.error("Prediction could not be completed.")
                st.exception(exc)

                st.info(
                    "Check that the checkpoint, model definition, "
                    "class name and checkpoint state dictionary match."
                )

    with tab_architecture:
        show_architecture_details(model_name)

# ============================================================
# HOME PAGE
# ============================================================

def home_page():
    st.markdown("""
    <div class="eyebrow">MISSION CONTROL / RESEARCH OVERVIEW</div>
    <h1>Mapping the morphology of galaxies</h1>
    <p>
    How effectively can different CNN architectures learn multiple
    morphological attributes from Hubble Space Telescope galaxy images,
    and how do architectural differences affect common and rare features?
    </p>
    """, unsafe_allow_html=True)

    st.markdown("## Research at a glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Image records", "80,632")
    c2.metric("Model architectures", "3")
    c3.metric("Morphology tasks", "10")
    c4.metric("Output logits", "15")

    st.markdown("## Explore the laboratory")

    cards = [
        (
            "Model 1",
            "Custom CNN V2",
            "A convolutional baseline with four feature-extraction blocks.",
            "Model 1 — Custom CNN V2",
        ),
        (
            "Model 2",
            "ResNet-18",
            "A residual network with an ImageNet-pretrained backbone.",
            "Model 2 — ResNet-18",
        ),
        (
            "Model 3",
            "MobileNetV3-Large",
            "An efficient convolutional architecture with a custom multi-task head.",
            "Model 3 — MobileNetV3-Large",
        ),
    ]

    cols = st.columns(3, gap="large")

    for col, (number, title, description, model_name) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"### ✦ {number}")
                st.subheader(title)
                st.write(description)

                if st.button(
                    "Open prediction →",
                    key="home_predict_" + number,
                    use_container_width=True,
                ):
                    st.session_state["page"] = model_name
                    st.rerun()

                if st.button(
                    "Open architecture ↗",
                    key="home_arch_" + number,
                    use_container_width=True,
                ):
                    st.session_state["page"] = model_name
                    st.session_state["open_architecture"] = True
                    st.rerun()

    st.markdown("## Research workflow")

    st.markdown("""
    <div class="arch-flow">
      <div class="arch-node">
        <span>01 / DATA</span>
        <b>Galaxy Zoo: Hubble</b>
        <small>Galaxy images and morphology annotations.</small>
      </div>
      <div class="arch-node">
        <span>02 / PREPROCESS</span>
        <b>Image preparation</b>
        <small>RGB, resize to 224×224, normalize to [-1, 1].</small>
      </div>
      <div class="arch-node">
        <span>03 / TRAIN</span>
        <b>Multi-task learning</b>
        <small>Eight binary tasks and two multiclass tasks.</small>
      </div>
      <div class="arch-node">
        <span>04 / EVALUATE</span>
        <b>Held-out test set</b>
        <small>Task-level metrics and confusion matrices.</small>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset and experimental design")

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Dataset split")

            split_data = pd.DataFrame({
                "Split": ["Training", "Validation", "Test"],
                "Images": [56442, 12095, 12095],
            })

            st.bar_chart(
                split_data.set_index("Split"),
                y="Images",
            )

            st.caption(
                "70% training, 15% validation and 15% test; "
                "random_state=42."
            )

    with right:
        with st.container(border=True):
            st.subheader("Ten morphology attributes")

            for task in BINARY:
                st.markdown(f"- **{LABELS[task]}:** binary classification")

            for task in MULTI:
                st.markdown(
                    f"- **{LABELS[task]}:** "
                    f"{len(CLASSES[task])}-class classification"
                )

    st.markdown("## Architecture at a glance")

    for model_name, config in MODELS.items():
        with st.expander(
            f"{config['architecture']} — architecture overview"
        ):
            st.write(config["description"])
            architecture_flow(model_name)

            if st.button(
                "Explore full architecture",
                key="home_full_arch_" + model_name,
            ):
                st.session_state["page"] = model_name
                st.session_state["open_architecture"] = True
                st.rerun()

    st.markdown("## Research principles")

    p1, p2, p3 = st.columns(3)

    with p1:
        with st.container(border=True):
            st.subheader("Consistent evaluation")
            st.write(
                "Use the same held-out test split and task definitions "
                "when comparing all three architectures."
            )

    with p2:
        with st.container(border=True):
            st.subheader("Class imbalance matters")
            st.write(
                "Accuracy alone can hide poor recognition of rare features. "
                "Inspect precision, recall, F1 and confusion matrices."
            )

    with p3:
        with st.container(border=True):
            st.subheader("Scientific limitations")
            st.write(
                "Predictions reflect the learned dataset labels and "
                "annotation thresholds. They are not ground truth "
                "for every individual galaxy."
            )


# ============================================================
# METRIC EXTRACTION
# ============================================================

def find_metric(metrics, candidates):
    """
    Search known metric names without assuming one particular JSON schema.
    """
    if not isinstance(metrics, dict):
        return None

    for key in candidates:
        if key in metrics and isinstance(
            metrics[key], (int, float)
        ):
            return metrics[key]

    return None


def get_model_score(metrics):
    """
    Try the documented summary keys first.
    Fall back to the task-level scores if necessary.
    """
    if not metrics:
        return None

    summary = metrics.get("summary", {})

    for key in [
        "overall_macro_task_f1",
        "overall_macro_f1",
        "macro_task_f1",
    ]:
        if key in summary:
            return summary[key]

        if key in metrics:
            return metrics[key]

    task_scores = []

    task_metrics = metrics.get("tasks", {})

    if isinstance(task_metrics, dict):
        for task, values in task_metrics.items():
            if isinstance(values, dict):
                score = find_metric(
                    values,
                    ["f1", "f1_score", "macro_f1"],
                )
                if score is not None:
                    task_scores.append(float(score))

    if task_scores:
        return sum(task_scores) / len(task_scores)

    return None


def find_task_metrics(metrics):
    """
    Support a few common JSON layouts:
    tasks -> task name -> metric dictionary,
    or task names directly at the root.
    """
    if not isinstance(metrics, dict):
        return {}

    if isinstance(metrics.get("tasks"), dict):
        return metrics["tasks"]

    found = {}

    for task in TASKS:
        value = metrics.get(task)

        if isinstance(value, dict):
            found[task] = value

    return found


# ============================================================
# MODEL COMPARISON PAGE
# ============================================================

def comparison_page():
    st.title("Three-model comparison")
    st.write(
        "Compare the custom CNN baseline, ResNet-18 and "
        "MobileNetV3-Large using the saved held-out test evaluations."
    )

    loaded = {}
    missing = []

    for model_name in MODELS:
        metrics, path = load_metrics(model_name)

        if metrics:
            loaded[model_name] = {
                "metrics": metrics,
                "path": path,
            }
        else:
            missing.append(model_name)

    if missing:
        st.warning(
            "Evaluation JSON not found for: "
            + ", ".join(missing)
            + ". Check the results folders and metric file names."
        )

    if not loaded:
        st.error(
            "No test_metrics.json files were found. "
            "The comparison dashboard needs the saved evaluation JSONs."
        )
        st.markdown("Expected locations include:")

        for model_name, config in MODELS.items():
            for path in config["metrics"]:
                st.markdown(f"- {model_name}: {path}")

        return

    st.markdown("## Overall performance")

    rows = []

    # Values recorded from the completed test evaluations.
    # These are used as fallbacks if a JSON summary uses another schema.
    recorded = {
        "Model 1 — Custom CNN V2": {
            "pooled_accuracy": 0.8341,
            "binary_macro_f1": 0.5262,
            "multiclass_macro_f1": 0.7363,
            "overall_macro_task_f1": 0.5682,
        },
        "Model 2 — ResNet-18": {
            "pooled_accuracy": 0.8748,
            "binary_macro_f1": 0.5889,
            "multiclass_macro_f1": 0.8011,
            "overall_macro_task_f1": 0.6313,
        },
        "Model 3 — MobileNetV3-Large": {
            "pooled_accuracy": 0.8926,
            "binary_macro_f1": 0.6152,
            "multiclass_macro_f1": 0.8289,
            "overall_macro_task_f1": 0.6579,
        },
    }

    for model_name, item in loaded.items():
        metrics = item["metrics"]
        summary = metrics.get("summary", {})
        fallback = recorded.get(model_name, {})

        def summary_value(keys, fallback_key):
            value = find_metric(summary, keys)

            if value is None:
                value = find_metric(metrics, keys)

            if value is None:
                value = fallback.get(fallback_key)

            return value

        rows.append({
            "Model": model_name,
            "Pooled accuracy": summary_value(
                [
                    "pooled_accuracy",
                    "overall_pooled_accuracy",
                    "test_pooled_accuracy",
                ],
                "pooled_accuracy",
            ),
            "Binary macro F1": summary_value(
                ["binary_macro_f1", "binary_f1_macro"],
                "binary_macro_f1",
            ),
            "Multiclass macro F1": summary_value(
                [
                    "multiclass_macro_f1",
                    "multiclass_f1_macro",
                ],
                "multiclass_macro_f1",
            ),
            "Overall macro task F1": summary_value(
                [
                    "overall_macro_task_f1",
                    "overall_macro_f1",
                    "macro_task_f1",
                ],
                "overall_macro_task_f1",
            ),
        })

    comparison = pd.DataFrame(rows)

    display_comparison = comparison.copy()

    for column in [
        "Pooled accuracy",
        "Binary macro F1",
        "Multiclass macro F1",
        "Overall macro task F1",
    ]:
        display_comparison[column] = display_comparison[column].apply(
            lambda value: (
                f"{float(value) * 100:.2f}%"
                if pd.notna(value) else "—"
            )
        )

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Pooled accuracy aggregates valid task predictions; it is not "
        "the same as accuracy for one mutually exclusive classification. "
        "Macro F1 averages task-level F1 scores. Fallback figures above "
        "are the previously recorded test results; the saved JSON is "
        "preferred whenever it exposes the relevant summary fields."
    )

    chart_column = "Overall macro task F1"

    if comparison[chart_column].notna().any():
        chart_data = comparison.dropna(
            subset=[chart_column]
        ).set_index("Model")[[chart_column]]

        st.markdown("### Overall macro task F1")
        st.bar_chart(chart_data)

    st.markdown("## Task-by-task comparison")

    task_rows = []

    for model_name, item in loaded.items():
        metrics = item["metrics"]
        task_metrics = find_task_metrics(metrics)

        for task in TASKS:
            values = task_metrics.get(task, {})

            if not isinstance(values, dict):
                values = {}

            f1 = find_metric(
                values,
                ["f1", "f1_score", "macro_f1"],
            )

            precision = find_metric(
                values,
                ["precision", "macro_precision"],
            )

            recall = find_metric(
                values,
                ["recall", "macro_recall"],
            )

            # Use recorded per-task F1 results if needed.
            recorded_f1 = {
                "Model 1 — Custom CNN V2": {
                    "featured": .9503,
                    "roundedness": .8286,
                    "symmetry": .7531,
                    "clumpy": .7123,
                    "bar": .1788,
                    "merger": .1634,
                    "disturbed": .0843,
                },
                "Model 2 — ResNet-18": {
                    "featured": .9660,
                    "edge_on": .8287,
                    "bar": .2742,
                    "spiral_arms": .7152,
                    "disturbed": .0751,
                    "merger": .3108,
                    "clumpy": .7756,
                    "symmetry": .7655,
                    "bulge": .7034,
                    "roundedness": .8988,
                },
                "Model 3 — MobileNetV3-Large": {
                    "featured": .9791,
                    "edge_on": .8392,
                    "bar": .3074,
                    "spiral_arms": .7156,
                    "disturbed": .0792,
                    "merger": .4274,
                    "clumpy": .7921,
                    "symmetry": .7815,
                    "bulge": .7221,
                    "roundedness": .9357,
                },
            }

            if f1 is None:
                f1 = recorded_f1.get(model_name, {}).get(task)

            task_rows.append({
                "Model": model_name,
                "Task": LABELS[task],
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
            })

    task_df = pd.DataFrame(task_rows)

    if not task_df.empty:
        chosen_task = st.selectbox(
            "Select a morphology attribute",
            [LABELS[t] for t in TASKS],
        )

        selected = task_df[task_df["Task"] == chosen_task].copy()

        if selected["F1"].notna().any():
            chart = selected.dropna(subset=["F1"]).set_index("Model")[["F1"]]
            st.bar_chart(chart)

        st.dataframe(
            selected,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("## Confusion matrices")

    chosen_model = st.selectbox(
        "Select a model",
        list(loaded.keys()),
        key="heatmap_model",
    )

    chosen_task = st.selectbox(
        "Select a task",
        TASKS,
        format_func=lambda task: LABELS[task],
        key="heatmap_task",
    )

    heatmap_dir = MODELS[chosen_model]["heatmaps"]

    possible_files = [
        heatmap_dir / f"{chosen_task}_confusion_matrix.png",
        heatmap_dir / f"{chosen_task}_confusion_matrix.jpg",
        heatmap_dir / f"{chosen_task}_confusion_matrix.png.png",
        heatmap_dir / f"{chosen_task}_cm.png",
    ]

    found_image = next(
        (path for path in possible_files if path.is_file()),
        None,
    )

    if found_image:
        st.image(
            str(found_image),
            caption=f"{chosen_model} — {LABELS[chosen_task]}",
            use_container_width=True,
        )
    else:
        st.info(
            "The heatmap file was not found at the expected path. "
            "Check the filenames in: " + str(heatmap_dir)
        )

    st.markdown("## Evaluation files")

    for model_name, item in loaded.items():
        with st.expander(model_name):
            st.write("Metrics file:", str(item["path"]))
            st.write("Architecture:", MODELS[model_name]["architecture"])
            st.write("Best epoch:", MODELS[model_name]["best_epoch"])


# ============================================================
# APP NAVIGATION
# ============================================================

PAGE_HOME = "Home"
PAGE_COMPARISON = "Three-model comparison"

PAGE_OPTIONS = [
    PAGE_HOME,
    "Model 1 — Custom CNN V2",
    "Model 2 — ResNet-18",
    "Model 3 — MobileNetV3-Large",
    PAGE_COMPARISON,
]

if "page" not in st.session_state:
    st.session_state["page"] = PAGE_HOME

if st.session_state["page"] not in PAGE_OPTIONS:
    st.session_state["page"] = PAGE_HOME

with st.sidebar:
    st.markdown("## ✦ GALAXAI")
    st.caption("Galaxy morphology research lab")
    st.divider()

    st.markdown("### Navigation")

    selected_page = st.radio(
        "Choose a destination",
        PAGE_OPTIONS,
        index=PAGE_OPTIONS.index(st.session_state["page"]),
        key="navigation_radio",
        label_visibility="collapsed",
    )

    if selected_page != st.session_state["page"]:
        st.session_state["page"] = selected_page
        st.session_state["open_architecture"] = False
        st.rerun()

    st.divider()

    st.markdown("### System status")

    st.markdown(
        "🟢 CUDA available"
        if torch.cuda.is_available()
        else "⚪ CPU inference"
    )

    st.caption(f"Device: {DEVICE}")

    st.markdown("### Checkpoints")

    for model_name, config in MODELS.items():
        if config["checkpoint"].is_file():
            st.markdown(f"🟢 {config['architecture']}")
        else:
            st.markdown(f"🔴 {config['architecture']}")

    st.divider()

    st.caption(
        "Research prototype · Galaxy Zoo: Hubble · "
        "Multi-task deep learning"
    )


# ============================================================
# PAGE ROUTER
# ============================================================

current_page = st.session_state["page"]

if current_page == PAGE_HOME:
    home_page()

elif current_page == PAGE_COMPARISON:
    comparison_page()

elif current_page in MODELS:
    prediction_page(current_page)

else:
    st.session_state["page"] = PAGE_HOME
    st.rerun()


# ============================================================
# ARCHITECTURE DEEP-LINK SUPPORT
# ============================================================

if st.session_state.get("open_architecture", False):
    st.session_state["open_architecture"] = False

    # The model page contains a dedicated architecture tab.
    # The session flag is consumed after navigation to avoid
    # repeatedly triggering reruns.
