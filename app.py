import os
import sys
import json

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# GALAXAI - MODEL 1
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

IMAGE_SIZE = 224


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "spiral_arms",
    "bar",
    "smooth_featured",
    "disturbed"
]


DISPLAY_NAMES = {
    "spiral_arms": "Spiral Arms",
    "bar": "Bar",
    "smooth_featured": "Smooth vs Featured",
    "disturbed": "Disturbed"
}


ICONS = {
    "spiral_arms": "🌀",
    "bar": "▰",
    "smooth_featured": "✨",
    "disturbed": "💥"
}


# ============================================================
# MODEL 1 - FINAL TEST PERFORMANCE
#
# These are the FINAL TEST results obtained after applying
# thresholds selected ONLY from the validation set.
# ============================================================

FINAL_METRICS = {

    "spiral_arms": {
        "accuracy": 0.8930,
        "precision": 0.6736,
        "recall": 0.6614,
        "f1": 0.6674,
        "roc_auc": 0.8680,
        "pr_auc": 0.7077,
    },

    "bar": {
        "accuracy": 0.8657,
        "precision": 0.0997,
        "recall": 0.3333,
        "f1": 0.1535,
        "roc_auc": 0.6844,
        "pr_auc": 0.0866,
    },

    "smooth_featured": {
        "accuracy": 0.9236,
        "precision": 0.9350,
        "recall": 0.9783,
        "f1": 0.9562,
        "roc_auc": 0.9251,
        "pr_auc": 0.9814,
    },

    "disturbed": {
        "accuracy": 0.6156,
        "precision": 0.0516,
        "recall": 0.5243,
        "f1": 0.0939,
        "roc_auc": 0.6301,
        "pr_auc": 0.0487,
    }
}


# ============================================================
# FINAL TEST CONFUSION MATRICES
#
# Format:
# [[TN, FP],
#  [FN, TP]]
# ============================================================

FINAL_CONFUSION_MATRICES = {

    "spiral_arms": [
        [2130, 141],
        [149, 291]
    ],

    "bar": [
        [2314, 298],
        [66, 33]
    ],

    "smooth_featured": [
        [247, 157],
        [50, 2257]
    ],

    "disturbed": [
        [1615, 993],
        [49, 54]
    ]
}


# ============================================================
# OVERALL MODEL 1 TEST PERFORMANCE
# ============================================================

OVERALL_METRICS = {
    "accuracy": 0.8245,
    "precision": 0.4400,
    "recall": 0.6243,
    "f1": 0.4677,
    "roc_auc": 0.7769,
    "pr_auc": 0.4561
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GalaxAI | Galaxy Morphology",
    page_icon="🌌",
    layout="wide"
)


# ============================================================
# ============================================================
# STANDARD STREAMLIT INTERFACE
# Custom CSS removed for maximum reliability.
# ============================================================


# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    src_path = os.path.join(
        BASE_DIR,
        "src"
    )


    if src_path not in sys.path:

        sys.path.insert(
            0,
            src_path
        )


    from models import CustomCNN


    model = CustomCNN(
        num_classes=4
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )


    model = model.to(
        device
    )


    model.eval()


    return model, device


# ============================================================
# LOAD THRESHOLDS
# ============================================================

@st.cache_data
def load_thresholds():

    with open(
        THRESHOLD_PATH,
        "r"
    ) as file:

        data = json.load(file)


    thresholds = {}


    for target in TARGETS:

        value = data[target]


        if isinstance(
            value,
            dict
        ):

            thresholds[target] = float(
                value["threshold"]
            )

        else:

            thresholds[target] = float(
                value
            )


    return thresholds


# ============================================================
# PREPROCESSING
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
# LOAD MODEL + THRESHOLDS
# ============================================================

try:

    model, device = load_model()

    thresholds = load_thresholds()

except Exception as e:

    st.error(
        f"Unable to load Model 1: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌌 GalaxAI"
)


st.subheader(
    "AI-Powered Multi-Attribute Galaxy Morphology Analysis"
)


st.write(
    "Deep learning analysis of Hubble Space Telescope "
    "galaxy images using a Custom Convolutional Neural Network."
)


# ============================================================
# MODEL STATUS
# ============================================================

if device.type == "cuda":

    gpu_name = torch.cuda.get_device_name(0)

    st.success(
        f"⚡ Model Ready • GPU Accelerated • {gpu_name}"
    )

else:

    st.warning(
        "Model Ready • Running on CPU"
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander(
    "🔬 About this system",
    expanded=False
):

    st.write(
        "**Dataset:** Galaxy Zoo: Hubble"
    )

    st.write(
        "**Architecture:** Custom CNN"
    )

    st.write(
        "**Input:** Galaxy image"
    )

    st.write(
        "**Outputs:** Four independent morphological attributes"
    )

    st.write(
        "**Inference device:** "
        +
        (
            "NVIDIA GPU"
            if device.type == "cuda"
            else "CPU"
        )
    )


# ============================================================
# UPLOAD
# ============================================================

st.divider()


st.header(
    "🔭 Analyze a Galaxy"
)


st.write(
    "Upload a JPG or PNG image and let the Custom CNN "
    "analyze its morphology."
)


uploaded_file = st.file_uploader(
    "Choose a galaxy image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE UPLOADED
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.divider()


    # ========================================================
    # IMAGE + CONTROL
    # ========================================================

    image_col, control_col = st.columns(
        [1.25, 1]
    )


    # --------------------------------------------------------
    # GALAXY IMAGE
    # --------------------------------------------------------

    with image_col:

        st.subheader(
            "🛰️ Galaxy Image"
        )


        st.image(
            image,
            use_container_width=True
        )


        st.caption(
            f"Original image: "
            f"{image.width} × "
            f"{image.height} pixels"
        )


    # --------------------------------------------------------
    # ANALYSIS CONTROL
    # --------------------------------------------------------

    with control_col:

        st.subheader(
            "🧠 AI Analysis"
        )


        st.write(
            "The Custom CNN evaluates four "
            "morphological attributes independently."
        )


        st.info(
            "Each attribute uses its own "
            "validation-derived decision threshold."
        )


        analyze = st.button(
            "🔬 ANALYZE GALAXY",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if analyze:

        with st.spinner(
            "Analyzing galaxy morphology..."
        ):

            input_tensor = transform(
                image
            ).unsqueeze(0)


            input_tensor = input_tensor.to(
                device
            )


            with torch.no_grad():

                outputs = model(
                    input_tensor
                )


                probabilities = torch.sigmoid(
                    outputs
                )[0].cpu().numpy()


        # ====================================================
        # MORPHOLOGY RESULTS
        # ====================================================

        st.divider()


        st.header(
            "🧬 Morphology Analysis"
        )


        st.write(
            "Four independent morphology predictions "
            "generated by Model 1."
        )


        for i, target in enumerate(TARGETS):

            probability = float(
                probabilities[i]
            )


            threshold = thresholds[target]


            prediction = (
                probability >= threshold
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

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


            # ------------------------------------------------
            # RESULT CONTAINER
            # ------------------------------------------------

            with st.container(
                border=True
            ):

                result_col1, result_col2 = st.columns(
                    [2, 1]
                )


                with result_col1:

                    st.subheader(
                        f"{ICONS[target]} "
                        f"{DISPLAY_NAMES[target]}"
                    )


                    if label in [
                        "YES",
                        "FEATURED"
                    ]:

                        st.success(
                            f"Prediction: {label}"
                        )

                    else:

                        st.info(
                            f"Prediction: {label}"
                        )


                with result_col2:

                    st.metric(
                        "Model score",
                        f"{probability * 100:.1f}%"
                    )


                st.progress(
                    min(
                        probability,
                        1.0
                    )
                )


                st.caption(
                    f"Decision threshold: "
                    f"{threshold:.2f}"
                )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.divider()


        st.subheader(
            "ℹ️ How to interpret the results"
        )


        st.info(
            "The four morphology attributes are evaluated "
            "independently. Their scores should NOT be compared "
            "against each other. Each attribute has its own "
            "validation-derived decision threshold."
        )


        # ====================================================
        # ANALYSIS SUMMARY
        # ====================================================

        st.divider()


        st.header(
            "🔬 Analysis Summary"
        )


        st.write(
            "Final morphological profile predicted by the "
            "Custom CNN."
        )


        summary_cols = st.columns(4)


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


            with summary_cols[i]:

                st.metric(

                    label=(
                        f"{ICONS[target]} "
                        f"{DISPLAY_NAMES[target]}"
                    ),

                    value=label,

                    delta=(
                        f"{probability * 100:.1f}% score"
                    )
                )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.divider()


        st.header(
            "📊 Model 1 Performance"
        )


        st.markdown(
            """
            <div class="performance-note">

            Performance of the Custom CNN on the final
            untouched TEST set. The decision thresholds were
            selected exclusively using the VALIDATION set.

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # OVERALL PERFORMANCE
        # ====================================================

        st.subheader(
            "🏆 Overall Test Performance"
        )


        overall_cols = st.columns(4)


        with overall_cols[0]:

            st.metric(
                "Average Accuracy",
                f"{OVERALL_METRICS['accuracy'] * 100:.2f}%"
            )


        with overall_cols[1]:

            st.metric(
                "Average Precision",
                f"{OVERALL_METRICS['precision'] * 100:.2f}%"
            )


        with overall_cols[2]:

            st.metric(
                "Average Recall",
                f"{OVERALL_METRICS['recall'] * 100:.2f}%"
            )


        with overall_cols[3]:

            st.metric(
                "Average F1-score",
                f"{OVERALL_METRICS['f1'] * 100:.2f}%"
            )


        # ====================================================
        # PER-MORPHOLOGY PERFORMANCE
        # ====================================================

        st.subheader(
            "🎯 Performance by Morphology"
        )


        performance_table = pd.DataFrame({

            "Morphology": [
                DISPLAY_NAMES[target]
                for target in TARGETS
            ],

            "Accuracy": [
                f"{FINAL_METRICS[target]['accuracy'] * 100:.2f}%"
                for target in TARGETS
            ],

            "Precision": [
                f"{FINAL_METRICS[target]['precision'] * 100:.2f}%"
                for target in TARGETS
            ],

            "Recall": [
                f"{FINAL_METRICS[target]['recall'] * 100:.2f}%"
                for target in TARGETS
            ],

            "F1-score": [
                f"{FINAL_METRICS[target]['f1'] * 100:.2f}%"
                for target in TARGETS
            ],

            "ROC-AUC": [
                f"{FINAL_METRICS[target]['roc_auc']:.4f}"
                for target in TARGETS
            ],

            "PR-AUC": [
                f"{FINAL_METRICS[target]['pr_auc']:.4f}"
                for target in TARGETS
            ]
        })


        st.dataframe(
            performance_table,
            hide_index=True,
            use_container_width=True
        )


        # ====================================================
        # CONFUSION MATRICES
        # ====================================================

        st.subheader(
            "🧩 Confusion Matrices"
        )


        st.caption(
            "Rows represent actual classes and columns represent "
            "predicted classes."
        )


        cm_cols = st.columns(2)


        for index, target in enumerate(TARGETS):

            matrix = FINAL_CONFUSION_MATRICES[
                target
            ]


            if target == "smooth_featured":

                class_labels = [
                    "SMOOTH",
                    "FEATURED"
                ]

            else:

                class_labels = [
                    "NO",
                    "YES"
                ]


            cm_df = pd.DataFrame(

                matrix,

                index=[
                    f"Actual {class_labels[0]}",
                    f"Actual {class_labels[1]}"
                ],

                columns=[
                    f"Predicted {class_labels[0]}",
                    f"Predicted {class_labels[1]}"
                ]
            )


            with cm_cols[index % 2]:

                st.markdown(
                    f"### {ICONS[target]} "
                    f"{DISPLAY_NAMES[target]}"
                )


                st.dataframe(
                    cm_df,
                    use_container_width=True
                )


                st.caption(
                    f"Accuracy: "
                    f"{FINAL_METRICS[target]['accuracy'] * 100:.2f}%"
                    f"  •  "
                    f"Precision: "
                    f"{FINAL_METRICS[target]['precision'] * 100:.2f}%"
                    f"  •  "
                    f"Recall: "
                    f"{FINAL_METRICS[target]['recall'] * 100:.2f}%"
                )


        # ====================================================
        # ADDITIONAL METRICS
        # ====================================================

        with st.expander(
            "📈 View ROC-AUC, PR-AUC and F1-score"
        ):

            additional_table = pd.DataFrame({

                "Morphology": [
                    DISPLAY_NAMES[target]
                    for target in TARGETS
                ],

                "F1-score": [
                    f"{FINAL_METRICS[target]['f1']:.4f}"
                    for target in TARGETS
                ],

                "ROC-AUC": [
                    f"{FINAL_METRICS[target]['roc_auc']:.4f}"
                    for target in TARGETS
                ],

                "PR-AUC": [
                    f"{FINAL_METRICS[target]['pr_auc']:.4f}"
                    for target in TARGETS
                ]
            })


            st.dataframe(
                additional_table,
                hide_index=True,
                use_container_width=True
            )


        # ====================================================
        # THRESHOLD INFORMATION
        # ====================================================

        with st.expander(
            "⚙️ View decision thresholds"
        ):

            threshold_table = pd.DataFrame({

                "Morphology": [
                    DISPLAY_NAMES[target]
                    for target in TARGETS
                ],

                "Validation-derived threshold": [
                    f"{thresholds[target]:.2f}"
                    for target in TARGETS
                ]
            })


            st.dataframe(
                threshold_table,
                hide_index=True,
                use_container_width=True
            )


# ============================================================
# ABOUT GALAXAI
# ============================================================

st.divider()


st.header(
    "🔬 About GalaxAI"
)


about_col1, about_col2, about_col3 = st.columns(3)


with about_col1:

    with st.container(
        border=True
    ):

        st.subheader(
            "🛰️ Dataset"
        )

        st.write(
            "Galaxy Zoo: Hubble"
        )

        st.caption(
            "Hubble Space Telescope galaxy imagery"
        )


with about_col2:

    with st.container(
        border=True
    ):

        st.subheader(
            "🧠 Architecture"
        )

        st.write(
            "Custom Convolutional Neural Network"
        )

        st.caption(
            "Four independent morphology outputs"
        )


with about_col3:

    with st.container(
        border=True
    ):

        st.subheader(
            "⚡ Inference"
        )

        st.write(
            "NVIDIA RTX 4050 Laptop GPU"
        )

        st.caption(
            "224 × 224 model input"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🌌 GalaxAI • Model 1: Custom CNN • "
    "Galaxy Zoo: Hubble • GPU Accelerated"
)