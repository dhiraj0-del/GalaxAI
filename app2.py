import streamlit as st
from PIL import Image
from pathlib import Path

from src.inference import load_model, predict, model_info


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LOSS_GRAPH = BASE_DIR / "results" / "model1_v2_loss_curve.png"
CONFUSION_DIR = BASE_DIR / "results" / "model1_v2"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GALAXAI — Galaxy Morphology",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

html, body {
    font-family: "Inter", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(72, 88, 190, 0.20),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 75%,
            rgba(130, 60, 190, 0.17),
            transparent 32%
        ),
        #03050d !important;

    color: #eef1ff;
}

[data-testid="stAppViewContainer"] {
    overflow-x: hidden;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* ============================================================
   STAR BACKGROUND
   ============================================================ */

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;

    pointer-events: none;

    opacity: 0.55;

    background-image:
        radial-gradient(
            circle,
            rgba(255,255,255,0.65) 1px,
            transparent 1px
        ),
        radial-gradient(
            circle,
            rgba(160,180,255,0.35) 1px,
            transparent 1px
        );

    background-size:
        110px 110px,
        170px 170px;

    background-position:
        0 0,
        40px 70px;

    animation: starMove 35s linear infinite;

    z-index: 0;
}

@keyframes starMove {
    from {
        transform: translate3d(0, 0, 0);
    }

    to {
        transform: translate3d(-80px, 100px, 0);
    }
}


/* ============================================================
   HERO
   ============================================================ */

.hero-wrapper {
    position: relative;
    overflow: hidden;

    padding: 48px;

    border-radius: 30px;

    margin-bottom: 30px;

    background:
        radial-gradient(
            circle at 80% 30%,
            rgba(112,95,255,0.24),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(15,21,48,0.96),
            rgba(7,10,25,0.94)
        );

    border: 1px solid rgba(145,160,255,0.22);

    box-shadow:
        0 30px 90px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.08);
}

.hero-wrapper::after {
    content: "";

    position: absolute;

    width: 280px;
    height: 280px;

    right: -80px;
    top: -90px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(125,140,255,0.20),
            rgba(70,40,150,0.08),
            transparent 70%
        );

    filter: blur(8px);

    animation: pulseOrb 5s ease-in-out infinite;
}

@keyframes pulseOrb {

    0%, 100% {
        transform: scale(0.9);
        opacity: 0.55;
    }

    50% {
        transform: scale(1.15);
        opacity: 0.9;
    }
}

.hero-kicker {
    color: #8e9bd0;

    font-size: 0.75rem;

    font-weight: 700;

    letter-spacing: 0.24em;

    text-transform: uppercase;

    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(3rem, 7vw, 6rem);

    line-height: 0.95;

    font-weight: 900;

    letter-spacing: 0.08em;

    margin: 0;

    background:
        linear-gradient(
            100deg,
            #ffffff 0%,
            #b9c7ff 45%,
            #d7b8ff 75%,
            #ffffff 100%
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 850px;

    margin-top: 20px;

    color: #a8b2cf;

    font-size: 1rem;

    line-height: 1.75;
}

.hero-tag {
    display: inline-block;

    margin-top: 20px;

    padding: 8px 15px;

    border-radius: 999px;

    color: #c7d0ff;

    background: rgba(115,130,255,0.10);

    border: 1px solid rgba(130,145,255,0.20);

    font-size: 0.72rem;

    letter-spacing: 0.12em;

    text-transform: uppercase;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-kicker {
    color: #727fa9;

    font-size: 0.70rem;

    font-weight: 800;

    letter-spacing: 0.20em;

    text-transform: uppercase;

    margin-bottom: 5px;
}

.section-title {
    color: #eef1ff;

    font-size: 1.65rem;

    font-weight: 800;

    margin-bottom: 18px;
}

.section-description {
    color: #7f8caf;

    font-size: 0.88rem;

    line-height: 1.7;

    margin-bottom: 20px;
}


/* ============================================================
   GLASS PANELS
   ============================================================ */

.glass-panel {
    padding: 25px;

    border-radius: 24px;

    background:
        linear-gradient(
            145deg,
            rgba(22,29,59,0.78),
            rgba(8,12,28,0.82)
        );

    border: 1px solid rgba(150,165,255,0.14);

    box-shadow:
        0 22px 60px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.05);

    backdrop-filter: blur(18px);
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {
    min-height: 145px;

    padding: 22px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(24,32,67,0.86),
            rgba(11,15,34,0.88)
        );

    border: 1px solid rgba(145,160,255,0.13);

    box-shadow:
        0 14px 40px rgba(0,0,0,0.22);
}

.metric-label {
    color: #7f8caf;

    font-size: 0.68rem;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.12em;
}

.metric-value {
    color: #f5f6ff;

    font-size: 1.7rem;

    font-weight: 900;

    margin-top: 10px;
}

.metric-description {
    color: #677493;

    font-size: 0.76rem;

    margin-top: 7px;

    line-height: 1.5;
}


/* ============================================================
   RESULT CARDS
   ============================================================ */

.result-card {
    position: relative;

    overflow: hidden;

    min-height: 145px;

    padding: 20px;

    border-radius: 19px;

    background:
        linear-gradient(
            145deg,
            rgba(24,32,67,0.86),
            rgba(11,15,34,0.88)
        );

    border: 1px solid rgba(145,160,255,0.13);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.22);

    transition:
        transform 0.22s ease,
        border-color 0.22s ease;
}

.result-card:hover {
    transform: translateY(-4px);

    border-color:
        rgba(150,170,255,0.30);
}

.result-card::before {
    content: "";

    position: absolute;

    left: 0;

    top: 0;

    bottom: 0;

    width: 3px;

    background:
        linear-gradient(
            180deg,
            #788aff,
            #bb8cff
        );

    opacity: 0.75;
}

.result-label {
    color: #7f8caf;

    font-size: 0.68rem;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.12em;
}

.result-value {
    color: #f5f6ff;

    font-size: 1.13rem;

    font-weight: 800;

    margin-top: 10px;
}

.result-score {
    color: #9ca9d4;

    font-size: 0.78rem;

    margin-top: 8px;
}

.status-detected {
    display: inline-block;

    color: #c7d5ff;

    background: rgba(86,110,255,0.14);

    border: 1px solid rgba(100,120,255,0.24);

    border-radius: 999px;

    padding: 5px 10px;

    font-size: 0.66rem;

    font-weight: 800;

    letter-spacing: 0.08em;
}

.status-negative {
    display: inline-block;

    color: #8994b3;

    background: rgba(120,130,160,0.08);

    border: 1px solid rgba(120,130,160,0.14);

    border-radius: 999px;

    padding: 5px 10px;

    font-size: 0.66rem;

    font-weight: 800;

    letter-spacing: 0.08em;
}

.score-track {
    height: 5px;

    width: 100%;

    margin-top: 12px;

    border-radius: 99px;

    background: rgba(255,255,255,0.07);

    overflow: hidden;
}

.score-fill {
    height: 100%;

    border-radius: 99px;

    background:
        linear-gradient(
            90deg,
            #7185ff,
            #ba8cff
        );
}


/* ============================================================
   IMAGE FRAME
   ============================================================ */

.image-frame {
    padding: 8px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(130,145,255,0.20),
            rgba(180,100,255,0.08)
        );

    box-shadow:
        0 20px 60px rgba(0,0,0,0.40);
}


/* ============================================================
   RESEARCH NOTE
   ============================================================ */

.research-note {
    padding: 22px;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(30,40,80,0.72),
            rgba(10,15,32,0.80)
        );

    border: 1px solid rgba(130,145,255,0.13);

    color: #8e9ab8;

    font-size: 0.84rem;

    line-height: 1.75;
}

.research-note strong {
    color: #dfe4ff;
}


/* ============================================================
   INFO BOX
   ============================================================ */

.info-box {
    padding: 18px;

    border-radius: 16px;

    background:
        rgba(35,45,85,0.42);

    border-left:
        3px solid #788aff;

    color: #8e9ab8;

    font-size: 0.83rem;

    line-height: 1.7;
}

.info-title {
    color: #e5e9ff;

    font-size: 0.9rem;

    font-weight: 800;

    margin-bottom: 7px;
}


/* ============================================================
   TASK TAGS
   ============================================================ */

.task-tag {
    display: inline-block;

    padding: 6px 10px;

    margin: 3px;

    border-radius: 9px;

    color: #c7d0ff;

    background:
        rgba(110,125,255,0.10);

    border:
        1px solid rgba(130,145,255,0.16);

    font-size: 0.68rem;

    font-weight: 700;
}


/* ============================================================
   PIPELINE
   ============================================================ */

.pipeline-box {
    padding: 22px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(24,32,67,0.82),
            rgba(11,15,34,0.88)
        );

    border:
        1px solid rgba(145,160,255,0.13);

    text-align: center;

    min-height: 150px;
}

.pipeline-icon {
    font-size: 2rem;

    margin-bottom: 10px;
}

.pipeline-title {
    color: #e9edff;

    font-size: 0.9rem;

    font-weight: 800;
}

.pipeline-description {
    color: #74809f;

    font-size: 0.73rem;

    margin-top: 7px;

    line-height: 1.5;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    padding: 55px 0 20px 0;

    color: #4c5878;

    font-size: 0.72rem;

    line-height: 1.8;
}

.divider {
    height: 1px;

    margin: 45px 0;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(148,163,184,0.16),
            transparent
        );
}


/* ============================================================
   STREAMLIT BUTTON
   ============================================================ */

.stButton > button {
    height: 52px;

    border-radius: 15px;

    border: 1px solid rgba(130,145,255,0.30);

    background:
        linear-gradient(
            135deg,
            rgba(87,105,255,0.30),
            rgba(150,90,255,0.22)
        );

    color: #eef1ff;

    font-weight: 800;

    letter-spacing: 0.04em;

    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color:
        rgba(160,175,255,0.55);

    transform: translateY(-2px);

    box-shadow:
        0 12px 35px rgba(70,80,200,0.22);
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background:
        rgba(10,15,32,0.72);

    border-radius: 18px;

    padding: 8px;

    border:
        1px dashed rgba(130,145,255,0.28);
}

[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(
            145deg,
            rgba(22,30,62,0.80),
            rgba(10,14,30,0.85)
        ) !important;

    border-radius: 16px !important;
}


/* ============================================================
   DATAFRAME / TABLE
   ============================================================ */

table {
    color: #dce2ff !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 800px) {

    .hero-wrapper {
        padding: 30px 24px;
    }

    .hero-title {
        font-size: 3rem;
    }

    .glass-panel {
        padding: 18px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


try:
    model = get_model()
    info = model_info()

except Exception as error:

    st.error("Unable to load the GALAXAI model.")
    st.exception(error)
    st.stop()


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero-wrapper">

        <div class="hero-kicker">
            Hubble Space Telescope · Deep Learning Research
        </div>

        <div class="hero-title">
            GALAXAI
        </div>

        <div class="hero-subtitle">
            Deep Learning-Based Multi-Attribute Galaxy Morphology
            Analysis Using Hubble Space Telescope Images.
            Analyze multiple visual structures of a galaxy through
            a single CNN-based inference pipeline.
        </div>

        <div class="hero-tag">
            MULTI-TASK CNN · 10 MORPHOLOGY ATTRIBUTES
        </div>

    </div>
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

st.html(
    """
    <div class="section-kicker">
        INPUT
    </div>

    <div class="section-title">
        Galaxy Observation
    </div>
    """
)


uploaded_file = st.file_uploader(
    "Upload a galaxy image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.html(
        """
        <div class="glass-panel">

            <div style="
                text-align:center;
                padding:35px 10px;
            ">

                <div style="
                    font-size:3rem;
                    margin-bottom:15px;
                ">
                    🌌
                </div>

                <div style="
                    color:#e9edff;
                    font-size:1.15rem;
                    font-weight:800;
                ">
                    Upload a Galaxy Image
                </div>

                <div style="
                    color:#7885a7;
                    margin-top:8px;
                    font-size:0.85rem;
                ">
                    JPG, PNG or WEBP · The model will automatically
                    preprocess the image to 224 × 224 RGB.
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# IMAGE UPLOADED
# ============================================================

else:

    image = Image.open(uploaded_file).convert("RGB")

    image_col, control_col = st.columns(
        [1.35, 1],
        gap="large",
    )


    # ========================================================
    # IMAGE
    # ========================================================

    with image_col:

        st.html(
            """
            <div class="section-kicker">
                OBSERVATION
            </div>

            <div class="section-title">
                Uploaded Galaxy
            </div>
            """
        )

        st.html(
            """
            <div class="image-frame">
            """
        )

        st.image(
            image,
            use_container_width=True,
        )

        # Do NOT close an HTML container in another Streamlit
        # element. This was the source of the raw HTML problem.
        st.html(
            """
            </div>
            """
        )

        st.caption(
            f"Original image · "
            f"{image.width} × {image.height} px · "
            f"{image.mode}"
        )


    # ========================================================
    # INFERENCE CONTROL
    # ========================================================

    with control_col:

        st.html(
            """
            <div class="section-kicker">
                INFERENCE ENGINE
            </div>

            <div class="section-title">
                Run Morphology Analysis
            </div>

            <div class="glass-panel">

                <div style="
                    color:#a2acc8;
                    line-height:1.7;
                    font-size:0.88rem;
                    margin-bottom:20px;
                ">

                    GALAXAI preprocesses the uploaded image
                    and passes it through the trained
                    Custom CNN V2.

                    <br><br>

                    The network independently evaluates

                    <b style="color:#e4e8ff;">
                    10 morphological attributes
                    </b>

                    across binary and multiclass tasks.

                </div>

            </div>
            """
        )

        analyze = st.button(
            "🚀  ANALYZE GALAXY",
            use_container_width=True,
        )


    # ========================================================
    # RUN INFERENCE
    # ========================================================

    if analyze:

        with st.spinner(
            "Extracting learned morphological features..."
        ):

            try:

                results = predict(
                    image,
                    model,
                )

                st.session_state["results"] = results

            except Exception as error:

                st.error("Inference failed.")
                st.exception(error)


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    st.html("<br>")

    st.html(
        """
        <div class="section-kicker">
            MODEL OUTPUT
        </div>

        <div class="section-title">
            Galaxy Morphology Profile
        </div>

        <div class="section-description">
            The Custom CNN V2 evaluates eight binary morphology
            attributes and two multiclass morphology attributes.
        </div>
        """
    )


    # ========================================================
    # BINARY TASKS
    # ========================================================

    binary_order = [
        "featured",
        "edge_on",
        "bar",
        "spiral_arms",
        "disturbed",
        "merger",
        "clumpy",
        "symmetry",
    ]

    binary_descriptions = {
        "featured":
            "Presence of featured/non-smooth morphology.",

        "edge_on":
            "Whether the galaxy appears edge-on.",

        "bar":
            "Presence of a central bar structure.",

        "spiral_arms":
            "Presence of visible spiral-arm structure.",

        "disturbed":
            "Evidence of disturbed morphology.",

        "merger":
            "Evidence of merger-like morphology.",

        "clumpy":
            "Presence of clumpy galaxy structure.",

        "symmetry":
            "Overall galaxy symmetry.",
    }


    cols = st.columns(
        4,
        gap="medium",
    )


    for index, task in enumerate(binary_order):

        result = results[task]

        score = float(result["score"])

        detected = bool(result["detected"])

        status = (
            "DETECTED"
            if detected
            else "NOT DETECTED"
        )

        status_class = (
            "status-detected"
            if detected
            else "status-negative"
        )

        with cols[index % 4]:

            st.html(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        {result["label"]}
                    </div>

                    <div style="margin-top:10px;">

                        <span class="{status_class}">
                            {status}
                        </span>

                    </div>

                    <div class="result-score">
                        Model score · {score:.3f}
                    </div>

                    <div class="score-track">

                        <div
                            class="score-fill"
                            style="width:{score * 100:.1f}%"
                        ></div>

                    </div>

                    <div style="
                        color:#687596;
                        font-size:0.70rem;
                        line-height:1.5;
                        margin-top:12px;
                    ">
                        {binary_descriptions[task]}
                    </div>

                </div>
                """
            )


    # ========================================================
    # MULTICLASS
    # ========================================================

    st.html("<br>")

    st.html(
        """
        <div class="section-kicker">
            MULTICLASS MORPHOLOGY
        </div>

        <div style="
            color:#8e9ab8;
            font-size:0.82rem;
            margin-bottom:15px;
        ">
            These attributes are predicted from multiple
            morphological categories.
        </div>
        """
    )


    multi_col1, multi_col2 = st.columns(
        2,
        gap="medium",
    )


    # ========================================================
    # BULGE
    # ========================================================

    bulge = results["bulge"]

    with multi_col1:

        st.html(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Bulge Prominence
                </div>

                <div class="result-value">
                    {bulge["prediction"]}
                </div>

                <div class="result-score">
                    Predicted class score ·
                    {float(bulge["score"]):.3f}
                </div>

                <div class="score-track">

                    <div
                        class="score-fill"
                        style="
                            width:
                            {float(bulge["score"]) * 100:.1f}%
                        "
                    ></div>

                </div>

                <div style="
                    color:#687596;
                    font-size:0.70rem;
                    line-height:1.5;
                    margin-top:12px;
                ">
                    None · Just noticeable · Obvious · Dominant
                </div>

            </div>
            """
        )


    # ========================================================
    # ROUNDEDNESS
    # ========================================================

    rounded = results["roundedness"]

    with multi_col2:

        st.html(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Roundedness
                </div>

                <div class="result-value">
                    {rounded["prediction"]}
                </div>

                <div class="result-score">
                    Predicted class score ·
                    {float(rounded["score"]):.3f}
                </div>

                <div class="score-track">

                    <div
                        class="score-fill"
                        style="
                            width:
                            {float(rounded["score"]) * 100:.1f}%
                        "
                    ></div>

                </div>

                <div style="
                    color:#687596;
                    font-size:0.70rem;
                    line-height:1.5;
                    margin-top:12px;
                ">
                    Completely rounded · In-between · Cigar-shaped
                </div>

            </div>
            """
        )


# ============================================================
# RESEARCH EVIDENCE
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True,
)

st.html(
    """
    <div class="section-kicker">
        RESEARCH EVIDENCE
    </div>

    <div class="section-title">
        Model 1 — Quantitative Evaluation
    </div>

    <div class="section-description">
        Held-out test-set evaluation of the Custom CNN V2 baseline.
    </div>
    """
)


# ============================================================
# KEY METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(
    4,
    gap="medium",
)


with metric1:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Overall Pooled Accuracy
            </div>

            <div class="metric-value">
                83.41%
            </div>

            <div class="metric-description">
                Across all valid task predictions.
                This is not a conventional single-label accuracy.
            </div>

        </div>
        """
    )


with metric2:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Overall Macro Task F1
            </div>

            <div class="metric-value">
                56.82%
            </div>

            <div class="metric-description">
                Mean F1 across the ten morphology tasks.
            </div>

        </div>
        """
    )


with metric3:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Best Validation Epoch
            </div>

            <div class="metric-value">
                27
            </div>

            <div class="metric-description">
                Selected using the lowest validation loss.
            </div>

        </div>
        """
    )


with metric4:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Best Validation Loss
            </div>

            <div class="metric-value">
                6.283238
            </div>

            <div class="metric-description">
                Best Model 1 checkpoint.
            </div>

        </div>
        """
    )


# ============================================================
# TRAINING / VALIDATION LOSS
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        TRAINING
    </div>

    <div class="section-title">
        Training & Validation Loss
    </div>

    <div class="section-description">
        Learning behaviour of Custom CNN V2 across the training epochs.
    </div>
    """
)


loss_col1, loss_col2 = st.columns(
    [1.55, 1],
    gap="large",
)


with loss_col1:

    if LOSS_GRAPH.exists():

        st.image(
            str(LOSS_GRAPH),
            use_container_width=True,
        )

    else:

        st.warning(
            "Loss graph not found. Expected file: "
            "results/model1_v2_loss_curve.png"
        )


with loss_col2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                WHAT THE GRAPH SHOWS
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Learning and generalization
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.84rem;
                line-height:1.75;
                margin-top:12px;
            ">
                The training curve shows how the optimization
                objective changes while the model learns from
                the training data.
            </p>

            <p style="
                color:#8e9ab8;
                font-size:0.84rem;
                line-height:1.75;
            ">
                The validation curve measures how the learned
                representation performs on unseen validation data.
            </p>

            <div class="info-box">

                <div class="info-title">
                    Checkpoint Selection
                </div>

                The final epoch was not automatically selected.
                The checkpoint with the lowest validation loss
                was selected to reduce the risk of overfitting.

            </div>

            <br>

            <div class="info-box">

                <div class="info-title">
                    Best Checkpoint
                </div>

                Epoch 27

                <br>

                Validation loss: 6.283238

            </div>

        </div>
        """
    )


# ============================================================
# MULTI-TASK LOSS DESIGN
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        LOSS FUNCTION
    </div>

    <div class="section-title">
        Multi-Task Loss Design
    </div>

    <div class="section-description">
        Different output types require different classification losses.
    </div>
    """
)


loss1, loss2, loss3 = st.columns(
    3,
    gap="medium",
)


with loss1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                8 BINARY TASKS
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                BCEWithLogitsLoss
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Featured, Edge-on, Bar, Spiral Arms,
                Disturbed, Merger, Clumpy Appearance
                and Galaxy Symmetry.
            </p>

        </div>
        """
    )


with loss2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                2 MULTICLASS TASKS
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                CrossEntropyLoss
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Bulge Prominence has four classes,
                while Roundedness has three classes.
            </p>

        </div>
        """
    )


with loss3:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                MASKING & WEIGHTING
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Hierarchical labels
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Task-specific masks exclude unavailable or
                ambiguous labels. Class weighting reduces
                the effect of severe class imbalance.
            </p>

        </div>
        """
    )


# ============================================================
# CONFUSION MATRIX ANALYSIS
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        TEST SET
    </div>

    <div class="section-title">
        Confusion Matrix Analysis
    </div>

    <div class="section-description">
        Confusion matrices show the distribution of actual and
        predicted classes for each morphology task.
    </div>
    """
)


cm_data = [

    {
        "task": "Featured",
        "file": "featured_confusion_matrix.png",
        "type": "Binary",
        "headline": "Strongest binary task",
        "metrics": "F1 95.03% · ROC-AUC 99.81% · Recall 97.85%",
        "point":
            "The model performs very strongly on Featured morphology."
    },

    {
        "task": "Edge-on",
        "file": "edge_on_confusion_matrix.png",
        "type": "Binary",
        "headline": "High positive recall",
        "metrics": "F1 68.69% · ROC-AUC 98.01% · Recall 95.38%",
        "point":
            "The model detects most positive edge-on galaxies, "
            "but produces more false positives."
    },

    {
        "task": "Bar",
        "file": "bar_confusion_matrix.png",
        "type": "Binary",
        "headline": "Challenging rare feature",
        "metrics": "F1 17.88% · ROC-AUC 80.22% · Recall 67.95%",
        "point":
            "Bar detection is difficult because the positive "
            "class is relatively rare."
    },

    {
        "task": "Spiral Arms",
        "file": "spiral_arms_confusion_matrix.png",
        "type": "Binary",
        "headline": "Meaningful spiral recognition",
        "metrics": "F1 68.10% · ROC-AUC 90.14% · Recall 72.59%",
        "point":
            "The model learns useful spiral-arm structure."
    },

    {
        "task": "Disturbed",
        "file": "disturbed_confusion_matrix.png",
        "type": "Binary",
        "headline": "Very challenging rare feature",
        "metrics": "F1 8.43% · ROC-AUC 75.01% · Recall 59.09%",
        "point":
            "The low F1 is strongly affected by severe class imbalance."
    },

    {
        "task": "Merger",
        "file": "merger_confusion_matrix.png",
        "type": "Binary",
        "headline": "Rare morphology challenge",
        "metrics": "F1 16.34% · ROC-AUC 85.06% · Recall 66.91%",
        "point":
            "Merger detection remains difficult because positive "
            "examples are rare."
    },

    {
        "task": "Clumpy Appearance",
        "file": "clumpy_confusion_matrix.png",
        "type": "Binary",
        "headline": "Good morphology detection",
        "metrics": "F1 71.23% · ROC-AUC 92.51% · Recall 86.16%",
        "point":
            "The model identifies clumpy appearance reasonably well."
    },

    {
        "task": "Galaxy Symmetry",
        "file": "symmetry_confusion_matrix.png",
        "type": "Binary",
        "headline": "Balanced performance",
        "metrics": "F1 75.31% · ROC-AUC 85.61% · Recall 78.46%",
        "point":
            "Symmetry is one of the stronger binary tasks."
    },

    {
        "task": "Bulge Prominence",
        "file": "bulge_confusion_matrix.png",
        "type": "Multiclass",
        "headline": "Four-class prediction",
        "metrics": "Accuracy 73.79% · Macro F1 64.39% · Balanced Acc. 72.54%",
        "point":
            "The dominant bulge class is recognized better than "
            "the rarer categories."
    },

    {
        "task": "Roundedness",
        "file": "roundedness_confusion_matrix.png",
        "type": "Multiclass",
        "headline": "Strong multiclass result",
        "metrics": "Accuracy 83.93% · Macro F1 82.86% · Balanced Acc. 87.26%",
        "point":
            "Roundedness is the strongest multiclass task."
    },

]


# ============================================================
# CONFUSION MATRIX DISPLAY
# ============================================================

for i in range(0, len(cm_data), 2):

    row = cm_data[i:i + 2]

    columns = st.columns(
        2,
        gap="large",
    )

    for col, item in zip(columns, row):

        with col:

            cm_path = (
                CONFUSION_DIR /
                item["file"]
            )

            st.html(
                f"""
                <div class="glass-panel">

                    <div class="section-kicker">
                        {item["type"]} TASK
                    </div>

                    <div style="
                        color:#eef1ff;
                        font-size:1.15rem;
                        font-weight:800;
                        margin-top:4px;
                    ">
                        {item["task"]}
                    </div>

                    <div style="
                        color:#b9c5ff;
                        font-size:0.76rem;
                        font-weight:800;
                        margin-top:7px;
                    ">
                        {item["headline"]}
                    </div>

                </div>
                """
            )

            if cm_path.exists():

                st.image(
                    str(cm_path),
                    use_container_width=True,
                )

            else:

                st.warning(
                    f"Matrix image not found: {item['file']}"
                )

            st.html(
                f"""
                <div class="research-note">

                    <strong>
                        {item["metrics"]}
                    </strong>

                    <br><br>

                    {item["point"]}

                </div>
                """
            )


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        PERFORMANCE SUMMARY
    </div>

    <div class="section-title">
        Strongest & Most Challenging Tasks
    </div>
    """
)


summary1, summary2 = st.columns(
    2,
    gap="large",
)


with summary1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                STRONGEST TASKS
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Common and visually distinctive morphology
            </div>

            <ul style="
                color:#8e9ab8;
                line-height:1.9;
                font-size:0.84rem;
            ">

                <li>
                    Featured F1:
                    <strong style="color:#dfe4ff;">
                        95.03%
                    </strong>
                </li>

                <li>
                    Roundedness accuracy:
                    <strong style="color:#dfe4ff;">
                        83.93%
                    </strong>
                </li>

                <li>
                    Symmetry F1:
                    <strong style="color:#dfe4ff;">
                        75.31%
                    </strong>
                </li>

                <li>
                    Clumpy F1:
                    <strong style="color:#dfe4ff;">
                        71.23%
                    </strong>
                </li>

            </ul>

        </div>
        """
    )


with summary2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                CHALLENGING TASKS
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Rare morphology features
            </div>

            <ul style="
                color:#8e9ab8;
                line-height:1.9;
                font-size:0.84rem;
            ">

                <li>
                    Bar F1:
                    <strong style="color:#dfe4ff;">
                        17.88%
                    </strong>
                </li>

                <li>
                    Merger F1:
                    <strong style="color:#dfe4ff;">
                        16.34%
                    </strong>
                </li>

                <li>
                    Disturbed F1:
                    <strong style="color:#dfe4ff;">
                        8.43%
                    </strong>
                </li>

                <li>
                    Severe class imbalance affects minority features.
                </li>

            </ul>

        </div>
        """
    )


# ============================================================
# METRIC INTERPRETATION
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        METRIC INTERPRETATION
    </div>

    <div class="section-title">
        Understanding the Evaluation
    </div>
    """
)


interpret1, interpret2, interpret3 = st.columns(
    3,
    gap="medium",
)


with interpret1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                MACRO TASK F1
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                56.82%
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Average F1 across all ten morphology tasks.
                It reflects the fact that common tasks perform
                much better than several rare tasks.
            </p>

        </div>
        """
    )


with interpret2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                POOLED ACCURACY
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                83.41%
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Calculated across all valid task predictions.
                It should not be interpreted as a conventional
                single-label classification accuracy.
            </p>

        </div>
        """
    )


with interpret3:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                ROC-AUC VS F1
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Ranking vs threshold
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                ROC-AUC evaluates ranking across thresholds,
                while F1 depends on a specific classification
                threshold. Therefore both metrics can differ
                substantially.
            </p>

        </div>
        """
    )


# ============================================================
# DATASET
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        DATASET
    </div>

    <div class="section-title">
        Galaxy Zoo: Hubble
    </div>
    """
)


data1, data2, data3 = st.columns(
    3,
    gap="medium",
)


with data1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                IMAGE RECORDS
            </div>

            <div class="metric-value">
                80,632
            </div>

            <div class="metric-description">
                Galaxy image records used in the project.
            </div>

        </div>
        """
    )


with data2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                DATA SPLIT
            </div>

            <div style="
                margin-top:15px;
            ">

                <span class="task-tag">
                    56,442 Train
                </span>

                <span class="task-tag">
                    12,095 Validation
                </span>

                <span class="task-tag">
                    12,095 Test
                </span>

            </div>

            <div class="metric-description">
                70 / 15 / 15 split.
            </div>

        </div>
        """
    )


with data3:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                LABEL STRATEGY
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Hierarchical annotations
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Task-specific masks exclude unavailable or
                ambiguous labels instead of treating them as
                negative examples.
            </p>

        </div>
        """
    )


# ============================================================
# MORPHOLOGY TASKS
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        TASK DEFINITION
    </div>

    <div class="section-title">
        Ten Morphology Attributes
    </div>
    """
)


task1, task2 = st.columns(
    2,
    gap="large",
)


with task1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                8 BINARY ATTRIBUTES
            </div>

            <div style="margin-top:15px;">

                <span class="task-tag">Featured</span>
                <span class="task-tag">Edge-on</span>
                <span class="task-tag">Bar</span>
                <span class="task-tag">Spiral Arms</span>
                <span class="task-tag">Disturbed</span>
                <span class="task-tag">Merger</span>
                <span class="task-tag">Clumpy Appearance</span>
                <span class="task-tag">Galaxy Symmetry</span>

            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
                margin-top:15px;
            ">
                Each binary task produces one output logit and
                is converted to a score using the sigmoid function.
            </p>

        </div>
        """
    )


with task2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                2 MULTICLASS ATTRIBUTES
            </div>

            <div style="margin-top:15px;">

                <strong style="color:#e9edff;">
                    Bulge Prominence
                </strong>

                <br>

                <span class="task-tag">None</span>
                <span class="task-tag">Just noticeable</span>
                <span class="task-tag">Obvious</span>
                <span class="task-tag">Dominant</span>

                <br><br>

                <strong style="color:#e9edff;">
                    Roundedness
                </strong>

                <br>

                <span class="task-tag">
                    Completely rounded
                </span>

                <span class="task-tag">
                    In-between
                </span>

                <span class="task-tag">
                    Cigar-shaped
                </span>

            </div>

        </div>
        """
    )


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        MODEL ARCHITECTURE
    </div>

    <div class="section-title">
        Custom CNN V2
    </div>

    <div class="section-description">
        Four convolutional blocks produce a shared feature
        representation which is connected to ten task-specific
        prediction heads.
    </div>
    """
)


architecture = [

    (
        "01",
        "Input",
        "224 × 224 × 3",
        "RGB galaxy image",
    ),

    (
        "02",
        "Convolution Block 1",
        "32 × 112 × 112",
        "3×3 convolution + BatchNorm + ReLU + MaxPool",
    ),

    (
        "03",
        "Convolution Block 2",
        "64 × 56 × 56",
        "Feature extraction",
    ),

    (
        "04",
        "Convolution Block 3",
        "128 × 28 × 28",
        "Higher-level morphology features",
    ),

    (
        "05",
        "Convolution Block 4",
        "256 × 14 × 14",
        "Deep morphological representation",
    ),

    (
        "06",
        "Adaptive Average Pool",
        "256 × 1 × 1",
        "Spatial aggregation",
    ),

    (
        "07",
        "Shared Fully Connected Layer",
        "256",
        "Shared feature embedding",
    ),

    (
        "08",
        "Task Heads",
        "15 logits",
        "8 binary + 2 multiclass outputs",
    ),

]


for number, title, shape, description in architecture:

    st.html(
        f"""
        <div class="glass-panel" style="
            margin-bottom:10px;
            padding:17px 22px;
        ">

            <div style="
                display:flex;
                align-items:center;
                gap:18px;
                flex-wrap:wrap;
            ">

                <div style="
                    min-width:45px;
                    height:45px;
                    border-radius:13px;

                    background:
                        rgba(99,102,241,0.12);

                    border:
                        1px solid rgba(129,140,248,0.20);

                    display:flex;
                    align-items:center;
                    justify-content:center;

                    color:#a5b4fc;

                    font-weight:800;
                ">
                    {number}
                </div>

                <div style="
                    flex:1;
                    min-width:180px;
                ">

                    <div style="
                        color:#e0e7ff;
                        font-size:0.9rem;
                        font-weight:800;
                    ">
                        {title}
                    </div>

                    <div style="
                        color:#65718f;
                        font-size:0.73rem;
                        margin-top:4px;
                    ">
                        {description}
                    </div>

                </div>

                <div style="
                    color:#a5b4fc;
                    font-family:monospace;
                    font-weight:700;
                    font-size:0.78rem;
                ">
                    {shape}
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        TRAINING
    </div>

    <div class="section-title">
        Training Configuration
    </div>
    """
)


train1, train2 = st.columns(
    2,
    gap="large",
)


with train1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                OPTIMIZATION
            </div>

            <ul style="
                color:#8e9ab8;
                line-height:1.9;
                font-size:0.84rem;
            ">

                <li>Optimizer: <strong>AdamW</strong></li>

                <li>Learning rate: <strong>0.001</strong></li>

                <li>Batch size: <strong>32</strong></li>

                <li>Maximum epochs: <strong>30</strong></li>

                <li>Best checkpoint: <strong>Epoch 27</strong></li>

            </ul>

        </div>
        """
    )


with train2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                DATA AUGMENTATION
            </div>

            <ul style="
                color:#8e9ab8;
                line-height:1.9;
                font-size:0.84rem;
            ">

                <li>Resize to 224 × 224</li>

                <li>Random horizontal flip</li>

                <li>Random rotation up to 10°</li>

                <li>Tensor conversion</li>

                <li>Normalization using mean/std = 0.5</li>

            </ul>

        </div>
        """
    )


# ============================================================
# INFERENCE PIPELINE
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        INFERENCE PIPELINE
    </div>

    <div class="section-title">
        How GALAXAI Analyzes a Galaxy
    </div>
    """
)


pipeline = [

    (
        "🌌",
        "Galaxy Image",
        "Uploaded observation",
    ),

    (
        "⚙️",
        "Preprocessing",
        "224 × 224 · RGB · Normalize",
    ),

    (
        "🧠",
        "CNN Feature Extraction",
        "Learned visual patterns",
    ),

    (
        "🔭",
        "Morphology Profile",
        "10 independent tasks",
    ),

]


pipeline_columns = st.columns(
    4,
    gap="medium",
)


for col, item in zip(
    pipeline_columns,
    pipeline,
):

    icon, title, description = item

    with col:

        st.html(
            f"""
            <div class="pipeline-box">

                <div class="pipeline-icon">
                    {icon}
                </div>

                <div class="pipeline-title">
                    {title}
                </div>

                <div class="pipeline-description">
                    {description}
                </div>

            </div>
            """
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        SYSTEM
    </div>

    <div class="section-title">
        Model Information
    </div>
    """
)


info1, info2, info3, info4 = st.columns(
    4,
    gap="medium",
)


with info1:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Architecture
            </div>

            <div class="metric-value">
                CNN V2
            </div>

            <div class="metric-description">
                Custom convolutional baseline.
            </div>

        </div>
        """
    )


with info2:

    parameters = info.get(
        "parameters",
        1242863,
    )

    st.html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Trainable Parameters
            </div>

            <div class="metric-value">
                {parameters:,}
            </div>

            <div class="metric-description">
                Learnable model parameters.
            </div>

        </div>
        """
    )


with info3:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Morphology Tasks
            </div>

            <div class="metric-value">
                10
            </div>

            <div class="metric-description">
                8 binary + 2 multiclass.
            </div>

        </div>
        """
    )


with info4:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Input Resolution
            </div>

            <div class="metric-value">
                224 × 224
            </div>

            <div class="metric-description">
                RGB image input.
            </div>

        </div>
        """
    )


# ============================================================
# RESEARCH INTERPRETATION
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        RESEARCH INTERPRETATION
    </div>

    <div class="section-title">
        What Model 1 Demonstrates
    </div>
    """
)


research1, research2 = st.columns(
    2,
    gap="large",
)


with research1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                STRENGTHS
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Strong recognition of common morphology
            </div>

            <ul style="
                color:#8e9ab8;
                line-height:1.85;
                font-size:0.84rem;
            ">

                <li>
                    Featured F1:
                    <strong style="color:#dfe4ff;">
                        95.03%
                    </strong>
                </li>

                <li>
                    Roundedness accuracy:
                    <strong style="color:#dfe4ff;">
                        83.93%
                    </strong>
                </li>

                <li>
                    Symmetry F1:
                    <strong style="color:#dfe4ff;">
                        75.31%
                    </strong>
                </li>

                <li>
                    Clumpy F1:
                    <strong style="color:#dfe4ff;">
                        71.23%
                    </strong>
                </li>

            </ul>

        </div>
        """
    )


with research2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                LIMITATIONS
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Rare features remain difficult
            </div>

            <ul style="
                color:#8e9ab8;
                line-height:1.85;
                font-size:0.84rem;
            ">

                <li>
                    Bar F1:
                    <strong style="color:#dfe4ff;">
                        17.88%
                    </strong>
                </li>

                <li>
                    Merger F1:
                    <strong style="color:#dfe4ff;">
                        16.34%
                    </strong>
                </li>

                <li>
                    Disturbed F1:
                    <strong style="color:#dfe4ff;">
                        8.43%
                    </strong>
                </li>

                <li>
                    Severe class imbalance affects minority features.
                </li>

            </ul>

        </div>
        """
    )


# ============================================================
# FUTURE WORK
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="section-kicker">
        NEXT EXPERIMENTS
    </div>

    <div class="section-title">
        Future Work
    </div>
    """
)


future1, future2, future3 = st.columns(
    3,
    gap="medium",
)


with future1:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                MODEL 2
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                ResNet
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Compare a residual architecture using the same
                dataset, task definitions and evaluation protocol.
            </p>

        </div>
        """
    )


with future2:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                MODEL 3
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                MobileNet
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Investigate the trade-off between computational
                efficiency and morphology classification performance.
            </p>

        </div>
        """
    )


with future3:

    st.html(
        """
        <div class="glass-panel">

            <div class="result-label">
                COMPARATIVE STUDY
            </div>

            <div style="
                color:#e9edff;
                font-size:1rem;
                font-weight:800;
                margin-top:12px;
            ">
                Architecture comparison
            </div>

            <p style="
                color:#8e9ab8;
                font-size:0.82rem;
                line-height:1.7;
            ">
                Compare F1, balanced accuracy, ROC-AUC, PR-AUC,
                parameter count and computational efficiency.
            </p>

        </div>
        """
    )


# ============================================================
# FINAL RESEARCH NOTE
# ============================================================

st.html("<br>")

st.html(
    """
    <div class="research-note">

        <strong>
            Research conclusion
        </strong>

        <br><br>

        Custom CNN V2 provides a useful multi-task baseline
        and learns several common galaxy morphology attributes
        effectively.

        However, performance on rare attributes remains limited,
        particularly for Bar, Merger and Disturbed morphology.

        The next research step is to keep the task definition
        and evaluation protocol fixed and compare the baseline
        with ResNet and MobileNet architectures.

        <br><br>

        <strong>
            Important metric distinction:
        </strong>

        83.41% is the pooled accuracy across valid task predictions,
        whereas 56.82% is the overall Macro Task F1.
        These metrics should not be treated as interchangeable.

        <br><br>

        <strong>
            Important inference note:
        </strong>

        Displayed model scores are model output scores and should
        not be interpreted as calibrated scientific probabilities
        or certainty.

    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <div class="divider"></div>

        <strong style="color:#7886aa;">
            GALAXAI
        </strong>

        <br>

        Deep Learning-Based Multi-Attribute Galaxy Morphology Analysis

        <br>

        Galaxy Zoo: Hubble · Hubble Space Telescope · Custom CNN V2

        <br><br>

        Research Prototype · Model 1 Baseline

    </div>
    """
)