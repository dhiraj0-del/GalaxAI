import streamlit as st
from PIL import Image

from src.inference import load_model, predict, model_info


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

    /* ======================================================
       GLOBAL
       ====================================================== */

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 20% 10%,
                rgba(70, 90, 190, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 70%,
                rgba(120, 55, 190, 0.16),
                transparent 32%
            ),
            #03050d !important;
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

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }


    /* ======================================================
       ANIMATED SPACE BACKGROUND
       ====================================================== */

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.65;
        background-image:
            radial-gradient(circle, rgba(255,255,255,0.65) 1px, transparent 1px),
            radial-gradient(circle, rgba(160,180,255,0.35) 1px, transparent 1px);
        background-size: 110px 110px, 170px 170px;
        background-position: 0 0, 40px 70px;
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


    /* ======================================================
       HERO
       ====================================================== */

    .hero-wrapper {
        position: relative;
        overflow: hidden;
        padding: 42px 46px;
        border-radius: 30px;
        margin-bottom: 28px;

        background:
            radial-gradient(
                circle at 80% 30%,
                rgba(112, 95, 255, 0.22),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                rgba(15, 21, 48, 0.95),
                rgba(7, 10, 25, 0.92)
            );

        border: 1px solid rgba(145, 160, 255, 0.22);

        box-shadow:
            0 30px 90px rgba(0, 0, 0, 0.45),
            inset 0 1px 0 rgba(255,255,255,0.08);

        transform: perspective(1200px) rotateX(1deg);
    }

    .hero-wrapper::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -70px;
        top: -80px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(125, 140, 255, 0.18),
                rgba(70, 40, 150, 0.08),
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
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: clamp(2.8rem, 6vw, 5.5rem);
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
        max-width: 800px;
        margin-top: 18px;

        color: #a8b2cf;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 20px;
        padding: 7px 13px;

        border-radius: 999px;

        color: #c7d0ff;
        background: rgba(115, 130, 255, 0.10);
        border: 1px solid rgba(130, 145, 255, 0.20);

        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }


    /* ======================================================
       SECTION TITLES
       ====================================================== */

    .section-kicker {
        color: #727fa9;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .section-title {
        color: #eef1ff;
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 18px;
    }


    /* ======================================================
       GLASS PANELS
       ====================================================== */

    .glass-panel {
        padding: 25px;

        border-radius: 24px;

        background:
            linear-gradient(
                145deg,
                rgba(22, 29, 59, 0.78),
                rgba(8, 12, 28, 0.82)
            );

        border: 1px solid rgba(150, 165, 255, 0.14);

        box-shadow:
            0 22px 60px rgba(0, 0, 0, 0.30),
            inset 0 1px 0 rgba(255,255,255,0.05);

        backdrop-filter: blur(18px);

        transition:
            transform 0.25s ease,
            border-color 0.25s ease;
    }

    .glass-panel:hover {
        transform:
            perspective(1000px)
            rotateX(1deg)
            translateY(-2px);

        border-color:
            rgba(150, 165, 255, 0.26);
    }


    /* ======================================================
       UPLOAD AREA
       ====================================================== */

    [data-testid="stFileUploader"] {
        background:
            rgba(10, 15, 32, 0.72);
        border-radius: 18px;
        padding: 8px;
        border: 1px dashed rgba(130, 145, 255, 0.28);
    }

    [data-testid="stFileUploaderDropzone"] {
        background:
            linear-gradient(
                145deg,
                rgba(22, 30, 62, 0.80),
                rgba(10, 14, 30, 0.85)
            ) !important;

        border-radius: 16px !important;
    }


    /* ======================================================
       IMAGE CONTAINER
       ====================================================== */

    .image-frame {
        padding: 8px;
        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(130, 145, 255, 0.20),
                rgba(180, 100, 255, 0.08)
            );

        box-shadow:
            0 20px 60px rgba(0,0,0,0.40);
    }


    /* ======================================================
       RESULT CARDS
       ====================================================== */

    .result-card {
        position: relative;
        overflow: hidden;

        min-height: 145px;

        padding: 18px;

        border-radius: 19px;

        background:
            linear-gradient(
                145deg,
                rgba(24, 32, 67, 0.86),
                rgba(11, 15, 34, 0.88)
            );

        border: 1px solid rgba(145, 160, 255, 0.13);

        box-shadow:
            0 12px 35px rgba(0,0,0,0.22);

        transition:
            transform 0.22s ease,
            border-color 0.22s ease,
            box-shadow 0.22s ease;
    }

    .result-card:hover {
        transform:
            perspective(800px)
            rotateX(2deg)
            translateY(-4px);

        border-color:
            rgba(150, 170, 255, 0.30);

        box-shadow:
            0 20px 45px rgba(0,0,0,0.32);
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

    .score-track {
        height: 5px;
        width: 100%;

        margin-top: 11px;

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
                #6f82ff,
                #b58cff
            );
    }


    /* ======================================================
       STATUS BADGES
       ====================================================== */

    .status-detected {
        display: inline-block;

        padding: 4px 9px;

        border-radius: 999px;

        background: rgba(91, 116, 255, 0.14);
        border: 1px solid rgba(115, 135, 255, 0.25);

        color: #b9c5ff;

        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .status-negative {
        display: inline-block;

        padding: 4px 9px;

        border-radius: 999px;

        background: rgba(100, 110, 140, 0.10);
        border: 1px solid rgba(130, 140, 165, 0.18);

        color: #8f99b5;

        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }


    /* ======================================================
       PIPELINE
       ====================================================== */

    .pipeline-grid {
        display: grid;

        grid-template-columns:
            1fr
            auto
            1fr
            auto
            1fr
            auto
            1fr;

        align-items: center;

        gap: 10px;

        margin-top: 12px;
    }

    .pipeline-box {
        min-height: 125px;

        display: flex;
        flex-direction: column;

        justify-content: center;
        align-items: center;

        text-align: center;

        padding: 18px;

        border-radius: 19px;

        background:
            linear-gradient(
                145deg,
                rgba(23, 31, 64, 0.85),
                rgba(9, 13, 29, 0.88)
            );

        border: 1px solid rgba(140, 155, 255, 0.13);

        box-shadow:
            0 15px 35px rgba(0,0,0,0.22);
    }

    .pipeline-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
    }

    .pipeline-title {
        color: #edf0ff;
        font-weight: 800;
        font-size: 0.92rem;
    }

    .pipeline-desc {
        color: #7e89aa;
        font-size: 0.70rem;
        margin-top: 6px;
    }

    .pipeline-arrow {
        color: #7584bd;
        font-size: 1.45rem;
        font-weight: 800;
    }


    /* ======================================================
       MODEL INFORMATION
       ====================================================== */

    .metric-card {
        padding: 20px;

        border-radius: 18px;

        background:
            rgba(15, 21, 43, 0.78);

        border: 1px solid rgba(145,160,255,0.12);

        text-align: center;
    }

    .metric-label {
        color: #7581a4;
        font-size: 0.67rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .metric-value {
        color: #f1f3ff;
        font-size: 1.35rem;
        font-weight: 850;
        margin-top: 7px;
    }


    /* ======================================================
       RESEARCH NOTE
       ====================================================== */

    .research-note {
        padding: 22px;

        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                rgba(35, 40, 82, 0.55),
                rgba(12, 16, 35, 0.70)
            );

        border: 1px solid rgba(145, 160, 255, 0.13);

        color: #9da8c5;

        line-height: 1.7;
    }

    .research-note strong {
        color: #dfe4ff;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {
        min-height: 52px;

        border-radius: 14px !important;

        border: 1px solid rgba(150,165,255,0.30) !important;

        background:
            linear-gradient(
                135deg,
                #596df5,
                #8b5de8
            ) !important;

        color: white !important;

        font-weight: 800 !important;

        letter-spacing: 0.06em;

        box-shadow:
            0 12px 30px rgba(87, 93, 220, 0.28);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 18px 38px rgba(87, 93, 220, 0.38);
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        margin-top: 55px;

        padding-top: 20px;

        border-top:
            1px solid rgba(120,135,180,0.10);

        text-align: center;

        color: #59627d;

        font-size: 0.72rem;

        letter-spacing: 0.08em;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 900px) {

        .pipeline-grid {
            grid-template-columns: 1fr;
        }

        .pipeline-arrow {
            transform: rotate(90deg);
            text-align: center;
        }

        .hero-wrapper {
            padding: 30px 25px;
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
except Exception as error:
    st.error(
        "Unable to load the GALAXAI model."
    )
    st.exception(error)
    st.stop()


info = model_info()


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
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    label_visibility="collapsed",
)


if uploaded_file is None:

    st.html(
        """
        <div class="glass-panel">

            <div style="
                text-align:center;
                padding:30px 10px;
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
                    JPG, PNG or WEBP · The model will
                    automatically preprocess the image
                    to 224 × 224 RGB.
                </div>

            </div>

        </div>
        """
    )


else:

    image = Image.open(uploaded_file)

    # Convert for consistent display/inference
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ========================================================
    # INPUT / ANALYSIS COLUMNS
    # ========================================================

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
            '<div class="image-frame">'
        )

        st.image(
            image,
            use_container_width=True,
        )

        st.html(
            '</div>'
        )

        st.caption(
            f"Original image · "
            f"{image.width} × {image.height} px · "
            f"{image.mode}"
        )

    # ========================================================
    # ANALYSIS CONTROL
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

            results = predict(
                image,
                model,
            )

        st.session_state["results"] = results


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    st.markdown("<br>", unsafe_allow_html=True)

    st.html(
        """
        <div class="section-kicker">
            MODEL OUTPUT
        </div>

        <div class="section-title">
            Galaxy Morphology Profile
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

    cols = st.columns(
        4,
        gap="medium",
    )

    for index, task in enumerate(
        binary_order
    ):

        result = results[task]

        score = result["score"]

        status = (
            "DETECTED"
            if result["detected"]
            else "NOT DETECTED"
        )

        status_class = (
            "status-detected"
            if result["detected"]
            else "status-negative"
        )

        with cols[index % 4]:

            st.html(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        {result["label"]}
                    </div>

                    <div style="margin-top:9px;">
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

                </div>
                """
            )


    # ========================================================
    # MULTICLASS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    multi_col1, multi_col2 = st.columns(
        2,
        gap="medium",
    )

    # --------------------------------------------------------
    # BULGE
    # --------------------------------------------------------

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
                    {bulge["score"]:.3f}
                </div>

                <div class="score-track">
                    <div
                        class="score-fill"
                        style="width:{bulge["score"] * 100:.1f}%"
                    ></div>
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # ROUNDEDNESS
    # --------------------------------------------------------

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
                    {rounded["score"]:.3f}
                </div>

                <div class="score-track">
                    <div
                        class="score-fill"
                        style="width:{rounded["score"] * 100:.1f}%"
                    ></div>
                </div>

            </div>
            """
        )


# ============================================================
# PIPELINE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.html(
    """
    <div class="section-kicker">
        INFERENCE PIPELINE
    </div>

    <div class="section-title">
        How GALAXAI Analyzes the Galaxy
    </div>

    <div class="pipeline-grid">

        <div class="pipeline-box">

            <div class="pipeline-icon">
                🌌
            </div>

            <div class="pipeline-title">
                Galaxy Image
            </div>

            <div class="pipeline-desc">
                Uploaded observation
            </div>

        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-box">

            <div class="pipeline-icon">
                ⚙️
            </div>

            <div class="pipeline-title">
                Preprocessing
            </div>

            <div class="pipeline-desc">
                224 × 224 · RGB · Normalize
            </div>

        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-box">

            <div class="pipeline-icon">
                🧠
            </div>

            <div class="pipeline-title">
                CNN Feature Extraction
            </div>

            <div class="pipeline-desc">
                Learned visual patterns
            </div>

        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-box">

            <div class="pipeline-icon">
                🔭
            </div>

            <div class="pipeline-title">
                Morphology Profile
            </div>

            <div class="pipeline-desc">
                10 independent tasks
            </div>

        </div>

    </div>
    """
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

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

m1, m2, m3, m4 = st.columns(
    4,
    gap="medium",
)

with m1:

    st.html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Architecture
            </div>

            <div class="metric-value">
                Custom CNN V2
            </div>

        </div>
        """
    )

with m2:

    st.html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Parameters
            </div>

            <div class="metric-value">
                {info["parameters"]:,}
            </div>

        </div>
        """
    )

with m3:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Morphology Tasks
            </div>

            <div class="metric-value">
                10
            </div>

        </div>
        """
    )

with m4:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Input Resolution
            </div>

            <div class="metric-value">
                224 × 224
            </div>

        </div>
        """
    )


# ============================================================
# RESEARCH NOTE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.html(
    """
    <div class="research-note">

        <strong>Research interpretation</strong>

        <br><br>

        GALAXAI performs multi-task galaxy morphology analysis.
        Each morphological attribute is evaluated using its
        corresponding learned prediction head.

        <br><br>

        Binary attributes are reported as detected or not detected,
        together with the model's output score.

        Bulge Prominence and Roundedness are multiclass predictions
        and are reported using their predicted morphological category.

        <br><br>

        <strong>Important:</strong>
        displayed model scores should not be interpreted as
        calibrated probabilities or scientific certainty.

    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        GALAXAI · Deep Learning-Based Multi-Attribute
        Galaxy Morphology Analysis

        <br><br>

        Research Prototype · Custom CNN V2

    </div>
    """
)