from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(page_title="GALAXAI | Galaxy Morphology", page_icon="🌌", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
:root { --space-ink:#050711; --space-line:rgba(130,164,255,.25); --space-cyan:#72efff; --space-text:#eef3ff; }
.stApp { color:var(--space-text); background:radial-gradient(ellipse at 15% 5%,rgba(49,69,155,.38),transparent 35%),radial-gradient(ellipse at 85% 20%,rgba(87,39,139,.26),transparent 32%),radial-gradient(ellipse at 50% 95%,rgba(14,77,110,.22),transparent 42%),#050711; background-attachment:fixed; }
.stApp:before { content:''; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.35; background-image:radial-gradient(1px 1px at 7% 18%,#fff 99%,transparent),radial-gradient(1px 1px at 24% 78%,#9eeaff 99%,transparent),radial-gradient(2px 2px at 44% 32%,#fff 99%,transparent),radial-gradient(1px 1px at 66% 14%,#fff 99%,transparent),radial-gradient(1px 1px at 83% 61%,#b7b3ff 99%,transparent),radial-gradient(1px 1px at 92% 28%,#fff 99%,transparent),radial-gradient(1px 1px at 56% 88%,#fff 99%,transparent); background-size:310px 240px; }
.block-container { max-width:1500px; padding-top:1.3rem; padding-bottom:3rem; position:relative; z-index:1; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(10,15,34,.97),rgba(7,9,21,.97)); border-right:1px solid var(--space-line); }
section[data-testid="stSidebar"] * { color:#e7edff; }
.hero { position:relative; overflow:hidden; padding:2.4rem 2.5rem; border:1px solid rgba(123,170,255,.4); border-radius:28px; background:linear-gradient(120deg,rgba(23,37,78,.94),rgba(12,15,39,.88) 58%,rgba(45,24,78,.8)); box-shadow:0 25px 65px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.11),0 0 38px rgba(83,119,255,.13); margin-bottom:1.5rem; transform:perspective(1200px) rotateX(.6deg); }
.hero:after { content:'✦'; position:absolute; right:8%; top:2%; font-size:10rem; color:rgba(125,206,255,.12); text-shadow:0 0 40px rgba(103,171,255,.35); transform:rotate(12deg); pointer-events:none; }
.eyebrow { color:var(--space-cyan); letter-spacing:.22em; font-size:.72rem; font-weight:700; text-transform:uppercase; }
.hero h1 { font-family:'Orbitron',sans-serif; color:#f7f8ff; font-size:clamp(2.4rem,5vw,4.1rem); letter-spacing:.04em; margin:.45rem 0; text-shadow:0 0 26px rgba(114,239,255,.28); }
.hero p { color:#c9d4f5; max-width:970px; font-size:1.03rem; line-height:1.7; }
h1,h2,h3 { font-family:'Orbitron','Inter',sans-serif !important; letter-spacing:.015em; }
h2 { color:#e6eaff; } h3 { color:#d9e5ff; }
p,li,label,.stMarkdown { line-height:1.65; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(25,38,75,.9),rgba(10,15,34,.9)); border:1px solid var(--space-line); padding:15px 17px; border-radius:18px; box-shadow:0 12px 28px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.07); transition:transform .22s ease,box-shadow .22s ease; }
div[data-testid="stMetric"]:hover { transform:translateY(-4px) rotateX(2deg); box-shadow:0 18px 32px rgba(0,0,0,.3),0 0 22px rgba(102,151,255,.13); }
div[data-testid="stMetricLabel"] { color:#b6c6ee !important; } div[data-testid="stMetricValue"] { color:#f2f6ff !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid rgba(130,164,255,.26) !important; border-radius:20px !important; background:linear-gradient(145deg,rgba(20,29,60,.72),rgba(9,13,29,.72)) !important; box-shadow:0 14px 32px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.05); transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; }
div[data-testid="stVerticalBlockBorderWrapper"]:hover { transform:translateY(-3px); border-color:rgba(114,239,255,.5) !important; box-shadow:0 18px 38px rgba(0,0,0,.32),0 0 24px rgba(94,143,255,.1); }
.stButton>button { border:1px solid rgba(117,211,255,.45); border-radius:12px; background:linear-gradient(135deg,rgba(43,72,133,.95),rgba(76,49,132,.95)); color:white; font-weight:650; box-shadow:0 7px 18px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.1); transition:all .2s ease; }
.stButton>button:hover { transform:translateY(-2px); border-color:var(--space-cyan); box-shadow:0 10px 24px rgba(0,0,0,.32),0 0 18px rgba(114,239,255,.18); color:white; }
.stButton>button[kind="primary"] { background:linear-gradient(110deg,#287cb4,#6851c6); }
.stProgress > div > div { background:linear-gradient(90deg,#5de4ff,#9b8cff); }
.stTabs [data-baseweb="tab-list"] { gap:.45rem; } .stTabs [data-baseweb="tab"] { background:rgba(18,26,53,.75); border:1px solid rgba(130,164,255,.2); border-radius:10px 10px 4px 4px; padding:.55rem .9rem; }
.stTabs [aria-selected="true"] { border-color:rgba(114,239,255,.6) !important; background:rgba(45,68,122,.8) !important; }
div[data-testid="stExpander"] { border:1px solid rgba(130,164,255,.23); border-radius:14px; background:rgba(13,19,41,.55); }
[data-testid="stDataFrame"] { border:1px solid rgba(130,164,255,.22); border-radius:12px; overflow:hidden; } hr { border-color:rgba(130,164,255,.22); }
small,.stCaption { color:#aebcdd !important; }
@media (prefers-reduced-motion: reduce) { *,*:before,*:after { transition:none !important; animation:none !important; } }
</style>
<div class="hero"><div class="eyebrow">HUBBLE SPACE TELESCOPE · MULTI-TASK AI RESEARCH</div>
<h1>✦ GALAXAI</h1><p><b>Explore the hidden morphology of galaxies.</b> A deep-learning research laboratory comparing a custom CNN, ResNet-18 and MobileNetV3-Large on ten galaxy morphology attributes. Upload an image, inspect each model's predictions, explore its architecture, and compare the full held-out test evaluation.</p></div>
""", unsafe_allow_html=True)

BINARY = ["featured", "edge_on", "bar", "spiral_arms", "disturbed", "merger", "clumpy", "symmetry"]
MULTI = ["bulge", "roundedness"]
TASKS = BINARY + MULTI
LABELS = {
    "featured":"Featured", "edge_on":"Edge-on", "bar":"Bar", "spiral_arms":"Spiral arms",
    "disturbed":"Disturbed", "merger":"Merger", "clumpy":"Clumpy", "symmetry":"Symmetry",
    "bulge":"Bulge prominence", "roundedness":"Roundedness",
}
CLASSES = {"bulge":["None","Just noticeable","Obvious","Dominant"],
           "roundedness":["Completely rounded","In-between","Cigar-shaped"]}
MODELS = {
    "Model 1 — Custom CNN V2": {
        "architecture":"Custom CNN V2", "description":"Four convolutional blocks (32, 64, 128, 256 channels), adaptive pooling, and a shared 256-unit head.",
        "metrics":[BASE_DIR/"models/model1_custom_cnn/results/test_metrics.json", BASE_DIR/"results/model1_v2/test_metrics.json"],
        "heatmaps":BASE_DIR/"models/model1_custom_cnn/results", "best_epoch":27, "parameters":"1,242,863",
        "checkpoint":BASE_DIR/"checkpoints/galaxai_model1_v2_best.pth"},
    "Model 2 — ResNet-18": {
        "architecture":"ImageNet-pretrained ResNet-18", "description":"ResNet-18 backbone (512 features), shared 256-unit head, and 10 task-specific prediction heads.",
        "metrics":[BASE_DIR/"models/model2_resnet/results/test_metrics.json", BASE_DIR/"results/model2_resnet/test_metrics.json"],
        "heatmaps":BASE_DIR/"models/model2_resnet/results", "best_epoch":2, "parameters":"Not recorded here",
        "checkpoint":BASE_DIR/"models/model2_resnet/checkpoints/galaxai_model2_resnet18_best.pth"},
    "Model 3 — MobileNetV3-Large": {
        "architecture":"ImageNet-pretrained MobileNetV3-Large", "description":"MobileNetV3-Large backbone (960 features), shared 256-unit head, and 10 task-specific prediction heads.",
        "metrics":[BASE_DIR/"models/model3_mobilenet/results/test_metrics.json", BASE_DIR/"results/model3_mobilenet/results/test_metrics.json"],
        "heatmaps":BASE_DIR/"models/model3_mobilenet/results", "best_epoch":3, "parameters":"3,221,823",
        "checkpoint":BASE_DIR/"models/model3_mobilenet/checkpoints/galaxai_model3_mobilenetv3_large_best.pth"},
}

@st.cache_data(show_spinner=False)
def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None

def load_metrics(name):
    for path in MODELS[name]["metrics"]:
        if path.is_file():
            return read_json(str(path)), path
    return None, None

def fmt(value, percent=False):
    try:
        number = float(value)
        return f"{number*100:.2f}%" if percent else f"{number:.4f}"
    except (TypeError, ValueError):
        return "—"

def summary(data):
    return data.get("summary", {}) if data else {}

def metric_for(record, key):
    aliases = {
        "F1 / Macro F1": ("f1", "macro_f1"),
        "Accuracy": ("accuracy",),
        "Balanced accuracy": ("balanced_accuracy",),
        "Precision": ("precision", "macro_precision"),
        "Recall": ("recall", "macro_recall"),
        "ROC-AUC": ("roc_auc",),
        "PR-AUC / AP": ("pr_auc", "average_precision"),
    }
    for candidate in aliases[key]:
        if candidate in record and record[candidate] is not None:
            return record[candidate]
    return None

def render_task_table(data, group):
    tasks = data.get(group, {}) if data else {}
    if not tasks:
        st.warning(f"No {group} metrics found in this JSON.")
        return
    rows = []
    for task in TASKS:
        if task not in tasks:
            continue
        m = tasks[task]
        row = {"Task": LABELS.get(task, task), "Valid samples": m.get("samples", "—")}
        if group == "binary":
            keys = [("Accuracy","accuracy"),("Balanced accuracy","balanced_accuracy"),
                    ("Precision","precision"),("Recall","recall"),("F1","f1"),
                    ("ROC-AUC","roc_auc"),("PR-AUC / AP","pr_auc")]
        else:
            keys = [("Accuracy","accuracy"),("Balanced accuracy","balanced_accuracy"),
                    ("Macro precision","macro_precision"),("Macro recall","macro_recall"),
                    ("Macro F1","macro_f1")]
            if isinstance(m.get("per_class_f1"), list):
                row["Per-class F1"] = ", ".join(fmt(x) for x in m["per_class_f1"])
        for label, key in keys:
            row[label] = fmt(m.get(key))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_confusion(name, data):
    config = MODELS[name]
    records = {}
    for group in ("binary", "multiclass"):
        if isinstance(data.get(group), dict):
            records.update(data[group])
    st.caption("All available task confusion matrices and saved heatmaps are shown below.")
    found_any = False
    for task in TASKS:
        if task not in records:
            continue
        record = records[task]
        cm = record.get("confusion_matrix")
        image_files = list(config["heatmaps"].glob("*.png")) if config["heatmaps"].is_dir() else []
        matching = [p for p in image_files if task.lower() in p.stem.lower()]
        with st.expander(LABELS.get(task, task), expanded=False):
            if isinstance(cm, list):
                labels = ["Negative", "Positive"] if task in BINARY else CLASSES.get(task, [str(i) for i in range(len(cm))])
                st.write("Confusion matrix (rows = true labels; columns = predicted labels)")
                st.dataframe(pd.DataFrame(cm, index=labels[:len(cm)], columns=labels[:len(cm[0])]), use_container_width=True)
                found_any = True
            if matching:
                st.image(str(matching[0]), caption=f"{LABELS.get(task, task)} confusion-matrix heatmap", use_container_width=True)
                found_any = True
            if not isinstance(cm, list) and not matching:
                st.info("No confusion-matrix values or saved heatmap were found for this task.")
    if not found_any:
        st.info("No confusion matrices were found in this model's evaluation JSON or results folder.")


def render_loss_history(name):
    config = MODELS[name]
    key = name.split("—")[0].strip().lower().replace(" ", "")
    history_candidates = []
    if key == "model1":
        history_candidates = [BASE_DIR/"results/model1_v2_history.json",
                              BASE_DIR/"models/model1_custom_cnn/results/history.json",
                              BASE_DIR/"models/model1_custom_cnn/results/model1_v2_history.json"]
    elif key == "model2":
        history_candidates = [BASE_DIR/"models/model2_resnet/results/model2_resnet18_history.json",
                              BASE_DIR/"models/model2_resnet/results/history.json"]
    elif key == "model3":
        history_candidates = [BASE_DIR/"models/model3_mobilenet/results/model3_mobilenetv3_large_history.json",
                              BASE_DIR/"models/model3_mobilenet/results/history.json"]
    history = None
    history_path = None
    for candidate in history_candidates:
        if candidate.is_file():
            history = read_json(str(candidate))
            if history is not None:
                history_path = candidate
                break
    if history is None:
        st.info("No recognized training-history JSON was found for this model. Loss plots are shown when the history file is available.")
        return
    rows = history.get("history", history.get("epochs", history.get("records", history)))
    if isinstance(rows, dict):
        # Some logs store metric series as parallel lists.
        keys = [k for k, v in rows.items() if isinstance(v, list)]
        if keys and all(isinstance(rows[k], list) for k in keys):
            length = max(len(rows[k]) for k in keys)
            frame = pd.DataFrame({k: rows[k] for k in keys if len(rows[k]) == length})
        else:
            st.json(history)
            return
    elif isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
        frame = pd.DataFrame(rows)
    else:
        st.json(history)
        return
    frame.columns = [str(c).lower() for c in frame.columns]
    epoch_col = next((c for c in frame.columns if c in ("epoch", "epochs")), None)
    train_col = next((c for c in frame.columns if "train" in c and "loss" in c and "task" not in c), None)
    val_col = next((c for c in frame.columns if ("val" in c or "valid" in c) and "loss" in c and "task" not in c), None)
    if epoch_col is not None:
        frame = frame.set_index(epoch_col)
    if train_col or val_col:
        chart_cols = [c for c in (train_col, val_col) if c and c in frame.columns]
        chart = frame[chart_cols].apply(pd.to_numeric, errors="coerce")
        chart.columns = [("Training loss" if "train" in c else "Validation loss") for c in chart.columns]
        st.line_chart(chart, use_container_width=True)
    else:
        st.caption("The history file was found, but its column names do not clearly identify train/validation loss.")
    st.caption(f"Training history source: {history_path}")
    st.dataframe(frame.reset_index(), use_container_width=True, hide_index=True)

def comparison_page():
    st.header("Three-model evaluation comparison")
    st.caption("Metrics are read from each model's saved test_metrics.json. No model is retrained by this dashboard.")
    loaded = {name: load_metrics(name) for name in MODELS}
    rows = []
    for name, (data, path) in loaded.items():
        s = summary(data)
        correct, valid = s.get("overall_correct_predictions"), s.get("overall_valid_predictions")
        rows.append({
            "Model":name, "Architecture":MODELS[name]["architecture"],
            "Pooled task accuracy":s.get("overall_accuracy", s.get("pooled_accuracy")),
            "Binary macro F1":s.get("binary_macro_f1"),
            "Multiclass macro F1":s.get("multiclass_macro_f1"),
            "Overall macro task F1":s.get("overall_macro_task_f1"),
            "Correct / valid predictions":f"{correct:,} / {valid:,}" if isinstance(correct,int) and isinstance(valid,int) else "—",
            "Metrics file":str(path) if path else "NOT FOUND",
        })
    st.subheader("Headline metrics")
    df = pd.DataFrame(rows)
    show = df.copy()
    for col in ["Pooled task accuracy","Binary macro F1","Multiclass macro F1","Overall macro task F1"]:
        show[col] = show[col].map(fmt)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Pooled task accuracy aggregates valid predictions across separate tasks; it is not the same as macro task F1 or conventional single-label image accuracy.")
    scored = [r for r in rows if isinstance(r["Overall macro task F1"], (int,float))]
    if scored:
        best = max(scored, key=lambda r:r["Overall macro task F1"])
        st.success(f"Highest overall macro task F1 in available JSON files: {best['Model']} ({best['Overall macro task F1']:.4f}).")

    st.subheader("Per-task comparison")
    metric = st.selectbox("Metric", ["F1 / Macro F1","Accuracy","Balanced accuracy","Precision","Recall","ROC-AUC","PR-AUC / AP"])
    chart_rows = []
    for task in TASKS:
        row = {"Task":LABELS[task]}
        group = "binary" if task in BINARY else "multiclass"
        for name, (data, _) in loaded.items():
            record = (data or {}).get(group, {}).get(task, {})
            row[name] = metric_for(record, metric)
        chart_rows.append(row)
    chart = pd.DataFrame(chart_rows).set_index("Task")
    st.bar_chart(chart, use_container_width=True)
    table = chart.reset_index()
    for name in MODELS:
        table[name] = table[name].map(fmt)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("All evaluation metrics")
    selected = st.selectbox("Model detail", list(MODELS), key="detail_model")
    data, path = loaded[selected]
    if data is None:
        st.error("Metrics JSON not found. Expected one of:\n" + "\n".join(str(p) for p in MODELS[selected]["metrics"]))
    else:
        tabs = st.tabs(["8 binary tasks","2 multiclass tasks","Confusion matrices","Training history","Architecture & files"])
        with tabs[0]:
            render_task_table(data, "binary")
        with tabs[1]:
            render_task_table(data, "multiclass")
        with tabs[2]:
            render_confusion(selected, data)
        with tabs[3]:
            render_loss_history(selected)
        with tabs[4]:
            config = MODELS[selected]
            st.markdown(f"**{config['architecture']}**")
            st.write(config["description"])
            st.write(f"Recorded best epoch: {config['best_epoch']}")
            st.write(f"Parameter count: {config['parameters']}")
            st.write(f"Checkpoint exists locally: {'Yes' if config['checkpoint'].is_file() else 'No / not included locally'}")
            st.code(str(path))
    with st.expander("Metric guide"):
        st.markdown("""
- **Precision:** proportion of predicted positives that are correct.
- **Recall:** proportion of actual positives detected.
- **F1:** harmonic mean of precision and recall.
- **Balanced accuracy:** average class recall, useful with imbalanced labels.
- **ROC-AUC:** ability to rank positives above negatives across thresholds.
- **PR-AUC / average precision:** particularly useful for rare positive classes.
- **Macro F1:** gives each class or task equal weight rather than letting common labels dominate.
- **Confusion matrix:** shows false positives and false negatives that a headline metric can hide.
""")
    st.warning("For a strict architecture-only comparison, verify the three evaluation JSONs use the same test split and metric definitions. Rare attributes such as disturbed, bar and merger should be judged using precision, recall, F1, PR-AUC and confusion matrices—not accuracy alone.")

def overview_page():
    st.header("Research overview")
    a,b,c = st.columns(3)
    a.metric("Architectures","3")
    b.metric("Morphology tasks","10")
    c.metric("Output logits","15")
    st.markdown("""
**Research question:** How effectively can different CNN architectures learn multiple morphological attributes from Hubble Space Telescope galaxy images, and how do architectural differences affect recognition of common and rare morphological features?

**Dataset:** Galaxy Zoo: Hubble. The project uses a 70/15/15 train-validation-test split (`random_state=42`) and task-specific masks for unavailable or ambiguous annotations.

- **8 binary tasks:** featured, edge-on, bar, spiral arms, disturbed, merger, clumpy and symmetry.
- **2 multiclass tasks:** bulge prominence (4 classes) and roundedness (3 classes).
- **15 logits total:** one logit for each binary task, four bulge logits and three roundedness logits.
""")
    for name, config in MODELS.items():
        with st.expander(name):
            st.markdown(f"**{config['architecture']}**")
            st.write(config["description"])
            st.write(f"Recorded best epoch: {config['best_epoch']} · Parameters: {config['parameters']}")
    st.info("Model 1 remains the frozen baseline. This app reads existing results and does not modify checkpoints or retrain models.")

# ------------------------- prediction helpers -------------------------
import importlib.util
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

MODEL_SPECS = {
    "Model 1 — Custom CNN V2": {
        "module_path": BASE_DIR / "src" / "models.py",
        "class_name": "GalaxyCNNV2",
        "checkpoint": BASE_DIR / "checkpoints" / "galaxai_model1_v2_best.pth",
        "architecture_page": "Model 1 prediction",
    },
    "Model 2 — ResNet-18": {
        "module_path": BASE_DIR / "models" / "model2_resnet" / "models.py",
        "class_name": "GalaxyResNet18",
        "checkpoint": BASE_DIR / "models" / "model2_resnet" / "checkpoints" / "galaxai_model2_resnet18_best.pth",
        "architecture_page": "Model 2 prediction",
    },
    "Model 3 — MobileNetV3-Large": {
        "module_path": BASE_DIR / "models" / "model3_mobilenet" / "models.py",
        "class_name": "GalaxyMobileNetV3",
        "checkpoint": BASE_DIR / "models" / "model3_mobilenet" / "checkpoints" / "galaxai_model3_mobilenetv3_large_best.pth",
        "architecture_page": "Model 3 prediction",
    },
}

@st.cache_resource(show_spinner=False)
def load_predictor(model_name):
    """Load one model and its local checkpoint. Does not retrain or download weights."""
    spec = MODEL_SPECS[model_name]
    if not spec["module_path"].is_file():
        raise FileNotFoundError(f"Model definition not found: {spec['module_path']}")
    if not spec["checkpoint"].is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {spec['checkpoint']}")

    if model_name.startswith("Model 1"):
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        from src.models import GalaxyCNNV2
        model = GalaxyCNNV2()
    else:
        module_key = "galaxai_" + ("resnet_model_module" if model_name.startswith("Model 2") else "mobilenet_model_module")
        module_spec = importlib.util.spec_from_file_location(module_key, spec["module_path"])
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not import model file: {spec['module_path']}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_key] = module
        module_spec.loader.exec_module(module)
        model_class = getattr(module, spec["class_name"])
        # Avoid downloading pretrained weights: the trained checkpoint supplies the parameters.
        try:
            model = model_class(pretrained=False)
        except TypeError:
            model = model_class()

    checkpoint = torch.load(spec["checkpoint"], map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict_all(model_name, image):
    model = load_predictor(model_name)
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        outputs = model(tensor)
    result = {}
    for task in BINARY:
        score = float(torch.sigmoid(outputs[task].reshape(-1)[0]).item())
        result[task] = {"score": score, "detected": score >= 0.5}
    for task in MULTI:
        probs = torch.softmax(outputs[task], dim=1)[0]
        idx = int(torch.argmax(probs).item())
        result[task] = {"score": float(probs[idx].item()), "class_index": idx,
                        "prediction": CLASSES[task][idx],
                        "probabilities": [float(v) for v in probs.tolist()]}
    return result


def architecture_section(model_name):
    config = MODELS[model_name]
    st.subheader("Architecture used")
    st.markdown(f"**{config['architecture']}**")
    st.write(config["description"])
    st.write(f"Recorded best epoch: {config['best_epoch']} · Parameter count: {config['parameters']}")
    if model_name.startswith("Model 1"):
        st.markdown("- Input: 224 × 224 RGB image\n- Four blocks: 3→32→64→128→256 channels; each block uses two 3×3 convolutions, batch normalization, ReLU and max pooling\n- Adaptive average pooling, flattening, shared 256-unit dense layer and 0.40 dropout\n- Eight binary heads and two multiclass heads (bulge: 4 classes; roundedness: 3 classes)")
    elif model_name.startswith("Model 2"):
        st.markdown("- Input: 224 × 224 RGB image\n- Torchvision ResNet-18 backbone with residual skip connections; final backbone feature size: 512\n- Original ImageNet classifier replaced by a shared 512→256 dense layer, ReLU and 0.40 dropout\n- Eight binary heads and two multiclass heads; 15 logits total")
    else:
        st.markdown("- Input: 224 × 224 RGB image\n- Torchvision MobileNetV3-Large backbone; final feature size: 960\n- Original ImageNet classifier replaced by a shared 960→256 dense layer, ReLU and 0.40 dropout\n- Eight binary heads and two multiclass heads; 15 logits total")
    st.caption(f"Checkpoint: {MODEL_SPECS[model_name]['checkpoint']}")


def model_prediction_page(model_name):
    st.header(model_name)
    st.write("Upload a galaxy image to estimate all ten morphology attributes using this model's own trained checkpoint.")
    st.caption(f"Inference device: {DEVICE} · Input preprocessing: RGB, resize to 224×224, normalize with mean/std [0.5, 0.5, 0.5].")
    uploaded = st.file_uploader("Upload a galaxy image", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"], key="upload_" + model_name)
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as exc:
            st.error(f"Could not open the image: {exc}")
    if image is not None:
        left, right = st.columns([1, 1.25])
        with left:
            st.image(image, caption=f"Uploaded image · {image.width} × {image.height}", use_container_width=True)
        with right:
            if st.button("🚀 Analyze galaxy", type="primary", use_container_width=True, key="analyze_" + model_name):
                try:
                    with st.spinner(f"Running {model_name} inference…"):
                        st.session_state["prediction_" + model_name] = predict_all(model_name, image)
                except Exception as exc:
                    st.error("Prediction failed. Check the model class name, checkpoint and installed PyTorch/torchvision versions.")
                    st.exception(exc)
            result = st.session_state.get("prediction_" + model_name)
            if isinstance(result, dict):
                st.subheader("Binary attributes")
                cols = st.columns(2)
                for i, task in enumerate(BINARY):
                    item = result[task]
                    with cols[i % 2]:
                        st.metric(LABELS[task], f"{item['score']:.1%}", "Detected" if item["detected"] else "Not detected")
                        st.progress(max(0.0, min(1.0, item["score"])))
                st.subheader("Multiclass attributes")
                for task in MULTI:
                    item = result[task]
                    st.markdown(f"**{LABELS[task]}:** {item['prediction']}  ·  model score {item['score']:.1%}")
                    st.dataframe(pd.DataFrame({"Class": CLASSES[task], "Score": item["probabilities"]}), hide_index=True, use_container_width=True)
                st.caption("Scores are model outputs and are not guaranteed to be calibrated probabilities. Binary predictions use a 0.50 threshold for display.")
    st.divider()
    architecture_section(model_name)


def overview_page():
    st.header("Home · Research overview")
    a, b, c = st.columns(3)
    a.metric("Architectures", "3")
    b.metric("Morphology tasks", "10")
    c.metric("Output logits", "15")
    st.markdown("""
**Research question:** How effectively can different CNN architectures learn multiple morphological attributes from Hubble Space Telescope galaxy images, and how do architectural differences affect recognition of common and rare morphological features?

**Dataset:** Galaxy Zoo: Hubble. The project uses a 70/15/15 train-validation-test split (`random_state=42`) and task-specific masks for unavailable or ambiguous annotations.

- **8 binary tasks:** featured, edge-on, bar, spiral arms, disturbed, merger, clumpy and symmetry.
- **2 multiclass tasks:** bulge prominence (4 classes) and roundedness (3 classes).
- **15 output logits:** 8 binary logits + 4 bulge logits + 3 roundedness logits.
""")
    st.subheader("Research pipeline")
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:.4rem 0 1.2rem">
      <div style="padding:17px;border:1px solid #344b80;border-radius:15px;background:linear-gradient(145deg,#17264a,#0d1228)"><div style="font-size:1.45rem">🛰️</div><b>01 · Hubble data</b><div style="color:#b8c8eb;font-size:.88rem">Galaxy Zoo: Hubble images and morphology annotations</div></div>
      <div style="padding:17px;border:1px solid #344b80;border-radius:15px;background:linear-gradient(145deg,#17264a,#0d1228)"><div style="font-size:1.45rem">🧪</div><b>02 · Data preparation</b><div style="color:#b8c8eb;font-size:.88rem">RGB · 224 × 224 · masked multi-task labels</div></div>
      <div style="padding:17px;border:1px solid #344b80;border-radius:15px;background:linear-gradient(145deg,#17264a,#0d1228)"><div style="font-size:1.45rem">🧠</div><b>03 · Model training</b><div style="color:#b8c8eb;font-size:.88rem">Three architectures, shared task definitions and data split</div></div>
      <div style="padding:17px;border:1px solid #344b80;border-radius:15px;background:linear-gradient(145deg,#17264a,#0d1228)"><div style="font-size:1.45rem">📊</div><b>04 · Evaluation</b><div style="color:#b8c8eb;font-size:.88rem">Per-task metrics, class balance and confusion matrices</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Dataset and experimental protocol")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Dataset scale**")
        st.write("The project uses 80,632 Galaxy Zoo: Hubble image records, split into 56,442 training, 12,095 validation and 12,095 test records (70/15/15; random state 42). The held-out test set is used for evaluation, not model fitting.")
        st.markdown("**Image preprocessing**")
        st.write("Images are converted to RGB, resized to 224 × 224, converted to tensors and normalized channel-wise with mean 0.5 and standard deviation 0.5. Training additionally uses random horizontal flips and rotations up to 10 degrees.")
    with d2:
        st.markdown("**Masked multi-task learning**")
        st.write("Each task has its own validity mask. Invalid or unavailable annotations are excluded from that task's loss, helping prevent missing labels from being treated as ground truth.")
        st.markdown("**Loss functions**")
        st.write("Binary attributes use binary cross-entropy with logits; bulge prominence and roundedness use multiclass cross-entropy. The task losses are combined for joint training.")
    with st.expander("Label definitions and prediction interpretation"):
        st.markdown("""
        - **Binary attributes:** featured, edge-on, bar, spiral arms, disturbed, merger, clumpy and symmetry.
        - **Bulge prominence:** None, Just noticeable, Obvious, Dominant.
        - **Roundedness:** Completely rounded, In-between, Cigar-shaped.
        - Training labels were thresholded from annotation fractions at 0.70; ambiguous annotations were masked out.
        - Live binary predictions in this app use a 0.50 score threshold for display. That is an inference display threshold, not the 0.70 annotation-label construction threshold.
        - Scores are model outputs and may not be calibrated probabilities. The tool is a research prototype, not an astronomical classification authority.
        """)
    st.subheader("Explore GALAXAI")
    cards = [
        ("Model 1 prediction", "Custom CNN V2", "Run the frozen baseline model."),
        ("Model 2 prediction", "ResNet-18", "Run inference with the residual network."),
        ("Model 3 prediction", "MobileNetV3-Large", "Run inference with the lightweight architecture."),
        ("Three-model comparison", "Evaluation dashboard", "Compare all saved test metrics, confusion matrices and histories."),
    ]
    cols = st.columns(2)
    for i, (page_name, title, desc) in enumerate(cards):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.write(desc)
                if st.button("Open page →", key="home_" + page_name, use_container_width=True):
                    st.session_state["navigation"] = page_name
                    st.rerun()
    st.subheader("Architecture quick links")
    cols = st.columns(3)
    for i, model_name in enumerate(MODELS):
        with cols[i]:
            st.markdown(f"**{MODELS[model_name]['architecture']}**")
            st.caption(MODELS[model_name]["description"])
            page_name = f"Model {i+1} prediction"
            if st.button("View architecture →", key="arch_" + model_name, use_container_width=True):
                st.session_state["navigation"] = page_name
                st.session_state["show_architecture_" + model_name] = True
                st.rerun()
    st.subheader("Research cautions")
    st.markdown("""
    - **Class imbalance matters:** a high pooled accuracy can hide poor detection of rare features. Always inspect F1, precision, recall, PR-AUC and confusion matrices for bar, disturbed and merger.
    - **Macro task F1:** gives each task equal weight; it is not the same as pooled task accuracy.
    - **Model comparison:** results are meaningful only when models use the same held-out test records, label masks and metric definitions.
    - **Reproducibility:** this dashboard reads existing metrics and checkpoints. It does not retrain the models or modify saved experiment artifacts.
    """)
    st.info("GALAXAI is an experimental multi-task galaxy morphology research prototype. Predictions should be interpreted alongside evaluation results and label uncertainty.")


PAGES = ["Home", "Model 1 prediction", "Model 2 prediction", "Model 3 prediction", "Three-model comparison"]
if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Home"
page = st.sidebar.radio("Navigation", PAGES, key="navigation")
if page == "Home":
    overview_page()
elif page == "Model 1 prediction":
    model_prediction_page("Model 1 — Custom CNN V2")
elif page == "Model 2 prediction":
    model_prediction_page("Model 2 — ResNet-18")
elif page == "Model 3 prediction":
    model_prediction_page("Model 3 — MobileNetV3-Large")
else:
    comparison_page()

st.divider()
st.caption("GALAXAI research prototype · Inspect per-task metrics and confusion matrices, especially for rare morphology attributes.")
