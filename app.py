"""
app.py
------
Streamlit Dashboard for the Digital Image Processing (DIP) autonomous driving project.

Features 6 distinct panels:
  Panel 1: Input Frame & Quality Vector
  Panel 2: DIP Stage (Enhancement & Classical Stages)
  Panel 3: Object / Region Stage (Connected Components)
  Panel 4: AI Stage (Deep Learning Segmentation)
  Panel 5: Evaluation (Aggregate Results & Benchmarks)
  Panel 6: Failure Analysis & Qualitative Review

All paths and parameters come from configs/*.yaml.
No invented numbers; missing checkpoints or tables report "not run yet".
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.classical.lane_hough import get_hough_stage_images
from src.classical.region import analyze_regions
from src.data.degrade import apply_fog, apply_low_light
from src.enhance import ENHANCEMENT_NAMES, get_enhancement
from src.quality.metrics import QualityVector, compute_quality_vector
from src.router.rule import RuleRouter
from src.viz.panels import (
    comparison_histogram_figure,
    draw_drivable_overlay,
    draw_probability_heatmap,
    evaluation_bar_chart_figure,
    failure_panel,
    lane_stages_grid_figure,
    mask_panel,
    quality_panel,
)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config & Theme Styling
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DIP Drivable Area & Enhancement Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Page Canvas ─────────────────────────────────────────────────────────── */
    .stApp {
        background-color: #FFFFFF !important;
        color: #1B3A2A !important;
    }

    /* ── Typography (General) ────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        color: #1E7B45;
        font-family: sans-serif;
        font-weight: 700;
    }

    h2:not(.num-head) {
        border-bottom: 2px solid #A8E0BC;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* ── Numbered Section Headings ───────────────────────────────────────────── */
    /* Main page numbered section headings */
    .num-head {
        background-color: #D5F2DF !important;
        color: #14532D !important;
        padding: 10px 16px !important;
        border-radius: 10px !important;
        border-left: 5px solid #2E9E5B !important;
        border-bottom: none !important;
        width: 100% !important;
        box-sizing: border-box !important;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        display: block !important;
    }
    .num-head * {
        color: #14532D !important;
    }

    /* Sidebar numbered section headings */
    .num-head-sb {
        background-color: #A8E0BC !important;
        color: #14532D !important;
        padding: 10px 16px !important;
        border-radius: 10px !important;
        border-left: 5px solid #14532D !important;
        border-bottom: none !important;
        width: 100% !important;
        box-sizing: border-box !important;
        margin-top: 18px !important;
        margin-bottom: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        display: block !important;
    }
    .num-head-sb * {
        color: #14532D !important;
    }

    /* ── Dark Green Sidebar ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        background-color: #14532D !important;
        border-right: 1px solid #2E9E5B !important;
        color: #F0FFF4 !important;
    }

    /* Sidebar scrollbar & collapse controls */
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 8px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background-color: #2E9E5B;
        border-radius: 4px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-track {
        background-color: #14532D;
    }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] button,
    [data-testid="collapsedControl"] svg,
    button[kind="header"] {
        color: #F0FFF4 !important;
        fill: #F0FFF4 !important;
    }

    /* Sidebar text: labels, paragraphs, captions, help, spans */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {
        color: #F0FFF4 !important;
    }

    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3:not(.num-head-sb),
    [data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: #2E9E5B !important;
    }

    /* Sidebar radio and checkbox controls */
    [data-testid="stSidebar"] input[type="radio"],
    [data-testid="stSidebar"] input[type="checkbox"] {
        accent-color: #6CC78B !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #F0FFF4 !important;
    }

    /* Sidebar Inputs: selectbox, text input, slider */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #A8E0BC !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #1B3A2A !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #A8E0BC !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] input {
        color: #1B3A2A !important;
        background-color: #FFFFFF !important;
    }

    /* Dropdown popover / menu lists (dark text on white/pale green) */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    ul[data-baseweb="menu"] li {
        background-color: #FFFFFF !important;
        color: #1B3A2A !important;
    }
    ul[data-baseweb="menu"] li:hover,
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #EAF7EE !important;
        color: #14532D !important;
    }

    /* File uploader in sidebar */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #EAF7EE !important;
        border: 1px dashed #2E9E5B !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * {
        color: #1B3A2A !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #2E9E5B !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    /* Sidebar Expander */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #1C693A !important;
        border: 1px solid #2E9E5B !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #14532D !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #F0FFF4 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] code {
        background-color: #1E7B45 !important;
        color: #FFFFFF !important;
    }

    /* ── Main Page Components ────────────────────────────────────────────────── */
    .stButton > button,
    button[kind="primary"],
    button[kind="secondary"],
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        background-color: #2E9E5B !important;
        color: #FFFFFF !important;
        border: 1px solid #1E7B45 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }

    .stButton > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background-color: #1E7B45 !important;
        border-color: #1E7B45 !important;
        color: #FFFFFF !important;
    }

    input[type="checkbox"],
    input[type="radio"] {
        accent-color: #2E9E5B;
    }

    /* Tables & DataFrames */
    [data-testid="stDataFrame"],
    .stTable,
    [data-testid="stTable"] {
        border: 1px solid #A8E0BC !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] th,
    .stTable th,
    [data-testid="stTable"] th {
        background-color: #EAF7EE !important;
        color: #1E7B45 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #A8E0BC !important;
    }

    [data-testid="stDataFrame"] td,
    .stTable td,
    [data-testid="stTable"] td {
        color: #1B3A2A !important;
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #EAF7EE !important;
    }

    /* Metric Cards & Expanders on Main Page */
    [data-testid="stMetric"] {
        background-color: #EAF7EE !important;
        border: 1px solid #A8E0BC !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 6px rgba(46, 158, 91, 0.08) !important;
    }

    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetricLabel"] {
        color: #1E7B45 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] {
        color: #1B3A2A !important;
        font-weight: 700 !important;
    }

    .main [data-testid="stExpander"] {
        background-color: #EAF7EE !important;
        border: 1px solid #A8E0BC !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(46, 158, 91, 0.06) !important;
        margin-bottom: 12px !important;
    }
    .main [data-testid="stExpander"] summary {
        color: #1E7B45 !important;
        font-weight: 600 !important;
    }
    .main [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        border-top: 1px solid #A8E0BC !important;
        border-radius: 0 0 10px 10px !important;
        padding: 14px !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    }
    [data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]),
    [data-testid="stAlert"][data-baseweb="notification"]:has([aria-label="info"]) {
        background-color: #EAF7EE !important;
        color: #1B3A2A !important;
        border-left: 5px solid #2E9E5B !important;
        border-top: 1px solid #A8E0BC !important;
        border-right: 1px solid #A8E0BC !important;
        border-bottom: 1px solid #A8E0BC !important;
    }
    [data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]),
    [data-testid="stAlert"][data-baseweb="notification"]:has([aria-label="warning"]) {
        background-color: #FEF9E7 !important;
        color: #5B4300 !important;
        border-left: 5px solid #2E9E5B !important;
        border-top: 1px solid #F3E5AB !important;
        border-right: 1px solid #F3E5AB !important;
        border-bottom: 1px solid #F3E5AB !important;
    }
    [data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]),
    [data-testid="stAlert"][data-baseweb="notification"]:has([aria-label="success"]) {
        background-color: #EAF7EE !important;
        color: #1E7B45 !important;
        border-left: 5px solid #1E7B45 !important;
        border-top: 1px solid #A8E0BC !important;
        border-right: 1px solid #A8E0BC !important;
        border-bottom: 1px solid #A8E0BC !important;
    }
    [data-testid="stAlert"]:has([data-testid="stNotificationContentError"]),
    [data-testid="stAlert"][data-baseweb="notification"]:has([aria-label="error"]) {
        background-color: #FDF2F2 !important;
        color: #7A1B1B !important;
        border-left: 5px solid #E06A6A !important;
        border-top: 1px solid #F5C6CB !important;
        border-right: 1px solid #F5C6CB !important;
        border-bottom: 1px solid #F5C6CB !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(text: str, sidebar: bool = False) -> None:
    """Render a styled numbered section heading with a light green background."""
    cls_name = "num-head-sb" if sidebar else "num-head"
    tag = "h3" if sidebar else "h2"
    html = f'<{tag} class="{cls_name}">{text}</{tag}>'
    if sidebar:
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached Configuration & Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_manifest(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_csv_table(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_resource
def get_cached_router(config_path: str = "configs/router.yaml") -> RuleRouter:
    return RuleRouter(config_path=config_path)


@st.cache_resource
def get_cached_model(checkpoint_path: str, allow_random: bool = False):
    """Load PyTorch ResNet-18 U-Net segmentation model."""
    try:
        from src.models.unet import build_smp_unet
        model, meta = build_smp_unet(checkpoint_path=checkpoint_path, allow_random_init=allow_random)
        return model, meta, "loaded"
    except FileNotFoundError as e:
        return None, None, f"not run yet: {e}"
    except Exception as e:
        return None, None, f"error loading model: {e}"


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """Compute Intersection-over-Union (IoU) and Dice coefficient for drivable area."""
    p_bool = pred_mask > 0
    g_bool = gt_mask > 0
    intersection = np.logical_and(p_bool, g_bool).sum()
    union = np.logical_or(p_bool, g_bool).sum()
    iou = float(intersection / union) if union > 0 else 1.0

    p_sum = p_bool.sum()
    g_sum = g_bool.sum()
    dice = float(2.0 * intersection / (p_sum + g_sum)) if (p_sum + g_sum) > 0 else 1.0
    return iou, dice


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Controls & Parameters
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Controls & Settings")

data_cfg = load_yaml_config("configs/data.yaml")
router_cfg = load_yaml_config("configs/router.yaml")
enhance_cfg = load_yaml_config("configs/enhance.yaml")
unet_cfg = load_yaml_config("configs/unet.yaml")

target_w = data_cfg.get("image_width", 640)
target_h = data_cfg.get("image_height", 352)
seed = data_cfg.get("seed", 42)

# Frame source
section_header("1. Frame Selection", sidebar=True)
frame_source = st.sidebar.radio("Frame Source:", ["Test Split Manifest", "Upload Image"], index=0)

selected_bgr: Optional[np.ndarray] = None
selected_gt_mask: Optional[np.ndarray] = None
selected_frame_id: str = "sample"

manifest_path = Path("data/manifests/test.csv")
test_df = load_manifest(str(manifest_path))

if frame_source == "Test Split Manifest":
    if test_df is not None and not test_df.empty:
        conditions = ["all"] + sorted(list(test_df["condition"].dropna().unique()))
        sel_cond = st.sidebar.selectbox("Filter Condition:", conditions, index=0)
        filtered_df = test_df if sel_cond == "all" else test_df[test_df["condition"] == sel_cond]

        options = filtered_df["image_id"].tolist()
        selected_frame_id = st.sidebar.selectbox("Select Image ID:", options, index=0)
        row = filtered_df[filtered_df["image_id"] == selected_frame_id].iloc[0]

        img_p = Path(row["img_path"])
        mask_p = Path(row["mask_path"]) if "mask_path" in row and pd.notna(row["mask_path"]) else None

        if img_p.exists():
            selected_bgr = cv2.imread(str(img_p))
        else:
            st.sidebar.error(f"Image not found at {img_p}")

        if mask_p and mask_p.exists():
            selected_gt_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    else:
        st.sidebar.warning("data/manifests/test.csv not found or empty.")
else:
    uploaded = st.sidebar.file_uploader("Upload Image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        selected_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        selected_frame_id = uploaded.name

# Fallback dummy frame if none loaded
if selected_bgr is None:
    selected_bgr = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    selected_gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    selected_gt_mask[200:350, 150:490] = 255
    st.sidebar.info("Using standard placeholder synthetic frame.")

# Ensure target dimensions
if selected_bgr.shape[:2] != (target_h, target_w):
    selected_bgr = cv2.resize(selected_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
if selected_gt_mask is not None and selected_gt_mask.shape[:2] != (target_h, target_w):
    selected_gt_mask = cv2.resize(selected_gt_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

# Synthetic Degradation
section_header("2. Synthetic Degradation", sidebar=True)
degradation_choice = st.sidebar.selectbox(
    "Apply Degradation:",
    ["none", "fog mild (beta=1.0)", "fog dense (beta=2.5)", "low light"],
    index=0,
)

processed_input_bgr = selected_bgr.copy()
if degradation_choice == "fog mild (beta=1.0)":
    processed_input_bgr = apply_fog(selected_bgr, beta=1.0, A=0.9, seed=seed)
elif degradation_choice == "fog dense (beta=2.5)":
    processed_input_bgr = apply_fog(selected_bgr, beta=2.5, A=0.9, seed=seed)
elif degradation_choice == "low light":
    processed_input_bgr = apply_low_light(selected_bgr, intensity_scale=0.25, gamma=1.8, seed=seed)

# Enhancement Strategy
section_header("3. Enhancement Selection", sidebar=True)
enhancement_options = ["adaptive (router)"] + list(ENHANCEMENT_NAMES)
enhancement_choice = st.sidebar.selectbox("Enhancement Method:", enhancement_options, index=0)

# Active Configuration Parameters Expander
with st.sidebar.expander("⚙️ Active Parameters", expanded=False):
    st.markdown(f"**Target Resolution**: `{target_w} × {target_h}`")
    st.markdown(f"**Random Seed**: `{seed}`")
    st.markdown("**Router Rules (configs/router.yaml)**:")
    st.json(router_cfg.get("rules", []))
    st.markdown(f"**Default Router Action**: `{router_cfg.get('default_action', 'none')}`")
    st.markdown("**Enhancement Configs (configs/enhance.yaml)**:")
    st.json(enhance_cfg)
    st.markdown(f"**Checkpoint Dir**: `{unet_cfg.get('checkpoint_dir', 'results/checkpoints/unet')}`")
    st.caption("ℹ️ Note: No lane F1 evaluated (BDD100K mirror drivable area only; IoU and Dice are used).")


# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Autonomous Driving DIP & Segmentation Dashboard")
st.markdown(
    "Interactive inspection system for Digital Image Processing (DIP) enhancement, "
    "classical image processing, and deep learning drivable area segmentation."
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1: INPUT
# ─────────────────────────────────────────────────────────────────────────────

section_header("1. Input Frame & Quality Metrics")
p1_col1, p1_col2 = st.columns([1, 1])

with p1_col1:
    st.image(
        cv2.cvtColor(processed_input_bgr, cv2.COLOR_BGR2RGB),
        caption=f"Input Frame: {selected_frame_id} (Degradation: {degradation_choice})",
        width="stretch",
    )
    h_in, w_in, c_in = processed_input_bgr.shape
    dtype_in = str(processed_input_bgr.dtype)
    bit_depth = "8-bit" if dtype_in == "uint8" else dtype_in

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Width", f"{w_in} px")
    m2.metric("Height", f"{h_in} px")
    m3.metric("Channels", f"{c_in} (BGR)")
    m4.metric("Bit Depth", bit_depth)

with p1_col2:
    st.subheader("5-Value Quality Vector")
    qv = compute_quality_vector(processed_input_bgr)
    qv_dict = qv.to_dict()

    q_col1, q_col2, q_col3 = st.columns(3)
    q_col1.metric("Brightness (L)", f"{qv.brightness:.2f}")
    q_col2.metric("Contrast (Std L)", f"{qv.contrast:.2f}")
    q_col3.metric("Sharpness (Lap)", f"{qv.sharpness:.2f}")

    q_col4, q_col5, _ = st.columns(3)
    q_col4.metric("Noise (Immerkaer)", f"{qv.noise:.2f}")
    q_col5.metric("Haze (Dark Ch.)", f"{qv.haze:.3f}")

    fig_qv = quality_panel(processed_input_bgr, qv_dict)
    st.pyplot(fig_qv)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2: DIP STAGE (Enhancement & Classical Hough)
# ─────────────────────────────────────────────────────────────────────────────

section_header("2. DIP Enhancement & Classical Stages")

# Determine active enhancement
router = get_cached_router()
if enhancement_choice == "adaptive (router)":
    active_enhancement, rule_reason = router.route_with_reason(qv)
    st.info(f"🧠 **Adaptive Router Selected**: `{active_enhancement}` — *{rule_reason}*")
else:
    active_enhancement = enhancement_choice
    rule_reason = f"Manually selected: {enhancement_choice}"
    st.info(f"⚙️ **Manual Enhancement Selected**: `{active_enhancement}`")

enhancement_fn = get_enhancement(active_enhancement)
enhanced_bgr = enhancement_fn(processed_input_bgr)

p2_col1, p2_col2 = st.columns(2)
with p2_col1:
    st.image(
        cv2.cvtColor(processed_input_bgr, cv2.COLOR_BGR2RGB),
        caption="Original / Degraded Input",
        width="stretch",
    )
with p2_col2:
    st.image(
        cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB),
        caption=f"Enhanced Frame: {active_enhancement}",
        width="stretch",
    )

# Histograms
st.subheader("Luminance Histograms (Before vs. After)")
fig_hist = comparison_histogram_figure(processed_input_bgr, enhanced_bgr, name=active_enhancement.upper())
st.pyplot(fig_hist)

# Intermediate Hough stages
with st.expander("🔍 Intermediate Classical Lane Detection Stages (5 Steps)", expanded=True):
    hough_stages = get_hough_stage_images(enhanced_bgr, use_birds_eye=False)
    fig_stages = lane_stages_grid_figure(hough_stages)
    st.pyplot(fig_stages)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3: OBJECT / REGION STAGE
# ─────────────────────────────────────────────────────────────────────────────

section_header("3. Object & Region Stage (Connected Components)")

# Use ground truth drivable mask if available, else derive from threshold/classical cleaned mask
if selected_gt_mask is not None and np.any(selected_gt_mask > 0):
    region_source_mask = selected_gt_mask
    mask_source_name = "Ground Truth Drivable Mask"
else:
    region_source_mask = hough_stages.get("08_cleaned_mask", np.zeros((target_h, target_w), dtype=np.uint8))
    mask_source_name = "Classical Extracted Mask"

stats_dict, color_labels, annotated_overlay = analyze_regions(region_source_mask, bgr_frame=enhanced_bgr)

p3_col1, p3_col2 = st.columns(2)
with p3_col1:
    st.image(
        color_labels,
        caption=f"Connected Components Label Map ({mask_source_name})",
        width="stretch",
    )
with p3_col2:
    st.image(
        cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB),
        caption="Extracted Regions with Bounding Boxes (Cyan) & Centroids (Red)",
        width="stretch",
    )

st.metric("Total Segments Detected", stats_dict.get("segment_count", 0))

regions_list = stats_dict.get("regions", [])
if regions_list:
    df_regions = pd.DataFrame(regions_list)
    st.dataframe(df_regions, width="stretch")
else:
    st.write("No connected regions detected.")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4: AI STAGE (Segmentation Model)
# ─────────────────────────────────────────────────────────────────────────────

section_header("4. AI Stage: Deep Learning Drivable Area Segmentation")

ckpt_path = unet_cfg.get("checkpoint_dir", "results/checkpoints/unet") + "/unet_drivable_best.pt"
allow_mock = st.checkbox("Enable Mock / Random-Init Model if Checkpoint Missing", value=False)

model, meta, load_status = get_cached_model(checkpoint_path=ckpt_path, allow_random=allow_mock)

if model is None:
    st.warning(
        f"⚠️ Checkpoint not found at `{ckpt_path}`.\n\n"
        "**Status**: `not run yet`.\n\n"
        "Please train the model on Google Colab using `colab_bundle.zip` and download the checkpoint to "
        "`results/checkpoints/unet/unet_drivable_best.pt`. (Check 'Enable Mock Model' above to preview UI)."
    )
else:
    from src.models.unet import predict_drivable_unet

    # Run predictions on unenhanced vs enhanced
    prob_unenh, mask_unenh = predict_drivable_unet(model, processed_input_bgr, meta=meta, device="cpu")
    prob_enh, mask_enh = predict_drivable_unet(model, enhanced_bgr, meta=meta, device="cpu")

    ai_c1, ai_c2 = st.columns(2)
    with ai_c1:
        st.subheader("Without Enhancement")
        heatmap_unenh = draw_probability_heatmap(prob_unenh)
        overlay_unenh = draw_drivable_overlay(processed_input_bgr, mask_unenh, color=(0, 255, 0), alpha=0.45)
        st.image(cv2.cvtColor(heatmap_unenh, cv2.COLOR_BGR2RGB), caption="Probability Heatmap (Raw Input)", width="stretch")
        st.image(cv2.cvtColor(overlay_unenh, cv2.COLOR_BGR2RGB), caption="Segmentation Overlay (Raw Input)", width="stretch")

        if selected_gt_mask is not None:
            iou_raw, dice_raw = compute_binary_metrics(mask_unenh, selected_gt_mask)
            r_c1, r_c2 = st.columns(2)
            r_c1.metric("Raw Frame IoU", f"{iou_raw:.4f}")
            r_c2.metric("Raw Frame Dice", f"{dice_raw:.4f}")

    with ai_c2:
        st.subheader(f"With Enhancement ({active_enhancement})")
        heatmap_enh = draw_probability_heatmap(prob_enh)
        overlay_enh = draw_drivable_overlay(enhanced_bgr, mask_enh, color=(0, 255, 0), alpha=0.45)
        st.image(cv2.cvtColor(heatmap_enh, cv2.COLOR_BGR2RGB), caption=f"Probability Heatmap ({active_enhancement})", width="stretch")
        st.image(cv2.cvtColor(overlay_enh, cv2.COLOR_BGR2RGB), caption=f"Segmentation Overlay ({active_enhancement})", width="stretch")

        if selected_gt_mask is not None:
            iou_enh, dice_enh = compute_binary_metrics(mask_enh, selected_gt_mask)
            e_c1, e_c2 = st.columns(2)
            e_c1.metric("Enhanced Frame IoU", f"{iou_enh:.4f}")
            e_c2.metric("Enhanced Frame Dice", f"{dice_enh:.4f}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 5: EVALUATION (Aggregate Results & Benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

section_header("5. Evaluation & Benchmark Summary")

# Evaluation tables
col_eval1, col_eval2 = st.columns(2)

with col_eval1:
    st.subheader("Synthetic Degradation PSNR & SSIM")
    synth_summary_path = "results/tables/summary_synthetic_psnr_ssim.csv"
    synth_df = load_csv_table(synth_summary_path)
    if synth_df is not None:
        st.dataframe(synth_df, width="stretch")
    else:
        st.caption("not run yet: run scripts/eval_synthetic_restoration.py")

with col_eval2:
    st.subheader("Condition-Stratified Test Performance (mIoU / Dice)")
    test_eval_summary_path = "results/tables/summary_test_metrics.csv"
    eval_df = load_csv_table(test_eval_summary_path)
    if eval_df is not None:
        st.dataframe(eval_df, width="stretch")
        if "condition" in eval_df.columns and "miou" in eval_df.columns:
            fig_bar = evaluation_bar_chart_figure(eval_df, group_col="condition", metric_col="miou")
            st.pyplot(fig_bar)
    else:
        st.caption("not run yet: run scripts/evaluate.py on test split with trained model.")

# Benchmarks & Router Accuracy
col_eval3, col_eval4 = st.columns(2)

with col_eval3:
    st.subheader("Runtime Per Stage Benchmark (ms / FPS)")
    runtime_path = "results/tables/runtime_benchmarks.csv"
    runtime_df = load_csv_table(runtime_path)
    if runtime_df is not None:
        st.dataframe(runtime_df, width="stretch")
    else:
        st.caption("not run yet: run scripts/benchmark_runtime.py")

with col_eval4:
    st.subheader("Router Accuracy vs. Oracle")
    router_acc_path = "results/tables/router_vs_oracle.csv"
    router_acc_df = load_csv_table(router_acc_path)
    if router_acc_df is not None:
        st.dataframe(router_acc_df, width="stretch")
    else:
        st.caption("not run yet: run scripts/evaluate_router.py")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 6: FAILURE ANALYSIS & QUALITATIVE REVIEW
# ─────────────────────────────────────────────────────────────────────────────

section_header("6. Failure Analysis & Qualitative Review")

# Test per-frame predictions table
test_preds_path = "results/tables/test_predictions.csv"
test_preds_df = load_csv_table(test_preds_path)

if test_preds_df is not None and not test_preds_df.empty:
    st.subheader("Test Split Case Inspection")
    c_cond, c_sort = st.columns(2)
    cond_opts = ["all"] + sorted(list(test_preds_df["condition"].dropna().unique()))
    sel_f_cond = c_cond.selectbox("Select Condition:", cond_opts, key="fail_cond")
    sel_sort = c_sort.selectbox("Sort Order:", ["Lowest IoU (Failures)", "Highest IoU (Successes)"], key="fail_sort")

    f_df = test_preds_df if sel_f_cond == "all" else test_preds_df[test_preds_df["condition"] == sel_f_cond]
    ascending = (sel_sort == "Lowest IoU (Failures)")
    f_df = f_df.sort_values(by="iou", ascending=ascending)

    st.dataframe(f_df.head(20), width="stretch")

    # Thumbnail gallery of worst 20
    records = f_df.to_dict(orient="records")
    st.subheader("Worst 20 Frames Gallery")
    fig_fail = failure_panel(records, k=min(20, len(records)))
    st.pyplot(fig_fail)
else:
    st.caption("Test per-frame predictions: not run yet (run scripts/evaluate.py).")

    # Current frame inspection
    if selected_gt_mask is not None and selected_bgr is not None:
        st.subheader(f"Current Inspection: {selected_frame_id}")
        pred_dummy = (cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY) > 100).astype(np.uint8) * 255
        fig_mask = mask_panel(enhanced_bgr, selected_gt_mask, pred_dummy, title=f"Frame: {selected_frame_id}")
        st.pyplot(fig_mask)

# Qualitative Notes
st.subheader("📝 Qualitative Edge-Case Notes")
notes_path = Path("results/failure_notes.json")
existing_notes: list[dict] = []
if notes_path.exists():
    try:
        with open(notes_path, "r", encoding="utf-8") as f:
            existing_notes = json.load(f)
    except Exception:
        existing_notes = []

with st.form("notes_form"):
    note_text = st.text_area(f"Add observation / failure reason for '{selected_frame_id}':")
    submitted = st.form_submit_button("Save Note to results/failure_notes.json")

    if submitted and note_text.strip():
        new_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_id": selected_frame_id,
            "condition": degradation_choice,
            "enhancement": active_enhancement,
            "note": note_text.strip(),
        }
        existing_notes.append(new_entry)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(existing_notes, f, indent=2)
        st.success(f"Saved note for {selected_frame_id}!")

if existing_notes:
    with st.expander(f"Saved Notes ({len(existing_notes)})", expanded=False):
        for item in reversed(existing_notes):
            st.markdown(
                f"- **{item.get('image_id')}** ({item.get('timestamp')[:19]}) | *{item.get('condition')} / {item.get('enhancement')}*:\n"
                f"  > {item.get('note')}"
            )
