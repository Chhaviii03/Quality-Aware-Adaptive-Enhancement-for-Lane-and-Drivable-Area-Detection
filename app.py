"""
app.py
------
Streamlit Dashboard for the Digital Image Processing (DIP) autonomous driving project.
Research Platform: DRIVING PERCEPTION LAB

Workstation Layout Architecture:
  1. Top Header: Typographic Brand & Subtitle
  2. Control Console: Structured 3-Group Workspace Panel
     - Group A: Input Configuration (Manifest / Upload & File Info)
     - Group B: Degradation Configuration (Atmospheric Simulation)
     - Group C: Enhancement Configuration (Adaptive Router & Algorithms)
  3. Horizontal Navigation Bar (01 Overview | 02 Quality | 03 Enhancement | 04 Classical DIP | 05 Deep Learning | 06 Evaluation)
  4. Main Workspace: Content of the Selected Dashboard Only

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
    page_title="Driving Perception Lab — DIP & Road Segmentation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* ── Global Canvas & Reset ────────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp {
        background-color: #F7F7F2 !important;
        color: #343A38 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* ── Typography (General) ────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
        color: #244638 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }

    /* ── Top Brand Wordmark ───────────────────────────────────────────────────── */
    .brand-container {
        padding: 4px 0 14px 0;
        margin-bottom: 8px;
        border-bottom: 1px solid #DCE1DC;
    }
    .brand-title {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        color: #173D2D !important;
        text-transform: uppercase !important;
        line-height: 1.2 !important;
    }
    .brand-subtitle {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.12em !important;
        color: #77837C !important;
        text-transform: uppercase !important;
        margin-top: 3px !important;
    }

    /* ── Workstation Control Console ─────────────────────────────────────────── */
    .console-container {
        background-color: #FFFFFF;
        border: 1px solid #DCE1DC;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .console-group-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #527D62;
        text-transform: uppercase;
        margin-bottom: 8px;
        border-bottom: 1px solid #F0F2F0;
        padding-bottom: 4px;
    }
    .console-info-row {
        font-size: 11.5px;
        color: #77837C;
        margin-top: 6px;
        line-height: 1.4;
    }
    .console-info-val {
        font-family: 'IBM Plex Mono', monospace;
        color: #173D2D;
        font-weight: 600;
    }

    /* ── Horizontal Navigation Bar (Tabs) ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent !important;
        border-bottom: 1px solid #DCE1DC !important;
        padding-bottom: 0px;
        margin-top: 4px;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        color: #77837C !important;
        padding: 9px 18px !important;
        background-color: transparent !important;
        border-radius: 4px 4px 0 0 !important;
        border: 1px solid transparent !important;
        border-bottom: none !important;
        transition: all 0.15s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #244638 !important;
        background-color: #EAEFEA !important;
    }
    .stTabs [aria-selected="true"] {
        color: #173D2D !important;
        background-color: #DDE9DF !important;
        border: 1px solid #DCE1DC !important;
        border-bottom: 2px solid #173D2D !important;
    }

    /* ── Section Headings in Workspaces ───────────────────────────────────────── */
    .section-header-block {
        margin: 10px 0 16px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #DCE1DC;
    }
    .section-badge {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        color: #173D2D !important;
        text-transform: uppercase !important;
        margin-bottom: 3px !important;
    }
    .section-desc {
        font-size: 13.5px !important;
        color: #77837C !important;
        margin: 0 !important;
    }

    /* ── Status Bar in Pipeline Overview ─────────────────────────────────────── */
    .status-bar-box {
        background-color: #FFFFFF;
        border: 1px solid #DCE1DC;
        border-left: 4px solid #527D62;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .status-bar-item {
        font-size: 13px;
        color: #343A38;
    }
    .status-bar-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #77837C;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .status-bar-val {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: #173D2D;
    }

    /* ── Analytical Cards & Data Tables ───────────────────────────────────────── */
    .research-card {
        background-color: #FFFFFF;
        border: 1px solid #DCE1DC;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .research-card-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #527D62;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #F0F2F0;
        font-size: 13px;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    .metric-row-label {
        color: #77837C;
    }
    .metric-row-val {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: #244638;
    }

    /* Metric Display Blocks */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE1DC !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 11.5px !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        color: #77837C !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] div {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 19px !important;
        font-weight: 600 !important;
        color: #244638 !important;
    }

    /* Tables & DataFrames */
    [data-testid="stDataFrame"],
    .stTable,
    [data-testid="stTable"] {
        border: 1px solid #DCE1DC !important;
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stDataFrame"] th,
    .stTable th,
    [data-testid="stTable"] th {
        background-color: #EAEFEA !important;
        color: #244638 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #DCE1DC !important;
    }
    [data-testid="stDataFrame"] td,
    .stTable td,
    [data-testid="stTable"] td {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12.5px !important;
        color: #343A38 !important;
    }

    /* Buttons */
    .stButton > button,
    button[kind="primary"],
    button[kind="secondary"] {
        background-color: #527D62 !important;
        color: #FFFFFF !important;
        border: 1px solid #244638 !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #244638 !important;
        color: #FFFFFF !important;
    }

    /* Inputs & Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE1DC !important;
        border-radius: 4px !important;
    }
    div[data-baseweb="select"] * {
        color: #343A38 !important;
    }
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE1DC !important;
        border-radius: 4px !important;
    }
    div[data-baseweb="input"] input {
        color: #343A38 !important;
    }

    /* Sidebar fallback (if opened) */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        background-color: #173D2D !important;
        border-right: 1px solid #244638 !important;
        color: #F0FFF4 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #DDE9DF !important;
        font-size: 12.5px !important;
    }

    /* Top header bar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {
        color: #244638 !important;
        fill: #244638 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_brand_header() -> None:
    """Render top typographic wordmark branding."""
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-title">DRIVING PERCEPTION LAB</div>
            <div class="brand-subtitle">IMAGE PROCESSING · COMPUTER VISION · ROAD SEGMENTATION RESEARCH WORKSTATION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(badge: str, description: str) -> None:
    """Render a clean, restrained scientific section heading."""
    st.markdown(
        f"""
        <div class="section-header-block">
            <div class="section-badge">{badge}</div>
            <p class="section-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
# Load Configurations & Target Dimensions
# ─────────────────────────────────────────────────────────────────────────────

data_cfg = load_yaml_config("configs/data.yaml")
router_cfg = load_yaml_config("configs/router.yaml")
enhance_cfg = load_yaml_config("configs/enhance.yaml")
unet_cfg = load_yaml_config("configs/unet.yaml")

target_w = data_cfg.get("image_width", 640)
target_h = data_cfg.get("image_height", 352)
seed = data_cfg.get("seed", 42)

manifest_path = Path("data/manifests/test.csv")
test_df = load_manifest(str(manifest_path))


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOP BRAND HEADER
# ─────────────────────────────────────────────────────────────────────────────

render_brand_header()


# ─────────────────────────────────────────────────────────────────────────────
# 2. WORKSTATION CONTROL CONSOLE (Collapsible Horizontal Drawer)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("WORKSTATION CONTROL CONSOLE · INPUT, DEGRADATION & ENHANCEMENT", expanded=True):
    col_in, col_deg, col_enh = st.columns([5.2, 3.4, 3.4])

    # ── Group A: Input Configuration ─────────────────────────────────────────
    with col_in:
        st.markdown('<div class="console-group-header">A. INPUT CONFIGURATION</div>', unsafe_allow_html=True)
        frame_source = st.radio(
            "Frame Source:",
            ["Test Split Manifest", "Upload Image"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="ctrl_frame_source",
        )

        selected_bgr: Optional[np.ndarray] = None
        selected_gt_mask: Optional[np.ndarray] = None
        selected_frame_id: str = "sample"

        orig_w: Optional[int] = None
        orig_h: Optional[int] = None
        orig_c: Optional[int] = None
        orig_bit_depth: str = "Unavailable"

        if frame_source == "Test Split Manifest":
            if test_df is not None and not test_df.empty:
                c_sub1, c_sub2 = st.columns(2)
                conditions = ["all"] + sorted(list(test_df["condition"].dropna().unique()))
                sel_cond = c_sub1.selectbox("Filter Condition:", conditions, index=0, key="ctrl_sel_cond")
                filtered_df = test_df if sel_cond == "all" else test_df[test_df["condition"] == sel_cond]

                options = filtered_df["image_id"].tolist()
                selected_frame_id = c_sub2.selectbox("Select Image ID:", options, index=0, key="ctrl_sel_img")
                row = filtered_df[filtered_df["image_id"] == selected_frame_id].iloc[0]

                img_p = Path(row["img_path"])
                mask_p = Path(row["mask_path"]) if "mask_path" in row and pd.notna(row["mask_path"]) else None

                if img_p.exists():
                    raw_disk_bgr = cv2.imread(str(img_p))
                    if raw_disk_bgr is not None:
                        orig_h, orig_w, orig_c = raw_disk_bgr.shape
                        orig_bit_depth = "8-bit per channel" if raw_disk_bgr.dtype == np.uint8 else str(raw_disk_bgr.dtype)
                        selected_bgr = raw_disk_bgr
                else:
                    st.error(f"Image not found at {img_p}")

                if mask_p and mask_p.exists():
                    selected_gt_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            else:
                st.warning("data/manifests/test.csv not found or empty.")
        else:
            uploaded = st.file_uploader(
                "Upload Driving Frame (JPG/PNG):",
                type=["jpg", "jpeg", "png"],
                key="ctrl_uploader",
            )
            if uploaded is not None:
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                raw_upload_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if raw_upload_bgr is not None:
                    orig_h, orig_w, orig_c = raw_upload_bgr.shape
                    orig_bit_depth = "8-bit per channel" if raw_upload_bgr.dtype == np.uint8 else str(raw_upload_bgr.dtype)
                    selected_bgr = raw_upload_bgr
                    selected_frame_id = uploaded.name

        # Fallback dummy frame if none loaded
        if selected_bgr is None:
            selected_bgr = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
            selected_gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            selected_gt_mask[200:350, 150:490] = 255
            orig_w, orig_h, orig_c = target_w, target_h, 3
            orig_bit_depth = "8-bit (Synthetic Placeholder)"
            st.info("Using standard placeholder synthetic frame.")

        # Ensure model target dimensions
        if selected_bgr.shape[:2] != (target_h, target_w):
            selected_bgr = cv2.resize(selected_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if selected_gt_mask is not None and selected_gt_mask.shape[:2] != (target_h, target_w):
            selected_gt_mask = cv2.resize(selected_gt_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # File metadata summary row
        orig_w_str = f"{orig_w} px" if orig_w is not None else "Unavailable"
        orig_h_str = f"{orig_h} px" if orig_h is not None else "Unavailable"
        st.markdown(
            f"""
            <div class="console-info-row">
                Selected: <span class="console-info-val">{selected_frame_id}</span> ·
                Raw Res: <span class="console-info-val">{orig_w_str} × {orig_h_str}</span> ·
                Bit: <span class="console-info-val">{orig_bit_depth}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Group B: Degradation Configuration ───────────────────────────────────
    with col_deg:
        st.markdown('<div class="console-group-header">B. DEGRADATION CONFIGURATION</div>', unsafe_allow_html=True)
        degradation_choice = st.selectbox(
            "Apply Synthetic Degradation:",
            ["none", "fog mild (beta=1.0)", "fog dense (beta=2.5)", "low light"],
            index=0,
            key="ctrl_degrade",
        )

        processed_input_bgr = selected_bgr.copy()
        if degradation_choice == "fog mild (beta=1.0)":
            processed_input_bgr = apply_fog(selected_bgr, beta=1.0, A=0.9, seed=seed)
            deg_desc = "Koschmieder atmospheric model (β=1.0, A=0.9)"
        elif degradation_choice == "fog dense (beta=2.5)":
            processed_input_bgr = apply_fog(selected_bgr, beta=2.5, A=0.9, seed=seed)
            deg_desc = "Heavy atmospheric scattering (β=2.5, A=0.9)"
        elif degradation_choice == "low light":
            processed_input_bgr = apply_low_light(selected_bgr, intensity_scale=0.25, gamma=1.8, seed=seed)
            deg_desc = "Non-linear illumination attenuation (γ=1.8, scale=0.25)"
        else:
            deg_desc = "Standard baseline (zero synthetic degradation)"

        st.markdown(
            f"""
            <div class="console-info-row">
                Atmosphere: <span class="console-info-val">{degradation_choice.upper()}</span><br>
                Model: <i>{deg_desc}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Group C: Enhancement Configuration ───────────────────────────────────
    with col_enh:
        st.markdown('<div class="console-group-header">C. ENHANCEMENT CONFIGURATION</div>', unsafe_allow_html=True)
        enhancement_options = ["adaptive (router)"] + list(ENHANCEMENT_NAMES)
        enhancement_choice = st.selectbox(
            "Enhancement Algorithm:",
            enhancement_options,
            index=0,
            key="ctrl_enhance",
        )

        # Compute quality vector & determine active enhancement once
        qv = compute_quality_vector(processed_input_bgr)
        qv_dict = qv.to_dict()

        router = get_cached_router()
        if enhancement_choice == "adaptive (router)":
            active_enhancement, rule_reason = router.route_with_reason(qv)
            router_mode_text = "Adaptive Rule-Based Router"
        else:
            active_enhancement = enhancement_choice
            rule_reason = f"Manual override: {enhancement_choice}"
            router_mode_text = "Manual Override"

        enhancement_fn = get_enhancement(active_enhancement)
        enhanced_bgr = enhancement_fn(processed_input_bgr)

        st.markdown(
            f"""
            <div class="console-info-row">
                Active: <span class="console-info-val">{active_enhancement.upper()}</span><br>
                Trigger: <i>{rule_reason}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sub-expander for technical specifications & JSON parameters
    with st.expander("Technical Parameters & Router Rules (JSON)", expanded=False):
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown(f"**Target Inference Resolution**: `{target_w} × {target_h}`")
            st.markdown(f"**Random Seed**: `{seed}`")
            st.markdown(f"**Default Router Action**: `{router_cfg.get('default_action', 'none')}`")
            st.markdown(f"**Checkpoint Dir**: `{unet_cfg.get('checkpoint_dir', 'results/checkpoints/unet')}`")
            st.caption("Evaluation metric note: BDD100K drivable area uses mIoU and Dice (no lane F1 evaluated).")
        with p_c2:
            st.markdown("**Router Rules (configs/router.yaml)**:")
            st.json(router_cfg.get("rules", []))
            st.markdown("**Enhancement Configs (configs/enhance.yaml)**:")
            st.json(enhance_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Cached Downstream Stage Computations (Run once, shared across all tabs)
# ─────────────────────────────────────────────────────────────────────────────

hough_stages = get_hough_stage_images(enhanced_bgr, use_birds_eye=False)

if selected_gt_mask is not None and np.any(selected_gt_mask > 0):
    region_source_mask = selected_gt_mask
    mask_source_name = "Ground Truth Drivable Mask"
else:
    region_source_mask = hough_stages.get("08_cleaned_mask", np.zeros((target_h, target_w), dtype=np.uint8))
    mask_source_name = "Classical Extracted Mask"

stats_dict, color_labels, annotated_overlay = analyze_regions(region_source_mask, bgr_frame=enhanced_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HORIZONTAL NAVIGATION BAR & WORKSPACES
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "01 Overview",
    "02 Quality",
    "03 Enhancement",
    "04 Classical DIP",
    "05 Deep Learning",
    "06 Evaluation",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: 01 OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    render_section_header(
        "01 / PIPELINE OVERVIEW",
        "End-to-end analysis of driving image quality, adaptive enhancement, and road perception.",
    )

    # Top Status Bar: Router Decision & Fired Rule
    st.markdown(
        f"""
        <div class="status-bar-box">
            <div class="status-bar-item">
                <span class="status-bar-label">ROUTER MODE:</span>
                <span class="status-bar-val"> {router_mode_text}</span>
            </div>
            <div class="status-bar-item">
                <span class="status-bar-label">SELECTED STRATEGY:</span>
                <span class="status-bar-val"> {active_enhancement.upper()}</span>
            </div>
            <div class="status-bar-item">
                <span class="status-bar-label">TRIGGER LOGIC:</span>
                <span class="status-bar-val"> {rule_reason}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Editorial Split Layout: Image on Left (60%), Quality & Metadata on Right (40%)
    col_img, col_metrics = st.columns([6, 4])

    with col_img:
        st.image(
            cv2.cvtColor(processed_input_bgr, cv2.COLOR_BGR2RGB),
            caption=f"Input Frame: {selected_frame_id} (Degradation: {degradation_choice})",
            width="stretch",
        )

    with col_metrics:
        # Quality Vector (Aligned rows)
        st.markdown(
            f"""
            <div class="research-card">
                <div class="research-card-title">
                    <span>5-Value Quality Vector</span>
                    <span>FRAME METRICS</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Brightness (Luminance Mean)</span>
                    <span class="metric-row-val">{qv.brightness:.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Contrast (Luminance Std Dev)</span>
                    <span class="metric-row-val">{qv.contrast:.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Sharpness (Laplacian Variance)</span>
                    <span class="metric-row-val">{qv.sharpness:.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Noise (Immerkaer Estimation)</span>
                    <span class="metric-row-val">{qv.noise:.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Haze (Dark Channel Density)</span>
                    <span class="metric-row-val">{qv.haze:.3f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Resolution Audit
        orig_w_str = f"{orig_w} px" if orig_w is not None else "Unavailable"
        orig_h_str = f"{orig_h} px" if orig_h is not None else "Unavailable"
        orig_c_str = f"{orig_c} (BGR)" if orig_c is not None else "Unavailable"

        st.markdown(
            f"""
            <div class="research-card">
                <div class="research-card-title">
                    <span>Resolution Audit</span>
                    <span>STANDARDS CHECK</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Original Uploaded Dimensions</span>
                    <span class="metric-row-val">{orig_w_str} × {orig_h_str}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Original Channels & Bit Depth</span>
                    <span class="metric-row-val">{orig_c_str} · {orig_bit_depth}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Model Standard Input Resolution</span>
                    <span class="metric-row-val">{target_w} × {target_h} px</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Model Input Channels & Precision</span>
                    <span class="metric-row-val">3 (BGR) · 8-bit uint8</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Downstream Stage Previews (3-Column Layout)
    st.markdown('<div class="console-group-header" style="margin-top: 18px;">DOWNSTREAM PIPELINE STAGE PREVIEWS</div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.markdown(
            f"""
            <div class="research-card" style="padding: 10px; margin-bottom: 8px;">
                <div class="research-card-title">
                    <span>DIP Enhancement</span>
                    <span>{active_enhancement.upper()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(
            cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB),
            caption=f"Restored via {active_enhancement}",
            width="stretch",
        )

    with p_col2:
        seg_count = stats_dict.get("segment_count", 0)
        st.markdown(
            f"""
            <div class="research-card" style="padding: 10px; margin-bottom: 8px;">
                <div class="research-card-title">
                    <span>Classical Lane Stage</span>
                    <span>{seg_count} SEGMENTS</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(
            cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB),
            caption=f"Hough & Morphology ({mask_source_name})",
            width="stretch",
        )

    with p_col3:
        ckpt_p = unet_cfg.get("checkpoint_dir", "results/checkpoints/unet") + "/unet_drivable_best.pt"
        ckpt_exists = Path(ckpt_p).exists()
        ckpt_badge = "WEIGHTS FOUND" if ckpt_exists else "WEIGHTS MISSING"
        st.markdown(
            f"""
            <div class="research-card" style="padding: 10px; margin-bottom: 8px;">
                <div class="research-card-title">
                    <span>Deep Learning Drivable Area</span>
                    <span>{ckpt_badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if selected_gt_mask is not None:
            st.image(
                selected_gt_mask,
                caption="Ground Truth Drivable Area Mask",
                width="stretch",
            )
        else:
            st.image(
                color_labels,
                caption="Connected Components Segment Map",
                width="stretch",
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: 02 QUALITY
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    render_section_header(
        "02 / QUALITY ANALYSIS",
        "Quantitative assessment of image degradation, distribution statistics, and perceptual clarity.",
    )

    # Resolution Audit Card: Original Image vs Model Input Specifications
    orig_w_str = f"{orig_w} px" if orig_w is not None else "Unavailable"
    orig_h_str = f"{orig_h} px" if orig_h is not None else "Unavailable"
    orig_c_str = f"{orig_c} (BGR)" if orig_c is not None else "Unavailable"

    meta_c1, meta_c2 = st.columns(2)
    with meta_c1:
        st.markdown(
            f"""
            <div class="research-card">
                <div class="research-card-title">Group A: Original Image Properties</div>
                <div class="metric-row">
                    <span class="metric-row-label">Actual Uploaded / Read Width</span>
                    <span class="metric-row-val">{orig_w_str}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Actual Uploaded / Read Height</span>
                    <span class="metric-row-val">{orig_h_str}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Original Channel Format</span>
                    <span class="metric-row-val">{orig_c_str}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Original Bit Depth</span>
                    <span class="metric-row-val">{orig_bit_depth}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with meta_c2:
        st.markdown(
            f"""
            <div class="research-card">
                <div class="research-card-title">Group B: Model Input Specifications</div>
                <div class="metric-row">
                    <span class="metric-row-label">Target Inference Width</span>
                    <span class="metric-row-val">{target_w} px</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Target Inference Height</span>
                    <span class="metric-row-val">{target_h} px</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Inference Channel Order</span>
                    <span class="metric-row-val">3 (BGR, ImageNet Normalized)</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Quantization / Data Type</span>
                    <span class="metric-row-val">8-bit unsigned integer (uint8)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5-Value Metric Row
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
    m_c1.metric("Brightness (L)", f"{qv.brightness:.2f}")
    m_c2.metric("Contrast (Std L)", f"{qv.contrast:.2f}")
    m_c3.metric("Sharpness (Lap)", f"{qv.sharpness:.2f}")
    m_c4.metric("Noise (Immerkaer)", f"{qv.noise:.2f}")
    m_c5.metric("Haze (Dark Ch.)", f"{qv.haze:.3f}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Quality Visualizations (Left: Input & Radar, Right: Histograms)
    q_vis_col1, q_vis_col2 = st.columns(2)

    with q_vis_col1:
        st.markdown('<div class="research-card-title" style="color: #244638;">QUALITY VECTOR VISUALIZATION</div>', unsafe_allow_html=True)
        fig_qv = quality_panel(processed_input_bgr, qv_dict)
        st.pyplot(fig_qv)

    with q_vis_col2:
        st.markdown('<div class="research-card-title" style="color: #244638;">LUMINANCE HISTOGRAM (BEFORE VS AFTER)</div>', unsafe_allow_html=True)
        fig_hist = comparison_histogram_figure(processed_input_bgr, enhanced_bgr, name=active_enhancement.upper())
        st.pyplot(fig_hist)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: 03 ENHANCEMENT
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    render_section_header(
        "03 / ENHANCEMENT COMPARISON",
        "Algorithmic restoration and contrast enhancement under adverse visual conditions.",
    )

    # Method & Justification Info Panel
    st.markdown(
        f"""
        <div class="status-bar-box">
            <div class="status-bar-item">
                <span class="status-bar-label">ENHANCEMENT STRATEGY:</span>
                <span class="status-bar-val"> {active_enhancement.upper()}</span>
            </div>
            <div class="status-bar-item">
                <span class="status-bar-label">ROUTER JUSTIFICATION:</span>
                <span class="status-bar-val"> {rule_reason}</span>
            </div>
            <div class="status-bar-item">
                <span class="status-bar-label">SOURCE FRAME:</span>
                <span class="status-bar-val"> {selected_frame_id} ({degradation_choice})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-Side Large Comparison Layout
    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        st.markdown('<div class="research-card-title" style="color: #244638;">ORIGINAL / DEGRADED INPUT</div>', unsafe_allow_html=True)
        st.image(
            cv2.cvtColor(processed_input_bgr, cv2.COLOR_BGR2RGB),
            caption="Original / Degraded Input Frame",
            width="stretch",
        )
    with comp_c2:
        st.markdown(f'<div class="research-card-title" style="color: #244638;">ENHANCED FRAME ({active_enhancement.upper()})</div>', unsafe_allow_html=True)
        st.image(
            cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB),
            caption=f"Enhanced Frame ({active_enhancement})",
            width="stretch",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="research-card-title" style="color: #244638;">LUMINANCE DISTRIBUTION DENSITY COMPARISON</div>', unsafe_allow_html=True)
    fig_hist_full = comparison_histogram_figure(processed_input_bgr, enhanced_bgr, name=active_enhancement.upper())
    st.pyplot(fig_hist_full)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: 04 CLASSICAL DIP
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    render_section_header(
        "04 / CLASSICAL DIP",
        "Multi-stage classical lane detection, edge extraction, and morphological region analysis.",
    )

    st.caption("ℹ️ Note: Classical lane detection provides qualitative edge and marking extraction. No ground-truth lane annotations exist in the mirror dataset; quantitative evaluation is conducted on drivable-area masks.")

    # Intermediate Classical Lane Detection Stages (5 Steps)
    st.markdown('<div class="research-card-title" style="color: #244638; margin-top: 8px;">INTERMEDIATE CLASSICAL LANE DETECTION STAGES (5 STEPS)</div>', unsafe_allow_html=True)
    fig_stages = lane_stages_grid_figure(hough_stages)
    st.pyplot(fig_stages)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="research-card-title" style="color: #244638;">CONNECTED COMPONENTS & MORPHOLOGICAL REGION STAGE</div>', unsafe_allow_html=True)

    r_c1, r_c2 = st.columns(2)
    with r_c1:
        st.image(
            color_labels,
            caption=f"Connected Components Label Map ({mask_source_name})",
            width="stretch",
        )
    with r_c2:
        st.image(
            cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB),
            caption="Extracted Regions with Bounding Boxes (Cyan) & Centroids (Red)",
            width="stretch",
        )

    st.metric("Total Connected Segments Detected", stats_dict.get("segment_count", 0))

    regions_list = stats_dict.get("regions", [])
    if regions_list:
        df_regions = pd.DataFrame(regions_list)
        st.dataframe(df_regions, width="stretch")
    else:
        st.caption("No connected regions detected.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: 05 DEEP LEARNING
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    render_section_header(
        "05 / DEEP LEARNING SEGMENTATION",
        "Semantic drivable-area segmentation using ResNet-18 U-Net.",
    )

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
            st.markdown('<div class="research-card-title" style="color: #244638;">WITHOUT ENHANCEMENT (RAW INPUT)</div>', unsafe_allow_html=True)
            heatmap_unenh = draw_probability_heatmap(prob_unenh)
            overlay_unenh = draw_drivable_overlay(processed_input_bgr, mask_unenh, color=(0, 255, 0), alpha=0.45)
            st.image(cv2.cvtColor(heatmap_unenh, cv2.COLOR_BGR2RGB), caption="Probability Heatmap (Raw Input)", width="stretch")
            st.image(cv2.cvtColor(overlay_unenh, cv2.COLOR_BGR2RGB), caption="Drivable Segmentation Overlay (Raw Input)", width="stretch")

            if selected_gt_mask is not None:
                iou_raw, dice_raw = compute_binary_metrics(mask_unenh, selected_gt_mask)
                r_c1, r_c2 = st.columns(2)
                r_c1.metric("Raw Frame IoU", f"{iou_raw:.4f}")
                r_c2.metric("Raw Frame Dice", f"{dice_raw:.4f}")

        with ai_c2:
            st.markdown(f'<div class="research-card-title" style="color: #244638;">WITH ENHANCEMENT ({active_enhancement.upper()})</div>', unsafe_allow_html=True)
            heatmap_enh = draw_probability_heatmap(prob_enh)
            overlay_enh = draw_drivable_overlay(enhanced_bgr, mask_enh, color=(0, 255, 0), alpha=0.45)
            st.image(cv2.cvtColor(heatmap_enh, cv2.COLOR_BGR2RGB), caption=f"Probability Heatmap ({active_enhancement})", width="stretch")
            st.image(cv2.cvtColor(overlay_enh, cv2.COLOR_BGR2RGB), caption=f"Drivable Segmentation Overlay ({active_enhancement})", width="stretch")

            if selected_gt_mask is not None:
                iou_enh, dice_enh = compute_binary_metrics(mask_enh, selected_gt_mask)
                e_c1, e_c2 = st.columns(2)
                e_c1.metric("Enhanced Frame IoU", f"{iou_enh:.4f}")
                e_c2.metric("Enhanced Frame Dice", f"{dice_enh:.4f}")

        # Model Architecture Specs Card
        st.markdown(
            """
            <div class="research-card" style="margin-top: 14px;">
                <div class="research-card-title">MODEL SPECIFICATIONS & INFERENCE ARCHITECTURE</div>
                <div class="metric-row">
                    <span class="metric-row-label">Architecture</span>
                    <span class="metric-row-val">ResNet-18 U-Net (Segmentation Models PyTorch)</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Encoder Backbone</span>
                    <span class="metric-row-val">ResNet-18 (ImageNet pre-trained weights)</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Decoder Features</span>
                    <span class="metric-row-val">Skip connections + transposed convolutions</span>
                </div>
                <div class="metric-row">
                    <span class="metric-row-label">Decision Threshold</span>
                    <span class="metric-row-val">0.50 (Sigmoid Binary Classification)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: 06 EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    render_section_header(
        "06 / EVALUATION & FAILURE ANALYSIS",
        "Quantitative benchmark results, condition-stratified performance, and edge-case inspection.",
    )

    # Section A: Performance Summary & Restoration Tables
    st.markdown('<div class="research-card-title" style="color: #244638;">A. BENCHMARK PERFORMANCE & RESTORATION SUMMARY</div>', unsafe_allow_html=True)
    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:
        st.markdown("**Synthetic Degradation PSNR & SSIM**")
        synth_summary_path = "results/tables/summary_synthetic_psnr_ssim.csv"
        synth_df = load_csv_table(synth_summary_path)
        if synth_df is not None:
            st.dataframe(synth_df, width="stretch")
        else:
            st.caption("not run yet: run scripts/eval_synthetic_restoration.py")

    with col_eval2:
        st.markdown("**Condition-Stratified Test Performance (mIoU / Dice)**")
        test_eval_summary_path = "results/tables/summary_test_metrics.csv"
        eval_df = load_csv_table(test_eval_summary_path)
        if eval_df is not None:
            st.dataframe(eval_df, width="stretch")
            if "condition" in eval_df.columns and "miou" in eval_df.columns:
                fig_bar = evaluation_bar_chart_figure(eval_df, group_col="condition", metric_col="miou")
                st.pyplot(fig_bar)
        else:
            st.caption("not run yet: run scripts/evaluate.py on test split with trained model.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Section B: Operational Benchmarks
    st.markdown('<div class="research-card-title" style="color: #244638;">B. OPERATIONAL BENCHMARKS & ROUTER ACCURACY</div>', unsafe_allow_html=True)
    col_eval3, col_eval4 = st.columns(2)

    with col_eval3:
        st.markdown("**Runtime Per Stage Benchmark (ms / FPS)**")
        runtime_path = "results/tables/runtime_benchmarks.csv"
        runtime_df = load_csv_table(runtime_path)
        if runtime_df is not None:
            st.dataframe(runtime_df, width="stretch")
        else:
            st.caption("not run yet: run scripts/benchmark_runtime.py")

    with col_eval4:
        st.markdown("**Router Accuracy vs. Oracle**")
        router_acc_path = "results/tables/router_vs_oracle.csv"
        router_acc_df = load_csv_table(router_acc_path)
        if router_acc_df is not None:
            st.dataframe(router_acc_df, width="stretch")
        else:
            st.caption("not run yet: run scripts/evaluate_router.py")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Section C: Test Split Case Inspection & Failure Gallery
    st.markdown('<div class="research-card-title" style="color: #244638;">C. TEST SPLIT CASE INSPECTION & FAILURE GALLERY</div>', unsafe_allow_html=True)
    test_preds_path = "results/tables/test_predictions.csv"
    test_preds_df = load_csv_table(test_preds_path)

    if test_preds_df is not None and not test_preds_df.empty:
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
        st.markdown('<div class="research-card-title" style="color: #244638; margin-top: 10px;">WORST 20 FRAMES GALLERY</div>', unsafe_allow_html=True)
        fig_fail = failure_panel(records, k=min(20, len(records)))
        st.pyplot(fig_fail)
    else:
        st.caption("Test per-frame predictions: not run yet (run scripts/evaluate.py).")

        # Current frame inspection
        if selected_gt_mask is not None and selected_bgr is not None:
            st.markdown(f"**Current Frame Inspection: {selected_frame_id}**")
            pred_dummy = (cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY) > 100).astype(np.uint8) * 255
            fig_mask = mask_panel(enhanced_bgr, selected_gt_mask, pred_dummy, title=f"Frame: {selected_frame_id}")
            st.pyplot(fig_mask)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Section D: Qualitative Edge-Case Notes
    st.markdown('<div class="research-card-title" style="color: #244638;">D. QUALITATIVE EDGE-CASE NOTES</div>', unsafe_allow_html=True)
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR (System Diagnostics & Status Monitor)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0; border-bottom: 1px solid #2A5944; margin-bottom: 14px;">
            <div style="font-family: 'IBM Plex Sans'; font-size: 14px; font-weight: 700; color: #FFFFFF; letter-spacing: 0.06em;">
                DRIVING PERCEPTION LAB
            </div>
            <div style="font-family: 'IBM Plex Mono'; font-size: 10px; color: #DDE9DF; letter-spacing: 0.08em; margin-top: 2px;">
                SYSTEM MONITOR & TELEMETRY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        **Active Telemetry**:
        - Source: `{frame_source}`
        - Frame ID: `{selected_frame_id}`
        - Atmosphere: `{degradation_choice}`
        - Enhancement: `{active_enhancement}`
        - Dimensions: `{target_w} × {target_h}`
        """
    )
    st.divider()
    st.markdown(
        """
        **Workstation Architecture**:
        - All controls are accessible in the **Workstation Control Console** at the top of the workspace.
        - Use the horizontal navigation tabs to switch between the 6 specialized dashboards.
        """
    )
