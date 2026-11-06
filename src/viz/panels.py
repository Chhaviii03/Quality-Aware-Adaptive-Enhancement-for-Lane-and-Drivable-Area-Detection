"""
src/viz/panels.py
-----------------
Visualization panels and dashboard figure builders for the DIP project (FR-13).

Functions
---------
quality_panel               — original + quality-vector horizontal bar chart
enhancement_panel           — side-by-side before / after
histogram_panel             — 2x2 grid before/after images and RGB/Luminance histograms
stage_panel                 — grid of all classical pipeline stages from directory
mask_panel                  — original + GT mask + predicted mask + overlay
failure_panel               — grid of worst-K frames
draw_drivable_overlay       — blend semi-transparent binary mask onto BGR frame
draw_probability_heatmap    — map float [0, 1] probability map to BGR Jet colormap
comparison_histogram_figure — side-by-side luminance histograms before and after
lane_stages_grid_figure     — 1x5 row of classical lane Hough stages
evaluation_bar_chart_figure — bar chart of mIoU across conditions or enhancements
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bgr2rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Quality panel
# ─────────────────────────────────────────────────────────────────────────────

def quality_panel(
    bgr: np.ndarray,
    qv_dict: dict[str, float],
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Show original image alongside a bar chart of quality measures."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(_bgr2rgb(bgr))
    axes[0].set_title("Input frame")
    axes[0].axis("off")

    keys   = list(qv_dict.keys())
    values = [qv_dict[k] for k in keys]
    axes[1].barh(keys, values, color="#4C9BE8")
    axes[1].set_title("Quality vector")
    axes[1].set_xlabel("Value")
    for i, v in enumerate(values):
        axes[1].text(v, i, f"  {v:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement panel
# ─────────────────────────────────────────────────────────────────────────────

def enhancement_panel(
    original: np.ndarray,
    enhanced: np.ndarray,
    enhancement_name: str,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side before / after enhancement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(_bgr2rgb(original))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(_bgr2rgb(enhanced))
    axes[1].set_title(f"Enhanced: {enhancement_name}")
    axes[1].axis("off")

    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Histogram Enhancement Comparison Panel
# ─────────────────────────────────────────────────────────────────────────────

def histogram_panel(
    original: np.ndarray,
    enhanced: np.ndarray,
    enhancement_name: str,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Show before/after enhancement alongside RGB & Luminance channel histograms.

    Layout (2x2 grid):
      [ Original Image ]       [ Enhanced Image ]
      [ Before Histograms ]    [ After Histograms ]
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"height_ratios": [1.2, 1.0]})

    # Top-left: Original Image
    axes[0, 0].imshow(_bgr2rgb(original))
    axes[0, 0].set_title("Original (Degraded / Input)", fontsize=12, pad=8)
    axes[0, 0].axis("off")

    # Top-right: Enhanced Image
    axes[0, 1].imshow(_bgr2rgb(enhanced))
    axes[0, 1].set_title(f"Enhanced: {enhancement_name.upper()}", fontsize=12, pad=8)
    axes[0, 1].axis("off")

    # Compute histograms
    colors = [("b", "blue", "Blue"), ("g", "green", "Green"), ("r", "red", "Red")]

    # Bottom-left: Histogram Before
    ax_hist_before = axes[1, 0]
    for idx, (col_key, plot_col, label) in enumerate(colors):
        hist = cv2.calcHist([original], [idx], None, [256], [0, 256])
        ax_hist_before.plot(hist, color=plot_col, alpha=0.7, label=label, linewidth=1.5)

    # Luminance histogram
    lab_orig = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    hist_l_orig = cv2.calcHist([lab_orig], [0], None, [256], [0, 256])
    ax_hist_before.plot(hist_l_orig, color="black", linestyle="--", alpha=0.8, label="Luminance (L)", linewidth=1.5)

    ax_hist_before.set_title("Histogram Before Enhancement", fontsize=11)
    ax_hist_before.set_xlabel("Pixel Intensity [0 - 255]", fontsize=10)
    ax_hist_before.set_ylabel("Pixel Count", fontsize=10)
    ax_hist_before.set_xlim([0, 256])
    ax_hist_before.grid(True, linestyle=":", alpha=0.5)
    ax_hist_before.legend(loc="upper right", fontsize=8)

    # Bottom-right: Histogram After
    ax_hist_after = axes[1, 1]
    for idx, (col_key, plot_col, label) in enumerate(colors):
        hist = cv2.calcHist([enhanced], [idx], None, [256], [0, 256])
        ax_hist_after.plot(hist, color=plot_col, alpha=0.7, label=label, linewidth=1.5)

    lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    hist_l_enh = cv2.calcHist([lab_enh], [0], None, [256], [0, 256])
    ax_hist_after.plot(hist_l_enh, color="black", linestyle="--", alpha=0.8, label="Luminance (L)", linewidth=1.5)

    ax_hist_after.set_title(f"Histogram After {enhancement_name.upper()}", fontsize=11)
    ax_hist_after.set_xlabel("Pixel Intensity [0 - 255]", fontsize=10)
    ax_hist_after.set_ylabel("Pixel Count", fontsize=10)
    ax_hist_after.set_xlim([0, 256])
    ax_hist_after.grid(True, linestyle=":", alpha=0.5)
    ax_hist_after.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Stage panel (classical pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def stage_panel(
    stage_dir: str | Path,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Load all saved stage images from `stage_dir` and show as a grid."""
    stage_dir = Path(stage_dir)
    imgs = sorted(stage_dir.glob("*.jpg")) + sorted(stage_dir.glob("*.png"))

    n = len(imgs)
    if n == 0:
        raise FileNotFoundError(f"No stage images found in {stage_dir}")

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = np.array(axes).flatten()

    for ax, img_path in zip(axes, imgs):
        img = cv2.imread(str(img_path))
        if img is not None:
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(_bgr2rgb(img))
        ax.set_title(img_path.stem.replace("_", " "), fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Mask panel
# ─────────────────────────────────────────────────────────────────────────────

def mask_panel(
    original:  np.ndarray,
    gt_mask:   np.ndarray,
    pred_mask: np.ndarray,
    title:     str = "",
    out_path:  Optional[str | Path] = None,
) -> plt.Figure:
    """Show original, GT drivable mask, predicted mask, and overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    axes[0].imshow(_bgr2rgb(original))
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Drivable")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Drivable")
    axes[2].axis("off")

    overlay = _bgr2rgb(original).copy()
    overlay[pred_mask > 0] = [255, 80, 80]  # red = prediction
    overlay[gt_mask > 0] = [80, 200, 80]   # green = GT
    # overlap where both are positive will be [80, 200, 80] unless overwritten
    overlap = (pred_mask > 0) & (gt_mask > 0)
    overlay[overlap] = [240, 220, 50]  # yellow = True Positive
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (G=GT, R=Pred, Y=Both)")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Failure panel
# ─────────────────────────────────────────────────────────────────────────────

def failure_panel(
    frame_records: list[dict],
    k: int = 8,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Grid of the k worst frames from a per-frame results list."""
    records = frame_records[:k]
    cols = min(4, max(1, k))
    rows = (len(records) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, max(1, rows) * 3))
    axes = np.array(axes).flatten()

    for ax, rec in zip(axes, records):
        img_path = rec.get("img_path", "")
        img = cv2.imread(img_path) if img_path and Path(img_path).exists() else None
        if img is None:
            ax.axis("off")
            continue
        img = cv2.resize(img, (640, 352))
        ax.imshow(_bgr2rgb(img))
        iou_val = rec.get("iou", rec.get("drivable_iou", 0.0))
        cond = rec.get("condition", "test")
        enh = rec.get("enhancement", "none")
        ax.set_title(f"IoU={iou_val:.3f} | {cond} | {enh}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle(f"Failure Cases (Lowest {len(records)} mIoU Frames)", fontsize=11)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Streamlit Helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_drivable_overlay(
    bgr: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a semi-transparent colored mask over the BGR frame."""
    H, W = bgr.shape[:2]
    if binary_mask.shape[:2] != (H, W):
        binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay = bgr.copy()
    mask_bool = (binary_mask > 0)
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask_bool] = (alpha * color_arr + (1.0 - alpha) * bgr[mask_bool]).astype(np.uint8)
    return overlay


def draw_probability_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """Convert float probability map in [0, 1] to a BGR color heatmap."""
    u8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def comparison_histogram_figure(original: np.ndarray, enhanced: np.ndarray, name: str = "Enhanced") -> plt.Figure:
    """Plot single-channel luminance / grayscale histograms before and after enhancement."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    lab_orig = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)[:, :, 0]
    lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)[:, :, 0]

    h_orig = cv2.calcHist([lab_orig], [0], None, [256], [0, 256])
    h_enh = cv2.calcHist([lab_enh], [0], None, [256], [0, 256])

    axes[0].plot(h_orig, color="black", linewidth=1.5)
    axes[0].set_title("Original Luminance Histogram", fontsize=11)
    axes[0].set_xlim([0, 256])
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(h_enh, color="royalblue", linewidth=1.5)
    axes[1].set_title(f"{name} Luminance Histogram", fontsize=11)
    axes[1].set_xlim([0, 256])
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


def lane_stages_grid_figure(stages: dict[str, np.ndarray]) -> plt.Figure:
    """Plot grid of classical lane detection intermediate stages with captions."""
    stage_keys = [
        ("02_hls_mask", "HLS Colour Mask"),
        ("03_canny", "Canny Edges (50/150)"),
        ("04_roi", "Trapezoid ROI Mask"),
        ("05_hough", "Probabilistic Hough Lines"),
        ("08_cleaned_mask", "Cleaned / Closed Lane Mask"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for ax, (k, title) in zip(axes, stage_keys):
        img = stages.get(k)
        if img is not None:
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(_bgr2rgb(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    return fig


def evaluation_bar_chart_figure(
    df: pd.DataFrame,
    group_col: str = "condition",
    metric_col: str = "miou",
    title: str = "Mean IoU Comparison",
) -> plt.Figure:
    """Bar chart of mIoU across conditions or enhancement methods."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = df[group_col].astype(str)
    y = df[metric_col].astype(float)
    bars = ax.bar(x, y, color="#2E86AB", width=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean IoU", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig
