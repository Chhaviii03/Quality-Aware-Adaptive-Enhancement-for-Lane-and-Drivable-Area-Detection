"""
src/viz/panels.py
-----------------
Visualization panels for the report (FR-13).

Functions
---------
quality_panel      — original + quality-vector bar chart
enhancement_panel  — side-by-side before / after
stage_panel        — grid of all classical pipeline stages
mask_panel         — original + GT mask + predicted mask + overlay
failure_panel      — grid of worst-K frames per condition
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


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
    gt_lane:   np.ndarray,
    pred_lane: np.ndarray,
    gt_road:   Optional[np.ndarray]   = None,
    pred_road: Optional[np.ndarray]   = None,
    title:     str = "",
    out_path:  Optional[str | Path]   = None,
) -> plt.Figure:
    """Show original, GT mask, prediction, and overlay."""
    n = 4 if gt_road is None else 6
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 3))

    axes[0].imshow(_bgr2rgb(original));  axes[0].set_title("Original");  axes[0].axis("off")
    axes[1].imshow(gt_lane,   cmap="gray"); axes[1].set_title("GT lane");   axes[1].axis("off")
    axes[2].imshow(pred_lane, cmap="gray"); axes[2].set_title("Pred lane"); axes[2].axis("off")

    overlay = _bgr2rgb(original).copy()
    overlay[pred_lane > 0] = [255, 80,  80]   # red = prediction
    overlay[gt_lane   > 0] = [80,  200, 80]   # green = GT
    axes[3].imshow(overlay); axes[3].set_title("Overlay (G=GT, R=Pred)"); axes[3].axis("off")

    if gt_road is not None and pred_road is not None:
        axes[4].imshow(gt_road,   cmap="gray"); axes[4].set_title("GT road");   axes[4].axis("off")
        axes[5].imshow(pred_road, cmap="gray"); axes[5].set_title("Pred road"); axes[5].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
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
    data_root: str,
    split: str,
    out_path: Optional[str | Path] = None,
    k: int = 8,
) -> plt.Figure:
    """Grid of the k worst frames from a per-frame results DataFrame."""
    import cv2
    from src.data.labels import load_lane_mask

    records = frame_records[:k]
    cols = min(4, k)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()

    for ax, rec in zip(axes, records):
        img = cv2.imread(rec["img_path"])
        if img is None:
            ax.axis("off")
            continue
        img = cv2.resize(img, (640, 352))
        gt  = load_lane_mask(data_root, split, rec["image_id"])
        overlay = _bgr2rgb(img).copy()
        overlay[gt > 0] = [80, 200, 80]
        ax.imshow(overlay)
        ax.set_title(
            f"F1={rec.get('lane_f1', 0):.2f} | {rec.get('condition', '')} | "
            f"{rec.get('enhancement', '')}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle(f"Failure cases (lowest {k} lane F1)", fontsize=11)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
