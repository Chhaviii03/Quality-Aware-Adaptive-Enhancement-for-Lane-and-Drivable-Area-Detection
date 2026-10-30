"""
src/eval/metrics.py
-------------------
Evaluation metrics used by the harness and oracle labeller.

All functions operate on numpy uint8 binary masks (0 or 255).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# ─────────────────────────────────────────────────────────────────────────────
# Drivable-area metrics
# ─────────────────────────────────────────────────────────────────────────────

def road_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union for drivable-area masks.

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary uint8 masks (0 or 255), same shape.

    Returns
    -------
    float in [0, 1].
    """
    pred_b = pred > 0
    gt_b   = gt   > 0
    inter  = (pred_b & gt_b).sum()
    union  = (pred_b | gt_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def road_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for drivable-area masks."""
    pred_b = pred > 0
    gt_b   = gt   > 0
    inter  = (pred_b & gt_b).sum()
    denom  = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter) / float(denom)


# ─────────────────────────────────────────────────────────────────────────────
# Lane F1 (5 px tolerance)
# ─────────────────────────────────────────────────────────────────────────────

def _within_tolerance(pred: np.ndarray, gt: np.ndarray, tol: int = 5) -> np.ndarray:
    """Return predicted lane pixels that are within `tol` px of any GT pixel."""
    pred_b = (pred > 0).astype(np.uint8)
    gt_b   = (gt   > 0).astype(np.uint8)

    if gt_b.sum() == 0:
        return np.zeros_like(pred_b)

    # Distance transform on the background of the GT mask
    dt_gt = distance_transform_edt(1 - gt_b)   # 0 where GT=1, distance elsewhere
    # Predicted pixels within tolerance of any GT pixel
    within = (pred_b > 0) & (dt_gt <= tol)
    return within.astype(np.uint8)


def lane_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 5) -> float:
    """Lane F1 with 5-pixel tolerance (our project definition).

    Precision = fraction of predicted lane pixels within `tol` px of GT.
    Recall    = fraction of GT lane pixels within `tol` px of prediction.

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary uint8 masks (0 or 255), same shape.
    tol : int
        Pixel tolerance for matching.

    Returns
    -------
    float in [0, 1].
    """
    pred_b = (pred > 0).astype(np.uint8)
    gt_b   = (gt   > 0).astype(np.uint8)

    n_pred = pred_b.sum()
    n_gt   = gt_b.sum()

    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    tp_prec = _within_tolerance(pred_b * 255, gt_b * 255, tol).sum()
    tp_rec  = _within_tolerance(gt_b  * 255, pred_b * 255, tol).sum()

    precision = float(tp_prec) / float(n_pred)
    recall    = float(tp_rec)  / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
# Image quality (for synthetic sets)
# ─────────────────────────────────────────────────────────────────────────────

def psnr(img: np.ndarray, ref: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def ssim(img: np.ndarray, ref: np.ndarray) -> float:
    """Structural Similarity Index (single-channel or multi-channel)."""
    from skimage.metrics import structural_similarity
    if img.ndim == 3:
        return float(structural_similarity(img, ref, channel_axis=2,
                                           data_range=255))
    return float(structural_similarity(img, ref, data_range=255))
