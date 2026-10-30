"""
src/enhance/gamma.py
--------------------
Adaptive gamma correction — pure function image → image.

The gamma exponent is chosen so that the mean luminance of the output
maps to 0.5 (mid-grey), then clamped to [gamma_min, gamma_max].

gamma = ln(0.5) / ln(mean_L)   where mean_L is in [0, 1]

Unit test: tests/test_enhance.py::test_gamma
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def apply_gamma(
    bgr: np.ndarray,
    gamma_min: float = 0.4,
    gamma_max: float = 1.0,
) -> np.ndarray:
    """Apply adaptive gamma correction to a BGR uint8 image.

    Parameters
    ----------
    bgr : np.ndarray
        Input image, dtype uint8, shape (H, W, 3).
    gamma_min : float
        Lower bound for the gamma exponent (prevents over-brightening).
    gamma_max : float
        Upper bound for the gamma exponent.

    Returns
    -------
    np.ndarray
        Gamma-corrected image, same dtype and shape as input.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    # Compute mean luminance on the L channel (normalised to [0, 1])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    mean_L = float(lab[:, :, 0].mean()) / 255.0

    # Guard against degenerate images (all black / all white)
    eps = 1e-6
    mean_L = max(eps, min(mean_L, 1.0 - eps))

    # Adaptive gamma
    gamma = math.log(0.5) / math.log(mean_L)
    gamma = float(np.clip(gamma, gamma_min, gamma_max))

    # Build lookup table for speed
    table = np.array(
        [(i / 255.0) ** gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    out = cv2.LUT(bgr, table)
    return out
