"""
src/quality/metrics.py
----------------------
Five quality measures returned as a named dataclass.

All measures operate on a uint8 BGR image (as loaded by OpenCV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Public data structure
# ─────────────────────────────────────────────────────────────────────────────

class QualityVector(dict):
    """Per-frame quality descriptor (dict-compatible dataclass).

    Fields
    ------
    brightness : float
        Mean of the L channel in CIE-LAB, range [0, 255].
    contrast : float
        Standard deviation of the L channel in CIE-LAB, range [0, 255].
    sharpness : float
        Variance of the Laplacian on the grayscale image (higher = sharper).
    noise : float
        Immerkaer fast-noise-sigma estimate on the grayscale image.
    haze : float
        Mean of the 15×15 dark channel, normalised to [0, 1].
    """
    brightness: float
    contrast:   float
    sharpness:  float
    noise:      float
    haze:       float

    def __init__(self, brightness: float, contrast: float, sharpness: float, noise: float, haze: float):
        super().__init__(
            brightness=float(brightness),
            contrast=float(contrast),
            sharpness=float(sharpness),
            noise=float(noise),
            haze=float(haze),
        )
        self.brightness = float(brightness)
        self.contrast   = float(contrast)
        self.sharpness  = float(sharpness)
        self.noise      = float(noise)
        self.haze       = float(haze)

    def to_dict(self) -> dict[str, float]:
        return dict(self)

    def to_array(self) -> np.ndarray:
        return np.array([self.brightness, self.contrast,
                         self.sharpness, self.noise, self.haze],
                        dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Individual measures
# ─────────────────────────────────────────────────────────────────────────────

def _brightness_contrast(bgr: np.ndarray) -> tuple[float, float]:
    """Return (brightness, contrast) from the L channel of CIE-LAB."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    return float(L.mean()), float(L.std())


def _sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian on the grayscale image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _noise_sigma(bgr: np.ndarray) -> float:
    """Immerkaer (1996) fast noise-sigma estimate.

    sigma = sqrt(pi/2) * 1/(6*(W-2)*(H-2)) * |sum(K * I)|
    where K is the 3×3 Laplacian-of-Gaussian kernel:
        [ 1  -2  1 ]
        [-2   4 -2 ]
        [ 1  -2  1 ]
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W = gray.shape
    if H < 3 or W < 3:
        return 0.0

    kernel = np.array([[ 1, -2,  1],
                        [-2,  4, -2],
                        [ 1, -2,  1]], dtype=np.float64)
    filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
    sigma = math.sqrt(math.pi / 2.0) / (6.0 * (W - 2) * (H - 2)) * np.abs(filtered).sum()
    return float(sigma)


def _haze(bgr: np.ndarray, patch: int = 15) -> float:
    """Mean of the dark channel, normalised to [0, 1].

    Dark channel = min over patch of min over RGB channels.
    """
    # minimum across colour channels → dark map (H, W)
    dark_map = bgr.min(axis=2).astype(np.float32) / 255.0
    # minimum over spatial patch via erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark_channel = cv2.erode(dark_map, kernel)
    return float(dark_channel.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality(bgr: np.ndarray) -> QualityVector:
    """Compute all five quality measures for a single BGR uint8 image.

    Parameters
    ----------
    bgr : np.ndarray
        Image in OpenCV BGR format, dtype uint8, shape (H, W, 3).

    Returns
    -------
    QualityVector
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) BGR image, got shape {bgr.shape}")
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {bgr.dtype}")

    brightness, contrast = _brightness_contrast(bgr)
    sharpness = _sharpness(bgr)
    noise = _noise_sigma(bgr)
    haze = _haze(bgr)

    return QualityVector(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        noise=noise,
        haze=haze,
    )


# Alias for compatibility
compute_quality_vector = compute_quality

