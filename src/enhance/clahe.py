"""
src/enhance/clahe.py
--------------------
CLAHE enhancement — pure function image → image.

Applies Contrast-Limited Adaptive Histogram Equalisation on the L channel
of CIE-LAB and converts back to BGR.

Unit test: tests/test_enhance.py::test_clahe
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L channel of a BGR image.

    Parameters
    ----------
    bgr : np.ndarray
        Input image, dtype uint8, shape (H, W, 3), BGR channel order.
    clip_limit : float
        Threshold for contrast limiting.  Default 2.0.
    tile_grid_size : tuple[int, int]
        Size of the grid for histogram equalisation.  Default (8, 8).

    Returns
    -------
    np.ndarray
        Enhanced image, same dtype, shape and channel order as input.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_eq = clahe.apply(L)

    lab_eq = cv2.merge([L_eq, a, b])
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return out
