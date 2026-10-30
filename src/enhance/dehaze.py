"""
src/enhance/dehaze.py
---------------------
Dark-channel prior dehazing — pure function image → image.

Reference: He et al. "Single Image Haze Removal Using Dark Channel Prior",
           CVPR 2009 / TPAMI 2011.

Implementation steps
--------------------
1. Compute the dark channel (15×15 minimum filter over RGB minimum).
2. Estimate atmospheric light A from the brightest 0.1 % of dark-channel pixels.
3. Estimate raw transmission t̃(x) = 1 − ω · dark_channel(I/A).
4. Refine transmission with a guided filter (uses opencv-contrib or a
   software fallback).
5. Recover the scene radiance: J = (I − A) / max(t, t_min) + A.

Unit test: tests/test_enhance.py::test_dehaze
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dark_channel(bgr: np.ndarray, patch: int = 15) -> np.ndarray:
    """Compute the dark channel of an image (float32, [0, 1])."""
    img_f = bgr.astype(np.float32) / 255.0
    min_c = img_f.min(axis=2)           # per-pixel minimum over RGB
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dc = cv2.erode(min_c, kernel)       # minimum over patch
    return dc


def _atm_light(bgr: np.ndarray, dark: np.ndarray, top_frac: float = 0.001) -> np.ndarray:
    """Estimate atmospheric light from the brightest dark-channel pixels."""
    H, W = dark.shape
    n_pixels = H * W
    n_top = max(1, int(n_pixels * top_frac))

    # Flatten and sort dark channel
    flat_dark = dark.flatten()
    indices = np.argsort(flat_dark)[-n_top:]   # indices of brightest pixels

    # Among those pixels, pick the one with the highest intensity in BGR
    bgr_f = bgr.astype(np.float32) / 255.0
    flat_bgr = bgr_f.reshape(-1, 3)
    A = flat_bgr[indices].max(axis=0)          # per-channel max
    return np.clip(A, 1e-6, 1.0)              # shape (3,)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Box-filter–based guided filter (Kaiming He et al., 2013).

    Falls back to a pure NumPy box-filter implementation so opencv-contrib
    is not required (though it is preferred for speed).
    """
    try:
        # opencv-contrib path
        gf = cv2.ximgproc.createGuidedFilter(guide, radius, eps)
        return gf.filter(src)
    except AttributeError:
        pass

    # Software fallback (approximation via box filter)
    def box(img: np.ndarray, r: int) -> np.ndarray:
        ksize = 2 * r + 1
        return cv2.boxFilter(img, cv2.CV_64F, (ksize, ksize))

    I = guide.astype(np.float64)
    p = src.astype(np.float64)

    N = box(np.ones_like(I), radius)
    mean_I = box(I, radius) / N
    mean_p = box(p, radius) / N
    mean_Ip = box(I * p, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box(I * I, radius) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a, radius) / N
    mean_b = box(b, radius) / N

    out = mean_a * I + mean_b
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_dehaze(
    bgr: np.ndarray,
    patch_size: int   = 15,
    omega: float      = 0.95,
    t_min: float      = 0.10,
    top_frac: float   = 0.001,
    guided_radius: int = 40,
    guided_eps: float  = 1e-3,
) -> np.ndarray:
    """Remove haze from a BGR uint8 image using the dark-channel prior.

    Parameters
    ----------
    bgr : np.ndarray
        Input image, dtype uint8, shape (H, W, 3).
    patch_size : int
        Side length of the dark-channel patch (must be odd or even; erosion
        kernel is `patch_size × patch_size`).
    omega : float
        Haze removal strength in (0, 1].  0.95 removes most haze while
        keeping a little depth cue.
    t_min : float
        Minimum transmission floor to avoid division by near-zero.
    top_frac : float
        Fraction of brightest dark-channel pixels used to estimate A.
    guided_radius : int
        Radius for the guided filter refinement of the transmission map.
    guided_eps : float
        Regularisation parameter for the guided filter.

    Returns
    -------
    np.ndarray
        Dehazed image, dtype uint8, same shape as input.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    img_f = bgr.astype(np.float32) / 255.0   # (H, W, 3), float [0,1]

    # 1. Dark channel
    dark = _dark_channel(bgr, patch=patch_size)

    # 2. Atmospheric light
    A = _atm_light(bgr, dark, top_frac=top_frac)   # shape (3,)

    # 3. Raw transmission
    # normalise each channel by A, then take dark channel
    img_norm = img_f / A[np.newaxis, np.newaxis, :]
    img_norm_u8 = np.clip(img_norm * 255, 0, 255).astype(np.uint8)
    dark_norm = _dark_channel(img_norm_u8, patch=patch_size)
    t_raw = 1.0 - omega * dark_norm            # (H, W), float

    # 4. Refine transmission with guided filter
    gray_guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_refined = _guided_filter(gray_guide, t_raw, guided_radius, guided_eps)
    t_refined = np.clip(t_refined, t_min, 1.0)

    # 5. Scene radiance recovery: J = (I - A) / t + A
    t_map = t_refined[:, :, np.newaxis]        # broadcast over channels
    J = (img_f - A) / t_map + A
    J = np.clip(J, 0.0, 1.0)

    out = (J * 255).astype(np.uint8)
    return out
