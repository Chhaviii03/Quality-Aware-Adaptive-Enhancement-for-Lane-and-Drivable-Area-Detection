"""
src/enhance/denoise.py
----------------------
Denoising enhancement — pure function image → image.

Supports two backends:
  * "nlm"      — Non-Local Means (cv2.fastNlMeansDenoisingColored)
  * "bilateral" — Bilateral filter (cv2.bilateralFilter)

Also exports `apply_denoise_clahe` which chains denoise → CLAHE for
frames that are both dark and noisy.

Unit test: tests/test_enhance.py::test_denoise, test_denoise_clahe
"""

from __future__ import annotations

import cv2
import numpy as np

from src.enhance.clahe import apply_clahe


def apply_denoise(
    bgr: np.ndarray,
    method: str   = "nlm",
    # NLM params
    nlm_h: int              = 10,
    nlm_template_size: int  = 7,
    nlm_search_size: int    = 21,
    # bilateral params
    bilateral_d: int          = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
) -> np.ndarray:
    """Denoise a BGR uint8 image.

    Parameters
    ----------
    bgr : np.ndarray
        Input image, dtype uint8, shape (H, W, 3).
    method : str
        "nlm" (non-local means) or "bilateral".
    nlm_h : int
        Filter strength for NLM (higher = more smoothing).
    nlm_template_size : int
        Template patch size for NLM (must be odd).
    nlm_search_size : int
        Search window size for NLM (must be odd).
    bilateral_d : int
        Diameter of each pixel neighbourhood for bilateral.
    bilateral_sigma_color : float
        Filter sigma in colour space.
    bilateral_sigma_space : float
        Filter sigma in coordinate space.

    Returns
    -------
    np.ndarray
        Denoised image, dtype uint8, same shape as input.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    if method == "nlm":
        out = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h=nlm_h,
            hColor=nlm_h,
            templateWindowSize=nlm_template_size,
            searchWindowSize=nlm_search_size,
        )
    elif method == "bilateral":
        out = cv2.bilateralFilter(
            bgr,
            bilateral_d,
            bilateral_sigma_color,
            bilateral_sigma_space,
        )
    else:
        raise ValueError(f"Unknown denoise method: {method!r}. Use 'nlm' or 'bilateral'.")

    return out


def apply_denoise_clahe(
    bgr: np.ndarray,
    *,
    # denoise
    method: str               = "nlm",
    nlm_h: int                = 10,
    nlm_template_size: int    = 7,
    nlm_search_size: int      = 21,
    bilateral_d: int          = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    # CLAHE
    clip_limit: float         = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Denoise then apply CLAHE — for dark, noisy frames.

    This two-step sequence prevents CLAHE from amplifying noise.
    """
    denoised = apply_denoise(
        bgr,
        method=method,
        nlm_h=nlm_h,
        nlm_template_size=nlm_template_size,
        nlm_search_size=nlm_search_size,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    return apply_clahe(denoised, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
