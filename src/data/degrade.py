"""
src/data/degrade.py
-------------------
Synthetic degradation generator (FR-03).

Two degradations with fixed seeds:
  * Fog  — atmospheric scattering model I = J·t + A·(1−t)
  * Low light — intensity scale + gamma + Poisson-Gaussian noise

Each function is a pure image → image transform.
Seeds are applied per-frame using (global_seed XOR image_index).

Usage
-----
    from src.data.degrade import apply_fog, apply_low_light

    foggy     = apply_fog(bgr, severity="mild", seed=42)
    dark_noisy = apply_low_light(bgr, seed=42)
"""

from __future__ import annotations

import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Fog
# ─────────────────────────────────────────────────────────────────────────────

def apply_fog(
    bgr: np.ndarray,
    beta: float  = 1.0,
    A:    float  = 0.9,
    seed: int    = 42,
) -> np.ndarray:
    """Apply synthetic fog using the atmospheric scattering model.

    I(x) = J(x)·t(x) + A·(1 − t(x))
    t(x) = exp(−β·d(x))

    The depth proxy d(x) increases linearly toward the horizon row and
    is constant below it (uniform depth below the horizon = distant road).

    Parameters
    ----------
    bgr : np.ndarray
        Clear-day input, uint8, (H, W, 3).
    beta : float
        Scattering coefficient. 1.0 = mild, 2.5 = dense.
    A : float
        Atmospheric light intensity ∈ (0, 1]. Default 0.9.
    seed : int
        Random seed (currently deterministic; seed kept for API consistency).

    Returns
    -------
    np.ndarray
        Fogged image, uint8, same shape.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    H, W = bgr.shape[:2]
    img_f = bgr.astype(np.float32) / 255.0

    # Depth proxy: rows 0..H/2 linearly 0→1; rows H/2..H constant 1
    horizon = H // 2
    rows    = np.arange(H, dtype=np.float32)
    depth   = np.where(rows < horizon,
                       rows / max(horizon, 1),
                       1.0)
    depth   = depth[:, np.newaxis]          # (H, 1) broadcast to (H, W)

    t = np.exp(-beta * depth).astype(np.float32)   # transmission (H, 1)
    t = np.broadcast_to(t, (H, W))[:, :, np.newaxis]  # (H, W, 1)

    fogged = img_f * t + A * (1.0 - t)
    fogged = np.clip(fogged, 0.0, 1.0)
    return (fogged * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Low light
# ─────────────────────────────────────────────────────────────────────────────

def apply_low_light(
    bgr: np.ndarray,
    intensity_scale: float = 0.25,
    gamma:           float = 1.8,
    read_noise_sigma: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a low-light image.

    Steps
    -----
    1. Scale intensity by `intensity_scale`.
    2. Apply gamma darkening (out = in^gamma).
    3. Add Poisson shot noise + Gaussian read noise.

    Parameters
    ----------
    bgr : np.ndarray
        Clear input, uint8, (H, W, 3).
    intensity_scale : float
        Multiply pixel values by this before gamma (0.20–0.35 per PRD).
    gamma : float
        Darkening gamma (1.5–2.2 per PRD).
    read_noise_sigma : float
        Standard deviation of additive Gaussian read noise (0.01–0.03).
    seed : int

    Returns
    -------
    np.ndarray
        Dark, noisy image, uint8, same shape.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    rng = np.random.default_rng(seed)

    img_f = bgr.astype(np.float32) / 255.0

    # 1. Scale
    img_f = img_f * intensity_scale

    # 2. Gamma darkening
    img_f = np.power(np.clip(img_f, 0, 1), gamma)

    # 3. Poisson shot noise (photon noise)
    #    Scale to photon counts, sample Poisson, scale back
    photons = 1000.0   # virtual photon count at full exposure
    img_counts = img_f * photons
    shot = rng.poisson(np.maximum(img_counts, 0)).astype(np.float32) / photons
    shot = np.clip(shot, 0, 1)

    # 4. Gaussian read noise
    read_noise = rng.normal(0, read_noise_sigma, shot.shape).astype(np.float32)
    noisy = np.clip(shot + read_noise, 0.0, 1.0)

    return (noisy * 255).astype(np.uint8)
