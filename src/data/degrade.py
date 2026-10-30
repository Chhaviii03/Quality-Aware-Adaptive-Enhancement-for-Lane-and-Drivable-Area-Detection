"""
src/data/degrade.py
-------------------
Synthetic degradation generator (FR-03).

Two degradations with fixed seeds:
  * Fog  — atmospheric scattering model I = J·t + A·(1−t)
            Depth proxy d(y) grows toward the horizon (bottom of image is close to vehicle,
            horizon is distant, sky has maximum depth).
            Supports beta = 1.0 (mild) and beta = 2.5 (dense).
  * Low light — intensity scale + gamma darkening + Poisson-Gaussian noise.

Includes a utility `save_degraded_beside_original` to save degraded versions
beside each original clear frame.

Usage
-----
    from src.data.degrade import apply_fog, apply_low_light, save_degraded_beside_original

    fog_mild   = apply_fog(bgr, beta=1.0, A=0.9, seed=42)
    fog_dense  = apply_fog(bgr, beta=2.5, A=0.9, seed=42)
    dark_noisy = apply_low_light(bgr, intensity_scale=0.25, gamma=1.8, seed=42)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Fog (Atmospheric Scattering Model)
# ─────────────────────────────────────────────────────────────────────────────

def apply_fog(
    bgr: np.ndarray,
    beta: float = 1.0,
    A: float = 0.9,
    horizon_ratio: float = 0.55,
    seed: int = 42,
) -> np.ndarray:
    """Apply synthetic fog using the Koschmieder atmospheric scattering model.

    I(x) = J(x)·t(x) + A·(1 − t(x))
    t(x) = exp(−β·d(x))

    The depth proxy d(x) grows from the bottom of the frame (nearest the vehicle,
    depth ~0.1) toward the horizon (distant road, depth = 1.0). Sky above the
    horizon has maximum distance (depth = 1.0).

    Parameters
    ----------
    bgr : np.ndarray
        Clear-day input, uint8, (H, W, 3).
    beta : float
        Scattering coefficient. Typically 1.0 = mild, 2.5 = dense.
    A : float
        Atmospheric airlight intensity in (0, 1]. Default 0.9.
    horizon_ratio : float
        Row fraction representing the horizon line (default: 0.55).
    seed : int
        Random seed (retained for deterministic interface).

    Returns
    -------
    np.ndarray
        Fogged image, uint8, same shape as input.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    H, W = bgr.shape[:2]
    img_f = bgr.astype(np.float32) / 255.0

    # Depth proxy:
    # Road rows [horizon .. H]: depth grows from ~0.10 (at row H, near car) to 1.00 (at horizon)
    # Sky rows [0 .. horizon]: maximum depth = 1.00 (infinite / sky background)
    horizon = int(H * horizon_ratio)
    rows = np.arange(H, dtype=np.float32)

    road_depth = 1.0 - (rows - horizon) / max(H - horizon, 1) * 0.90
    depth_1d = np.where(rows < horizon, 1.0, road_depth)
    depth_1d = np.clip(depth_1d, 0.0, 1.0)
    depth = depth_1d[:, np.newaxis]  # shape (H, 1)

    # Transmission map t = exp(-beta * depth)
    t = np.exp(-beta * depth).astype(np.float32)
    t_3d = np.broadcast_to(t, (H, W))[:, :, np.newaxis]

    fogged = img_f * t_3d + A * (1.0 - t_3d)
    fogged = np.clip(fogged, 0.0, 1.0)
    return (fogged * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Low Light (Poisson-Gaussian Noise)
# ─────────────────────────────────────────────────────────────────────────────

def apply_low_light(
    bgr: np.ndarray,
    intensity_scale: float = 0.25,
    gamma: float = 1.8,
    read_noise_sigma: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Simulate low-light nighttime driving conditions with sensor noise.

    Steps
    -----
    1. Scale intensity by `intensity_scale` (0.20–0.35).
    2. Apply non-linear sensor gamma darkening (1.5–2.2).
    3. Add Poisson shot noise (signal-dependent photon arrival noise).
    4. Add Gaussian read noise (signal-independent sensor readout noise).

    Parameters
    ----------
    bgr : np.ndarray
        Clear input, uint8, (H, W, 3).
    intensity_scale : float
        Brightness attenuation factor.
    gamma : float
        Darkening gamma exponent.
    read_noise_sigma : float
        Standard deviation of Gaussian read noise.
    seed : int
        Fixed random seed for reproducible noise generation.

    Returns
    -------
    np.ndarray
        Dark, noisy image, uint8, same shape as input.
    """
    if bgr.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {bgr.dtype}")

    rng = np.random.default_rng(seed)
    img_f = bgr.astype(np.float32) / 255.0

    # 1. Intensity scaling
    img_scaled = img_f * intensity_scale

    # 2. Gamma darkening
    img_gamma = np.power(np.clip(img_scaled, 0.0, 1.0), gamma)

    # 3. Poisson shot noise
    photons = 1000.0  # reference photon scale
    counts = img_gamma * photons
    shot = rng.poisson(np.maximum(counts, 0.0)).astype(np.float32) / photons
    shot = np.clip(shot, 0.0, 1.0)

    # 4. Additive Gaussian read noise
    read_noise = rng.normal(0.0, read_noise_sigma, shot.shape).astype(np.float32)
    noisy = np.clip(shot + read_noise, 0.0, 1.0)

    return (noisy * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Save Degraded Frames Beside Originals
# ─────────────────────────────────────────────────────────────────────────────

def save_degraded_beside_original(
    orig_path: str | Path,
    out_dir: Optional[str | Path] = None,
    betas: tuple[float, float] = (1.0, 2.5),
    low_light_scale: float = 0.25,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate and save degraded versions of an image beside the original.

    Saves:
      - {stem}_fog_mild.jpg   (beta = 1.0)
      - {stem}_fog_dense.jpg  (beta = 2.5)
      - {stem}_low_light.jpg  (intensity scale + Poisson-Gaussian noise)

    Parameters
    ----------
    orig_path : str | Path
        Path to clear original image.
    out_dir : str | Path | None
        Directory where degraded images are saved. Defaults to same parent as orig_path.
    betas : tuple[float, float]
        (mild_beta, dense_beta) for fog.
    low_light_scale : float
        Scale for low-light simulation.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    dict mapping degradation key to saved Path.
    """
    orig_path = Path(orig_path)
    bgr = cv2.imread(str(orig_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {orig_path}")

    target_dir = Path(out_dir) if out_dir is not None else orig_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = orig_path.stem

    saved = {}

    # Fog mild (beta=1.0)
    fog_mild = apply_fog(bgr, beta=betas[0], A=0.9, seed=seed)
    p_mild = target_dir / f"{stem}_fog_mild.jpg"
    cv2.imwrite(str(p_mild), fog_mild)
    saved["fog_mild"] = p_mild

    # Fog dense (beta=2.5)
    fog_dense = apply_fog(bgr, beta=betas[1], A=0.9, seed=seed)
    p_dense = target_dir / f"{stem}_fog_dense.jpg"
    cv2.imwrite(str(p_dense), fog_dense)
    saved["fog_dense"] = p_dense

    # Low light
    dark = apply_low_light(bgr, intensity_scale=low_light_scale, seed=seed)
    p_dark = target_dir / f"{stem}_low_light.jpg"
    cv2.imwrite(str(p_dark), dark)
    saved["low_light"] = p_dark

    return saved
