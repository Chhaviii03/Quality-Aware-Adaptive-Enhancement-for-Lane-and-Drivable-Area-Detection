"""
src/enhance/__init__.py
-----------------------
Enhancement dispatcher.

Usage
-----
from src.enhance import get_enhancement, ENHANCEMENT_NAMES

fn = get_enhancement("clahe")
enhanced = fn(bgr_image)
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from src.enhance.clahe   import apply_clahe
from src.enhance.gamma   import apply_gamma
from src.enhance.dehaze  import apply_dehaze
from src.enhance.denoise import apply_denoise, apply_denoise_clahe

# All valid enhancement names
ENHANCEMENT_NAMES = ("none", "clahe", "gamma", "dehaze", "denoise", "denoise_clahe")


def _identity(bgr: np.ndarray) -> np.ndarray:
    """No-op enhancement — returns a copy."""
    return bgr.copy()


_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none":          _identity,
    "clahe":         apply_clahe,
    "gamma":         apply_gamma,
    "dehaze":        apply_dehaze,
    "denoise":       apply_denoise,
    "denoise_clahe": apply_denoise_clahe,
}


def get_enhancement(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the enhancement function for the given name.

    Parameters
    ----------
    name : str
        One of ENHANCEMENT_NAMES.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Pure function that accepts a uint8 BGR image and returns a uint8
        BGR image of the same shape.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown enhancement {name!r}. "
            f"Valid options: {ENHANCEMENT_NAMES}"
        )
    return _REGISTRY[name]
