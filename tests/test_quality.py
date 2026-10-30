"""
tests/test_quality.py
---------------------
Unit tests for the quality analyzer (src/quality/metrics.py).

Run with:
    pytest tests/test_quality.py -v
"""

import numpy as np
import pytest

from src.quality.metrics import compute_quality, QualityVector


def _make(shape=(352, 640, 3), seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=dtype)


class TestComputeQuality:
    def test_returns_quality_vector(self):
        img = _make()
        qv  = compute_quality(img)
        assert isinstance(qv, QualityVector)

    def test_all_fields_finite(self):
        img = _make()
        qv  = compute_quality(img)
        for field, val in qv.to_dict().items():
            assert np.isfinite(val), f"{field} is not finite: {val}"

    def test_brightness_range(self):
        img = _make()
        qv  = compute_quality(img)
        assert 0.0 <= qv.brightness <= 255.0

    def test_contrast_nonneg(self):
        img = _make()
        qv  = compute_quality(img)
        assert qv.contrast >= 0.0

    def test_sharpness_nonneg(self):
        img = _make()
        qv  = compute_quality(img)
        assert qv.sharpness >= 0.0

    def test_noise_nonneg(self):
        img = _make()
        qv  = compute_quality(img)
        assert qv.noise >= 0.0

    def test_haze_range(self):
        img = _make()
        qv  = compute_quality(img)
        assert 0.0 <= qv.haze <= 1.0

    def test_dark_image_low_brightness(self):
        dark = np.zeros((352, 640, 3), dtype=np.uint8)
        qv   = compute_quality(dark)
        assert qv.brightness < 10.0

    def test_white_image_high_brightness(self):
        white = np.full((352, 640, 3), 255, dtype=np.uint8)
        qv    = compute_quality(white)
        assert qv.brightness > 200.0

    def test_blurred_low_sharpness(self):
        import cv2
        img     = _make()
        blurred = cv2.GaussianBlur(img, (31, 31), 10)
        qv_orig = compute_quality(img)
        qv_blur = compute_quality(blurred)
        assert qv_blur.sharpness < qv_orig.sharpness

    def test_wrong_shape_raises(self):
        img = np.zeros((352, 640), dtype=np.uint8)   # grayscale, not (H,W,3)
        with pytest.raises(ValueError):
            compute_quality(img)

    def test_wrong_dtype_raises(self):
        img = np.zeros((352, 640, 3), dtype=np.float32)
        with pytest.raises(TypeError):
            compute_quality(img)

    def test_to_array_shape(self):
        img = _make()
        qv  = compute_quality(img)
        arr = qv.to_array()
        assert arr.shape == (5,)
        assert arr.dtype == np.float32
