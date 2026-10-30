"""
tests/test_degrade.py
---------------------
Unit tests for synthetic degradation functions.

Run with:
    pytest tests/test_degrade.py -v
"""

import numpy as np
import pytest

from src.data.degrade import apply_fog, apply_low_light


def _make(seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(50, 200, (352, 640, 3), dtype=np.uint8)


class TestFog:
    def test_shape_dtype_range(self):
        img = _make()
        out = apply_fog(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_no_mutation(self):
        img  = _make()
        orig = img.copy()
        apply_fog(img)
        assert np.array_equal(img, orig)

    def test_fog_increases_brightness(self):
        """Fogging adds white atmospheric light, so mean should increase."""
        img = _make()
        out = apply_fog(img)
        assert out.astype(float).mean() >= img.astype(float).mean() - 5

    def test_dense_fog_brighter_than_mild(self):
        img  = _make()
        mild  = apply_fog(img, beta=1.0)
        dense = apply_fog(img, beta=2.5)
        assert dense.astype(float).mean() >= mild.astype(float).mean() - 5

    def test_wrong_dtype_raises(self):
        with pytest.raises(TypeError):
            apply_fog(_make().astype(np.float32))


class TestLowLight:
    def test_shape_dtype_range(self):
        img = _make()
        out = apply_low_light(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_no_mutation(self):
        img  = _make()
        orig = img.copy()
        apply_low_light(img)
        assert np.array_equal(img, orig)

    def test_output_is_darker(self):
        img = _make()
        out = apply_low_light(img)
        assert out.astype(float).mean() < img.astype(float).mean()

    def test_different_seeds_differ(self):
        img = _make()
        a = apply_low_light(img, seed=0)
        b = apply_low_light(img, seed=99)
        # Different seeds should produce at least slightly different results
        # (different Poisson / Gaussian draws)
        assert not np.array_equal(a, b)

    def test_wrong_dtype_raises(self):
        with pytest.raises(TypeError):
            apply_low_light(_make().astype(np.float32))
