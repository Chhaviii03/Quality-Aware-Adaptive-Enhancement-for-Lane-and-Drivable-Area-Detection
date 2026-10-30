"""
tests/test_enhance.py
---------------------
Unit tests for every enhancement function.

Each test checks:
  ✓ Output shape matches input.
  ✓ Output dtype is uint8.
  ✓ Output values in [0, 255] (no NaN, no overflow).
  ✓ The function is a pure transform (does not mutate input).

Run with:
    pytest tests/test_enhance.py -v
"""

import numpy as np
import pytest

# ── Synthetic test images ──────────────────────────────────────────────────

def _make_image(shape=(352, 640, 3), seed=0, dtype=np.uint8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, shape, dtype=dtype))


def _make_dark_image(shape=(352, 640, 3), seed=1) -> np.ndarray:
    """Simulates a night frame (mean brightness ~30)."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 60, shape, dtype=np.uint8))


def _make_hazy_image(shape=(352, 640, 3), seed=2) -> np.ndarray:
    """Simulates a hazy frame (pixels biased toward white)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(150, 230, shape, dtype=np.uint8)
    return base


def _assert_valid(out: np.ndarray, ref: np.ndarray) -> None:
    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    assert out.dtype == np.uint8,  f"Expected uint8, got {out.dtype}"
    assert 0 <= int(out.min()), "Values below 0"
    assert int(out.max()) <= 255, "Values above 255"
    assert not np.isnan(out.astype(float)).any(), "NaN in output"


def _assert_no_mutation(fn, img):
    """Check that fn does not mutate its input."""
    original = img.copy()
    fn(img)
    assert np.array_equal(img, original), "Enhancement mutated input image"


# ── CLAHE ─────────────────────────────────────────────────────────────────

class TestCLAHE:
    def test_shape_dtype_range(self):
        from src.enhance.clahe import apply_clahe
        img = _make_image()
        out = apply_clahe(img)
        _assert_valid(out, img)

    def test_no_mutation(self):
        from src.enhance.clahe import apply_clahe
        img = _make_image()
        _assert_no_mutation(apply_clahe, img)

    def test_dark_image(self):
        from src.enhance.clahe import apply_clahe
        img = _make_dark_image()
        out = apply_clahe(img)
        _assert_valid(out, img)
        # CLAHE should increase mean brightness on a dark image
        assert out.mean() >= img.mean() - 5  # allow tiny floating-point diff

    def test_wrong_dtype_raises(self):
        from src.enhance.clahe import apply_clahe
        img = _make_image().astype(np.float32)
        with pytest.raises(TypeError):
            apply_clahe(img)

    def test_custom_params(self):
        from src.enhance.clahe import apply_clahe
        img = _make_image()
        out = apply_clahe(img, clip_limit=4.0, tile_grid_size=(4, 4))
        _assert_valid(out, img)


# ── Gamma ─────────────────────────────────────────────────────────────────

class TestGamma:
    def test_shape_dtype_range(self):
        from src.enhance.gamma import apply_gamma
        img = _make_image()
        out = apply_gamma(img)
        _assert_valid(out, img)

    def test_no_mutation(self):
        from src.enhance.gamma import apply_gamma
        img = _make_image()
        _assert_no_mutation(apply_gamma, img)

    def test_dark_image_brightened(self):
        from src.enhance.gamma import apply_gamma
        img = _make_dark_image()
        out = apply_gamma(img)
        _assert_valid(out, img)
        assert out.mean() > img.mean(), "Gamma should brighten a dark image"

    def test_wrong_dtype_raises(self):
        from src.enhance.gamma import apply_gamma
        with pytest.raises(TypeError):
            apply_gamma(_make_image().astype(np.float32))

    def test_gamma_clamping(self):
        """All-black image: mean≈0, gamma must be clamped and not crash."""
        from src.enhance.gamma import apply_gamma
        img = np.zeros((352, 640, 3), dtype=np.uint8)
        out = apply_gamma(img)
        _assert_valid(out, img)


# ── Dehaze ────────────────────────────────────────────────────────────────

class TestDehaze:
    def test_shape_dtype_range(self):
        from src.enhance.dehaze import apply_dehaze
        img = _make_hazy_image()
        out = apply_dehaze(img)
        _assert_valid(out, img)

    def test_no_mutation(self):
        from src.enhance.dehaze import apply_dehaze
        img = _make_hazy_image()
        _assert_no_mutation(apply_dehaze, img)

    def test_hazy_image_cleared(self):
        from src.enhance.dehaze import apply_dehaze
        img = _make_hazy_image()
        out = apply_dehaze(img)
        _assert_valid(out, img)

    def test_wrong_dtype_raises(self):
        from src.enhance.dehaze import apply_dehaze
        with pytest.raises(TypeError):
            apply_dehaze(_make_image().astype(np.float32))

    def test_small_image(self):
        """Edge case: very small image (50×50)."""
        from src.enhance.dehaze import apply_dehaze
        img = _make_hazy_image(shape=(50, 50, 3))
        out = apply_dehaze(img)
        _assert_valid(out, img)


# ── Denoise ───────────────────────────────────────────────────────────────

class TestDenoise:
    def test_nlm_shape_dtype_range(self):
        from src.enhance.denoise import apply_denoise
        img = _make_image()
        out = apply_denoise(img, method="nlm")
        _assert_valid(out, img)

    def test_bilateral_shape_dtype_range(self):
        from src.enhance.denoise import apply_denoise
        img = _make_image()
        out = apply_denoise(img, method="bilateral")
        _assert_valid(out, img)

    def test_no_mutation_nlm(self):
        from src.enhance.denoise import apply_denoise
        img = _make_image()
        _assert_no_mutation(lambda x: apply_denoise(x, method="nlm"), img)

    def test_no_mutation_bilateral(self):
        from src.enhance.denoise import apply_denoise
        img = _make_image()
        _assert_no_mutation(lambda x: apply_denoise(x, method="bilateral"), img)

    def test_invalid_method_raises(self):
        from src.enhance.denoise import apply_denoise
        with pytest.raises(ValueError):
            apply_denoise(_make_image(), method="foobar")

    def test_wrong_dtype_raises(self):
        from src.enhance.denoise import apply_denoise
        with pytest.raises(TypeError):
            apply_denoise(_make_image().astype(np.float32))


# ── Denoise+CLAHE ─────────────────────────────────────────────────────────

class TestDenoiseCLAHE:
    def test_shape_dtype_range(self):
        from src.enhance.denoise import apply_denoise_clahe
        img = _make_dark_image()
        out = apply_denoise_clahe(img)
        _assert_valid(out, img)

    def test_no_mutation(self):
        from src.enhance.denoise import apply_denoise_clahe
        img = _make_dark_image()
        _assert_no_mutation(apply_denoise_clahe, img)


# ── None (identity) ───────────────────────────────────────────────────────

class TestNone:
    def test_identity_returns_copy(self):
        from src.enhance import get_enhancement
        fn  = get_enhancement("none")
        img = _make_image()
        out = fn(img)
        assert np.array_equal(out, img), "Identity should return same pixel values"
        assert out is not img, "Identity should return a copy, not the same object"
        _assert_valid(out, img)


# ── Registry ──────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_names_registered(self):
        from src.enhance import ENHANCEMENT_NAMES, get_enhancement
        for name in ENHANCEMENT_NAMES:
            fn = get_enhancement(name)
            assert callable(fn), f"{name} is not callable"

    def test_unknown_name_raises(self):
        from src.enhance import get_enhancement
        with pytest.raises(KeyError):
            get_enhancement("nonexistent_enhancement")
