"""
tests/test_classical.py
-----------------------
Unit tests for the classical Hough lane detector (src/classical/lane_hough.py).

Run with:
    pytest tests/test_classical.py -v
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.classical.lane_hough import (
    detect_lanes,
    get_perspective_transform,
    birds_eye_warp,
    birds_eye_unwarp,
)


def _make_lane_image(H: int = 352, W: int = 640) -> np.ndarray:
    """Create a synthetic road image with white left and yellow right lane markings."""
    img = np.full((H, W, 3), 50, dtype=np.uint8)  # dark asphalt road

    # Left white lane line (slanted)
    cv2.line(img, (int(0.20 * W), H), (int(0.45 * W), int(0.55 * H)), (255, 255, 255), 8)
    # Right yellow lane line (slanted) in BGR: yellow is (0, 255, 255)
    cv2.line(img, (int(0.80 * W), H), (int(0.55 * W), int(0.55 * H)), (0, 255, 255), 8)
    return img


class TestLaneHough:
    def test_output_shape_and_dtype(self):
        img = _make_lane_image()
        mask = detect_lanes(img, save_stages=False)
        assert mask.shape == (352, 640)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_detects_lanes_on_synthetic_road(self):
        img = _make_lane_image()
        mask = detect_lanes(img, save_stages=False)
        # Should detect lane pixels
        assert mask.sum() > 0

    def test_save_stages_writes_stage_files(self, tmp_path):
        img = _make_lane_image()
        stage_dir = tmp_path / "stages"
        detect_lanes(img, save_stages=True, stage_dir=stage_dir, use_birds_eye=True)

        expected_files = [
            "00_original.jpg",
            "01_blur.jpg",
            "02_hls_mask.jpg",
            "03_canny.jpg",
            "04_roi.jpg",
            "04b_birds_eye.jpg",
            "05_hough.jpg",
            "06_fit.jpg",
            "07_raw_mask.jpg",
            "08_final_mask.jpg",
            "09_overlay.jpg",
        ]
        for f in expected_files:
            stage_file = stage_dir / f
            assert stage_file.exists(), f"Missing stage file: {f}"
            assert stage_file.stat().st_size > 0

    def test_birds_eye_warp_roundtrip(self):
        H, W = 352, 640
        M, Minv = get_perspective_transform(H, W)
        img = _make_lane_image(H, W)

        warped = birds_eye_warp(img, M)
        assert warped.shape == img.shape

        unwarped = birds_eye_unwarp(warped, Minv)
        assert unwarped.shape == img.shape

    def test_blank_image_returns_blank_mask(self):
        blank = np.zeros((352, 640, 3), dtype=np.uint8)
        mask = detect_lanes(blank, save_stages=False)
        assert mask.shape == (352, 640)
        assert mask.sum() == 0
