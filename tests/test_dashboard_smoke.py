"""
tests/test_dashboard_smoke.py
-----------------------------
Smoke tests for the Streamlit dashboard components (Stage 7).

Verifies that all 6 panels and their visual / analytical helpers run without error
on synthetic and manifest data, without requiring a GPU or live Streamlit server.
"""

from __future__ import annotations

import json
import py_compile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import pytest

from src.classical.lane_hough import get_hough_stage_images
from src.classical.region import analyze_regions
from src.data.degrade import apply_fog, apply_low_light
from src.enhance import ENHANCEMENT_NAMES, get_enhancement
from src.quality.metrics import QualityVector, compute_quality_vector
from src.router.rule import RuleRouter
from src.viz.panels import (
    comparison_histogram_figure,
    draw_drivable_overlay,
    draw_probability_heatmap,
    evaluation_bar_chart_figure,
    failure_panel,
    lane_stages_grid_figure,
    mask_panel,
    quality_panel,
)


@pytest.fixture
def sample_bgr() -> np.ndarray:
    """Create a realistic synthetic road frame (352, 640, 3)."""
    img = np.full((352, 640, 3), 100, dtype=np.uint8)
    # Sky region
    img[:160, :] = [210, 180, 140]
    # Road region
    img[160:, :] = [70, 70, 70]
    # Drivable lane markings
    cv2.line(img, (200, 352), (290, 190), (255, 255, 255), 4)
    cv2.line(img, (440, 352), (350, 190), (0, 255, 255), 4)
    return img


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create a synthetic drivable area mask (352, 640)."""
    mask = np.zeros((352, 640), dtype=np.uint8)
    pts = np.array([[200, 352], [440, 352], [350, 190], [290, 190]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def test_app_compilation():
    """Verify that app.py compiles cleanly with no syntax errors."""
    compiled = py_compile.compile("app.py", doraise=True)
    assert compiled is not None


def test_panel1_input_and_quality(sample_bgr):
    """Panel 1: Input dimensions and 5-value quality vector computation."""
    H, W, C = sample_bgr.shape
    assert (H, W, C) == (352, 640, 3)
    assert sample_bgr.dtype == np.uint8

    qv = compute_quality_vector(sample_bgr)
    assert isinstance(qv, QualityVector)
    qv_dict = qv.to_dict()

    for k in ["brightness", "contrast", "sharpness", "noise", "haze"]:
        assert k in qv_dict
        assert np.isfinite(qv_dict[k])

    fig = quality_panel(sample_bgr, qv_dict)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_panel2_dip_enhancement_and_classical(sample_bgr):
    """Panel 2: Rule router routing, enhancement execution, and intermediate Hough stages."""
    router = RuleRouter("configs/router.yaml")
    qv = compute_quality_vector(sample_bgr)
    action, reason = router.route_with_reason(qv)

    assert action in ENHANCEMENT_NAMES
    assert isinstance(reason, str) and len(reason) > 0

    enh_fn = get_enhancement(action)
    enhanced = enh_fn(sample_bgr)
    assert enhanced.shape == sample_bgr.shape
    assert enhanced.dtype == np.uint8

    fig_hist = comparison_histogram_figure(sample_bgr, enhanced, name=action.upper())
    assert isinstance(fig_hist, plt.Figure)
    plt.close(fig_hist)

    stages = get_hough_stage_images(enhanced, use_birds_eye=False)
    for expected_key in ["01_blur", "02_hls_mask", "03_canny", "04_roi", "05_hough", "08_cleaned_mask"]:
        assert expected_key in stages
        assert stages[expected_key] is not None

    fig_grid = lane_stages_grid_figure(stages)
    assert isinstance(fig_grid, plt.Figure)
    plt.close(fig_grid)


def test_panel3_region_analysis(sample_mask, sample_bgr):
    """Panel 3: Connected components analysis and bounding box / centroid generation."""
    stats_dict, color_labels, overlay = analyze_regions(sample_mask, bgr_frame=sample_bgr)

    assert "segment_count" in stats_dict
    assert stats_dict["segment_count"] >= 1
    assert "regions" in stats_dict

    reg = stats_dict["regions"][0]
    assert "region_id" in reg
    assert "area" in reg and reg["area"] > 0
    assert "perimeter" in reg
    assert "bbox" in reg and len(reg["bbox"]) == 4
    assert "centroid" in reg and len(reg["centroid"]) == 2

    assert color_labels.shape == sample_bgr.shape
    assert overlay.shape == sample_bgr.shape


def test_panel4_ai_stage(sample_bgr):
    """Panel 4: SMP U-Net model building, prediction, and visual overlays."""
    from src.models.unet import build_smp_unet, predict_drivable_unet

    # Random init mode for testing when checkpoint is not trained yet
    model, meta = build_smp_unet(checkpoint_path=None, allow_random_init=True)
    assert model is not None
    assert meta["architecture"] == "smp_unet_resnet18"

    prob_map, binary_mask = predict_drivable_unet(model, sample_bgr, meta=meta, device="cpu")

    assert prob_map.shape == sample_bgr.shape[:2]
    assert prob_map.dtype == np.float32 or prob_map.dtype == np.float64
    assert 0.0 <= prob_map.min() and prob_map.max() <= 1.0

    assert binary_mask.shape == sample_bgr.shape[:2]
    assert binary_mask.dtype == np.uint8
    assert set(np.unique(binary_mask)).issubset({0, 255})

    heatmap = draw_probability_heatmap(prob_map)
    assert heatmap.shape == sample_bgr.shape

    overlay = draw_drivable_overlay(sample_bgr, binary_mask)
    assert overlay.shape == sample_bgr.shape


def test_panel5_evaluation_tables():
    """Panel 5: Loading evaluation summary tables and plotting bar chart."""
    synth_path = Path("results/tables/summary_synthetic_psnr_ssim.csv")
    if synth_path.exists():
        df = pd.read_csv(synth_path)
        assert not df.empty

    demo_eval_df = pd.DataFrame({
        "condition": ["day_clear", "night", "rain"],
        "miou": [0.852, 0.791, 0.764],
    })
    fig_bar = evaluation_bar_chart_figure(demo_eval_df, group_col="condition", metric_col="miou")
    assert isinstance(fig_bar, plt.Figure)
    plt.close(fig_bar)


def test_panel6_failure_analysis(sample_bgr, sample_mask, tmp_path):
    """Panel 6: Worst frame thumbnails, mask overlay panel, and failure notes JSON."""
    records = [
        {"image_id": "test_001.jpg", "iou": 0.35, "condition": "night", "enhancement": "gamma", "img_path": ""},
        {"image_id": "test_002.jpg", "iou": 0.42, "condition": "rain", "enhancement": "dehaze", "img_path": ""},
    ]
    fig_fail = failure_panel(records, k=2)
    assert isinstance(fig_fail, plt.Figure)
    plt.close(fig_fail)

    fig_mask = mask_panel(sample_bgr, sample_mask, sample_mask, title="Test Mask Panel")
    assert isinstance(fig_mask, plt.Figure)
    plt.close(fig_mask)

    # Test failure notes JSON appending
    notes_file = tmp_path / "failure_notes.json"
    sample_notes = [{"image_id": "test_001.jpg", "note": "False negative on dark asphalt."}]
    with open(notes_file, "w", encoding="utf-8") as f:
        json.dump(sample_notes, f)

    with open(notes_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["image_id"] == "test_001.jpg"
