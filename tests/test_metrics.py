"""
tests/test_metrics.py
---------------------
Unit tests for evaluation metrics (src/eval/metrics.py).

Run with:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from src.eval.metrics import (
    iou,
    road_iou,
    dice,
    road_dice,
    miou,
    lane_f1,
    psnr,
    ssim,
)


def _ones(shape=(352, 640)):
    return np.full(shape, 255, dtype=np.uint8)

def _zeros(shape=(352, 640)):
    return np.zeros(shape, dtype=np.uint8)


class TestRoadIoU:
    def test_perfect(self):
        m = _ones()
        assert road_iou(m, m) == pytest.approx(1.0)

    def test_zero(self):
        assert road_iou(_ones(), _zeros()) == pytest.approx(0.0)

    def test_both_empty(self):
        assert road_iou(_zeros(), _zeros()) == pytest.approx(1.0)

    def test_partial(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        gt   = np.zeros((100, 100), dtype=np.uint8)
        pred[:50, :] = 255
        gt[25:75, :] = 255
        val = road_iou(pred, gt)
        assert 0.0 < val < 1.0


class TestRoadDice:
    def test_perfect(self):
        m = _ones()
        assert road_dice(m, m) == pytest.approx(1.0)

    def test_zero(self):
        assert road_dice(_ones(), _zeros()) == pytest.approx(0.0)

    def test_both_empty(self):
        assert road_dice(_zeros(), _zeros()) == pytest.approx(1.0)


class TestLaneF1:
    def test_perfect(self):
        m = _ones()
        assert lane_f1(m, m) == pytest.approx(1.0)

    def test_no_pred_no_gt(self):
        assert lane_f1(_zeros(), _zeros()) == pytest.approx(1.0)

    def test_no_pred(self):
        assert lane_f1(_zeros(), _ones()) == pytest.approx(0.0)

    def test_no_gt(self):
        assert lane_f1(_ones(), _zeros()) == pytest.approx(0.0)

    def test_within_tolerance(self):
        """A prediction shifted by 3 px (< 5 px tol) should give high F1."""
        gt   = np.zeros((100, 200), dtype=np.uint8)
        pred = np.zeros((100, 200), dtype=np.uint8)
        gt  [:, 100]   = 255   # thin vertical line at col 100
        pred[:, 103]   = 255   # shifted 3 px right
        score = lane_f1(pred, gt, tol=5)
        assert score > 0.8, f"Expected F1 > 0.8, got {score:.3f}"

    def test_outside_tolerance(self):
        """Shift of 10 px (> 5 px tol) should give low F1."""
        gt   = np.zeros((100, 200), dtype=np.uint8)
        pred = np.zeros((100, 200), dtype=np.uint8)
        gt  [:, 100] = 255
        pred[:, 115] = 255
        score = lane_f1(pred, gt, tol=5)
        assert score < 0.3, f"Expected F1 < 0.3, got {score:.3f}"


class TestPSNR:
    def test_identical(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert psnr(img, img) == float("inf")

    def test_different(self):
        ref  = np.zeros((100, 100, 3), dtype=np.uint8)
        noisy = np.full((100, 100, 3), 10, dtype=np.uint8)
        val = psnr(noisy, ref)
        assert 0 < val < 100


class TestSSIM:
    def test_identical(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-3)

    def test_different(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.full((100, 100, 3), 255, dtype=np.uint8)
        val = ssim(a, b)
        assert val < 0.5


class TestIoUAndDiceAliases:
    def test_iou_alias(self):
        m = _ones()
        assert iou(m, m) == road_iou(m, m)

    def test_dice_alias(self):
        m = _ones()
        assert dice(m, m) == road_dice(m, m)


class TestMIoU:
    def test_miou_list(self):
        m1 = _ones()
        z1 = _zeros()
        # IoU(m1, m1) = 1.0, IoU(z1, m1) = 0.0 -> mean = 0.5
        val = miou([m1, z1], [m1, m1])
        assert val == pytest.approx(0.5)

    def test_miou_dict(self):
        m1 = _ones()
        z1 = _zeros()
        preds = {"road": m1, "lane": z1}
        gts   = {"road": m1, "lane": m1}
        val = miou(preds, gts)
        assert val == pytest.approx(0.5)
