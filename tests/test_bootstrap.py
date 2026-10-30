"""
tests/test_bootstrap.py
-----------------------
Unit tests for bootstrap confidence intervals and per-frame CSV writer.

Run with:
    pytest tests/test_bootstrap.py -v
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.eval.bootstrap import bootstrap_ci, paired_diff_ci, aggregate_by_condition
from src.eval.writer import PerFrameCSVWriter, write_per_frame_csv


class TestBootstrapCI:
    def test_bootstrap_ci_coverage(self):
        np.random.seed(42)
        # Normal distribution with mean 50.0
        data = np.random.normal(loc=50.0, scale=5.0, size=100)
        mean, lo, hi = bootstrap_ci(data, n=1000, ci=0.95, seed=42)
        assert lo < mean < hi
        assert 48.0 < mean < 52.0

    def test_paired_bootstrap_ci(self):
        np.random.seed(42)
        # Method A is consistently better than B by ~5 units
        b = np.random.normal(loc=50.0, scale=3.0, size=100)
        a = b + np.random.normal(loc=5.0, scale=1.0, size=100)

        # Using paired_with argument
        diff_mean, lo, hi = bootstrap_ci(a, paired_with=b, n=1000, ci=0.95, seed=42)
        assert lo < diff_mean < hi
        assert 4.0 < diff_mean < 6.0
        assert lo > 0  # Statistically significant improvement

        # Using paired_diff_ci directly
        p_mean, p_lo, p_hi = paired_diff_ci(a, b, n=1000, ci=0.95, seed=42)
        assert p_mean == pytest.approx(diff_mean)
        assert p_lo == pytest.approx(lo)
        assert p_hi == pytest.approx(hi)

    def test_aggregate_by_condition(self):
        df = pd.DataFrame({
            "condition": ["day_clear"] * 50 + ["night"] * 50,
            "lane_f1": np.concatenate([np.random.uniform(0.7, 0.9, 50), np.random.uniform(0.4, 0.6, 50)]),
            "road_iou": np.concatenate([np.random.uniform(0.8, 0.95, 50), np.random.uniform(0.6, 0.75, 50)]),
        })
        agg = aggregate_by_condition(df, metric_cols=["lane_f1", "road_iou"], n=1000)
        assert len(agg) == 2
        assert "lane_f1_mean" in agg.columns
        assert "lane_f1_lo" in agg.columns
        assert "lane_f1_hi" in agg.columns
        assert "road_iou_mean" in agg.columns


class TestPerFrameCSVWriter:
    def test_writer_save_and_load(self, tmp_path):
        out_csv = tmp_path / "test_eval.csv"
        writer = PerFrameCSVWriter(out_csv)
        writer.add_record(
            frame_id="frame_001.jpg",
            condition="day_clear",
            road_iou=0.85,
            road_dice=0.92,
            lane_f1=0.78,
            miou=0.81,
            enhancement="none",
            ms_quality=1.2,
            ms_router=0.1,
            ms_enhance=0.0,
            ms_model=12.5,
            brightness=110.0,
        )
        saved_path = writer.save()
        assert saved_path.exists()

        df = pd.read_csv(saved_path)
        assert len(df) == 1
        assert df.iloc[0]["frame_id"] == "frame_001.jpg"
        assert df.iloc[0]["road_iou"] == pytest.approx(0.85)
        assert df.iloc[0]["lane_f1"] == pytest.approx(0.78)
        assert df.iloc[0]["miou"] == pytest.approx(0.81)

    def test_write_per_frame_csv_helper(self, tmp_path):
        out_csv = tmp_path / "helper_eval.csv"
        records = [
            {
                "frame_id": "f1.jpg",
                "condition": "rain",
                "road_iou": 0.70,
                "road_dice": 0.82,
                "lane_f1": 0.65,
                "miou": 0.675,
                "enhancement": "clahe",
            }
        ]
        df = write_per_frame_csv(records, out_csv)
        assert len(df) == 1
        assert out_csv.exists()
