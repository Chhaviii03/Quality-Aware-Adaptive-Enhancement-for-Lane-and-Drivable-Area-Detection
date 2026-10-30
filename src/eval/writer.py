"""
src/eval/writer.py
------------------
Per-frame evaluation results CSV writer (FR-11).

Logs per-frame predictions and metrics:
    frame_id, condition, road_iou, road_dice, lane_f1, miou,
    enhancement, ms_quality, ms_router, ms_enhance, ms_model,
    brightness, contrast, sharpness, noise, haze
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import pandas as pd


STANDARD_COLUMNS = [
    "frame_id",
    "condition",
    "road_iou",
    "road_dice",
    "lane_f1",
    "miou",
    "enhancement",
    "ms_quality",
    "ms_router",
    "ms_enhance",
    "ms_model",
]


class PerFrameCSVWriter:
    """Buffers and writes evaluation results on a per-frame basis."""

    def __init__(self, out_path: str | Path | None = None) -> None:
        self.out_path = Path(out_path) if out_path is not None else None
        self.records: list[dict[str, Any]] = []

    def add_record(
        self,
        frame_id: str,
        condition: str,
        road_iou: float,
        road_dice: float,
        lane_f1: float,
        miou: float,
        enhancement: str,
        ms_quality: float = 0.0,
        ms_router: float = 0.0,
        ms_enhance: float = 0.0,
        ms_model: float = 0.0,
        **extra_metrics: Any,
    ) -> None:
        """Add a single frame evaluation result."""
        rec = {
            "frame_id": frame_id,
            "condition": condition,
            "road_iou": float(road_iou),
            "road_dice": float(road_dice),
            "lane_f1": float(lane_f1),
            "miou": float(miou),
            "enhancement": str(enhancement),
            "ms_quality": float(ms_quality),
            "ms_router": float(ms_router),
            "ms_enhance": float(ms_enhance),
            "ms_model": float(ms_model),
            **extra_metrics,
        }
        self.records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to pandas DataFrame with proper column ordering."""
        df = pd.DataFrame(self.records)
        if df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        # Ensure standard columns appear first, then any extra columns
        existing_std = [c for c in STANDARD_COLUMNS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in STANDARD_COLUMNS]
        return df[existing_std + extra_cols]

    def save(self, out_path: str | Path | None = None) -> Path:
        """Write records to CSV.

        Parameters
        ----------
        out_path : str | Path | None
            Destination path. Defaults to `self.out_path`.

        Returns
        -------
        Path where CSV was saved.
        """
        target = Path(out_path) if out_path is not None else self.out_path
        if target is None:
            raise ValueError("No output path specified for PerFrameCSVWriter.")

        target.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(target, index=False)
        return target

    def __len__(self) -> int:
        return len(self.records)


def write_per_frame_csv(
    records: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> pd.DataFrame:
    """Convenience helper to write a sequence of per-frame records to CSV.

    Parameters
    ----------
    records : Sequence[dict[str, Any]]
        List of per-frame result dictionaries.
    out_path : str | Path
        Output CSV file path.

    Returns
    -------
    pd.DataFrame containing the written records.
    """
    writer = PerFrameCSVWriter(out_path)
    for rec in records:
        writer.records.append(dict(rec))
    writer.save()
    return writer.to_dataframe()
