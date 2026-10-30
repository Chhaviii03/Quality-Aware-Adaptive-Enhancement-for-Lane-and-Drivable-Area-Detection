"""
src/eval
--------
Evaluation package:
  - metrics: iou, road_iou, dice, road_dice, miou, lane_f1, psnr, ssim
  - writer: PerFrameCSVWriter, write_per_frame_csv
  - bootstrap: bootstrap_ci, paired_diff_ci, aggregate_by_condition
  - run_eval: run_eval
"""

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
from src.eval.writer import (
    PerFrameCSVWriter,
    write_per_frame_csv,
)
from src.eval.bootstrap import (
    bootstrap_ci,
    paired_diff_ci,
    aggregate_by_condition,
)

__all__ = [
    "iou",
    "road_iou",
    "dice",
    "road_dice",
    "miou",
    "lane_f1",
    "psnr",
    "ssim",
    "PerFrameCSVWriter",
    "write_per_frame_csv",
    "bootstrap_ci",
    "paired_diff_ci",
    "aggregate_by_condition",
]
