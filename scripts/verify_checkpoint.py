"""
scripts/verify_checkpoint.py
----------------------------
Verifies the trained U-Net checkpoint against stored Colab validation metrics.

Loads results/checkpoints/unet_drivable_best.pt, prints checkpoint metadata,
runs local inference across data/manifests/val.csv, and checks that local mIoU
matches the stored Colab mIoU within 0.01.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.models.unet import build_smp_unet, predict_drivable_unet


def compute_eval_metrics(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]):
    """Compute dataset-level IoU drivable, IoU background, mIoU, and Dice."""
    total_inter_fg = 0
    total_union_fg = 0
    total_inter_bg = 0
    total_union_bg = 0
    total_pred_fg = 0
    total_gt_fg = 0

    for p, g in zip(pred_masks, gt_masks):
        p_fg = (p > 0)
        g_fg = (g > 0)
        p_bg = ~p_fg
        g_bg = ~g_fg

        inter_fg = np.logical_and(p_fg, g_fg).sum()
        union_fg = np.logical_or(p_fg, g_fg).sum()
        inter_bg = np.logical_and(p_bg, g_bg).sum()
        union_bg = np.logical_or(p_bg, g_bg).sum()

        total_inter_fg += inter_fg
        total_union_fg += union_fg
        total_inter_bg += inter_bg
        total_union_bg += union_bg

        total_pred_fg += p_fg.sum()
        total_gt_fg += g_fg.sum()

    iou_fg = float(total_inter_fg / total_union_fg) if total_union_fg > 0 else 1.0
    iou_bg = float(total_inter_bg / total_union_bg) if total_union_bg > 0 else 1.0
    miou = (iou_fg + iou_bg) / 2.0
    dice = float(2.0 * total_inter_fg / (total_pred_fg + total_gt_fg)) if (total_pred_fg + total_gt_fg) > 0 else 1.0

    return {
        "iou_drivable": iou_fg,
        "iou_background": iou_bg,
        "miou": miou,
        "dice": dice,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify U-Net checkpoint")
    parser.add_argument("--checkpoint", default="results/checkpoints/unet_drivable_best.pt")
    parser.add_argument("--manifest", default="data/manifests/val.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"FAIL: Checkpoint not found at '{ckpt_path}'")
        sys.exit(1)

    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt_raw.get("epoch", "N/A")
    config = ckpt_raw.get("config", {})
    stored_val = ckpt_raw.get("val", {})

    print("=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)
    print(f"Checkpoint Path: {ckpt_path}")
    print(f"Trained Epoch:   {epoch}")
    print(f"Stored Config:   {config}")
    print(f"Stored Colab Val Metrics: {stored_val}")
    print("-" * 60)

    model, meta = build_smp_unet(checkpoint_path=str(ckpt_path), device=args.device)

    df_val = pd.read_csv(args.manifest)
    if args.limit:
        df_val = df_val.head(args.limit)

    pred_masks = []
    gt_masks = []
    times = []

    print(f"Running inference on {len(df_val)} frames...")
    for idx, row in df_val.iterrows():
        bgr = cv2.imread(row["img_path"])
        gt = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

        t0 = time.perf_counter()
        prob, mask = predict_drivable_unet(model, bgr, meta=meta, device=args.device)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)
        pred_masks.append(mask)
        gt_masks.append(gt)

    local_metrics = compute_eval_metrics(pred_masks, gt_masks)
    avg_ms = np.mean(times)

    print("-" * 60)
    print("COMPARISON: Stored Colab Metrics vs. Locally Computed Metrics")
    print("-" * 60)
    print(f"{'Metric':<20} {'Stored Colab':<15} {'Locally Computed':<15} {'Difference':<15}")
    print("-" * 60)
    for m in ["miou", "iou_drivable", "iou_background", "dice"]:
        s_val = stored_val.get(m, float("nan"))
        l_val = local_metrics.get(m, float("nan"))
        diff = abs(s_val - l_val)
        print(f"{m:<20} {s_val:<15.4f} {l_val:<15.4f} {diff:<15.4f}")

    print("-" * 60)
    print(f"Mean CPU Inference Time per frame: {avg_ms:.2f} ms ({1000.0/avg_ms:.1f} FPS)")

    diff_miou = abs(stored_val.get("miou", 0.0) - local_metrics["miou"])
    if diff_miou < 0.01:
        print(f"\nRESULT: PASS (mIoU diff {diff_miou:.5f} < 0.01)")
        sys.exit(0)
    else:
        print(f"\nRESULT: FAIL (mIoU diff {diff_miou:.5f} >= 0.01)")
        sys.exit(1)


if __name__ == "__main__":
    main()

