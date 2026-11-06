#!/usr/bin/env python
"""
scripts/evaluate.py
-------------------
Evaluation script for U-Net drivable area segmentation on VAL and TEST splits.

Usage:
    python scripts/evaluate.py --split val --ckpt results/checkpoints/unet/unet_drivable_best.pt
    python scripts/evaluate.py --split test --ckpt results/checkpoints/unet/unet_drivable_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from src.models.unet import build_smp_unet
from src.enhance import get_enhancement
from src.quality.metrics import compute_quality
from src.router.rule import RuleRouter


def compute_metrics(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> dict[str, float]:
    """Compute dataset-level cumulative micro IoU and Dice."""
    tot_inter_fg = 0
    tot_union_fg = 0
    tot_inter_bg = 0
    tot_union_bg = 0
    tot_p_fg = 0
    tot_g_fg = 0

    frame_mious = []
    frame_dices = []

    for p, g in zip(pred_masks, gt_masks):
        p_fg = (p > 0)
        g_fg = (g > 0)
        p_bg = ~p_fg
        g_bg = ~g_fg

        inter_fg = np.logical_and(p_fg, g_fg).sum()
        union_fg = np.logical_or(p_fg, g_fg).sum()
        inter_bg = np.logical_and(p_bg, g_bg).sum()
        union_bg = np.logical_or(p_bg, g_bg).sum()

        tot_inter_fg += inter_fg
        tot_union_fg += union_fg
        tot_inter_bg += inter_bg
        tot_union_bg += union_bg

        p_fg_sum = p_fg.sum()
        g_fg_sum = g_fg.sum()
        tot_p_fg += p_fg_sum
        tot_g_fg += g_fg_sum

        f_iou_fg = float(inter_fg / union_fg) if union_fg > 0 else (1.0 if inter_fg == 0 else 0.0)
        f_iou_bg = float(inter_bg / union_bg) if union_bg > 0 else (1.0 if inter_bg == 0 else 0.0)
        f_miou = (f_iou_fg + f_iou_bg) / 2.0
        f_dice = float(2.0 * inter_fg / (p_fg_sum + g_fg_sum)) if (p_fg_sum + g_fg_sum) > 0 else 1.0

        frame_mious.append(f_miou)
        frame_dices.append(f_dice)

    ds_iou_fg = float(tot_inter_fg / tot_union_fg) if tot_union_fg > 0 else 1.0
    ds_iou_bg = float(tot_inter_bg / tot_union_bg) if tot_union_bg > 0 else 1.0
    ds_miou = (ds_iou_fg + ds_iou_bg) / 2.0
    ds_dice = float(2.0 * tot_inter_fg / (tot_p_fg + tot_g_fg)) if (tot_p_fg + tot_g_fg) > 0 else 1.0

    return {
        "dataset_iou_drivable": ds_iou_fg,
        "dataset_iou_background": ds_iou_bg,
        "dataset_miou": ds_miou,
        "dataset_dice": ds_dice,
        "frame_miou_mean": float(np.mean(frame_mious)),
        "frame_dice_mean": float(np.mean(frame_dices)),
        "frame_mious": frame_mious,
        "frame_dices": frame_dices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on validation or test split.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--ckpt", default="results/checkpoints/unet/unet_drivable_best.pt", help="Path to checkpoint")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames evaluated")
    parser.add_argument("--model", default="unet", choices=["unet", "hough"], help="Model type")
    parser.add_argument("--router_cfg", default="configs/router.yaml", help="Path to router config")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify checkpoint path
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        # Fallback to results/checkpoints/unet_drivable_best.pt if exists
        alt_path = Path("results/checkpoints/unet_drivable_best.pt")
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            print(f"Error: checkpoint not found at '{ckpt_path}' or '{alt_path}'")
            sys.exit(1)

    print("=" * 65)
    print(f"EVALUATION ON {args.split.upper()} SPLIT")
    print("=" * 65)
    print(f"Checkpoint Path: {ckpt_path}")
    print(f"Device:          {args.device}")

    # Load raw checkpoint dictionary to inspect stored metrics
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt_raw.get("epoch", "N/A")
    stored_val = ckpt_raw.get("val", {})
    print(f"Trained Epoch:   {epoch}")
    print(f"Stored Val Metrics in Checkpoint: {stored_val}")
    print("-" * 65)

    # Build model
    model, meta = build_smp_unet(str(ckpt_path), device=args.device)
    model.eval()

    mean = np.array(meta.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(meta.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    # Load manifest
    manifest_path = Path(f"data/manifests/{args.split}.csv")
    if not manifest_path.exists():
        print(f"Error: manifest not found at '{manifest_path}'")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    if args.limit:
        df = df.head(args.limit)
    n_frames = len(df)
    print(f"Loaded {n_frames} frames from {manifest_path}")

    # Router setup
    router = RuleRouter(args.router_cfg)

    # Run inference
    pred_masks_raw = []
    pred_masks_router = []
    gt_masks = []
    frame_records = []

    t_start = time.time()
    with torch.no_grad():
        for idx, row in df.iterrows():
            fid = row["image_id"]
            img_p = row["img_path"]
            mask_p = row["mask_path"]
            cond = row.get("condition", "unlabeled")

            bgr = cv2.imread(img_p)
            gt = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if bgr is None or gt is None:
                continue

            gt_masks.append(gt)

            # Raw image inference (baseline)
            rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            norm_raw = (rgb_raw.astype(np.float32) / 255.0 - mean) / std
            t_raw = torch.from_numpy(norm_raw.transpose(2, 0, 1)).unsqueeze(0).float().to(args.device)
            logit_raw = model(t_raw)
            prob_raw = torch.sigmoid(logit_raw).squeeze().cpu().numpy()
            mask_raw = (prob_raw > 0.5).astype(np.uint8) * 255
            pred_masks_raw.append(mask_raw)

            # Router image inference
            qv = compute_quality(bgr)
            action, reason = router.route_with_reason(qv)
            enh_fn = get_enhancement(action)
            enh_bgr = enh_fn(bgr)
            rgb_enh = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
            norm_enh = (rgb_enh.astype(np.float32) / 255.0 - mean) / std
            t_enh = torch.from_numpy(norm_enh.transpose(2, 0, 1)).unsqueeze(0).float().to(args.device)
            logit_enh = model(t_enh)
            prob_enh = torch.sigmoid(logit_enh).squeeze().cpu().numpy()
            mask_enh = (prob_enh > 0.5).astype(np.uint8) * 255
            pred_masks_router.append(mask_enh)

            # Per-frame metric for test predictions
            p_fg = mask_raw > 0
            g_fg = gt > 0
            inter = np.logical_and(p_fg, g_fg).sum()
            union = np.logical_or(p_fg, g_fg).sum()
            iou_f = float(inter / union) if union > 0 else 1.0
            dice_f = float(2.0 * inter / (p_fg.sum() + g_fg.sum())) if (p_fg.sum() + g_fg.sum()) > 0 else 1.0

            frame_records.append({
                "image_id": fid,
                "condition": cond,
                "iou": iou_f,
                "dice": dice_f,
                "router_action": action,
                "img_path": img_p,
                "mask_path": mask_p,
            })

    elapsed = time.time() - t_start
    print(f"Inference completed in {elapsed:.2f} s ({elapsed / n_frames * 1000:.1f} ms/frame)")

    # Compute metrics
    metrics_raw = compute_metrics(pred_masks_raw, gt_masks)
    metrics_router = compute_metrics(pred_masks_router, gt_masks)

    print("\n" + "=" * 65)
    print(f"RESULTS FOR {args.split.upper()} SPLIT ({n_frames} frames)")
    print("=" * 65)
    print(f"Baseline (None):")
    print(f"  Dataset mIoU:     {metrics_raw['dataset_miou']:.6f}")
    print(f"  Drivable IoU:     {metrics_raw['dataset_iou_drivable']:.6f}")
    print(f"  Background IoU:   {metrics_raw['dataset_iou_background']:.6f}")
    print(f"  Dataset Dice:     {metrics_raw['dataset_dice']:.6f}")
    print(f"  Frame Mean mIoU:  {metrics_raw['frame_miou_mean']:.6f}")
    print("-" * 65)
    print(f"Adaptive Router:")
    print(f"  Dataset mIoU:     {metrics_router['dataset_miou']:.6f}")
    print(f"  Drivable IoU:     {metrics_router['dataset_iou_drivable']:.6f}")
    print(f"  Background IoU:   {metrics_router['dataset_iou_background']:.6f}")
    print(f"  Dataset Dice:     {metrics_router['dataset_dice']:.6f}")
    print(f"  Frame Mean mIoU:  {metrics_router['frame_miou_mean']:.6f}")
    print("-" * 65)

    if args.split == "val" and stored_val:
        stored_miou = stored_val.get("miou", None)
        stored_dice = stored_val.get("dice", None)
        local_miou = metrics_raw["dataset_miou"]
        local_dice = metrics_raw["dataset_dice"]

        print("VAL CHECKPOINT VERIFICATION COMPARISON:")
        print(f"  Stored Colab mIoU: {stored_miou:.6f}" if stored_miou is not None else "  Stored mIoU: N/A")
        print(f"  Local Val mIoU:    {local_miou:.6f}")
        diff_miou = abs(local_miou - stored_miou) if stored_miou is not None else 0.0
        print(f"  Difference (mIoU): {diff_miou:.6f}")
        print(f"  Stored Colab Dice: {stored_dice:.6f}" if stored_dice is not None else "  Stored Dice: N/A")
        print(f"  Local Val Dice:    {local_dice:.6f}")
        diff_dice = abs(local_dice - stored_dice) if stored_dice is not None else 0.0
        print(f"  Difference (Dice): {diff_dice:.6f}")
        print("-" * 65)

        if diff_miou > 0.01:
            print("WARNING: Difference exceeds 0.01!")
        else:
            print("SANITY CHECK PASSED: Difference <= 0.01")

    # If test split, write tables that dashboard reads
    if args.split == "test":
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)

        # 1. summary_test_metrics.csv
        summary_rows = [
            {
                "strategy": "none",
                "condition": "all",
                "display_name": "Baseline (None)",
                "miou": metrics_raw["dataset_miou"],
                "dice": metrics_raw["dataset_dice"],
                "iou_drivable": metrics_raw["dataset_iou_drivable"],
                "iou_background": metrics_raw["dataset_iou_background"],
            },
            {
                "strategy": "adaptive_router",
                "condition": "all",
                "display_name": "Proposed Adaptive Router",
                "miou": metrics_router["dataset_miou"],
                "dice": metrics_router["dataset_dice"],
                "iou_drivable": metrics_router["dataset_iou_drivable"],
                "iou_background": metrics_router["dataset_iou_background"],
            },
        ]
        sum_df = pd.DataFrame(summary_rows)
        sum_p = tables_dir / "summary_test_metrics.csv"
        sum_df.to_csv(sum_p, index=False)
        print(f"Wrote dashboard table: {sum_p}")

        # 2. test_predictions.csv
        pred_df = pd.DataFrame(frame_records)
        pred_p = tables_dir / "test_predictions.csv"
        pred_df.to_csv(pred_p, index=False)
        print(f"Wrote dashboard table: {pred_p}")


if __name__ == "__main__":
    main()

