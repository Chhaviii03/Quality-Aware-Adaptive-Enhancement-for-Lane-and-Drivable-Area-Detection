"""
scripts/evaluate_final_test.py
------------------------------
Final held-out test evaluation on BDD100K test split (data/manifests/test.csv).

Compares 7 strategies:
  1. No enhancement (baseline)
  2. CLAHE
  3. Gamma correction
  4. Dehazing
  5. Denoising
  6. Denoising + CLAHE
  7. Frozen rule-based router (adaptive)

Computes for each:
  - Drivable-area IoU
  - Background IoU
  - mIoU
  - Dice
  - Delta vs baseline
  - Bootstrap 95% confidence intervals

Saves all outputs under results/final_test/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.enhance import ENHANCEMENT_NAMES, get_enhancement
from src.eval.bootstrap import bootstrap_ci, paired_diff_ci
from src.models.unet import build_smp_unet, predict_drivable_unet
from src.quality.metrics import compute_quality
from src.router.rule import RuleRouter
from src.viz.panels import mask_panel, _bgr2rgb


def compute_frame_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Compute per-frame IoU drivable, IoU background, mIoU, Dice."""
    p_fg = (pred_mask > 0)
    g_fg = (gt_mask > 0)
    p_bg = ~p_fg
    g_bg = ~g_fg

    inter_fg = np.logical_and(p_fg, g_fg).sum()
    union_fg = np.logical_or(p_fg, g_fg).sum()
    inter_bg = np.logical_and(p_bg, g_bg).sum()
    union_bg = np.logical_or(p_bg, g_bg).sum()

    iou_fg = float(inter_fg / union_fg) if union_fg > 0 else (1.0 if inter_fg == 0 else 0.0)
    iou_bg = float(inter_bg / union_bg) if union_bg > 0 else (1.0 if inter_bg == 0 else 0.0)
    frame_miou = (iou_fg + iou_bg) / 2.0

    fg_sum = p_fg.sum() + g_fg.sum()
    dice_fg = float(2.0 * inter_fg / fg_sum) if fg_sum > 0 else 1.0

    return {
        "iou_drivable": iou_fg,
        "iou_background": iou_bg,
        "miou": frame_miou,
        "dice": dice_fg,
        "inter_fg": int(inter_fg),
        "union_fg": int(union_fg),
        "inter_bg": int(inter_bg),
        "union_bg": int(union_bg),
        "p_fg_sum": int(p_fg.sum()),
        "g_fg_sum": int(g_fg.sum()),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / "results" / "final_test").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = (repo_root / "results" / "checkpoints" / "unet_drivable_best.pt").resolve()
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt_bytes = ckpt_path.read_bytes()
    ckpt_hash = hashlib.sha256(ckpt_bytes).hexdigest()
    ckpt_size = len(ckpt_bytes)

    print("=" * 70)
    print("FINAL HELD-OUT TEST EVALUATION (150 FRAMES)")
    print("=" * 70)
    print(f"Checkpoint Path:   {ckpt_path}")
    print(f"Checkpoint SHA256: {ckpt_hash}")
    print(f"Checkpoint Size:   {ckpt_size:,} bytes")
    print(f"Platform:          {platform.platform()}, Python {platform.python_version()}")
    print("-" * 70)

    # Load frozen model & router
    model, meta = build_smp_unet(str(ckpt_path), device="cpu")
    model.eval()
    router = RuleRouter(str((repo_root / "configs" / "router.yaml").resolve()))

    mean = np.array(meta.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(meta.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    manifest_p = (repo_root / "data" / "manifests" / "test.csv").resolve()
    test_df = pd.read_csv(manifest_p)
    n_frames = len(test_df)
    print(f"Loaded test split manifest: {manifest_p} ({n_frames} frames)")

    strategies = [
        "none",
        "clahe",
        "gamma",
        "dehaze",
        "denoise",
        "denoise_clahe",
        "adaptive_router",
    ]

    per_frame_records = []
    # Store predictions for qualitative analysis
    stored_preds = {s: {} for s in strategies}
    stored_images = {}
    stored_gts = {}

    t_start = time.time()
    print("Executing evaluation across all 7 strategies...")

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            fid = row["image_id"]
            img_p = row["img_path"]
            mask_p = row["mask_path"]

            bgr = cv2.imread(img_p)
            gt = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            stored_images[fid] = bgr
            stored_gts[fid] = gt

            # 1. Quality vector & router
            qv = compute_quality(bgr)
            r_action, r_reason = router.route_with_reason(qv)

            # 2. Batch forward pass for all 6 fixed enhancements
            enh_imgs = []
            enh_bgrs = {}
            for a in ENHANCEMENT_NAMES:
                fn = get_enhancement(a)
                enh = fn(bgr)
                enh_bgrs[a] = enh
                rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
                norm = (rgb.astype(np.float32) / 255.0 - mean) / std
                enh_imgs.append(norm.transpose(2, 0, 1))

            batch_t = torch.from_numpy(np.stack(enh_imgs)).float()
            logits = model(batch_t)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            # Record metrics for each fixed strategy
            frame_strat_metrics = {}
            for a_idx, a in enumerate(ENHANCEMENT_NAMES):
                mask = (probs[a_idx] > 0.5).astype(np.uint8) * 255
                stored_preds[a][fid] = mask
                m = compute_frame_metrics(mask, gt)
                frame_strat_metrics[a] = m

            # Router strategy takes prediction from chosen action
            stored_preds["adaptive_router"][fid] = stored_preds[r_action][fid]
            frame_strat_metrics["adaptive_router"] = frame_strat_metrics[r_action]

            # Collect record
            rec = {
                "image_id": fid,
                "condition": row.get("condition", "unlabeled"),
                "router_chosen_action": r_action,
                "router_reason": r_reason,
                "brightness": qv.brightness,
                "contrast": qv.contrast,
                "sharpness": qv.sharpness,
                "noise": qv.noise,
                "haze": qv.haze,
            }
            for s in strategies:
                sm = frame_strat_metrics[s]
                rec[f"{s}_iou_drivable"] = sm["iou_drivable"]
                rec[f"{s}_iou_background"] = sm["iou_background"]
                rec[f"{s}_miou"] = sm["miou"]
                rec[f"{s}_dice"] = sm["dice"]
                rec[f"{s}_inter_fg"] = sm["inter_fg"]
                rec[f"{s}_union_fg"] = sm["union_fg"]
                rec[f"{s}_inter_bg"] = sm["inter_bg"]
                rec[f"{s}_union_bg"] = sm["union_bg"]
                rec[f"{s}_p_fg_sum"] = sm["p_fg_sum"]
                rec[f"{s}_g_fg_sum"] = sm["g_fg_sum"]

            per_frame_records.append(rec)
            if (idx + 1) % 30 == 0 or (idx + 1) == n_frames:
                print(f"  Processed {idx + 1}/{n_frames} frames...")

    t_elapsed = time.time() - t_start
    print(f"Inference completed in {t_elapsed:.2f} s ({t_elapsed/n_frames*1000:.1f} ms/frame across 7 strategies).")

    df_per_frame = pd.DataFrame(per_frame_records)
    per_frame_csv = out_dir / "per_frame_metrics.csv"
    df_per_frame.to_csv(per_frame_csv, index=False)
    print(f"Saved per-frame metrics to: {per_frame_csv}")

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate Comparison Metrics
    # ─────────────────────────────────────────────────────────────────────────
    summary_rows = []
    base_miou_vals = df_per_frame["none_miou"].values
    base_dice_vals = df_per_frame["none_dice"].values

    base_tot_inter_fg = df_per_frame["none_inter_fg"].sum()
    base_tot_union_fg = df_per_frame["none_union_fg"].sum()
    base_tot_inter_bg = df_per_frame["none_inter_bg"].sum()
    base_tot_union_bg = df_per_frame["none_union_bg"].sum()
    base_ds_iou_fg = base_tot_inter_fg / base_tot_union_fg
    base_ds_iou_bg = base_tot_inter_bg / base_tot_union_bg
    base_ds_miou = (base_ds_iou_fg + base_ds_iou_bg) / 2.0
    base_ds_dice = 2.0 * base_tot_inter_fg / (df_per_frame["none_p_fg_sum"].sum() + df_per_frame["none_g_fg_sum"].sum())

    for s in strategies:
        # Dataset-level cumulative metrics
        tot_inter_fg = df_per_frame[f"{s}_inter_fg"].sum()
        tot_union_fg = df_per_frame[f"{s}_union_fg"].sum()
        tot_inter_bg = df_per_frame[f"{s}_inter_bg"].sum()
        tot_union_bg = df_per_frame[f"{s}_union_bg"].sum()
        tot_p_fg = df_per_frame[f"{s}_p_fg_sum"].sum()
        tot_g_fg = df_per_frame[f"{s}_g_fg_sum"].sum()

        ds_iou_fg = float(tot_inter_fg / tot_union_fg) if tot_union_fg > 0 else 1.0
        ds_iou_bg = float(tot_inter_bg / tot_union_bg) if tot_union_bg > 0 else 1.0
        ds_miou = (ds_iou_fg + ds_iou_bg) / 2.0
        ds_dice = float(2.0 * tot_inter_fg / (tot_p_fg + tot_g_fg)) if (tot_p_fg + tot_g_fg) > 0 else 1.0

        # Frame-level mean + bootstrap 95% CI
        miou_vals = df_per_frame[f"{s}_miou"].values
        dice_vals = df_per_frame[f"{s}_dice"].values

        m_mean, m_lo, m_hi = bootstrap_ci(miou_vals, n=1000, ci=0.95, seed=42)
        d_mean, d_lo, d_hi = bootstrap_ci(dice_vals, n=1000, ci=0.95, seed=42)

        # Paired delta vs baseline
        if s == "none":
            delta_miou = 0.0
            delta_dice = 0.0
            p_diff_mean, p_diff_lo, p_diff_hi = 0.0, 0.0, 0.0
        else:
            delta_miou = ds_miou - base_ds_miou
            delta_dice = ds_dice - base_ds_dice
            p_diff_mean, p_diff_lo, p_diff_hi = paired_diff_ci(miou_vals, base_miou_vals, n=1000, ci=0.95, seed=42)

        display_name = {
            "none": "1. Baseline (No Enhancement)",
            "clahe": "2. Fixed CLAHE",
            "gamma": "3. Fixed Gamma Correction",
            "dehaze": "4. Fixed Dehazing",
            "denoise": "5. Fixed Denoising",
            "denoise_clahe": "6. Fixed Denoise + CLAHE",
            "adaptive_router": "7. Proposed Adaptive Router",
        }[s]

        summary_rows.append({
            "strategy": s,
            "display_name": display_name,
            "dataset_iou_drivable": ds_iou_fg,
            "dataset_iou_background": ds_iou_bg,
            "dataset_miou": ds_miou,
            "dataset_dice": ds_dice,
            "delta_miou_vs_baseline": delta_miou,
            "delta_dice_vs_baseline": delta_dice,
            "frame_miou_mean": m_mean,
            "frame_miou_ci95": f"[{m_lo:.4f}, {m_hi:.4f}]",
            "frame_dice_mean": d_mean,
            "frame_dice_ci95": f"[{d_lo:.4f}, {d_hi:.4f}]",
            "paired_delta_miou_ci95": f"{p_diff_mean:+.4f} [{p_diff_lo:+.4f}, {p_diff_hi:+.4f}]",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary comparison table to: {summary_csv}")

    # ─────────────────────────────────────────────────────────────────────────
    # Print Formatted Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print("FINAL HELD-OUT TEST RESULTS (150 FRAMES)")
    print("=" * 105)
    header = f"{'Strategy':<30} | {'Drivable IoU':^13} | {'Bg IoU':^10} | {'Dataset mIoU':^13} | {'Dice':^10} | {'Delta mIoU':^11}"
    print(header)
    print("-" * len(header))
    for _, r in summary_df.iterrows():
        name = r["display_name"]
        fg_iou = f"{r['dataset_iou_drivable']:.4f}"
        bg_iou = f"{r['dataset_iou_background']:.4f}"
        ds_miou = f"{r['dataset_miou']:.4f}"
        dice = f"{r['dataset_dice']:.4f}"
        delta = f"{r['delta_miou_vs_baseline']:+.4f}" if r["strategy"] != "none" else "baseline"
        print(f"{name:<30} | {fg_iou:^13} | {bg_iou:^10} | {ds_miou:^13} | {dice:^10} | {delta:^11}")
    print("=" * len(header))

    # Router Action Distribution
    print("\nRouter Action Distribution on Test Set:")
    r_counts = df_per_frame["router_chosen_action"].value_counts()
    for act, cnt in r_counts.items():
        print(f"  {act:<15}: {cnt:>3} frames ({cnt/n_frames*100:>5.1f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # Representative Qualitative Figures
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating representative qualitative figures...")

    # 1. Largest improvement frame by router over baseline
    df_per_frame["router_gain"] = df_per_frame["adaptive_router_miou"] - df_per_frame["none_miou"]
    top_gain = df_per_frame.sort_values(by="router_gain", ascending=False).iloc[0]
    top_gain_id = top_gain["image_id"]
    top_gain_act = top_gain["router_chosen_action"]

    bgr_gain = stored_images[top_gain_id]
    gt_gain = stored_gts[top_gain_id]
    pred_base = stored_preds["none"][top_gain_id]
    pred_router = stored_preds["adaptive_router"][top_gain_id]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    axes[0].imshow(_bgr2rgb(bgr_gain)); axes[0].set_title(f"Input ({top_gain_id[:12]})"); axes[0].axis("off")
    axes[1].imshow(gt_gain, cmap="gray"); axes[1].set_title("Ground Truth Mask"); axes[1].axis("off")
    axes[2].imshow(pred_base, cmap="gray"); axes[2].set_title(f"Baseline (None) IoU={top_gain['none_iou_drivable']:.3f}"); axes[2].axis("off")
    axes[3].imshow(pred_router, cmap="gray"); axes[3].set_title(f"Router ({top_gain_act}) IoU={top_gain['adaptive_router_iou_drivable']:.3f}"); axes[3].axis("off")
    fig.suptitle(f"Top Improvement: Router Gain = +{top_gain['router_gain']:.4f} mIoU", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "qualitative_top_gain.png", dpi=150)
    plt.close(fig)

    # 2. Typical Gamma Correction frame (low light / dark condition)
    gamma_frames = df_per_frame[df_per_frame["router_chosen_action"] == "gamma"]
    if not gamma_frames.empty:
        gamma_sample = gamma_frames.iloc[0]
        gid = gamma_sample["image_id"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        axes[0].imshow(_bgr2rgb(stored_images[gid])); axes[0].set_title("Input Frame (Low Brightness)"); axes[0].axis("off")
        axes[1].imshow(stored_gts[gid], cmap="gray"); axes[1].set_title("Ground Truth Mask"); axes[1].axis("off")
        axes[2].imshow(stored_preds["none"][gid], cmap="gray"); axes[2].set_title(f"Baseline IoU={gamma_sample['none_iou_drivable']:.3f}"); axes[2].axis("off")
        axes[3].imshow(stored_preds["gamma"][gid], cmap="gray"); axes[3].set_title(f"Gamma IoU={gamma_sample['gamma_iou_drivable']:.3f}"); axes[3].axis("off")
        fig.suptitle(f"Router Decision: Gamma Enhancement (L={gamma_sample['brightness']:.1f})", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "qualitative_gamma_decision.png", dpi=150)
        plt.close(fig)

    # 3. Clean daylight frame where Router correctly chose 'none'
    none_frames = df_per_frame[df_per_frame["router_chosen_action"] == "none"]
    if not none_frames.empty:
        none_sample = none_frames.sort_values(by="none_iou_drivable", ascending=False).iloc[0]
        nid = none_sample["image_id"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        axes[0].imshow(_bgr2rgb(stored_images[nid])); axes[0].set_title("Clear Input Frame"); axes[0].axis("off")
        axes[1].imshow(stored_gts[nid], cmap="gray"); axes[1].set_title("Ground Truth Mask"); axes[1].axis("off")
        axes[2].imshow(stored_preds["none"][nid], cmap="gray"); axes[2].set_title(f"Baseline IoU={none_sample['none_iou_drivable']:.3f}"); axes[2].axis("off")
        axes[3].imshow(stored_preds["adaptive_router"][nid], cmap="gray"); axes[3].set_title(f"Router (None) IoU={none_sample['adaptive_router_iou_drivable']:.3f}"); axes[3].axis("off")
        fig.suptitle("Router Decision: None (Clean Road Preserved)", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "qualitative_clean_none.png", dpi=150)
        plt.close(fig)

    # 4. Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_labels = [s.replace("_", " ").title() for s in strategies]
    y_vals = summary_df["dataset_miou"].values
    bars = ax.bar(x_labels, y_vals, color=["#708090", "#4682B4", "#5F9EA0", "#6495ED", "#4169E1", "#1E90FF", "#2E8B57"], width=0.55)
    ax.set_ylim(0.70, 0.95)
    ax.set_ylabel("Dataset mIoU", fontsize=11)
    ax.set_title("Comparative Evaluation Across Strategies on Held-Out Test Split (150 Frames)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005, f"{h:.4f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_comparison_barchart.png", dpi=150)
    plt.close(fig)

    print(f"Saved qualitative figures to: {fig_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save Metadata
    # ─────────────────────────────────────────────────────────────────────────
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_size_bytes": ckpt_size,
        "test_manifest": str(manifest_p),
        "n_frames_evaluated": n_frames,
        "total_inference_time_seconds": round(t_elapsed, 2),
        "router_rules_source": "configs/router.yaml",
        "enhance_params_source": "configs/enhance.yaml",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    with open(out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved experiment metadata to: {out_dir / 'experiment_metadata.json'}")
    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()

