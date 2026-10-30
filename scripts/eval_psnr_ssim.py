#!/usr/bin/env python
"""
scripts/eval_psnr_ssim.py
-------------------------
Evaluation of Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)
on synthetic degradation sets (FR-03, FR-11).

Compares clean reference frames against:
  1. Synthetic degraded frames (fog mild, fog dense, low light)
  2. Enhanced frames (clahe, gamma, dehaze, denoise, denoise_clahe, adaptive)

Outputs:
  - Per-frame CSV: results/tables/synthetic_psnr_ssim.csv
  - Formatted condition summary table with bootstrap 95% CIs (1,000 resamples)

Usage
-----
    python scripts/eval_psnr_ssim.py
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

from src.data.degrade import apply_fog, apply_low_light
from src.enhance import get_enhancement, ENHANCEMENT_NAMES
from src.eval.metrics import psnr, ssim
from src.eval.bootstrap import bootstrap_ci
from src.quality.metrics import compute_quality
from src.router.rule import RuleRouter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PSNR/SSIM on synthetic degradation sets.")
    p.add_argument("--manifest", default="data/manifests/val.csv",
                   help="Manifest path to select clear reference frames")
    p.add_argument("--router_cfg", default="configs/router.yaml")
    p.add_argument("--out_csv", default="results/tables/synthetic_psnr_ssim.csv")
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap iterations")
    return p.parse_args()


def run_synthetic_eval(args: argparse.Namespace) -> pd.DataFrame:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path, comment="#")
    # Select clean day_clear frames to serve as ground-truth reference
    clean_rows = df[df["condition"] == "day_clear"]
    if clean_rows.empty:
        clean_rows = df.head(5)

    router = RuleRouter(args.router_cfg)

    enhancements_to_test = ["none", "clahe", "gamma", "dehaze", "denoise", "denoise_clahe", "adaptive"]

    records = []
    print("\n=========================================================================================")
    print("  Evaluating PSNR and SSIM on Synthetic Degradation Sets")
    print(f"  Reference clear frames: {len(clean_rows)}")
    print(f"  Enhancement candidates: {enhancements_to_test}")
    print("=========================================================================================\n")

    for idx, row in clean_rows.iterrows():
        img_path = Path(row["img_path"])
        ref_bgr = cv2.imread(str(img_path))
        if ref_bgr is None:
            continue

        ref_id = row["image_id"]

        # Synthetic degradation dictionary
        degradations = {
            "fog_mild": apply_fog(ref_bgr, beta=1.0, A=0.9, seed=42 + idx),
            "fog_dense": apply_fog(ref_bgr, beta=2.5, A=0.9, seed=42 + idx),
            "low_light": apply_low_light(ref_bgr, intensity_scale=0.25, gamma=1.8, seed=42 + idx),
        }

        for deg_name, deg_img in degradations.items():
            base_psnr = psnr(deg_img, ref_bgr)
            base_ssim = ssim(deg_img, ref_bgr)

            # Record unenhanced baseline
            records.append({
                "ref_id": ref_id,
                "degradation": deg_name,
                "enhancement": "none (degraded)",
                "psnr": base_psnr,
                "ssim": base_ssim,
                "delta_psnr": 0.0,
                "delta_ssim": 0.0,
            })

            # Evaluate each enhancement action
            qv = compute_quality(deg_img)
            adaptive_action = router.route(qv)

            for enh_name in ["clahe", "gamma", "dehaze", "denoise", "denoise_clahe", "adaptive"]:
                actual_action = adaptive_action if enh_name == "adaptive" else enh_name
                enh_fn = get_enhancement(actual_action)
                enh_img = enh_fn(deg_img)

                cur_psnr = psnr(enh_img, ref_bgr)
                cur_ssim = ssim(enh_img, ref_bgr)

                records.append({
                    "ref_id": ref_id,
                    "degradation": deg_name,
                    "enhancement": enh_name,
                    "psnr": cur_psnr,
                    "ssim": cur_ssim,
                    "delta_psnr": cur_psnr - base_psnr,
                    "delta_ssim": cur_ssim - base_ssim,
                })

    results_df = pd.DataFrame(records)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"Per-frame results saved to: {out_path} ({len(results_df)} evaluations)")

    # ── Summary Table with Bootstrap CIs (1,000 resamples) ─────────────────
    print("\n=========================================================================================")
    print(f"  SYNTHETIC ENHANCEMENT QUALITY SUMMARY (Bootstrap 95% CI, {args.n_boot} resamples)")
    print("=========================================================================================")
    header = f"{'Degradation':<12} | {'Enhancement':<16} | {'PSNR (dB)':^22} | {'SSIM':^22}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for (deg, enh), grp in results_df.groupby(["degradation", "enhancement"], sort=False):
        psnr_vals = grp["psnr"].values
        ssim_vals = grp["ssim"].values

        p_mean, p_lo, p_hi = bootstrap_ci(psnr_vals, n=args.n_boot, ci=0.95, seed=42)
        s_mean, s_lo, s_hi = bootstrap_ci(ssim_vals, n=args.n_boot, ci=0.95, seed=42)

        p_str = f"{p_mean:5.2f} [{p_lo:5.2f}, {p_hi:5.2f}]"
        s_str = f"{s_mean:5.3f} [{s_lo:5.3f}, {s_hi:5.3f}]"

        print(f"{deg:<12} | {enh:<16} | {p_str:^22} | {s_str:^22}")
        summary_rows.append({
            "degradation": deg,
            "enhancement": enh,
            "psnr_mean": p_mean,
            "psnr_lo": p_lo,
            "psnr_hi": p_hi,
            "ssim_mean": s_mean,
            "ssim_lo": s_lo,
            "ssim_hi": s_hi,
        })

    print("=" * len(header))
    summary_csv = out_path.parent / f"summary_{out_path.name}"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Summary saved to: {summary_csv}\n")

    return results_df


if __name__ == "__main__":
    args = parse_args()
    run_synthetic_eval(args)
