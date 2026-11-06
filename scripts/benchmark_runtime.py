#!/usr/bin/env python
"""
scripts/benchmark_runtime.py
----------------------------
Benchmarks the CPU runtime of each stage in the perception pipeline:
  1. Quality Scoring (5-D vector)
  2. Rule Router
  3. DIP Enhancement (CLAHE, Gamma, Dehaze, Denoise)
  4. Deep U-Net Segmentation (ResNet-18)
  5. Classical Hough Lane Detection
  6. Connected-Component Region Analysis

Computes per-stage milliseconds (ms) and total pipeline FPS on CPU.
Outputs results/tables/runtime_benchmarks.csv for the Streamlit dashboard.

Usage:
    python scripts/benchmark_runtime.py --manifest data/manifests/val.csv --num_frames 50
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

from src.models.unet import build_smp_unet, predict_drivable_unet
from src.enhance import get_enhancement, ENHANCEMENT_NAMES
from src.quality.metrics import compute_quality
from src.router.rule import RuleRouter
from src.classical.lane_hough import detect_lanes
from src.classical.region import analyze_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark perception pipeline stage latencies on CPU.")
    parser.add_argument("--manifest", default="data/manifests/val.csv", help="Path to manifest")
    parser.add_argument("--ckpt", default="results/checkpoints/unet/unet_drivable_best.pt", help="Path to checkpoint")
    parser.add_argument("--router_cfg", default="configs/router.yaml", help="Path to router config")
    parser.add_argument("--device", default="cpu", help="Device (cpu)")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to benchmark")
    parser.add_argument("--out_csv", default="results/tables/runtime_benchmarks.csv", help="Output summary CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_p = Path(args.manifest)
    if not manifest_p.exists():
        print(f"Error: manifest not found at {manifest_p}")
        sys.exit(1)

    ckpt_p = Path(args.ckpt)
    if not ckpt_p.exists():
        ckpt_p = Path("results/checkpoints/unet_drivable_best.pt")

    print("=" * 65)
    print("RUNTIME BENCHMARK (CPU ONLY)")
    print("=" * 65)
    print(f"Manifest:       {manifest_p}")
    print(f"Checkpoint:     {ckpt_p}")
    print(f"Frames to test: {args.num_frames}")
    print(f"Device:         {args.device}")
    print("-" * 65)

    # Load model and router
    model, meta = build_smp_unet(str(ckpt_p), device=args.device)
    model.eval()
    router = RuleRouter(args.router_cfg)

    df = pd.read_csv(manifest_p).head(args.num_frames)

    # Load images into RAM first to isolate I/O from computation timing
    images = []
    for _, row in df.iterrows():
        bgr = cv2.imread(row["img_path"])
        if bgr is not None:
            images.append(bgr)

    n_valid = len(images)
    print(f"Loaded {n_valid} images into memory for benchmarking.\n")

    # Warmup pass
    if images:
        warmup = images[0]
        qv = compute_quality(warmup)
        act = router.route(qv)
        _ = get_enhancement(act)(warmup)
        _, m = predict_drivable_unet(model, warmup, meta=meta, device=args.device)
        _ = detect_lanes(warmup)
        _ = analyze_regions(m)

    times_quality = []
    times_router = []
    times_enhance = []
    times_unet = []
    times_hough = []
    times_region = []
    times_total = []

    for bgr in images:
        t0 = time.perf_counter()

        # 1. Quality scoring
        t_q0 = time.perf_counter()
        qv = compute_quality(bgr)
        t_q1 = time.perf_counter()
        times_quality.append((t_q1 - t_q0) * 1000.0)

        # 2. Rule router
        t_r0 = time.perf_counter()
        act = router.route(qv)
        t_r1 = time.perf_counter()
        times_router.append((t_r1 - t_r0) * 1000.0)

        # 3. Enhancement
        t_e0 = time.perf_counter()
        enh_fn = get_enhancement(act)
        enh = enh_fn(bgr)
        t_e1 = time.perf_counter()
        times_enhance.append((t_e1 - t_e0) * 1000.0)

        # 4. Deep U-Net segmentation
        t_u0 = time.perf_counter()
        _, mask = predict_drivable_unet(model, enh, meta=meta, device=args.device)
        t_u1 = time.perf_counter()
        times_unet.append((t_u1 - t_u0) * 1000.0)

        # 5. Classical Hough lane detection
        t_h0 = time.perf_counter()
        _ = detect_lanes(enh)
        t_h1 = time.perf_counter()
        times_hough.append((t_h1 - t_h0) * 1000.0)

        # 6. Region analysis
        t_reg0 = time.perf_counter()
        _ = analyze_regions(mask)
        t_reg1 = time.perf_counter()
        times_region.append((t_reg1 - t_reg0) * 1000.0)

        t_end = time.perf_counter()
        times_total.append((t_end - t0) * 1000.0)

    # Compute statistics
    mean_q = float(np.mean(times_quality))
    mean_r = float(np.mean(times_router))
    mean_e = float(np.mean(times_enhance))
    mean_u = float(np.mean(times_unet))
    mean_h = float(np.mean(times_hough))
    mean_reg = float(np.mean(times_region))
    mean_tot = float(np.mean(times_total))
    fps = 1000.0 / mean_tot if mean_tot > 0 else 0.0

    print("=" * 65)
    print("BENCHMARK RESULTS (CPU Timings)")
    print("=" * 65)
    print(f"{'Pipeline Stage':<35} | {'Latency (ms)':>15}")
    print("-" * 55)
    print(f"{'1. Quality Scoring (5-D vector)':<35} | {mean_q:>12.2f} ms")
    print(f"{'2. Rule-Based Router':<35} | {mean_r:>12.2f} ms")
    print(f"{'3. DIP Enhancement (Adaptive)':<35} | {mean_e:>12.2f} ms")
    print(f"{'4. U-Net Drivable Segmentation':<35} | {mean_u:>12.2f} ms")
    print(f"{'5. Classical Hough Lane Detection':<35} | {mean_h:>12.2f} ms")
    print(f"{'6. Region & Connected Components':<35} | {mean_reg:>12.2f} ms")
    print("=" * 55)
    print(f"{'TOTAL PIPELINE LATENCY':<35} | {mean_tot:>12.2f} ms")
    print(f"{'TOTAL THROUGHPUT (FPS)':<35} | {fps:>12.2f} FPS")
    print("=" * 65)

    # Format table for Dashboard Panel 5
    summary_data = [
        {"Stage": "1. Quality Scoring", "Latency (ms)": f"{mean_q:.2f}", "Throughput (FPS)": f"{1000.0 / mean_q:.1f}" if mean_q > 0 else "N/A"},
        {"Stage": "2. Rule Router", "Latency (ms)": f"{mean_r:.2f}", "Throughput (FPS)": f"{1000.0 / mean_r:.1f}" if mean_r > 0 else "N/A"},
        {"Stage": "3. DIP Enhancement", "Latency (ms)": f"{mean_e:.2f}", "Throughput (FPS)": f"{1000.0 / mean_e:.1f}" if mean_e > 0 else "N/A"},
        {"Stage": "4. Deep U-Net Segmentation", "Latency (ms)": f"{mean_u:.2f}", "Throughput (FPS)": f"{1000.0 / mean_u:.1f}" if mean_u > 0 else "N/A"},
        {"Stage": "5. Classical Hough Lanes", "Latency (ms)": f"{mean_h:.2f}", "Throughput (FPS)": f"{1000.0 / mean_h:.1f}" if mean_h > 0 else "N/A"},
        {"Stage": "6. Region Analysis", "Latency (ms)": f"{mean_reg:.2f}", "Throughput (FPS)": f"{1000.0 / mean_reg:.1f}" if mean_reg > 0 else "N/A"},
        {"Stage": "TOTAL PIPELINE (CPU)", "Latency (ms)": f"{mean_tot:.2f}", "Throughput (FPS)": f"{fps:.2f}"},
    ]

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_data).to_csv(out_p, index=False)
    print(f"\nWrote dashboard benchmark table: {out_p}")


if __name__ == "__main__":
    main()

