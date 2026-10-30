#!/usr/bin/env python
"""
scripts/cache_enhanced.py
-------------------------
Precomputes enhanced val and test frames for every enhancement action and saves them.
This avoids recomputing expensive operations (such as dark-channel dehaze and NLM denoise)
during deep learning training or evaluation loops on Colab / local machines.

Actions cached:
  - clahe
  - gamma
  - dehaze
  - denoise (NLM)
  - denoise_clahe

Usage
-----
    python scripts/cache_enhanced.py --splits val test
    python scripts/cache_enhanced.py --splits val --enhancements dehaze denoise
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
import pandas as pd
from src.enhance import get_enhancement, ENHANCEMENT_NAMES

ACTIONS_TO_CACHE = ["clahe", "gamma", "dehaze", "denoise", "denoise_clahe"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute and cache enhanced frames.")
    p.add_argument("--splits", nargs="+", default=["val"],
                   choices=["train", "val", "test"],
                   help="Splits to process (default: val)")
    p.add_argument("--enhancements", nargs="+", default=ACTIONS_TO_CACHE,
                   choices=ENHANCEMENT_NAMES,
                   help="Enhancement actions to precompute")
    p.add_argument("--manifest_dir", default="data/manifests",
                   help="Directory with {split}.csv manifests")
    p.add_argument("--out_dir", default="data/cached_enhanced",
                   help="Directory to save precomputed images")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional frame limit for testing/debugging")
    return p.parse_args()


def cache_enhanced_frames(args: argparse.Namespace) -> None:
    manifest_dir = Path(args.manifest_dir)
    out_root = Path(args.out_dir)

    print("=================================================================")
    print("  Precomputing Enhanced Frame Cache")
    print(f"  Splits:        {args.splits}")
    print(f"  Enhancements:  {args.enhancements}")
    print(f"  Output Root:   {out_root}")
    print("=================================================================\n")

    total_cached = 0
    t_start = time.perf_counter()

    for split in args.splits:
        manifest_path = manifest_dir / f"{split}.csv"
        if not manifest_path.exists():
            print(f"WARNING: Manifest {manifest_path} not found. Skipping {split}.")
            continue

        df = pd.read_csv(manifest_path, comment="#")
        if df.empty:
            print(f"WARNING: Manifest {manifest_path} is empty. Skipping {split}.")
            continue

        if args.limit:
            df = df.head(args.limit)

        print(f"--- Processing Split '{split}' ({len(df)} frames) ---")

        for action in args.enhancements:
            if action == "none":
                continue

            enh_fn = get_enhancement(action)
            action_dir = out_root / split / action
            action_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            count = 0

            for _, row in df.iterrows():
                img_path = Path(row["img_path"])
                if not img_path.exists():
                    continue

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue

                target_path = action_dir / row["image_id"]
                # Skip if already cached
                if target_path.exists():
                    count += 1
                    continue

                enhanced = enh_fn(bgr)
                cv2.imwrite(str(target_path), enhanced)
                count += 1

            dt = time.perf_counter() - t0
            rate = count / max(dt, 1e-4)
            print(f"  [{action:<14}] Cached {count:>3} frames in {dt:5.2f}s ({rate:5.1f} fps) -> {action_dir}")
            total_cached += count

    total_time = time.perf_counter() - t_start
    print(f"\nCache generation complete: {total_cached} total files saved in {total_time:.2f}s.")
    print(f"Saved under: {out_root.resolve()}\n")


if __name__ == "__main__":
    args = parse_args()
    cache_enhanced_frames(args)
