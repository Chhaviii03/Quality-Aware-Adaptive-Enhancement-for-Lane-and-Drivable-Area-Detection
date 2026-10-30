#!/usr/bin/env python
"""
scripts/panel_compare.py
------------------------
Visual comparison panel showing:
  - Original (degraded or raw frame)
  - Enhanced frame
  - RGB & Luminance histogram before enhancement
  - RGB & Luminance histogram after enhancement

Usage
-----
    python scripts/panel_compare.py --img data/sample_val/val_fog_000.jpg --enhancement dehaze
    python scripts/panel_compare.py --img data/sample_val/val_night_000.jpg --enhancement clahe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
from src.enhance import get_enhancement, ENHANCEMENT_NAMES
from src.viz.panels import histogram_panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate before/after enhancement histogram panels.")
    p.add_argument("--img", required=True, help="Path to input image")
    p.add_argument("--enhancement", default="clahe", choices=ENHANCEMENT_NAMES,
                   help="Enhancement function to apply")
    p.add_argument("--out", default=None, help="Output path for the panel image")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    enh_fn = get_enhancement(args.enhancement)
    enhanced = enh_fn(bgr)

    out_path = args.out
    if out_path is None:
        out_dir = Path("results/figures/panels")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}_{args.enhancement}_hist_panel.png"

    fig = histogram_panel(
        original=bgr,
        enhanced=enhanced,
        enhancement_name=args.enhancement,
        out_path=out_path,
    )
    print(f"Panel successfully saved to: {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
