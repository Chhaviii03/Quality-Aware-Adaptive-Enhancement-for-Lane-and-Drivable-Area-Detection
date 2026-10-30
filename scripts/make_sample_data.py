"""
scripts/make_sample_data.py
---------------------------
Generates a lightweight, reproducible sample validation set for local CPU testing & evaluation.
Generates balanced frames across 4 conditions:
  - day_clear
  - night
  - rain
  - fog (synthetic atmospheric scattering)

Together with corresponding ground-truth lane and drivable-area masks.
Images are stored locally and ignored by .gitignore (only manifests are committed).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.data.degrade import apply_fog, apply_low_light


def generate_sample_val_data(
    out_dir: str | Path = "data/sample_val",
    manifest_csv: str | Path = "data/manifests/val.csv",
    frames_per_condition: int = 5,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lane_mask_dir = Path("data/labels/lane/masks/val")
    road_mask_dir = Path("data/labels/drivable/masks/val")
    lane_mask_dir.mkdir(parents=True, exist_ok=True)
    road_mask_dir.mkdir(parents=True, exist_ok=True)

    H, W = 352, 640
    conditions = ["day_clear", "night", "rain", "fog"]
    manifest_rows = []

    for cond in conditions:
        for idx in range(frames_per_condition):
            frame_id = f"val_{cond}_{idx:03d}.jpg"
            stem = f"val_{cond}_{idx:03d}"
            img_path = out_dir / frame_id

            # 1. Base clean driving scene (daytime clear)
            # Sky / horizon background
            base_img = np.zeros((H, W, 3), dtype=np.uint8)
            base_img[: int(0.55 * H), :] = (180, 150, 120)  # sky / background hills

            # Road surface (asphalt gray)
            road_pts = np.array([
                [int(0.05 * W), H],
                [int(0.42 * W), int(0.55 * H)],
                [int(0.58 * W), int(0.55 * H)],
                [int(0.95 * W), H],
            ], dtype=np.int32)
            cv2.fillPoly(base_img, [road_pts], (55, 55, 55))

            # GT road mask (drivable)
            road_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(road_mask, [road_pts], 255)

            # Left & right lane endpoints with small random lateral jitter per frame
            jitter = rng.integers(-10, 10)
            left_bot = (int(0.20 * W) + jitter, H)
            left_top = (int(0.45 * W) + jitter // 2, int(0.56 * H))
            right_bot = (int(0.80 * W) + jitter, H)
            right_top = (int(0.55 * W) + jitter // 2, int(0.56 * H))

            # Draw lane lines on base image
            # Left white solid line
            cv2.line(base_img, left_bot, left_top, (240, 240, 240), 8)
            # Right yellow solid line
            cv2.line(base_img, right_bot, right_top, (0, 220, 240), 8)

            # GT lane mask (8px binary mask)
            lane_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.line(lane_mask, left_bot, left_top, 255, 8)
            cv2.line(lane_mask, right_bot, right_top, 255, 8)

            # 2. Condition-specific rendering
            if cond == "day_clear":
                frame_bgr = base_img.copy()
            elif cond == "night":
                # Low light + headlight beam
                frame_bgr = apply_low_light(
                    base_img,
                    intensity_scale=0.25,
                    gamma=1.8,
                    read_noise_sigma=0.015,
                    seed=seed + idx,
                )
                # Headlight glow on lower road
                glow = np.zeros((H, W), dtype=np.float32)
                cv2.ellipse(glow, (int(0.5 * W), H), (int(0.35 * W), int(0.4 * H)), 0, 0, 180, 1.0, -1)
                glow = cv2.GaussianBlur(glow, (99, 99), 0)
                frame_bgr = np.clip(frame_bgr.astype(np.float32) + glow[:, :, None] * 70, 0, 255).astype(np.uint8)
            elif cond == "rain":
                # Rain: darker road, water reflection, rain streaks
                frame_bgr = cv2.convertScaleAbs(base_img, alpha=0.75, beta=-15)
                # Add streaks
                for _ in range(80):
                    rx = rng.integers(0, W)
                    ry = rng.integers(0, H - 30)
                    cv2.line(frame_bgr, (rx, ry), (rx + rng.integers(-4, -1), ry + rng.integers(15, 25)), (200, 200, 200), 1)
            elif cond == "fog":
                # Fog: atmospheric scattering
                frame_bgr = apply_fog(base_img, beta=1.8, A=0.88)
            else:
                frame_bgr = base_img.copy()

            # Save frame and masks
            cv2.imwrite(str(img_path), frame_bgr)
            cv2.imwrite(str(lane_mask_dir / f"{stem}.png"), lane_mask)
            cv2.imwrite(str(road_mask_dir / f"{stem}.png"), road_mask)

            manifest_rows.append({
                "image_id": frame_id,
                "condition": cond,
                "split": "val",
                "img_path": str(img_path).replace("\\", "/"),
            })

    # Write manifest CSV
    import pandas as pd
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = Path(manifest_csv)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated {len(manifest_df)} sample frames across {len(conditions)} conditions.")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    generate_sample_val_data()
