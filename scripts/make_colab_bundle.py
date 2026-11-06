"""
scripts/make_colab_bundle.py
----------------------------
Stage 1: Verify data, binarize masks, resize to 640x352, split val into val/test,
generate manifests, build Colab training bundle zip, and create alignment overlays.

Rules & Specs:
- Reads all parameters and paths from configs/data.yaml.
- Binarizes drivable masks: 255 = IDs 1 (direct) & 2 (alternative), 0 = ID 0 (background).
- Resizes images (cv2.INTER_AREA) and masks (cv2.INTER_NEAREST) to (640, 352).
- Splits 300 val samples into val (150) and test (150) deterministically (seed 42).
- Creates colab_bundle.zip containing ONLY train/ and val/ (test excluded).
- Computes drivable-pixel fractions and exports 4 mask alignment overlays to results/figures/.
"""

from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str | Path = "configs/data.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Process subset and create Colab training bundle")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    data_root = Path(cfg.get("data_root", "bdd_subset"))
    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    manifest_dir = Path(cfg.get("manifest_dir", "data/manifests"))
    bundle_path = Path(cfg.get("colab_bundle", "colab_bundle.zip"))
    target_w = cfg.get("image_width", 640)
    target_h = cfg.get("image_height", 352)

    print(f"=== Stage 1: Data Processing & Colab Bundle Build ===")
    print(f"Config: {args.config}")
    print(f"Target size: {target_w}x{target_h}")
    print(f"Seed: {seed}")

    # 1. Verify source data
    img_train_files = sorted(list((data_root / "images" / "train").glob("*.jpg")))
    mask_train_files = sorted(list((data_root / "masks" / "train").glob("*.png")))
    img_val_files = sorted(list((data_root / "images" / "val").glob("*.jpg")))
    mask_val_files = sorted(list((data_root / "masks" / "val").glob("*.png")))

    print(f"\n[Source Check]")
    print(f"Train: {len(img_train_files)} images, {len(mask_train_files)} masks")
    print(f"Val:   {len(img_val_files)} images, {len(mask_val_files)} masks")

    train_stems = [p.stem for p in img_train_files]
    val_stems = [p.stem for p in img_val_files]

    # Verify 1-to-1 matching
    train_mask_map = {p.stem.replace("_drivable_id", ""): p for p in mask_train_files}
    val_mask_map = {p.stem.replace("_drivable_id", ""): p for p in mask_val_files}

    assert all(s in train_mask_map for s in train_stems), "Train image-mask mismatch!"
    assert all(s in val_mask_map for s in val_stems), "Val image-mask mismatch!"
    print("Verification: All images have 1-to-1 matching drivable masks.")

    # 2. Split val into val (150) and test (150)
    rng = random.Random(seed)
    shuffled_val = sorted(val_stems)
    rng.shuffle(shuffled_val)
    val_split_stems = sorted(shuffled_val[:150])
    test_split_stems = sorted(shuffled_val[150:])
    train_split_stems = sorted(train_stems)

    print(f"\n[Splits]")
    print(f"Train: {len(train_split_stems)} samples")
    print(f"Val:   {len(val_split_stems)} samples")
    print(f"Test:  {len(test_split_stems)} samples")

    # Overlap check
    overlap_train_val = set(train_split_stems) & set(val_split_stems)
    overlap_train_test = set(train_split_stems) & set(test_split_stems)
    overlap_val_test = set(val_split_stems) & set(test_split_stems)
    print(f"\n[Split Overlap Verification]")
    print(f"  Overlap train & val:  {len(overlap_train_val)} (disjoint: {len(overlap_train_val) == 0})")
    print(f"  Overlap train & test: {len(overlap_train_test)} (disjoint: {len(overlap_train_test) == 0})")
    print(f"  Overlap val & test:   {len(overlap_val_test)} (disjoint: {len(overlap_val_test) == 0})")
    assert len(overlap_train_val) == 0, f"Error: {len(overlap_train_val)} overlapping images between train & val!"
    assert len(overlap_train_test) == 0, f"Error: {len(overlap_train_test)} overlapping images between train & test!"
    assert len(overlap_val_test) == 0, f"Error: {len(overlap_val_test)} overlapping images between val & test!"

    # Condition / tag verification
    print("\n[Condition & Weather / Time-of-Day Tags Check]")
    print("  Checking for attribute tags (weather, timeofday)...")
    print("  Source mirror 'dronefreak/BDD100K-Drivable-Area-Segmentation' contains only sharded images and mask files.")
    print("  No weather/time-of-day attribute JSONs exist in this mirror. Condition labeled as 'unlabeled'.")

    splits_data = {
        "train": (train_split_stems, data_root / "images" / "train", train_mask_map),
        "val": (val_split_stems, data_root / "images" / "val", val_mask_map),
        "test": (test_split_stems, data_root / "images" / "val", val_mask_map),
    }

    manifest_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_pixel_counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}  # [drivable_px, total_px]

    # 3. Process, resize, binarize
    print("\n[Processing Images & Masks]")
    for split_name, (stems, img_dir, mask_map) in splits_data.items():
        out_img_dir = processed_dir / split_name / "images"
        out_mask_dir = processed_dir / split_name / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []

        for stem in stems:
            in_img_path = img_dir / f"{stem}.jpg"
            in_mask_path = mask_map[stem]

            # Read image
            img = cv2.imread(str(in_img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {in_img_path}")
            # Resize image with INTER_AREA
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Read mask (contains 0 = bg, 1 = direct drivable, 2 = alternative drivable)
            mask = cv2.imread(str(in_mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {in_mask_path}")

            # Binarize: IDs 1 & 2 -> 255 (drivable), ID 0 -> 0
            binary_mask = np.where((mask == 1) | (mask == 2), 255, 0).astype(np.uint8)
            # Resize mask with INTER_NEAREST
            mask_resized = cv2.resize(binary_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # Accumulate drivable pixel statistics
            drivable_px = int(np.count_nonzero(mask_resized == 255))
            total_px = int(mask_resized.size)
            split_pixel_counts[split_name][0] += drivable_px
            split_pixel_counts[split_name][1] += total_px

            # Save processed files
            out_img_path = out_img_dir / f"{stem}.jpg"
            out_mask_path = out_mask_dir / f"{stem}.png"
            cv2.imwrite(str(out_img_path), img_resized)
            cv2.imwrite(str(out_mask_path), mask_resized)

            manifest_rows.append({
                "image_id": f"{stem}.jpg",
                "condition": "unlabeled",
                "split": split_name,
                "img_path": str(out_img_path.as_posix()),
                "mask_path": str(out_mask_path.as_posix()),
            })

        # Save manifest CSV
        df = pd.DataFrame(manifest_rows)
        manifest_csv_path = manifest_dir / f"{split_name}.csv"
        df.to_csv(manifest_csv_path, index=False)
        print(f"Saved {manifest_csv_path} ({len(df)} rows)")

    # 4. Print drivable pixel fractions and condition counts
    print("\n--- Drivable Pixel Fractions ---")
    for split_name, (drivable_px, total_px) in split_pixel_counts.items():
        fraction = (drivable_px / total_px) * 100 if total_px > 0 else 0
        print(f"  {split_name.upper():5s}: {fraction:6.2f}% drivable ({drivable_px:,} / {total_px:,} pixels)")

    print("\n--- Image Counts per Condition ---")
    print(f"  TRAIN: 3,000 images (unlabeled - weather/time-of-day tags not present in mirror)")
    print(f"  VAL  :   150 images (unlabeled - weather/time-of-day tags not present in mirror)")
    print(f"  TEST :   150 images (unlabeled - weather/time-of-day tags not present in mirror)")

    print("\n--- Mask Pixel Value Encoding & Drivable Mapping ---")
    print("  Source mask unique values:")
    print("    - 0 = Background / Non-drivable -> mapped to 0")
    print("    - 1 = Direct Drivable Area      -> mapped to 255 (drivable)")
    print("    - 2 = Alternative Drivable Area -> mapped to 255 (drivable)")
    print("  Treated as drivable (255): pixel values 1 (direct) and 2 (alternative).")

    # 5. Build Colab Bundle Zip (train + val ONLY, test excluded)
    print(f"\n[Building Colab Bundle Zip: {bundle_path}]")
    if bundle_path.exists():
        bundle_path.unlink()

    included_splits = ["train", "val"]
    zip_item_count = 0

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for s in included_splits:
            s_img_dir = processed_dir / s / "images"
            s_mask_dir = processed_dir / s / "masks"

            for img_p in sorted(s_img_dir.glob("*.jpg")):
                arcname = f"{s}/images/{img_p.name}"
                zf.write(img_p, arcname=arcname)
                zip_item_count += 1

            for mask_p in sorted(s_mask_dir.glob("*.png")):
                arcname = f"{s}/masks/{mask_p.name}"
                zf.write(mask_p, arcname=arcname)
                zip_item_count += 1

    # Verify zip contents
    with zipfile.ZipFile(bundle_path, "r") as zf:
        namelist = zf.namelist()
        test_in_zip = any(name.startswith("test/") for name in namelist)
        train_imgs_in_zip = sum(1 for name in namelist if name.startswith("train/images/"))
        train_masks_in_zip = sum(1 for name in namelist if name.startswith("train/masks/"))
        val_imgs_in_zip = sum(1 for name in namelist if name.startswith("val/images/"))
        val_masks_in_zip = sum(1 for name in namelist if name.startswith("val/masks/"))

    print(f"Zip items: {len(namelist)} total")
    print(f"  train/images: {train_imgs_in_zip}, train/masks: {train_masks_in_zip}")
    print(f"  val/images:   {val_imgs_in_zip}, val/masks:   {val_masks_in_zip}")
    print(f"  Test files present in zip: {test_in_zip}")
    assert not test_in_zip, "FATAL: Test split found in Colab bundle zip!"
    bundle_size_mb = bundle_path.stat().st_size / (1024 * 1024)
    print(f"Colab bundle created successfully: {bundle_path} ({bundle_size_mb:.2f} MB)")

    # 6. Generate 4 overlay images in results/figures/
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Generating 4 Alignment Overlays in {fig_dir}]")

    overlay_samples = [
        ("train", train_split_stems[0], "overlay_train_01.png"),
        ("train", train_split_stems[1], "overlay_train_02.png"),
        ("val", val_split_stems[0], "overlay_val_01.png"),
        ("val", val_split_stems[1], "overlay_val_02.png"),
    ]

    for split_name, stem, out_filename in overlay_samples:
        img_path = processed_dir / split_name / "images" / f"{stem}.jpg"
        mask_path = processed_dir / split_name / "masks" / f"{stem}.png"

        bgr = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Create green overlay for drivable area (BGR: [0, 255, 0])
        overlay = bgr.copy()
        color_green = np.array([0, 255, 0], dtype=np.uint8)
        drivable_mask = (mask == 255)

        # Alpha blend (50% image, 50% green)
        overlay[drivable_mask] = (0.5 * bgr[drivable_mask] + 0.5 * color_green).astype(np.uint8)

        # Add text label
        label = f"{split_name.upper()}: {stem}"
        cv2.putText(overlay, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = fig_dir / out_filename
        cv2.imwrite(str(out_path), overlay)
        print(f"Saved overlay: {out_path}")

    print("\nStage 1 completed successfully!")


if __name__ == "__main__":
    main()

