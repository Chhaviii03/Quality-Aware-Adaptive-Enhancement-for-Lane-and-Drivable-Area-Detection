"""
scripts/make_stage2_panels.py
-----------------------------
Generates Stage 2 comparison panels for 3 sample frames:
  - Frame 1: Day Clear
  - Frame 2: Foggy / Degraded (Koschmieder atmospheric scattering model)
  - Frame 3: Low-Light / Night (intensity scale + gamma + Poisson-Gaussian noise)

Each panel visualizes:
  - Row 1: Original and all 5 enhancements (CLAHE, Gamma, Dehaze, Denoise, Denoise+CLAHE)
  - Row 2: RGB and Luminance histograms for each corresponding image.
Outputs saved to results/figures/stage_panel_sample{1,2,3}.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data.degrade import apply_fog, apply_low_light
from src.enhance import get_enhancement, ENHANCEMENT_NAMES
from src.quality.metrics import compute_quality


def generate_panel_for_frame(
    bgr_input: np.ndarray,
    frame_title: str,
    out_path: str | Path,
) -> None:
    enhancements = ["none", "clahe", "gamma", "dehaze", "denoise", "denoise_clahe"]
    labels = ["Original", "CLAHE", "Gamma", "Dehaze", "Denoise", "Denoise+CLAHE"]

    enhanced_images = []
    for enh in enhancements:
        fn = get_enhancement(enh)
        img_out = fn(bgr_input)
        enhanced_images.append(img_out)

    fig, axes = plt.subplots(2, 6, figsize=(22, 7), gridspec_kw={"height_ratios": [1.1, 1.0]})

    colors = [("b", "blue", "B"), ("g", "green", "G"), ("r", "red", "R")]

    for i, (img, name) in enumerate(zip(enhanced_images, labels)):
        # Row 0: Image
        ax_img = axes[0, i]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(rgb)
        qv = compute_quality(img)
        ax_img.set_title(f"{name}\nL:{qv['brightness']:.1f} | S:{qv['sharpness']:.0f} | H:{qv['haze']:.2f}", fontsize=10)
        ax_img.axis("off")

        # Row 1: Histogram
        ax_hist = axes[1, i]
        for idx, (col_key, plot_col, col_lbl) in enumerate(colors):
            hist = cv2.calcHist([img], [idx], None, [256], [0, 256])
            ax_hist.plot(hist, color=plot_col, alpha=0.6, linewidth=1.2, label=col_lbl)

        # Luminance
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab], [0], None, [256], [0, 256])
        ax_hist.plot(hist_l, color="black", linestyle="--", alpha=0.8, linewidth=1.2, label="L")

        ax_hist.set_xlim([0, 256])
        ax_hist.set_xlabel("Intensity", fontsize=9)
        if i == 0:
            ax_hist.set_ylabel("Pixel Count", fontsize=9)
            ax_hist.legend(loc="upper right", fontsize=7)
        ax_hist.grid(True, linestyle=":", alpha=0.4)
        ax_hist.tick_params(axis="both", labelsize=8)

    fig.suptitle(f"Stage 2 Enhancement & Histogram Panel: {frame_title}", fontsize=14, y=0.98)
    fig.tight_layout()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_p}")


def main():
    val_images = sorted(list(Path("data/processed/val/images").glob("*.jpg")))
    if not val_images:
        # Fallback to sample_val if processed is missing
        val_images = sorted(list(Path("data/sample_val").glob("*.jpg")))

    assert len(val_images) >= 3, "Need at least 3 images to create panels."

    # Sample 1: Clean Day
    img1_path = val_images[0]
    img1 = cv2.imread(str(img1_path))

    # Sample 2: Synthetic Fog (Koschmieder model, beta=2.0, A=0.9)
    img2_base = cv2.imread(str(val_images[1]))
    img2 = apply_fog(img2_base, beta=2.0, A=0.9, seed=42)

    # Sample 3: Synthetic Low-Light (intensity_scale=0.25, gamma=1.8, Poisson-Gaussian noise)
    img3_base = cv2.imread(str(val_images[2]))
    img3 = apply_low_light(img3_base, intensity_scale=0.25, gamma=1.8, seed=42)

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Stage 2 panels for 3 sample frames...")
    generate_panel_for_frame(img1, f"Sample 1 - Daylight Clear ({img1_path.name})", out_dir / "stage_panel_sample1.png")
    generate_panel_for_frame(img2, f"Sample 2 - Synthetic Fog / Hazy ({val_images[1].name})", out_dir / "stage_panel_sample2.png")
    generate_panel_for_frame(img3, f"Sample 3 - Synthetic Low-Light / Night ({val_images[2].name})", out_dir / "stage_panel_sample3.png")
    print("All 3 stage-panel images created successfully!")


if __name__ == "__main__":
    main()

