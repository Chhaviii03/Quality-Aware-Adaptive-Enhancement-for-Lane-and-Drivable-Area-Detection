"""
src/classical/lane_hough.py
---------------------------
Classical lane detector (Branch A, FR-08).

Pipeline
--------
1. Gaussian blur (5×5).
2. HLS colour threshold (white + yellow markings).
3. Canny edge detection (low=50, high=150).
4. Trapezoid ROI mask.
5. Probabilistic Hough (rho=1, theta=1°, threshold=40,
   min_length=40, max_gap=20).
6. Slope filter → left / right line groups.
7. Least-squares line fit per group.
8. Morphological closing on the rendered 8 px mask.

Every intermediate image is optionally saved to `stage_dir`.

Usage
-----
    from src.classical.lane_hough import detect_lanes

    lane_mask = detect_lanes(bgr_image, save_stages=True,
                              stage_dir="results/figures/stages/frame_001")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(img: np.ndarray, path: Path, name: str) -> None:
    """Write an intermediate stage image."""
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / name), img)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-steps
# ─────────────────────────────────────────────────────────────────────────────

def _blur(bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(bgr, (5, 5), 0)


def _hls_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of white and yellow lane markings in HLS space."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    # White: high lightness, low saturation
    white = cv2.inRange(hls,
                        np.array([0,   180, 0],   dtype=np.uint8),
                        np.array([255, 255, 255],  dtype=np.uint8))

    # Yellow: hue ~15–35, moderate-to-high saturation
    yellow = cv2.inRange(hls,
                         np.array([10,  50, 70],   dtype=np.uint8),
                         np.array([40, 255, 255],  dtype=np.uint8))

    return cv2.bitwise_or(white, yellow)


def _canny(mask: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    return cv2.Canny(mask, low, high)


def _roi_mask(edges: np.ndarray) -> np.ndarray:
    """Apply a trapezoid ROI centred on the lower half of the image."""
    H, W = edges.shape[:2]
    roi = np.zeros_like(edges)

    # Trapezoid vertices (approximate lane area)
    pts = np.array([[
        (int(0.05 * W), H),
        (int(0.45 * W), int(0.55 * H)),
        (int(0.55 * W), int(0.55 * H)),
        (int(0.95 * W), H),
    ]], dtype=np.int32)

    cv2.fillPoly(roi, pts, 255)
    return cv2.bitwise_and(edges, roi)


def _hough_lines(edges: np.ndarray) -> list[tuple]:
    """Run Probabilistic Hough and return list of (x1,y1,x2,y2) segments."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.deg2rad(1),
        threshold=40,
        minLineLength=40,
        maxLineGap=20,
    )
    if lines is None:
        return []
    return [tuple(l[0]) for l in lines]


def _slope_filter(
    lines: list[tuple],
    slope_min: float = 0.3,
    slope_max: float = 3.0,
) -> tuple[list, list]:
    """Split segments into left / right groups by slope sign."""
    left, right = [], []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < slope_min or abs(slope) > slope_max:
            continue
        if slope < 0:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))
    return left, right


def _fit_line(segments: list[tuple], H: int) -> Optional[tuple[int, int, int, int]]:
    """Fit a single line through all segment endpoints; return (x1,y1,x2,y2)."""
    if not segments:
        return None
    pts = np.array([(x, y) for x1, y1, x2, y2 in segments for x, y in [(x1, y1), (x2, y2)]])
    xs, ys = pts[:, 0].astype(float), pts[:, 1].astype(float)
    if len(xs) < 2:
        return None
    coeffs = np.polyfit(ys, xs, 1)   # fit x = m*y + b
    y_bot = H
    y_top = int(H * 0.58)
    x_bot = int(np.polyval(coeffs, y_bot))
    x_top = int(np.polyval(coeffs, y_top))
    return x_bot, y_bot, x_top, y_top


def _render_mask(
    lines_left:  Optional[tuple],
    lines_right: Optional[tuple],
    H: int, W: int,
    thickness: int = 8,
) -> np.ndarray:
    """Render fitted left/right lines as an 8 px binary mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for seg in (lines_left, lines_right):
        if seg is not None:
            x1, y1, x2, y2 = seg
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    return mask


def _morph_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_lanes(
    bgr: np.ndarray,
    save_stages: bool = False,
    stage_dir: str | Path = "results/figures/stages",
) -> np.ndarray:
    """Run the full classical lane detection pipeline.

    Parameters
    ----------
    bgr : np.ndarray
        Input frame, uint8, (H, W, 3).
    save_stages : bool
        If True, writes every intermediate image to `stage_dir`.
    stage_dir : str | Path
        Directory for stage images (used only if save_stages=True).

    Returns
    -------
    np.ndarray
        Binary lane mask, uint8, (H, W), 255 = lane.
    """
    H, W = bgr.shape[:2]
    sdir = Path(stage_dir)

    # Stage 0: original
    if save_stages:
        _save(bgr, sdir, "00_original.jpg")

    # Stage 1: Gaussian blur
    blurred = _blur(bgr)
    if save_stages:
        _save(blurred, sdir, "01_blur.jpg")

    # Stage 2: HLS colour threshold
    colour_mask = _hls_mask(blurred)
    if save_stages:
        _save(colour_mask, sdir, "02_hls_mask.jpg")

    # Stage 3: Canny edges
    edges = _canny(colour_mask)
    if save_stages:
        _save(edges, sdir, "03_canny.jpg")

    # Stage 4: ROI mask
    roi_edges = _roi_mask(edges)
    if save_stages:
        _save(roi_edges, sdir, "04_roi.jpg")

    # Stage 5: Probabilistic Hough
    raw_lines = _hough_lines(roi_edges)
    hough_vis = bgr.copy()
    for x1, y1, x2, y2 in raw_lines:
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if save_stages:
        _save(hough_vis, sdir, "05_hough.jpg")

    # Stage 6: Slope filter + line fit
    left_segs, right_segs = _slope_filter(raw_lines)
    left_line  = _fit_line(left_segs,  H)
    right_line = _fit_line(right_segs, H)
    fit_vis = bgr.copy()
    for seg in (left_line, right_line):
        if seg:
            cv2.line(fit_vis, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 4)
    if save_stages:
        _save(fit_vis, sdir, "06_fit.jpg")

    # Stage 7: Render 8 px mask
    raw_mask = _render_mask(left_line, right_line, H, W)
    if save_stages:
        _save(raw_mask, sdir, "07_raw_mask.jpg")

    # Stage 8: Morphological close
    final_mask = _morph_close(raw_mask)
    if save_stages:
        _save(final_mask, sdir, "08_final_mask.jpg")

    # Stage 9: Overlay for report
    overlay = bgr.copy()
    overlay[final_mask > 0] = (0, 200, 255)
    if save_stages:
        _save(overlay, sdir, "09_overlay.jpg")

    return final_mask
