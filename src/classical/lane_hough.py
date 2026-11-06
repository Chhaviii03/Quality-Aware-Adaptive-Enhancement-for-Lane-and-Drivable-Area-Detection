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
5. Optional bird's-eye warp (Inverse Perspective Mapping).
6. Probabilistic Hough (rho=1, theta=1°, threshold=40,
   min_length=40, max_gap=20).
7. Slope filter → left / right line groups.
8. Least-squares line fit per group.
9. Render 8 px binary mask (unwarped if bird's-eye was active).
10. Morphological closing on the rendered 8 px mask.

Every intermediate image is saved when `save_stages=True`.

Usage
-----
    from src.classical.lane_hough import detect_lanes

    lane_mask = detect_lanes(
        bgr_image,
        save_stages=True,
        stage_dir="results/figures/stages/frame_001",
        use_birds_eye=False,
    )
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
    """Stage 1: Gaussian blur with 5x5 kernel."""
    return cv2.GaussianBlur(bgr, (5, 5), 0)


def _hls_mask(bgr: np.ndarray) -> np.ndarray:
    """Stage 2: Binary mask of white and yellow lane markings in HLS space."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    # White: high lightness, low saturation
    white = cv2.inRange(
        hls,
        np.array([0, 180, 0], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )

    # Yellow: hue ~10–40, moderate-to-high saturation and lightness
    yellow = cv2.inRange(
        hls,
        np.array([10, 50, 70], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )

    return cv2.bitwise_or(white, yellow)


def _canny(mask: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Stage 3: Canny edge detector."""
    return cv2.Canny(mask, low, high)


def _roi_mask(edges: np.ndarray) -> np.ndarray:
    """Stage 4: Apply a trapezoid ROI centred on the lower half of the image."""
    H, W = edges.shape[:2]
    roi = np.zeros_like(edges)

    # Trapezoid vertices (approximate driving lane field of view)
    pts = np.array([[
        (int(0.05 * W), H),
        (int(0.45 * W), int(0.55 * H)),
        (int(0.55 * W), int(0.55 * H)),
        (int(0.95 * W), H),
    ]], dtype=np.int32)

    cv2.fillPoly(roi, pts, 255)
    return cv2.bitwise_and(edges, roi)


def get_perspective_transform(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute perspective transform matrices for bird's-eye warp (IPM).

    Returns
    -------
    M, Minv : tuple[np.ndarray, np.ndarray]
        Forward (camera -> bird's-eye) and inverse (bird's-eye -> camera) 3x3 matrices.
    """
    src = np.float32([
        [0.10 * W, H],
        [0.43 * W, 0.60 * H],
        [0.57 * W, 0.60 * H],
        [0.90 * W, H],
    ])
    dst = np.float32([
        [0.20 * W, H],
        [0.20 * W, 0],
        [0.80 * W, 0],
        [0.80 * W, H],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def birds_eye_warp(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Warp image or mask to top-down bird's-eye view."""
    H, W = img.shape[:2]
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpPerspective(img, M, (W, H), flags=interp)


def birds_eye_unwarp(img: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    """Unwarp bird's-eye view back to vehicle camera perspective."""
    H, W = img.shape[:2]
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpPerspective(img, Minv, (W, H), flags=interp)


def _hough_lines(edges: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Stage 5: Run Probabilistic Hough and return list of (x1, y1, x2, y2) segments."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.deg2rad(1),
        threshold=40,
        minLineLength=40,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return []
    lines_arr = lines.reshape(-1, 4)
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines_arr]


def _slope_filter(
    lines: list[tuple[int, int, int, int]],
    slope_min: float = 0.2,
    slope_max: float = 4.0,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Stage 6: Split line segments into left and right groups by slope sign."""
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


def _fit_line(
    segments: list[tuple[int, int, int, int]],
    H: int,
    y_top_ratio: float = 0.58,
) -> Optional[tuple[int, int, int, int]]:
    """Fit a single line through segment endpoints; returns (x1, y1, x2, y2)."""
    if not segments:
        return None
    pts = np.array([(x, y) for x1, y1, x2, y2 in segments for x, y in [(x1, y1), (x2, y2)]])
    xs, ys = pts[:, 0].astype(float), pts[:, 1].astype(float)
    if len(xs) < 2:
        return None
    try:
        coeffs = np.polyfit(ys, xs, 1)  # fit x = m*y + b
    except np.linalg.LinAlgError:
        return None

    y_bot = H
    y_top = int(H * y_top_ratio)
    x_bot = int(np.polyval(coeffs, y_bot))
    x_top = int(np.polyval(coeffs, y_top))
    return x_bot, y_bot, x_top, y_top


def _render_mask(
    lines_left: Optional[tuple[int, int, int, int]],
    lines_right: Optional[tuple[int, int, int, int]],
    H: int,
    W: int,
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
    """Morphological closing to bridge small gaps and smooth edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_lanes(
    bgr: np.ndarray,
    save_stages: bool = False,
    stage_dir: str | Path = "results/figures/stages",
    use_birds_eye: bool = False,
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
    use_birds_eye : bool
        If True, applies perspective bird's-eye warp (IPM) before line fitting,
        then unwarps the detected mask back to perspective camera coordinates.

    Returns
    -------
    np.ndarray
        Binary lane mask, uint8, (H, W), 255 = lane, 0 = background.
    """
    H, W = bgr.shape[:2]
    sdir = Path(stage_dir)

    # Stage 0: original input
    if save_stages:
        _save(bgr, sdir, "00_original.jpg")

    # Stage 1: Gaussian blur (5x5)
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

    # Stage 4: Trapezoid ROI mask
    roi_edges = _roi_mask(edges)
    if save_stages:
        _save(roi_edges, sdir, "04_roi.jpg")

    # Optional bird's-eye warp
    M, Minv = get_perspective_transform(H, W)
    if use_birds_eye:
        hough_input = birds_eye_warp(roi_edges, M)
        if save_stages:
            bev_bgr = birds_eye_warp(bgr, M)
            _save(bev_bgr, sdir, "04b_birds_eye.jpg")
    else:
        hough_input = roi_edges

    # Stage 5: Probabilistic Hough lines
    raw_lines = _hough_lines(hough_input)
    hough_vis = (birds_eye_warp(bgr, M) if use_birds_eye else bgr).copy()
    for x1, y1, x2, y2 in raw_lines:
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if save_stages:
        _save(hough_vis, sdir, "05_hough.jpg")

    # Stage 6: Slope filter + line fit
    y_top_ratio = 0.0 if use_birds_eye else 0.58
    left_segs, right_segs = _slope_filter(raw_lines)
    left_line  = _fit_line(left_segs,  H, y_top_ratio=y_top_ratio)
    right_line = _fit_line(right_segs, H, y_top_ratio=y_top_ratio)

    fit_vis = (birds_eye_warp(bgr, M) if use_birds_eye else bgr).copy()
    for seg in (left_line, right_line):
        if seg:
            cv2.line(fit_vis, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 4)
    if save_stages:
        _save(fit_vis, sdir, "06_fit.jpg")

    # Stage 7: Render 8 px mask (and unwarp if bird's-eye active)
    rendered = _render_mask(left_line, right_line, H, W, thickness=8)
    if use_birds_eye:
        raw_mask = birds_eye_unwarp(rendered, Minv)
    else:
        raw_mask = rendered
    if save_stages:
        _save(raw_mask, sdir, "07_raw_mask.jpg")

    # Stage 8: Morphological closing
    final_mask = _morph_close(raw_mask)
    if save_stages:
        _save(final_mask, sdir, "08_final_mask.jpg")

    # Stage 9: Overlay for visual inspection & report figures
    overlay = bgr.copy()
    overlay[final_mask > 0] = (0, 200, 255)  # yellow-orange overlay
    if save_stages:
        _save(overlay, sdir, "09_overlay.jpg")

    return final_mask


def get_hough_stage_images(bgr: np.ndarray, use_birds_eye: bool = False) -> dict[str, np.ndarray]:
    """Run classical lane detector and return a dictionary of intermediate stage images.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping stage names to images for display:
        '01_blur', '02_hls_mask', '03_canny', '04_roi', '05_hough', '06_fit', '08_final_mask', '09_overlay'
    """
    H, W = bgr.shape[:2]
    stages: dict[str, np.ndarray] = {}

    stages["00_original"] = bgr.copy()
    blurred = _blur(bgr)
    stages["01_blur"] = blurred

    colour_mask = _hls_mask(blurred)
    stages["02_hls_mask"] = colour_mask

    edges = _canny(colour_mask)
    stages["03_canny"] = edges

    roi_edges = _roi_mask(edges)
    stages["04_roi"] = roi_edges

    M, Minv = get_perspective_transform(H, W)
    hough_input = birds_eye_warp(roi_edges, M) if use_birds_eye else roi_edges

    raw_lines = _hough_lines(hough_input)
    hough_vis = (birds_eye_warp(bgr, M) if use_birds_eye else bgr).copy()
    for x1, y1, x2, y2 in raw_lines:
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    stages["05_hough"] = hough_vis

    y_top_ratio = 0.0 if use_birds_eye else 0.58
    left_segs, right_segs = _slope_filter(raw_lines)
    left_line = _fit_line(left_segs, H, y_top_ratio=y_top_ratio)
    right_line = _fit_line(right_segs, H, y_top_ratio=y_top_ratio)

    fit_vis = (birds_eye_warp(bgr, M) if use_birds_eye else bgr).copy()
    for seg in (left_line, right_line):
        if seg:
            cv2.line(fit_vis, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 4)
    stages["06_fit"] = fit_vis

    rendered = _render_mask(left_line, right_line, H, W, thickness=8)
    raw_mask = birds_eye_unwarp(rendered, Minv) if use_birds_eye else rendered
    stages["07_raw_mask"] = raw_mask

    final_mask = _morph_close(raw_mask)
    stages["08_cleaned_mask"] = final_mask

    overlay = bgr.copy()
    overlay[final_mask > 0] = (0, 200, 255)
    stages["09_overlay"] = overlay

    return stages
