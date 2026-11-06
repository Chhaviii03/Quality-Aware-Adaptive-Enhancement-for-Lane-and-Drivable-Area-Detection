"""
src/classical/region.py
-----------------------
Object / Region Stage: Connected components analysis on binary masks.

Computes:
  - Connected-components label map (color-coded)
  - Region statistics: ID, Area, Perimeter, Bounding Box (x, y, w, h), Centroid (cx, cy)
  - Total segment count
  - Visual overlay with bounding boxes and centroids drawn on the image frame.
"""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np


def analyze_regions(
    binary_mask: np.ndarray,
    bgr_frame: Optional[np.ndarray] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run connected components analysis on a binary mask (uint8, 0 or 255).

    Parameters
    ----------
    binary_mask : np.ndarray
        Single-channel uint8 mask (0 = bg, >0 = foreground).
    bgr_frame : Optional[np.ndarray]
        Original BGR frame on which to draw bounding boxes and centroids.
        If None, a black canvas is used.

    Returns
    -------
    stats_dict : dict
        Contains 'segment_count', 'regions' list (id, area, perimeter, bbox, centroid),
        and 'total_area'.
    color_labels : np.ndarray
        Color-coded RGB/BGR image of the connected component labels.
    overlay_frame : np.ndarray
        Frame with colored bounding boxes and centroids annotated.
    """
    if binary_mask.ndim == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    mask_bin = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    H, W = mask_bin.shape[:2]
    if bgr_frame is not None:
        overlay_frame = bgr_frame.copy()
        if overlay_frame.shape[:2] != (H, W):
            overlay_frame = cv2.resize(overlay_frame, (W, H), interpolation=cv2.INTER_AREA)
    else:
        overlay_frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Generate color-coded label visualization
    # Map labels 0..num_labels-1 to distinct colors (label 0 = black)
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(max(num_labels, 1), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background is black
    color_labels = colors[labels]

    regions = []
    total_area = 0

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Compute perimeter using contours for region i
        region_bin = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(region_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(cnt, closed=True) for cnt in contours))

        total_area += area

        regions.append({
            "region_id": i,
            "area": area,
            "perimeter": round(perimeter, 2),
            "bbox": (x, y, w, h),
            "centroid": (round(cx, 1), round(cy, 1)),
        })

        # Draw on overlay: bounding box in cyan/green, centroid in red
        cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(overlay_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(
            overlay_frame,
            f"#{i}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    stats_dict = {
        "segment_count": num_labels - 1,
        "total_area": total_area,
        "regions": regions,
    }

    return stats_dict, color_labels, overlay_frame

