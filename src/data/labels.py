"""
src/data/labels.py
------------------
Convert BDD100K lane polylines and drivable-area polygons to binary masks.

BDD100K lane labels live in:   labels/lane/masks/{train,val}/
BDD100K seg labels live in:    labels/drivable/masks/{train,val}/
(Already rasterized PNGs in the official distribution — we just resize.)

If using the raw JSON annotation format instead, polylines are rasterized
here at 8 px width.

Usage
-----
    from src.data.labels import load_lane_mask, load_road_mask

    lane = load_lane_mask("/data/bdd100k", "train", "abc123.jpg",
                          width=640, height=352, lane_px=8)
    road = load_road_mask("/data/bdd100k", "train", "abc123.jpg",
                          width=640, height=352)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_lane_mask(
    data_root: str | Path,
    split: str,
    image_id: str,
    width: int  = 640,
    height: int = 352,
    lane_px: int = 8,
) -> np.ndarray:
    """Load or rasterize a lane binary mask.

    Tries the pre-rasterized PNG first; falls back to JSON polylines.

    Returns
    -------
    np.ndarray
        uint8, shape (height, width), 255 = lane, 0 = background.
    """
    data_root = Path(data_root)
    stem = Path(image_id).stem

    # Path 1: pre-rasterized mask PNG
    mask_path = data_root / "labels" / "lane" / "masks" / split / f"{stem}.png"
    if not mask_path.exists():
        mask_path = Path("data") / "labels" / "lane" / "masks" / split / f"{stem}.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    # Path 2: JSON annotation (raw labels)
    ann_path = data_root / "labels" / "lane" / "polygons" / split / f"{stem}.json"
    if ann_path.exists():
        import json
        with open(ann_path) as f:
            data = json.load(f)
        canvas = np.zeros((height, width), dtype=np.uint8)
        for ann in data.get("frames", [{}])[0].get("objects", []):
            if ann.get("category") in ("lane", "road line"):
                pts_raw = ann.get("poly2d", [{}])[0].get("vertices", [])
                if pts_raw:
                    pts = np.array([[int(p[0] * width / 1280),
                                     int(p[1] * height / 720)] for p in pts_raw],
                                   dtype=np.int32)
                    cv2.polylines(canvas, [pts], isClosed=False,
                                  color=255, thickness=lane_px)
        return canvas

    # No label found → return empty mask (will count as FN)
    return np.zeros((height, width), dtype=np.uint8)


def load_road_mask(
    data_root: str | Path,
    split: str,
    image_id: str,
    width: int  = 640,
    height: int = 352,
) -> np.ndarray:
    """Load the drivable-area binary mask.

    Returns
    -------
    np.ndarray
        uint8, shape (height, width), 255 = drivable, 0 = background.
    """
    data_root = Path(data_root)
    stem = Path(image_id).stem

    # BDD100K drivable masks: pixel value 0 = drivable, 1 = alt-drivable, 255 = bg
    mask_path = data_root / "labels" / "drivable" / "masks" / split / f"{stem}.png"
    if not mask_path.exists():
        mask_path = Path("data") / "labels" / "drivable" / "masks" / split / f"{stem}.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        # 0 and 1 → drivable; 255 → background
        drivable = ((mask == 0) | (mask == 1)).astype(np.uint8) * 255
        return drivable

    return np.zeros((height, width), dtype=np.uint8)
