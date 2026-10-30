"""
src/eval/run_eval.py
--------------------
Unified evaluation harness (FR-11).

Writes one per-frame CSV:
    frame_id, condition, road_iou, road_dice, lane_f1,
    enhancement, ms_quality, ms_router, ms_enhance, ms_model

Usage
-----
    python src/eval/run_eval.py \
        --model    unet \
        --config   configs/unet.yaml \
        --enhancement adaptive \
        --split    test \
        [--debug]

Flags
-----
--model      : unet | segformer | hough
--enhancement: none | clahe | gamma | dehaze | denoise | denoise_clahe | adaptive
--split      : train | val | test
--debug      : 20 images, CPU
--resume     : path to checkpoint (overrides config)
--out_csv    : output path (default: results/tables/<model>_<enh>_<split>.csv)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.quality.metrics import compute_quality
from src.enhance import get_enhancement, ENHANCEMENT_NAMES
from src.router.rule import RuleRouter
from src.eval.metrics import road_iou, road_dice, lane_f1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run evaluation harness.")
    p.add_argument("--model",        required=True,  choices=["unet", "segformer", "hough"])
    p.add_argument("--config",       default=None,   help="Path to model YAML config")
    p.add_argument("--enhance_cfg",  default="configs/enhance.yaml")
    p.add_argument("--router_cfg",   default="configs/router.yaml")
    p.add_argument("--data_cfg",     default="configs/data.yaml")
    p.add_argument("--enhancement",  default="none",
                   help=f"One of {ENHANCEMENT_NAMES} or 'adaptive'")
    p.add_argument("--split",        default="test", choices=["train", "val", "test"])
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--resume",       default=None,   help="Checkpoint path")
    p.add_argument("--out_csv",      default=None)
    return p.parse_args()


def _load_model(args, cfg: dict) -> torch.nn.Module | None:
    """Load the appropriate model (returns None for hough)."""
    device = "cpu"
    if args.model == "hough":
        return None

    if args.model == "unet":
        from src.models.unet import UNet
        model = UNet(
            in_channels=cfg.get("in_channels", 3),
            base_width=cfg.get("base_width", 32),
            depth=cfg.get("depth", 4),
            out_channels=cfg.get("out_channels", 2),
        )
    else:  # segformer
        from src.models.segformer import build_segformer
        model = build_segformer(
            backbone=cfg.get("backbone", "nvidia/mit-b0"),
            out_channels=cfg.get("out_channels", 2),
        )

    resume = args.resume or cfg.get("resume_from")
    if resume and Path(resume).exists():
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        print(f"Resumed from {resume}")
    else:
        print("WARNING: No checkpoint found. Running with random weights.")

    model.eval()
    return model


def _run_hough(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run classical Hough pipeline. Returns (lane_mask, road_mask)."""
    from src.classical.lane_hough import detect_lanes
    lane_mask = detect_lanes(bgr, save_stages=False)
    road_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)  # Hough has no road head
    return lane_mask, road_mask


def _run_model(
    model: torch.nn.Module,
    bgr: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run U-Net or SegFormer. Returns (road_mask, lane_mask)."""
    inp = torch.from_numpy(bgr).permute(2, 0, 1).float() / 255.0
    inp = inp.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    preds = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    road_mask = ((preds[0] > 0.5) * 255).astype(np.uint8)
    lane_mask = ((preds[1] > 0.5) * 255).astype(np.uint8)
    return road_mask, lane_mask


def run_eval(args: argparse.Namespace) -> pd.DataFrame:
    # Load configs
    with open(args.data_cfg)   as f: data_cfg = yaml.safe_load(f)
    with open(args.enhance_cfg) as f: enh_cfg  = yaml.safe_load(f)

    model_cfg = {}
    config_path = args.config or f"configs/{args.model}.yaml"
    if Path(config_path).exists():
        with open(config_path) as f:
            model_cfg = yaml.safe_load(f)

    # Router
    router = None
    if args.enhancement == "adaptive":
        router = RuleRouter(args.router_cfg)

    # Model
    model = _load_model(args, model_cfg)
    device = "cpu"

    # Load manifest
    manifest_path = Path(data_cfg.get("manifest_dir", "data/manifests")) / f"{args.split}.csv"
    df_manifest = pd.read_csv(manifest_path)
    if args.debug:
        df_manifest = df_manifest.head(20)
        print(f"[DEBUG] Running on {len(df_manifest)} frames.")

    records = []

    for _, row in df_manifest.iterrows():
        import cv2
        img_path = row["img_path"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"WARNING: could not read {img_path}; skipping.")
            continue

        H = data_cfg.get("image_height", 352)
        W = data_cfg.get("image_width",  640)
        bgr = cv2.resize(bgr, (W, H))

        # ── Quality ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        qv = compute_quality(bgr)
        ms_quality = (time.perf_counter() - t0) * 1000

        # ── Router ───────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if args.enhancement == "adaptive":
            chosen_enh = router.route(qv)
        else:
            chosen_enh = args.enhancement
        ms_router = (time.perf_counter() - t0) * 1000

        # ── Enhancement ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        enh_fn   = get_enhancement(chosen_enh)
        enhanced = enh_fn(bgr)
        ms_enhance = (time.perf_counter() - t0) * 1000

        # ── Model / Hough ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if args.model == "hough":
            lane_pred, road_pred = _run_hough(enhanced)
        else:
            road_pred, lane_pred = _run_model(model, enhanced, device)
        ms_model = (time.perf_counter() - t0) * 1000

        # ── Ground truth ─────────────────────────────────────────────────────
        from src.data.labels import load_lane_mask, load_road_mask
        lane_gt = load_lane_mask(
            data_cfg["data_root"], args.split,
            row["image_id"], W, H,
            data_cfg.get("lane_mask_width", 8)
        )
        road_gt = load_road_mask(
            data_cfg["data_root"], args.split, row["image_id"], W, H
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        r_iou  = road_iou( road_pred, road_gt)
        r_dice = road_dice(road_pred, road_gt)
        l_f1   = lane_f1(  lane_pred, lane_gt)

        records.append({
            "frame_id":   row["image_id"],
            "condition":  row["condition"],
            "road_iou":   r_iou,
            "road_dice":  r_dice,
            "lane_f1":    l_f1,
            "enhancement": chosen_enh,
            "ms_quality":  ms_quality,
            "ms_router":   ms_router,
            "ms_enhance":  ms_enhance,
            "ms_model":    ms_model,
            **qv.to_dict(),
        })

    result_df = pd.DataFrame(records)

    # Write CSV
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = (f"results/tables/{args.model}_{args.enhancement}_{args.split}.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    print(f"Results written to {out_csv}  ({len(result_df)} frames)")

    return result_df


if __name__ == "__main__":
    args = _parse_args()
    run_eval(args)
