"""
src/eval/run_eval.py
--------------------
Unified evaluation harness (FR-11).

Writes one per-frame CSV:
    frame_id, condition, road_iou, road_dice, lane_f1, miou,
    enhancement, ms_quality, ms_router, ms_enhance, ms_model,
    brightness, contrast, sharpness, noise, haze

And aggregates per-condition metrics with bootstrap confidence intervals (1,000 resamples).

Usage
-----
    python src/eval/run_eval.py \
        --model        hough \
        --enhancement  none \
        --split        val \
        [--save_stages] [--birds_eye] [--debug]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import yaml

from src.quality.metrics import compute_quality
from src.enhance import get_enhancement, ENHANCEMENT_NAMES
from src.router.rule import RuleRouter
from src.eval.metrics import road_iou, road_dice, lane_f1, miou
from src.eval.writer import PerFrameCSVWriter
from src.eval.bootstrap import aggregate_by_condition, bootstrap_ci


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified evaluation harness.")
    p.add_argument("--model",        required=True,  choices=["unet", "segformer", "hough"],
                   help="Model / detector to evaluate")
    p.add_argument("--config",       default=None,   help="Path to model YAML config")
    p.add_argument("--enhance_cfg",  default="configs/enhance.yaml")
    p.add_argument("--router_cfg",   default="configs/router.yaml")
    p.add_argument("--data_cfg",     default="configs/data.yaml")
    p.add_argument("--enhancement",  default="none",
                   help=f"One of {ENHANCEMENT_NAMES} or 'adaptive'")
    p.add_argument("--split",        default="val", choices=["train", "val", "test"])
    p.add_argument("--debug",        action="store_true", help="Run on first 20 frames")
    p.add_argument("--save_stages",  action="store_true", help="Save intermediate pipeline stages")
    p.add_argument("--stage_dir",    default="results/figures/stages", help="Directory for stages")
    p.add_argument("--birds_eye",    action="store_true", help="Enable bird's-eye warp for Hough")
    p.add_argument("--resume",       default=None,   help="Checkpoint path")
    p.add_argument("--out_csv",      default=None,   help="Output CSV path")
    return p.parse_args()


def _load_model(args, cfg: dict):
    """Load U-Net or SegFormer (returns None for Hough)."""
    if args.model == "hough":
        return None

    import torch
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
        print("INFO: Running model in CPU evaluation mode.")

    model.eval()
    return model


def _run_hough(
    bgr: np.ndarray,
    save_stages: bool = False,
    stage_dir: str | Path = "results/figures/stages",
    use_birds_eye: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run classical Hough pipeline. Returns (lane_mask, road_mask)."""
    from src.classical.lane_hough import detect_lanes
    lane_mask = detect_lanes(
        bgr,
        save_stages=save_stages,
        stage_dir=stage_dir,
        use_birds_eye=use_birds_eye,
    )
    # Classical Hough detects lanes; drivable road mask is zero
    road_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    return lane_mask, road_mask


def _run_model(
    model,
    bgr: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run U-Net or SegFormer. Returns (road_mask, lane_mask)."""
    import torch
    inp = torch.from_numpy(bgr).permute(2, 0, 1).float() / 255.0
    inp = inp.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    preds = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    road_mask = ((preds[0] > 0.5) * 255).astype(np.uint8)
    lane_mask = ((preds[1] > 0.5) * 255).astype(np.uint8)
    return road_mask, lane_mask


def run_eval(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load configs
    with open(args.data_cfg) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.enhance_cfg) as f:
        enh_cfg = yaml.safe_load(f)

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
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df_manifest = pd.read_csv(manifest_path, comment="#")
    if df_manifest.empty:
        raise ValueError(
            f"Manifest {manifest_path} is empty. Run scripts/make_sample_data.py or scripts/make_subset.sh first."
        )

    if args.debug:
        df_manifest = df_manifest.head(20)
        print(f"[DEBUG] Running on {len(df_manifest)} frames.")

    # Experiment Tag
    exp_tag = "B1" if (args.model == "hough" and args.enhancement == "none") else f"{args.model}_{args.enhancement}"
    print(f"\n========================================================")
    print(f"  Running Evaluation: {exp_tag.upper()} ({args.model} | {args.enhancement} | {args.split})")
    print(f"  Frames: {len(df_manifest)}")
    print(f"========================================================\n")

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = f"results/tables/{args.model}_{args.enhancement}_{args.split}.csv"
    writer = PerFrameCSVWriter(out_csv)

    stages_saved_count = 0
    max_stages_to_save = 4  # Save 1 stage progression per condition if requested

    for idx, row in df_manifest.iterrows():
        img_path = Path(row["img_path"])
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"WARNING: could not read {img_path}; skipping.")
            continue

        H = data_cfg.get("image_height", 352)
        W = data_cfg.get("image_width",  640)
        bgr = cv2.resize(bgr, (W, H))

        # ── 1. Quality Analysis ──────────────────────────────────────────────
        t0 = time.perf_counter()
        qv = compute_quality(bgr)
        ms_quality = (time.perf_counter() - t0) * 1000

        # ── 2. Router ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if args.enhancement == "adaptive":
            chosen_enh = router.route(qv)
        else:
            chosen_enh = args.enhancement
        ms_router = (time.perf_counter() - t0) * 1000

        # ── 3. Enhancement ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        enh_fn = get_enhancement(chosen_enh)
        enhanced = enh_fn(bgr)
        ms_enhance = (time.perf_counter() - t0) * 1000

        # ── 4. Model / Hough Detector ────────────────────────────────────────
        # Save stages for first sample of each condition when --save_stages is on
        should_save_stage = args.save_stages and (stages_saved_count < max_stages_to_save)
        frame_stage_dir = Path(args.stage_dir) / Path(row["image_id"]).stem if should_save_stage else args.stage_dir

        t0 = time.perf_counter()
        if args.model == "hough":
            lane_pred, road_pred = _run_hough(
                enhanced,
                save_stages=should_save_stage,
                stage_dir=frame_stage_dir,
                use_birds_eye=args.birds_eye,
            )
            if should_save_stage:
                stages_saved_count += 1
        else:
            road_pred, lane_pred = _run_model(model, enhanced, device)
        ms_model = (time.perf_counter() - t0) * 1000

        # ── 5. Ground truth labels ───────────────────────────────────────────
        from src.data.labels import load_lane_mask, load_road_mask
        lane_gt = load_lane_mask(
            data_cfg["data_root"], args.split,
            row["image_id"], W, H,
            data_cfg.get("lane_mask_width", 8),
        )
        road_gt = load_road_mask(
            data_cfg["data_root"], args.split, row["image_id"], W, H
        )

        # ── 6. Compute Metrics ───────────────────────────────────────────────
        r_iou  = road_iou(road_pred, road_gt)
        r_dice = road_dice(road_pred, road_gt)
        l_f1   = lane_f1(lane_pred, lane_gt, tol=5)
        # For Hough, mIoU reflects lane IoU & road IoU
        f_miou = miou({"road": road_pred, "lane": lane_pred}, {"road": road_gt, "lane": lane_gt})

        writer.add_record(
            frame_id=row["image_id"],
            condition=row["condition"],
            road_iou=r_iou,
            road_dice=r_dice,
            lane_f1=l_f1,
            miou=f_miou,
            enhancement=chosen_enh,
            ms_quality=ms_quality,
            ms_router=ms_router,
            ms_enhance=ms_enhance,
            ms_model=ms_model,
            **qv.to_dict(),
        )

    # Save per-frame CSV
    result_df = writer.to_dataframe()
    saved_csv = writer.save()
    print(f"Per-frame results saved to: {saved_csv} ({len(result_df)} frames)")

    # ── 7. Aggregate per condition with Bootstrap CI (1,000 resamples) ─────
    metric_cols = ["lane_f1", "road_iou", "road_dice", "miou"]
    summary_df = aggregate_by_condition(result_df, metric_cols=metric_cols, n=1000, ci=0.95, seed=42)

    # Compute Overall Row
    overall_rec = {"condition": "ALL (overall)", "n_frames": len(result_df)}
    for col in metric_cols:
        vals = result_df[col].dropna().values
        mean, lo, hi = bootstrap_ci(vals, n=1000, ci=0.95, seed=42)
        overall_rec[f"{col}_mean"] = mean
        overall_rec[f"{col}_lo"] = lo
        overall_rec[f"{col}_hi"] = hi

    summary_df = pd.concat([summary_df, pd.DataFrame([overall_rec])], ignore_index=True)

    # Save summary table CSV
    summary_csv = Path(out_csv).parent / f"summary_{Path(out_csv).name}"
    summary_df.to_csv(summary_csv, index=False)

    # ── 8. Display B1 Results Table ───────────────────────────────────────
    print(f"\n=========================================================================================")
    print(f"  EXPERIMENT {exp_tag.upper()} RESULTS PER CONDITION (Bootstrap 95% CI, 1,000 resamples)")
    print(f"  Model: {args.model} | Enhancement: {args.enhancement} | Split: {args.split}")
    print(f"=========================================================================================")
    header = f"{'Condition':<15} | {'N':>4} | {'Lane F1 (5px)':^20} | {'Road IoU':^20} | {'mIoU':^20}"
    print(header)
    print("-" * len(header))

    for _, r in summary_df.iterrows():
        c_name = r["condition"]
        n_f = int(r["n_frames"])
        f1_str  = f"{r['lane_f1_mean']:.3f} [{r['lane_f1_lo']:.3f}, {r['lane_f1_hi']:.3f}]"
        iou_str = f"{r['road_iou_mean']:.3f} [{r['road_iou_lo']:.3f}, {r['road_iou_hi']:.3f}]"
        miou_str = f"{r['miou_mean']:.3f} [{r['miou_lo']:.3f}, {r['miou_hi']:.3f}]"
        print(f"{c_name:<15} | {n_f:>4} | {f1_str:^20} | {iou_str:^20} | {miou_str:^20}")

    print("=" * len(header))
    print(f"Summary table saved to: {summary_csv}\n")

    return result_df, summary_df


def main():
    args = _parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
