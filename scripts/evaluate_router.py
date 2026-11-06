#!/usr/bin/env python
"""
scripts/evaluate_router.py
--------------------------
Evaluates the Rule-Based Router against the Exhaustive Oracle on the FULL validation split (150 frames).

Computes:
  - Agreement with oracle (%)
  - Mean regret (Oracle score - Router score)
  - Per-enhancement choice counts for Router and Oracle
  - Writes results/tables/router_vs_oracle.csv for the Streamlit dashboard

Usage:
    python scripts/evaluate_router.py --manifest data/manifests/val.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from src.models.unet import build_smp_unet
from src.enhance import ENHANCEMENT_NAMES, get_enhancement
from src.quality.metrics import compute_quality
from src.router.rule import RuleRouter
from src.eval.metrics import road_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate router vs oracle on validation split.")
    parser.add_argument("--manifest", default="data/manifests/val.csv", help="Path to validation manifest")
    parser.add_argument("--ckpt", default="results/checkpoints/unet/unet_drivable_best.pt", help="Path to checkpoint")
    parser.add_argument("--router_cfg", default="configs/router.yaml", help="Path to router config")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--out_csv", default="results/tables/router_vs_oracle.csv", help="Output summary CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit frames (default: None, full split)")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_p = Path(args.manifest)
    if not manifest_p.exists():
        print(f"Error: manifest not found at {manifest_p}")
        sys.exit(1)

    cached_p = Path("results/tables/router_oracle_val150.csv")

    # If cached full evaluation exists and no limit specified, we can reuse or verify
    if cached_p.exists() and args.limit is None:
        print(f"Reading cached 150-frame router/oracle evaluation from: {cached_p}")
        df_oracle = pd.read_csv(cached_p)
    else:
        # Compute freshly
        ckpt_p = Path(args.ckpt)
        if not ckpt_p.exists():
            ckpt_p = Path("results/checkpoints/unet_drivable_best.pt")
        model, meta = build_smp_unet(str(ckpt_p), device=args.device)
        model.eval()

        mean = np.array(meta.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(meta.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        router = RuleRouter(args.router_cfg)
        val_df = pd.read_csv(manifest_p)
        if args.limit:
            val_df = val_df.head(args.limit)

        records = []
        with torch.no_grad():
            for idx, row in val_df.iterrows():
                fid = row["image_id"]
                img = cv2.imread(row["img_path"])
                gt = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
                if img is None or gt is None:
                    continue

                qv = compute_quality(img)
                r_act = router.route(qv)

                scores = {}
                for a in ENHANCEMENT_NAMES:
                    fn = get_enhancement(a)
                    enh = fn(img)
                    rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
                    norm = (rgb.astype(np.float32) / 255.0 - mean) / std
                    t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(args.device)
                    prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
                    p_mask = (prob > 0.5).astype(np.uint8) * 255
                    scores[a] = road_iou(p_mask, gt)

                best_act = max(scores, key=scores.get)
                r_score = scores[r_act]
                o_score = scores[best_act]
                regret = max(0.0, o_score - r_score)

                records.append({
                    "frame_id": fid,
                    "condition": row.get("condition", "unlabeled"),
                    "router_action": r_act,
                    "oracle_action": best_act,
                    "router_score": r_score,
                    "oracle_score": o_score,
                    "regret": regret,
                    "match": (r_act == best_act),
                })
        df_oracle = pd.DataFrame(records)

    n_total = len(df_oracle)
    matches = df_oracle["match"].sum()
    agreement_pct = (matches / n_total) * 100.0
    mean_regret = df_oracle["regret"].mean()

    router_counts = Counter(df_oracle["router_action"])
    oracle_counts = Counter(df_oracle["oracle_action"])

    print("\n" + "=" * 65)
    print(f"ROUTER VS ORACLE EVALUATION ON FULL VAL SPLIT ({n_total} FRAMES)")
    print("=" * 65)
    print(f"Total Frames Evaluated:      {n_total}")
    print(f"Oracle Agreement Matches:    {matches} / {n_total}")
    print(f"Router Agreement with Oracle: {agreement_pct:.2f}%")
    print(f"Mean Regret (IoU loss):      {mean_regret:.6f}")
    print("-" * 65)
    print("Per-Enhancement Choice Counts:")
    print(f"{'Action':<16} | {'Router Count':^14} | {'Oracle Count':^14}")
    print("-" * 50)
    for a in ENHANCEMENT_NAMES:
        rc = router_counts.get(a, 0)
        oc = oracle_counts.get(a, 0)
        print(f"{a:<16} | {rc:^14} | {oc:^14}")
    print("=" * 65)

    # Format summary table for Dashboard Panel 5 (router_vs_oracle.csv)
    summary_rows = [
        {"Metric": "Total Frames Evaluated", "Value": str(n_total)},
        {"Metric": "Oracle Agreement (%)", "Value": f"{agreement_pct:.2f}%"},
        {"Metric": "Mean Regret (IoU)", "Value": f"{mean_regret:.6f}"},
        {"Metric": "Router Matches Count", "Value": f"{matches} / {n_total}"},
    ]
    for a in ENHANCEMENT_NAMES:
        summary_rows.append({
            "Metric": f"Action Choice: {a}",
            "Value": f"Router: {router_counts.get(a, 0)} | Oracle: {oracle_counts.get(a, 0)}",
        })

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_p, index=False)
    print(f"\nWrote dashboard summary table: {out_p}")


if __name__ == "__main__":
    main()

