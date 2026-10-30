"""
src/router/oracle.py
--------------------
Oracle labeller (FR-07).

For each validation frame, applies every enhancement, runs the frozen
model, and records the best-scoring enhancement as the oracle label.

Usage (intended to be called from a script, not run standalone):

    from src.router.oracle import build_oracle_labels
    oracle_df = build_oracle_labels(
        model=unet,
        val_loader=loader,
        enhance_cfg=enhance_cfg,
        out_csv="results/tables/oracle_labels.csv",
        device="cpu",
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.enhance import ENHANCEMENT_NAMES, get_enhancement
from src.eval.metrics import lane_f1, road_iou


def build_oracle_labels(
    model: torch.nn.Module,
    val_loader,
    enhance_cfg: dict,
    out_csv: str | Path = "results/tables/oracle_labels.csv",
    device: str = "cpu",
) -> pd.DataFrame:
    """Apply every enhancement to every val frame, score with model, save best.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen trained model (already loaded, in eval mode).
    val_loader : DataLoader
        Yields (bgr_numpy, lane_mask, road_mask, frame_id, condition).
    enhance_cfg : dict
        Enhancement hyperparameters from configs/enhance.yaml.
    out_csv : str | Path
        Where to write the oracle label CSV.
    device : str

    Returns
    -------
    pd.DataFrame
        Columns: frame_id, condition, oracle_action, oracle_score,
                 scores_<action> for each action.
    """
    model.eval()
    records = []

    with torch.no_grad():
        for batch in val_loader:
            bgr_np, lane_mask, road_mask, frame_id, condition = batch

            # bgr_np: (B, H, W, 3) uint8 numpy
            B = bgr_np.shape[0]

            for i in range(B):
                img = bgr_np[i]           # (H, W, 3) uint8
                lm  = lane_mask[i]        # (H, W) uint8
                rm  = road_mask[i]        # (H, W) uint8
                fid = frame_id[i]
                cond = condition[i]

                best_action = "none"
                best_score  = -1.0
                scores: dict[str, float] = {}

                for action in ENHANCEMENT_NAMES:
                    fn = get_enhancement(action)
                    enhanced = fn(img)

                    # Prepare tensor
                    inp = torch.from_numpy(enhanced).permute(2, 0, 1).float() / 255.0
                    inp = inp.unsqueeze(0).to(device)

                    logits = model(inp)            # (1, 2, H, W)
                    preds  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                    pred_road = (preds[0] > 0.5).astype(np.uint8)
                    pred_lane = (preds[1] > 0.5).astype(np.uint8)

                    r_iou = road_iou(pred_road, rm.numpy() if hasattr(rm, 'numpy') else rm)
                    l_f1  = lane_f1(pred_lane,  lm.numpy() if hasattr(lm, 'numpy') else lm)
                    score = 0.5 * r_iou + 0.5 * l_f1

                    scores[action] = score
                    if score > best_score:
                        best_score  = score
                        best_action = action

                rec = {
                    "frame_id":      fid,
                    "condition":     cond,
                    "oracle_action": best_action,
                    "oracle_score":  best_score,
                }
                rec.update({f"score_{a}": scores[a] for a in ENHANCEMENT_NAMES})
                records.append(rec)

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
