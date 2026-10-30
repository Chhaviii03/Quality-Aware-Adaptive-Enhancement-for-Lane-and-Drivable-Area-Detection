"""
src/models/train.py
-------------------
Training script for U-Net and SegFormer.

Features
--------
* Reads all hyperparameters from a YAML config.
* Supports --debug (20 images, 2 epochs, CPU).
* Supports --resume <checkpoint.pth>.
* Logs config snapshot + git commit hash + hardware string.
* Early stopping on val loss.
* Saves checkpoint every N epochs and at the best val metric.

Usage
-----
    # Debug run (CPU, 20 imgs, 2 epochs)
    python src/models/train.py --model unet --config configs/unet.yaml --debug

    # Colab / GPU run (override paths via YAML or flags)
    python src/models/train.py --model unet --config configs/unet.yaml \
        --data_root /content/drive/MyDrive/bdd100k \
        --checkpoint_dir /content/drive/MyDrive/checkpoints/unet

DO NOT run this script locally without --debug; it requires a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.models.unet import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(-2, -1))
    denom  = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def combined_loss(
    logits: torch.Tensor,
    masks:  torch.Tensor,
    lane_pos_weight: float = 8.0,
    bce_w: float = 0.5,
    dice_w: float = 0.5,
) -> torch.Tensor:
    """BCE + Dice with higher positive weight on the lane channel."""
    B, C, H, W = logits.shape
    total = torch.zeros(1, device=logits.device)

    for ch in range(C):
        pos_w = lane_pos_weight if ch == 1 else 1.0
        bce_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_w, device=logits.device)
        )
        bce = bce_fn(logits[:, ch], masks[:, ch].float())
        dl  = dice_loss(logits[:, ch], masks[:, ch].float()).mean()
        total = total + bce_w * bce + dice_w * dl

    return total / C


# ─────────────────────────────────────────────────────────────────────────────
# Minimal dataset (real DataLoader to be wired in by bdd_subset / labels)
# ─────────────────────────────────────────────────────────────────────────────

class BDDDataset(torch.utils.data.Dataset):
    """Thin wrapper — loads images and masks from a manifest CSV."""

    def __init__(
        self,
        manifest_csv: str,
        data_root:    str,
        split:        str,
        image_width:  int = 640,
        image_height: int = 352,
        max_n:        int | None = None,
    ) -> None:
        import pandas as pd
        import cv2
        self._cv2 = cv2
        self.data_root    = data_root
        self.split        = split
        self.W            = image_width
        self.H            = image_height
        df = pd.read_csv(manifest_csv)
        if max_n is not None:
            df = df.head(max_n)
        self.records = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        bgr = self._cv2.imread(row["img_path"])
        if bgr is None:
            bgr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        bgr = self._cv2.resize(bgr, (self.W, self.H))

        # Load masks
        from src.data.labels import load_lane_mask, load_road_mask
        lane = load_lane_mask(self.data_root, self.split, row["image_id"],
                              self.W, self.H)
        road = load_road_mask(self.data_root, self.split, row["image_id"],
                              self.W, self.H)

        # To tensor
        img_t  = torch.from_numpy(bgr).permute(2, 0, 1).float() / 255.0
        lane_t = torch.from_numpy((lane > 0).astype(np.float32))
        road_t = torch.from_numpy((road > 0).astype(np.float32))
        mask_t = torch.stack([road_t, lane_t], dim=0)   # (2, H, W)

        return img_t, mask_t, row["image_id"], row["condition"]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, args: argparse.Namespace) -> None:
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() and not args.debug else "cpu"
    print(f"Device: {device}")

    # Paths
    ckpt_dir   = Path(args.checkpoint_dir or cfg["checkpoint_dir"])
    log_dir    = Path(cfg.get("log_dir", "results/logs/unet"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = Path(args.manifest_dir or cfg.get("manifest_dir", "data/manifests"))
    data_root    = args.data_root or cfg["data_root"]
    H, W         = cfg["image_height"], cfg["image_width"]

    debug_cfg = cfg.get("debug", {})
    max_n     = debug_cfg.get("n_images", None)  if args.debug else None
    epochs    = debug_cfg.get("epochs",   2)     if args.debug else cfg["epochs"]
    batch     = debug_cfg.get("batch_size", 2)   if args.debug else cfg["batch_size"]

    # Datasets
    train_ds = BDDDataset(manifest_dir / "train.csv", data_root, "train", W, H, max_n)
    val_ds   = BDDDataset(manifest_dir / "val.csv",   data_root, "val",   W, H, max_n)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True,
                                               num_workers=0, pin_memory=False)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch, shuffle=False,
                                               num_workers=0, pin_memory=False)

    # Model
    if args.model == "unet":
        model = UNet(
            in_channels=cfg.get("in_channels", 3),
            base_width=cfg.get("base_width", 32),
            depth=cfg.get("depth", 4),
            out_channels=cfg.get("out_channels", 2),
        ).to(device)
    else:
        from src.models.segformer import build_segformer
        model = build_segformer(
            backbone=cfg.get("backbone", "nvidia/mit-b0"),
            out_channels=cfg.get("out_channels", 2),
        ).to(device)

    # Optimiser
    lr  = cfg.get("lr", 1e-3)
    wd  = cfg.get("weight_decay", 1e-4)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0
    best_val    = float("inf")
    patience    = cfg.get("early_stopping_patience", 7)
    no_improve  = 0

    # Resume
    resume = args.resume or cfg.get("resume_from")
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val_loss", best_val)
        print(f"Resumed from epoch {start_epoch} ({resume})")

    # Log config
    run_meta = {
        "config": cfg,
        "args": vars(args),
        "device": device,
    }
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        run_meta["git_commit"] = repo.head.commit.hexsha
    except Exception:
        run_meta["git_commit"] = "unavailable"

    with open(log_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    loss_cfg = cfg.get("loss", {})

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, masks, *_ in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            opt.zero_grad()
            logits = model(imgs)
            loss = combined_loss(
                logits, masks,
                lane_pos_weight=loss_cfg.get("lane_pos_weight", 8.0),
                bce_w=loss_cfg.get("bce_weight", 0.5),
                dice_w=loss_cfg.get("dice_weight", 0.5),
            )
            loss.backward()
            opt.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, *_ in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = combined_loss(
                    logits, masks,
                    lane_pos_weight=loss_cfg.get("lane_pos_weight", 8.0),
                    bce_w=loss_cfg.get("bce_weight", 0.5),
                    dice_w=loss_cfg.get("dice_weight", 0.5),
                )
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:03d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"({elapsed:.1f}s)")

        # Checkpoint every N epochs
        save_every = cfg.get("save_every_n_epochs", 5)
        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_val_loss": best_val,
                "val_loss": val_loss,
            }, ckpt_path)

        # Best checkpoint
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_val_loss": best_val,
                "val_loss": val_loss,
            }, ckpt_dir / "best.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"Training complete. Best val loss: {best_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",          required=True, choices=["unet", "segformer"])
    p.add_argument("--config",         default=None)
    p.add_argument("--data_root",      default=None)
    p.add_argument("--manifest_dir",   default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--resume",         default=None)
    p.add_argument("--debug",          action="store_true",
                   help="20 images, 2 epochs, CPU — for local testing")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    config_path = args.config or f"configs/{args.model}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args)
