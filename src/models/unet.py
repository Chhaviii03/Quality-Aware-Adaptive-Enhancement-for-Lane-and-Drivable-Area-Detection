"""
src/models/unet.py
------------------
U-Net from scratch with two sigmoid output channels:
    channel 0 = drivable area
    channel 1 = lane

Architecture: 4 down / 4 up levels, base_width=32 (≈8M parameters).

Usage
-----
    from src.models.unet import UNet
    model = UNet(in_channels=3, base_width=32, depth=4, out_channels=2)
    logits = model(x)   # x: (B, 3, 352, 640), logits: (B, 2, 352, 640)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    """MaxPool → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net from scratch.

    Parameters
    ----------
    in_channels : int   Input channels (3 for RGB/BGR).
    base_width  : int   Feature maps in the first encoder block (doubles each level).
    depth       : int   Number of down / up levels (4 recommended).
    out_channels: int   Output channels (2: drivable + lane).
    """

    def __init__(
        self,
        in_channels:  int = 3,
        base_width:   int = 32,
        depth:        int = 4,
        out_channels: int = 2,
    ) -> None:
        super().__init__()

        widths = [base_width * (2 ** i) for i in range(depth + 1)]
        # widths = [32, 64, 128, 256, 512] for base_width=32, depth=4

        # Encoder
        self.inc = _DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList([
            _Down(widths[i], widths[i + 1]) for i in range(depth)
        ])

        # Decoder
        self.ups = nn.ModuleList([
            _Up(widths[depth - i], widths[depth - i - 1], widths[depth - i - 1])
            for i in range(depth)
        ])

        # Output head
        self.out_conv = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Remove the bottleneck from skips (it's `x`)
        x = skips.pop()          # bottleneck
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        return self.out_conv(x)  # raw logits; apply sigmoid externally


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained / SMP ResNet18 U-Net Wrapper (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────

def build_smp_unet(
    checkpoint_path: str | Path | None = "results/checkpoints/unet_drivable_best.pt",
    allow_random_init: bool = False,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Build segmentation_models_pytorch Unet(resnet18, in_channels=3, classes=1).

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to checkpoint .pt containing 'state_dict', 'mean', 'std', 'H', 'W'.
    allow_random_init : bool
        If True, returns a randomly initialized model when checkpoint is missing (for smoke tests).
        If False, raises FileNotFoundError with a clear message.
    device : str
        Target device ('cpu' or 'cuda').

    Returns
    -------
    model : nn.Module
    meta : dict
        Contains 'mean', 'std', 'H', 'W', 'checkpoint_path', and 'is_mock'.
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    cp_path = Path(checkpoint_path) if checkpoint_path else None
    if cp_path and cp_path.exists():
        state = torch.load(cp_path, map_location=device)
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict)
        meta = {
            "architecture": "smp_unet_resnet18",
            "mean": state.get("mean", [0.485, 0.456, 0.406]),
            "std": state.get("std", [0.229, 0.224, 0.225]),
            "H": state.get("H", 352),
            "W": state.get("W", 640),
            "checkpoint_path": str(cp_path),
            "is_mock": False,
        }
    elif allow_random_init:
        meta = {
            "architecture": "smp_unet_resnet18",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "H": 352,
            "W": 640,
            "checkpoint_path": str(cp_path) if cp_path else "None",
            "is_mock": True,
        }
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please train the model on Google Colab and place the resulting "
            "'unet_drivable_best.pt' checkpoint in 'results/checkpoints/'."
        )

    model.to(device)
    model.eval()
    return model, meta


def predict_drivable_unet(
    model: nn.Module,
    bgr_img: np.ndarray,
    meta: dict,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run preprocessing, inference, and postprocessing for drivable area segmentation.

    Steps:
      1. BGR to RGB
      2. Resize to 640x352 with cv2.INTER_AREA
      3. Normalise by dividing by 255.0, then subtract mean and divide by std
      4. Forward pass + sigmoid
      5. Threshold sigmoid > 0.5 to get binary mask
      6. Resize back to original frame size with cv2.INTER_NEAREST (mask) and INTER_LINEAR (prob)

    Returns
    -------
    prob_map : np.ndarray
        Float32 probability map in [0, 1] at original image resolution.
    binary_mask : np.ndarray
        Uint8 binary mask (0 or 255) at original image resolution.
    """
    import cv2

    orig_h, orig_w = bgr_img.shape[:2]
    target_w = meta.get("W", 640)
    target_h = meta.get("H", 352)
    mean = np.array(meta.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(meta.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    # 1. BGR -> RGB
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # 2. Resize to target size (640x352) with INTER_AREA
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 3. / 255 and normalize
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - mean) / std

    # 4. Torch tensor
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # shape (target_h, target_w)

    mask = (prob > 0.5).astype(np.uint8) * 255

    # 6. Resize back if needed
    if (orig_h, orig_w) != (target_h, target_w):
        prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_orig = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        prob_orig = prob
        mask_orig = mask

    return prob_orig, mask_orig
