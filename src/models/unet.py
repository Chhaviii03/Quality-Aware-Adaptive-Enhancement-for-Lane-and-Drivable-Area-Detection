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
