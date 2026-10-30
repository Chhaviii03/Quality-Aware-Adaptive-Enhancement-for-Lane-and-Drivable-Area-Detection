"""
src/models/segformer.py
-----------------------
SegFormer-B0 fine-tuning wrapper using HuggingFace Transformers.

The pretrained encoder (nvidia/mit-b0) is loaded from HuggingFace Hub.
A lightweight two-channel segmentation head replaces the original head.
Logits are bilinearly upsampled to the full 640×352 resolution.

Usage
-----
    from src.models.segformer import build_segformer
    model = build_segformer("nvidia/mit-b0", out_channels=2)
    logits = model(x)   # x: (B, 3, 352, 640) → logits: (B, 2, 352, 640)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHead(nn.Module):
    """Lightweight MLP decode head that fuses multi-scale features."""

    def __init__(self, in_channels: list[int], embed_dim: int, out_channels: int) -> None:
        super().__init__()
        self.linear_fuse = nn.Conv2d(len(in_channels) * embed_dim, embed_dim, 1, bias=False)
        self.bn   = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(0.1)
        self.cls  = nn.Conv2d(embed_dim, out_channels, 1)

        self.linear_c = nn.ModuleList([
            nn.Conv2d(ch, embed_dim, 1, bias=False) for ch in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Upsample all feature maps to the finest resolution
        target_size = features[0].shape[-2:]
        outs = []
        for linear, feat in zip(self.linear_c, features):
            x = linear(feat)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.relu(self.bn(self.linear_fuse(x)))
        x = self.drop(x)
        return self.cls(x)


class SegFormerSeg(nn.Module):
    """SegFormer encoder + custom two-channel head."""

    def __init__(self, backbone_model, out_channels: int = 2, embed_dim: int = 256) -> None:
        super().__init__()
        self.encoder = backbone_model
        # B0 hidden sizes: [32, 64, 160, 256]
        hidden_sizes = backbone_model.config.hidden_sizes
        self.head = SegFormerHead(hidden_sizes, embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        # HuggingFace SegFormer encoder returns pixel_values → hidden states
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states    # tuple of 4 tensors
        logits = self.head(list(hidden_states))  # (B, out_ch, H/4, W/4)
        # Upsample to input resolution
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def build_segformer(
    backbone: str  = "nvidia/mit-b0",
    out_channels: int = 2,
    embed_dim: int    = 256,
) -> SegFormerSeg:
    """Build a SegFormer model with a custom segmentation head.

    Downloads the pretrained backbone from HuggingFace Hub on first call.

    Parameters
    ----------
    backbone : str
        HuggingFace model ID, e.g. "nvidia/mit-b0" or "nvidia/mit-b1".
    out_channels : int
        Number of output channels (2: drivable + lane).
    embed_dim : int
        Hidden dimension in the decode head.

    Returns
    -------
    SegFormerSeg
    """
    from transformers import SegformerModel

    encoder = SegformerModel.from_pretrained(backbone)
    return SegFormerSeg(encoder, out_channels=out_channels, embed_dim=embed_dim)
