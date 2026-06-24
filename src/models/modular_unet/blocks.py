"""
Decoder building blocks (Phase 6).

A DecoderBlock upsamples the running decoder feature ×2, optionally fuses the
encoder skip via SkipFusion, then applies two Conv-BN-ReLU layers. This mirrors
a standard UNet decoder block but routes the skip through the configurable
fusion / skip module instead of a hard-coded concat.
"""

import torch.nn as nn
import torch.nn.functional as F

from src.models.modular_unet.fusion import SkipFusion


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, skip_module, level,
                 placement="before_concat", mode="concat"):
        super().__init__()
        self.level = level
        self.has_skip = skip_channels > 0
        self.fusion = (
            SkipFusion(skip_module, mode=mode, placement=placement)
            if self.has_skip else None
        )
        # After fusion the channel width is in_channels + skip_channels
        # (skip module preserves channel count in v1).
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.has_skip and skip is not None:
            x = self.fusion(x, skip, self.level)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
