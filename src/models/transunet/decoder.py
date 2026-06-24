"""
Cascaded upsampling decoder for TransUNet (Phase 5).

Reshapes the transformer output back to a spatial grid (``conv_more``) and
upsamples ×2 per block, concatenating CNN skip features (hybrid design).
Self-contained (own conv block) so the package has no cross-model dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class TransUNetDecoder(nn.Module):
    def __init__(self, hidden_size, head_channels, decoder_channels, skip_channels):
        super().__init__()
        self.conv_more = Conv2dReLU(hidden_size, head_channels)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList([
            DecoderBlock(i, s, o)
            for i, s, o in zip(in_channels, skip_channels, decoder_channels)
        ])

    def forward(self, fmap, skips):
        x = self.conv_more(fmap)
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        return x
