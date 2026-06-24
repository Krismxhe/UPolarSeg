"""
ModularUNet decoder (Phase 6).

Standard UNet-style decoder, but each skip connection is routed through a
configurable SkipFusion + SkipModule (built from ``model.skip``). Channel
bookkeeping follows the usual SMP UNet convention:

    encoder out_channels (incl input) -> drop input -> reverse (deepest first)
    head        = deepest encoder channels
    in_channels = [head] + decoder_channels[:-1]
    skip        = encoder stages (shallower) + [0] for the last (full-res) block
    out         = decoder_channels
"""

import torch.nn as nn

from src.models.modular_unet.blocks import DecoderBlock
from src.models.modular_unet.skip_modules import build_skip_module


class ModularUnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, skip_cfg):
        super().__init__()
        enc = list(encoder_channels)[1:][::-1]      # drop input, deepest first
        head = enc[0]
        in_channels = [head] + list(decoder_channels[:-1])
        skip_channels = list(enc[1:]) + [0]         # last block has no skip
        out_channels = list(decoder_channels)

        placement = str((skip_cfg or {}).get("placement", "before_concat"))
        # One shared skip module receives the level index at each block.
        skip_module = build_skip_module(
            skip_cfg, channels_by_level=[c for c in skip_channels if c > 0]
        )

        self.blocks = nn.ModuleList([
            DecoderBlock(i, s, o, skip_module, level, placement=placement)
            for level, (i, s, o) in enumerate(zip(in_channels, skip_channels, out_channels))
        ])

    def forward(self, features):
        feats = features[1:][::-1]                  # drop input, deepest first
        x, skips = feats[0], feats[1:]
        for level, block in enumerate(self.blocks):
            skip = skips[level] if level < len(skips) else None
            x = block(x, skip)
        return x
