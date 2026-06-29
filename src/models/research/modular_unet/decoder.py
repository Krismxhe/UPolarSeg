"""
ModularUNet decoder.

This version supports channel-expanding SkipModules. The skip module is built
from the original encoder skip channel widths, then queried for its output
channel widths via ``out_channels_by_level``. DecoderBlock.conv1 is constructed
with those expanded skip widths.
"""

import torch.nn as nn

from src.models.research.modular_unet.blocks import DecoderBlock
from src.models.research.modular_unet.skip_modules import build_skip_module


class ModularUnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, skip_cfg):
        super().__init__()

        enc = list(encoder_channels)[1:][::-1]  # drop input, deepest first
        head = enc[0]

        in_channels = [head] + list(decoder_channels[:-1])
        skip_in_channels = list(enc[1:]) + [0]  # last block has no skip
        out_channels = list(decoder_channels)

        placement = str((skip_cfg or {}).get("placement", "before_concat"))

        active_skip_in_channels = [c for c in skip_in_channels if c > 0]
        skip_module = build_skip_module(
            skip_cfg,
            channels_by_level=active_skip_in_channels,
        )

        if hasattr(skip_module, "out_channels_by_level"):
            active_skip_out_channels = list(
                skip_module.out_channels_by_level(active_skip_in_channels)
            )
        else:
            active_skip_out_channels = active_skip_in_channels

        if len(active_skip_out_channels) != len(active_skip_in_channels):
            raise ValueError(
                "skip_module.out_channels_by_level(...) must return one output "
                "channel count for each input skip level. "
                f"Got {len(active_skip_out_channels)} for {len(active_skip_in_channels)} levels."
            )

        # Reinsert 0 for the final decoder block without skip.
        skip_out_iter = iter(active_skip_out_channels)
        skip_out_channels = [next(skip_out_iter) if c > 0 else 0 for c in skip_in_channels]

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    i,
                    s_out,
                    o,
                    skip_module,
                    level,
                    placement=placement,
                )
                for level, (i, s_out, o) in enumerate(
                    zip(in_channels, skip_out_channels, out_channels)
                )
            ]
        )

    def forward(self, features):
        feats = features[1:][::-1]  # drop input, deepest first
        x, skips = feats[0], feats[1:]
        for level, block in enumerate(self.blocks):
            skip = skips[level] if level < len(skips) else None
            x = block(x, skip)
        return x
