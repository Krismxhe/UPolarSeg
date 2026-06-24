"""
Skip fusion (Phase 6).

Decides how a (possibly transformed) skip feature is combined with the upsampled
decoder feature. v1 supports only ``placement="before_concat"`` with
``mode="concat"``; other placements/modes raise NotImplementedError (TODO).
"""

import torch
import torch.nn as nn


class SkipFusion(nn.Module):
    def __init__(self, skip_module, mode: str = "concat", placement: str = "before_concat"):
        super().__init__()
        self.skip_module = skip_module
        self.mode = mode
        self.placement = placement

    def forward(self, decoder_feature, skip_feature, level: int = 0):
        if self.placement != "before_concat":
            raise NotImplementedError(
                f"SkipFusion placement='{self.placement}' not implemented (TODO); "
                "v1 supports 'before_concat'."
            )
        if self.mode != "concat":
            raise NotImplementedError(
                f"SkipFusion mode='{self.mode}' not implemented (TODO); v1 supports 'concat'."
            )
        # before_concat: transform the skip first, then concat (channels preserved
        # by the skip module, so concat width = decoder_channels + skip_channels).
        skip_feature = self.skip_module(skip_feature, decoder_feature, level)
        return torch.cat([decoder_feature, skip_feature], dim=1)
