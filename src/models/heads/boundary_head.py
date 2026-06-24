"""
BoundaryHead skeleton (Phase 9b).

A minimal head that maps a decoder feature map to boundary logits. It is a
building block for future multi-task models; it is intentionally simple (a 1×1
projection) and is not wired into the SMP baselines.

Contract: forward(feature: B×C×H×W) -> boundary_logits: B×out_channels×H×W.
"""

import torch.nn as nn


class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature):
        return self.proj(feature)
