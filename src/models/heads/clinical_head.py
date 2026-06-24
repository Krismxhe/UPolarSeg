"""
ClinicalHead skeleton (Phase 9b).

A minimal head that regresses a fixed-size clinical vector from a feature map
(global-pooled → linear). Building block for a future *learned* clinical task;
the first-version clinical metrics are computed deterministically from the
predicted mask instead (see src.metrics.clinical_metrics), so this head is not
wired into the baseline.

Contract: forward(feature: B×C×H×W) -> B×num_outputs.
"""

import torch.nn as nn


class ClinicalHead(nn.Module):
    def __init__(self, in_channels: int, num_outputs: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_outputs)

    def forward(self, feature):
        pooled = self.pool(feature).flatten(1)
        return self.fc(pooled)
