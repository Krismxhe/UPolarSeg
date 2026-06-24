"""
ModularUNet (Phase 6).

A controllable UNet for skip-connection module research: SMP encoder backbone +
our own decoder whose skip connections are routed through a configurable
SkipFusion / SkipModule. v1 ships the identity skip (ablation control).

forward(x) returns a logits tensor B×num_classes×H×W (binary → B×1×H×W),
restored to the input spatial size. SegModule wraps it via
``normalize_model_output`` exactly like the SMP baselines, so training /
evaluation are unchanged.
"""

import torch.nn as nn
import torch.nn.functional as F

from src.models.modular_unet.decoder import ModularUnetDecoder
from src.models.modular_unet.encoder import build_encoder


class ModularUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights=None, in_channels=3,
                 num_classes=1, decoder_channels=(256, 128, 64, 32, 16), skip_cfg=None):
        super().__init__()
        self.encoder = build_encoder(
            encoder_name, in_channels=in_channels,
            depth=len(decoder_channels), weights=encoder_weights,
        )
        self.decoder = ModularUnetDecoder(
            self.encoder.out_channels, list(decoder_channels), skip_cfg or {}
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.encoder(x)
        decoded = self.decoder(features)
        logits = self.segmentation_head(decoded)
        # Guarantee output spatial size == input (covers sizes not divisible by 32).
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


def build_modular_unet(model_cfg, dataset_cfg) -> ModularUNet:
    return ModularUNet(
        encoder_name=model_cfg.get("encoder", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", None),
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=dataset_cfg.num_classes,
        decoder_channels=list(model_cfg.get("decoder_channels", [256, 128, 64, 32, 16])),
        skip_cfg=model_cfg.get("skip", {}),
    )
