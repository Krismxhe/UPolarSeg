import torch.nn as nn
import torch.nn.functional as F

from src.models.research.modular_unet.decoder import ModularUnetDecoder
from src.models.research.modular_unet.encoder import build_encoder
from src.models.heads import BoundaryHead
from segmentation_models_pytorch.base import initialization as init


class ModularUNet(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        decoder_channels=(256, 128, 64, 32, 16),
        skip_cfg=None,
        boundary_enabled=False,
    ):
        super().__init__()

        self.boundary_enabled = bool(boundary_enabled)

        self.encoder = build_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(decoder_channels),
            weights=encoder_weights,
        )

        self.decoder = ModularUnetDecoder(
            self.encoder.out_channels,
            list(decoder_channels),
            skip_cfg or {},
        )

        final_ch = decoder_channels[-1]
        self.segmentation_head = nn.Conv2d(
            final_ch,
            num_classes,
            kernel_size=3,
            padding=1,
        )

        self.boundary_head = (
            BoundaryHead(final_ch, out_channels=1)
            if self.boundary_enabled
            else None
        )

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.boundary_head is not None:
            init.initialize_head(self.boundary_head)

    def forward(self, x, return_features=False):
        h, w = x.shape[-2:]

        features = self.encoder(x)
        decoded = self.decoder(features)

        seg_logits = self.segmentation_head(decoded)
        if seg_logits.shape[-2:] != (h, w):
            seg_logits = F.interpolate(
                seg_logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        if not self.boundary_enabled and not return_features:
            return seg_logits

        if self.boundary_enabled:
            boundary_logits = self.boundary_head(decoded)
            if boundary_logits.shape[-2:] != (h, w):
                boundary_logits = F.interpolate(
                    boundary_logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

            return {
                "seg_logits": seg_logits,
                "boundary_logits": boundary_logits,
            }

        return {
            "seg_logits": seg_logits,
            "features": {
                "decoder_final": decoded,
            },
        }


def build_modular_unet(model_cfg, dataset_cfg) -> ModularUNet:
    heads_cfg = model_cfg.get("heads", {}) or {}
    boundary_cfg = heads_cfg.get("boundary", {}) or {}

    return ModularUNet(
        encoder_name=model_cfg.get("encoder", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", None),
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=dataset_cfg.num_classes,
        decoder_channels=list(model_cfg.get("decoder_channels", [256, 128, 64, 32, 16])),
        skip_cfg=model_cfg.get("skip", {}),
        boundary_enabled=bool(boundary_cfg.get("enabled", False)),
    )