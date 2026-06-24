"""
TransUNet (Phase 5) — first non-SMP baseline, validating the model factory's
extensibility.

Hybrid design (original, self-contained implementation):
  CNN backbone (SMP encoder via get_encoder — no SMP decoder internals touched)
  → 1×1 patch embedding of the stride-16 feature map → ViT transformer encoder
  → reshape to grid → cascaded upsampling decoder with CNN skip connections
  → 1×1 segmentation head → restore to input spatial size.

forward(x) returns logits B×num_classes×H×W (binary → B×1×H×W). SegModule wraps
it via normalize_model_output exactly like the other providers.
"""

import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder

from src.models.baselines.transunet.decoder import TransUNetDecoder
from src.models.baselines.transunet.vit import PatchEmbedding, TransformerEncoder

# Map ViT preset names to the CNN backbone used as the hybrid encoder.
_VIT_TO_ENCODER = {
    "R50-ViT-B_16": "resnet50",
}


class TransUNet(nn.Module):
    def __init__(self, img_size=224, in_channels=3, num_classes=1,
                 vit_name="R50-ViT-B_16", patch_size=16, n_skip=3,
                 hidden_size=768, mlp_dim=3072, num_heads=12, num_layers=12,
                 dropout_rate=0.0, decoder_channels=(256, 128, 64, 16),
                 encoder_weights=None):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_skip = n_skip
        self.grid = img_size // patch_size

        encoder_name = _VIT_TO_ENCODER.get(vit_name, "resnet50")
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights)
        enc_ch = self.encoder.out_channels   # e.g. resnet50: (3,64,256,512,1024,2048)

        # ViT consumes the stride-16 feature (index 4); skips come from /8,/4,/2.
        self.vit_feature_index = 4
        self.skip_indices = [3, 2, 1]
        skip_src = [enc_ch[3], enc_ch[2], enc_ch[1]]

        n_patches = self.grid * self.grid
        self.patch_embed = PatchEmbedding(enc_ch[4], hidden_size, n_patches, dropout_rate)
        self.transformer = TransformerEncoder(hidden_size, num_layers, num_heads, mlp_dim, dropout_rate)

        decoder_channels = list(decoder_channels)
        skip_channels = [
            (skip_src[i] if (i < n_skip and i < len(skip_src)) else 0)
            for i in range(len(decoder_channels))
        ]
        skip_channels[-1] = 0  # last (full-res) block has no skip
        self.decoder = TransUNetDecoder(
            hidden_size, head_channels=512,
            decoder_channels=decoder_channels, skip_channels=skip_channels,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder(x)

        tokens = self.patch_embed(feats[self.vit_feature_index])
        tokens = self.transformer(tokens)
        b, n, hidden = tokens.shape
        fmap = tokens.transpose(1, 2).reshape(b, hidden, self.grid, self.grid)

        skips = []
        for i in range(len(self.decoder.blocks)):
            if i < self.n_skip and i < len(self.skip_indices):
                skips.append(feats[self.skip_indices[i]])
            else:
                skips.append(None)

        decoded = self.decoder(fmap, skips)
        logits = self.segmentation_head(decoded)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


def build_transunet(model_cfg, dataset_cfg) -> TransUNet:
    params = model_cfg.get("params", {}) or {}
    model = TransUNet(
        img_size=int(model_cfg.img_size),
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=dataset_cfg.num_classes,
        vit_name=model_cfg.get("vit_name", "R50-ViT-B_16"),
        patch_size=model_cfg.get("patch_size", 16),
        n_skip=model_cfg.get("n_skip", 3),
        hidden_size=params.get("hidden_size", 768),
        mlp_dim=params.get("mlp_dim", 3072),
        num_heads=params.get("num_heads", 12),
        num_layers=params.get("num_layers", 12),
        dropout_rate=params.get("dropout_rate", 0.0),
        encoder_weights=model_cfg.get("encoder_weights", None),
    )
    pretrained_path = model_cfg.get("pretrained_path", None)
    if pretrained_path:
        from src.models.baselines.transunet.load_pretrained import load_transunet_weights
        load_transunet_weights(model, pretrained_path)
    return model
