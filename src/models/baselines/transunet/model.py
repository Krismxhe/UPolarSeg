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
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# Map ViT preset names to the CNN backbone used as the hybrid encoder.
_VIT_TO_ENCODER = {
    "R50-ViT-B_16": "resnet50",
}

def build_transunet(model_cfg, dataset_cfg):
    params = model_cfg.get("params", {}) or {}
    config_vit = CONFIGS_ViT_seg[model_cfg.vit_name]
    config_vit.n_classes = dataset_cfg.num_classes
    config_vit.n_skip = model_cfg.get("n_skip", 3)
    if model_cfg.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(model_cfg.img_size / model_cfg.get("patch_size", 16)), int(model_cfg.img_size / model_cfg.get("patch_size", 16)))
    model = ViT_seg(config_vit, img_size=model_cfg.img_size, num_classes=config_vit.n_classes)
    pretrained_path = model_cfg.get("pretrained_path", None)
    if pretrained_path:
        from src.models.baselines.transunet.load_pretrained import load_transunet_weights
        load_transunet_weights(model, pretrained_path)
    return model
