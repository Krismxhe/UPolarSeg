"""
Encoder builder (Phase 6).

Reuses segmentation-models-pytorch's public encoder factory (``get_encoder``) so
ModularUNet inherits 100+ backbones and ImageNet weights WITHOUT touching any
SMP decoder internals. The encoder returns a list of feature maps; its
``.out_channels`` describes the channel count at each stage (including the input
stage at index 0).
"""

from segmentation_models_pytorch.encoders import get_encoder


def build_encoder(name: str, in_channels: int = 3, depth: int = 5, weights=None):
    return get_encoder(name, in_channels=in_channels, depth=depth, weights=weights)
