"""
Pretrained-weight loading for TransUNet (Phase 5).

SCOPE / HONEST LIMITATION:
  This loads a torch ``state_dict`` checkpoint of THIS TransUNet implementation
  (``.pt`` / ``.pth``), partially (``strict=False``). It does **NOT** convert the
  official Google ViT / R50 ``.npz`` weights — that requires bespoke key mapping
  and is not implemented here. So "pretrained loading" is intentionally partial:
  the CNN backbone can still be ImageNet-initialised via ``model.encoder_weights``,
  but the transformer is trained from scratch unless a compatible torch
  checkpoint is supplied.
"""

import torch


def load_transunet_weights(model, pretrained_path):
    """Partially load a torch state_dict into ``model`` (no-op if path is None)."""
    if not pretrained_path:
        return model
    state = torch.load(pretrained_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[TransUNet] loaded weights from {pretrained_path}: "
        f"{len(missing)} missing / {len(unexpected)} unexpected keys "
        f"(strict=False; official .npz weights are not supported)."
    )
    return model
