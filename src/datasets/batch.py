"""
Batch unpacking helper.

Bridges the new dict-style batch (image / mask / metadata) with the legacy
2-tuple batch, so the LightningModule does not need to know which one it is
handed. This keeps old checkpoints / callers working while enabling per-case
evaluation and prediction saving downstream.
"""

from __future__ import annotations

from typing import Any


def unpack_batch(batch: Any):
    """Return (images, masks, metadata) from either a dict batch or a legacy tuple batch.

    - dict batch  → (batch["image"], batch["mask"], batch)
    - 2-tuple/list → (images, masks, {})

    The metadata dict is empty for legacy tuple batches.
    """
    if isinstance(batch, dict):
        return batch["image"], batch["mask"], batch
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, masks = batch
        return images, masks, {}
    raise TypeError(f"Unsupported batch type: {type(batch)}")

def original_size_from_metadata(meta, index: int):
    """Return one original image size as (H, W), or None when not recorded.

    PyTorch's default_collate transposes each dataset tuple ``(H, W)`` into
    ``[heights_tensor, widths_tensor]``. Explicit Bx2 arrays/tensors and an
    unbatched (H, W) pair are also accepted. Keep this helper framework-free.
    """
    if not meta or meta.get("orig_size") is None:
        return None
    sizes = meta["orig_size"]
    ndim = getattr(sizes, "ndim", None)
    if ndim == 2:
        if sizes.shape[1] != 2:
            raise ValueError("orig_size array must have shape (B, 2)")
        pair = sizes[index]
    elif ndim == 1:
        if len(sizes) != 2:
            raise ValueError("Unbatched orig_size must be an (H, W) pair")
        pair = sizes
    elif isinstance(sizes, (list, tuple)) and len(sizes) == 2:
        if all(getattr(v, "ndim", 0) == 1 for v in sizes):
            pair = sizes[0][index], sizes[1][index]
        elif all(getattr(v, "ndim", 0) == 0 and not isinstance(v, (list, tuple)) for v in sizes):
            pair = sizes
        else:
            raise ValueError("orig_size must be collated H/W vectors, a Bx2 array, or an (H, W) pair")
    else:
        raise ValueError("orig_size must be collated H/W vectors, a Bx2 array, or an (H, W) pair")

    result = []
    for value in pair:
        value = value.item() if hasattr(value, "item") else value
        try:
            size = int(value)
            valid = not isinstance(value, bool) and size > 0 and float(value) == size
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"orig_size must contain positive integer dimensions, got {pair!r}")
        result.append(size)
    return tuple(result)