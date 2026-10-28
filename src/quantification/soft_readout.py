"""Differentiable probability-weighted radial occupation.

This readout is a soft surrogate, not a hard-boundary measurement. It is
independent of which model or training method produced the probabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from src.quantification.polar_readout import PolarReadout


def read_soft_probabilities(
    readout: PolarReadout,
    probabilities: Tensor,
) -> dict[str, Tensor]:
    """Differentiable probability-weighted radial occupation.

    This is a soft surrogate. Do not detach probabilities here.
    """
    if (
        probabilities.ndim != 4 or probabilities.shape[1] != 3
    ):
        raise ValueError(
            "Expected probabilities [B, 3, angle, radius]."
        )

    expected = (
        readout.angle_bins,
        readout.radius_bins,
    )

    if tuple(probabilities.shape[-2:]) != expected:
        raise ValueError(
            "Prediction grid differs from cfg.polar.geometry."
        )

    p = probabilities.float()
    w = readout.ray_weights.float()

    od = (
        p[:, readout.od_class] * w
    ).sum(dim=-1)

    width = (
        p[:, readout.ppa_class] * w
    ).sum(dim=-1)

    return readout._summarize(od, width)
