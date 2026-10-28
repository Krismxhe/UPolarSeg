"""Hard polar-boundary measurements and their geometric validity flags."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.quantification.polar_readout import PolarReadout


@torch.no_grad()
def measure_hard_mask(
    readout: PolarReadout,
    masks: Tensor,
) -> dict[str, Tensor]:
    """Hard radial-boundary measurement with explicit validity flags.

    reference_valid:
        OD is centred and radial-prefix shaped;
        foreground is not clipped or outside the valid grid;
        the diameter selected by dv_mode is nondegenerate.

    aux_valid:
        additionally requires OD|PPA radial prefixes.
        Only then does one-hot occupancy equal boundary measurement.

    Invalid protected numeric values must not be reported as valid
    measurements.
    """

    readout._check_mask(masks)

    od = (
        (masks == readout.od_class)
        & readout.valid_grid
    )

    op = (
        (
            (masks == readout.od_class)
            | (masks == readout.ppa_class)
        )
        & readout.valid_grid
    )

    r = readout.radii.float()

    r_od = torch.where(
        od,
        r,
        0.0,
    ).amax(dim=-1)

    r_op = torch.where(
        op, 
        r,
        0.0,
    ).amax(dim=-1)

    result = readout._summarize(
        r_od,
        (r_op - r_od).clamp_min(0.0),
    )

    centred = od[..., 0].all(dim=-1)

    # Foreground re-entry after background means a broken radial prefix
    od_reentry = (
        (~od[..., :-1] & od[..., 1:])
    ).any(dim=(-2, -1))

    op_reentry = (
        (~op[..., :-1]) & op[..., 1:]
    ).any(dim=(-2, -1))

    edge = readout.last_valid[
        None, :, None
    ].expand(masks.shape[0], -1, 1)

    clipped = op.gather(
        -1,
        edge,
    ).any(dim=(-2, -1))

    outside = (
        (masks != 0) & ~readout.valid_grid
    ).any(dim=(-2, -1))

    # A disconnected OD is not accepted as the intended central chord.
    od_prefix = centred & ~od_reentry

    result["reference_valid"] = (
        od_prefix
        & ~clipped
        & ~outside
        & (
            result["dv"]
            >= readout.denominator_floor_px
        )
    )

    result["integral_compatible"] = (
        od_prefix & ~op_reentry
    )

    result["aux_valid"] = (
        result["reference_valid"]
        & result["integral_compatible"]
    )

    return result
