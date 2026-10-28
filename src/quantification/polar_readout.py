"""Shared polar geometry and compatible hard/soft readout entry points.

Geometry buffers remain non-persistent and belong directly to this module.
The readout has no trainable parameters and does not define a training loss.
"""

from __future__ import annotations

import math
from typing import TypeVar

import torch
from torch import Tensor, nn

from src.quantification.hard_readout import measure_hard_mask
from src.quantification.soft_readout import read_soft_probabilities


def _opposite_vertical(
        profile: Tensor,
        angle_deg: float = 90.0,
)-> Tensor:
    """Sum opposite radii on a fixed axis, with periodic interpolation."""

    n = profile.shape[-1]
    values = []

    for angle in (angle_deg, angle_deg + 180.0):
        x = (angle % 360.0) * n / 360.0
        i = math.floor(x)
        t = x - i

        values.append(
            (1.0 - t) * profile[..., i % n] + t * profile[..., (i+1) % n] 
        )
    
    return values[0] + values[1]


def _max_center_chord(profile: Tensor) -> Tensor:
    """Longest opposite-ray sum through the polar origin.

    ``profile[..., i]`` is the OD radius at angle ``2*pi*i/A``. Even angular
    grids have an exact opposite row. On odd grids, the opposite direction
    lies halfway between two rows and uses periodic linear interpolation.
    Maximize the paired sum, rather than choosing the longest single radius.
    """
    if profile.ndim < 1 or profile.shape[-1] == 0:
        raise ValueError("OD profile must have a nonempty angle dimension.")
    angles = profile.shape[-1]
    half = angles // 2
    opposite = torch.roll(profile, shifts=-half, dims=-1)
    if angles % 2:
        opposite = 0.5 * (
            opposite + torch.roll(profile, shifts=-(half + 1), dims=-1)
        )
    return (profile + opposite).amax(dim=-1)


class PolarReadout(nn.Module):
    """One geometry shared by hard measurement and differentiable occupation.

    Call measure_mask for integer masks or read_probabilities for class
    probabilities. Choosing a readout is independent of the model identity.
    """

    def __init__(self, 
                 valid_grid: Tensor,
                 max_radius: float,
                 od_class: int = 1,
                 ppa_class: int = 2,
                 denominator_floor_px: float = 1.0,
                 vertical_angle_deg: float = 90.0,
                 dv_mode: str = "fixed_vertical",
    ):
        super().__init__()

        if valid_grid.ndim != 2 or min(valid_grid.shape) < 4:
            raise ValueError(
                "valid_grid must be [angle, radius], both >= 4."
            )

        if not math.isfinite(max_radius) or max_radius <= 0:
            raise ValueError(
                "max_radius must be finite and positive."
            )

        if (
            not math.isfinite(denominator_floor_px)
            or denominator_floor_px <= 0
        ):
            raise ValueError(
                "denominator_floor_px must be finite and positive."
            )

        if not math.isfinite(vertical_angle_deg):
            raise ValueError(
                "vertical_angle_deg must be finite."
            )

        self.dv_mode = str(dv_mode).strip().lower()
        if self.dv_mode not in {"fixed_vertical", "max_center_chord"}:
            raise ValueError(
                "quantification.dv_mode must be 'fixed_vertical' or "
                f"'max_center_chord', got {dv_mode!r}."
            )

        if {od_class, ppa_class} != {1, 2}:
            raise ValueError(
                "Requires background=0 and OD/PPA=1/2."
            )

        valid = valid_grid.bool()

        if not bool(valid[:, 0].all()):
            raise ValueError(
                "Polar origin must be inside the image."
            )

        if bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError(
                "Image-valid samples must form radial prefixes."
            )
        
        self.od_class = int(od_class)
        self.ppa_class = int(ppa_class)
        self.max_radius = float(max_radius)

        self.denominator_floor_px = float(
            denominator_floor_px
        )

        # Used only by fixed_vertical. An axis is undirected: theta and
        # theta+180 are identical. Retain the attribute for old callers.
        self.vertical_angle_deg = (
            float(vertical_angle_deg) % 180
        )

        self.angle_bins, self.radius_bins = valid.shape

        dr = self.max_radius / self.radius_bins

        j = torch.arange(
            self.radius_bins,
            device = valid.device
        )

        weights = valid.float() * dr

        # Last-sample convention
        # OD at indices 0..k has radius K * dr not (k+1) * dr
        weights[:, 0] = 0.0

        last_valid = torch.where(
            valid,
            j,
            -1,
        ).amax(dim=-1)

        # Geometry only: no trainable parameters or persistent weight keys
        for name, value in (
            ("valid_grid", valid),
            ("ray_weights", weights),
            ("radii", j.float() * dr),
            ("last_valid", last_valid),
        ):
            self.register_buffer(
                name,
                value,
                persistent=False,
            )

        # Descriptive provenance only; never a parameter or persistent buffer.
        # A config-built readout fills this with resolved image geometry.
        self.geometry_metadata: dict[str, object] = {}

    def _check_mask(self, masks: Tensor) -> None:
        expected = (
            self.angle_bins,
            self.radius_bins,
        )

        if (
            masks.ndim != 3
            or tuple(masks.shape[-2:]) != expected
        ):
            raise ValueError(
                "Expected masks [B, angle_bins, radius_bins]."
            )

        if (
            masks.is_floating_point()
            or masks.dtype == torch.bool
        ):
            raise TypeError(
                "Masks must contain integer class IDs."
            )

        if bool(((masks < 0) | (masks > 2)).any()):
            raise ValueError(
                "Only labels 0/1/2 are supported; no ignore labels."
            )

    def _summarize(
        self,
        od: Tensor,
        width: Tensor,
    ) -> dict[str, Tensor]:
        pmw = width.amax(dim=-1)

        if self.dv_mode == "max_center_chord":
            dv = _max_center_chord(od)
        else:
            dv = _opposite_vertical(od, self.vertical_angle_deg)

        return {
            # primary measurements
            "pmw": pmw,
            "dv": dv,
            "pvdr": pmw / dv.clamp_min(
                self.denominator_floor_px
            ),

            # candidate profiles
            "od_profile": od,
            "width_profile": width,
        }

    def read_probabilities(self, probabilities: Tensor) -> dict[str, Tensor]:
        """Read soft radial occupation without detaching probabilities."""
        return read_soft_probabilities(self, probabilities)

    @torch.no_grad()
    def measure_mask(self, masks: Tensor) -> dict[str, Tensor]:
        """Read hard boundaries and validity flags without tracking gradients."""
        return measure_hard_mask(self, masks)


_ReadoutT = TypeVar("_ReadoutT", bound=PolarReadout)


def _build_polar_readout(cfg, readout_type: type[_ReadoutT]) -> _ReadoutT:
    """Share the existing config-to-geometry construction with loss builders.

    This helper deliberately does not inspect the training-loss enable flag
    or its weight. It imports image-warp dependencies only when called.
    """
    q = cfg.get("quantification") or {}

    dataset = cfg["dataset"]

    if (
        int(dataset["num_classes"]) != 3
        or dataset["mask_mode"] != "index"
    ):
        raise ValueError(
            "Requires 3-class index masks."
        )

    polar = cfg.get("polar") or {}

    if not polar.get("enabled", False):
        raise ValueError(
            "Quantification requires polar=enabled."
        )

    g = polar.get("geometry") or {}

    if g.get("mode", "linear") != "linear":
        raise ValueError(
            "This readout does not support logarithmic radial grids."
        )


    size = int(cfg["train"]["img_size"])
    center = g.get("center")

    if center is not None:
        center = tuple(float(x) for x in center)

        if (
            len(center) != 2
            or not all(
                math.isfinite(x) and 0 <= x < size
                for x in center
            )
        ):
            raise ValueError(
                "polar.geometry.center must be "
                "(x,y) inside resized ROI."
            )

    axis = float(q.get("vertical_angle_deg", 90.0))

    if not math.isfinite(axis):
        raise ValueError(
            "vertical_angle_deg must be finite."
        )


    # Lazy imports: constructing a readout is an explicit operation.
    import numpy as np
    from src.transforms.polar_transforms import PolarTransform

    warp = PolarTransform(
        radius_bins=int(g.get("radius_bins") or size),
        angle_bins=int(g.get("angle_bins") or size),
        center=center,
        max_radius=g.get("max_radius"),
        pt_type=g.get("pt_type") or "corner",
        mode="linear",
    )

    ones = np.ones(
        (size, size),
        dtype=np.uint8,
    )

    _, resolved_center, max_radius = warp._resolve_geometry(ones)

    if (
        not math.isfinite(max_radius)
        or max_radius <= 0
    ):
        raise ValueError(
            "Resolved polar max_radius must be positive and finite."
        )

    valid = torch.from_numpy(
        warp.apply_to_mask(ones).copy()
    ).bool()

    readout = readout_type(
        valid_grid=valid,
        max_radius=max_radius,
        od_class=int(q.get("od_class", 1)),
        ppa_class=int(q.get("ppa_class", 2)),
        denominator_floor_px=float(
            q.get("denominator_floor_px", 1.0)
        ),
        vertical_angle_deg=axis,
        dv_mode=q.get("dv_mode", "fixed_vertical"),
    )
    readout.geometry_metadata = {
        "img_size": size,
        "center": tuple(float(value) for value in resolved_center),
        "mode": "linear",
        "angle_bins": readout.angle_bins,
        "radius_bins": readout.radius_bins,
        "max_radius": readout.max_radius,
        "vertical_angle_deg": readout.vertical_angle_deg,
        "denominator_floor_px": readout.denominator_floor_px,
        "od_class": readout.od_class,
        "ppa_class": readout.ppa_class,
    }
    if readout.dv_mode == "max_center_chord":
        # The configured fixed axis has no effect on a full-angle maximum.
        # Record the new definition instead, so report protocols cannot mix it
        # with legacy fixed-axis measurements.
        readout.geometry_metadata.pop("vertical_angle_deg", None)
        readout.geometry_metadata.update({
            "dv_mode": readout.dv_mode,
            "diameter_definition": "max_theta[r(theta)+r(theta+pi)]",
            "opposite_angle_deg": 180.0,
            "opposite_interpolation": "periodic_linear",
        })
    return readout


def build_polar_readout(cfg) -> PolarReadout:
    """Build a measurement readout independently of quantification training.

    The same 3-class index-mask and enabled linear-polar requirements as the
    original loss apply. quantification.enabled and quantification.weight
    do not control this read-only geometry construction.
    """
    return _build_polar_readout(cfg, PolarReadout)
