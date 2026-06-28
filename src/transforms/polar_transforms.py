import albumentations as A
import cv2
import numpy as np
import torch

import math
from typing import Any, Optional, Sequence, Tuple

# ── polar transformation ────────────────────────────────────────
class PolarTransform(A.DualTransform):
    """
    Convert image and mask from Cartesian coordinates to polar coordinates.

    Output shape:
        image/mask height = angle_bins
        image/mask width  = radius_bins

    Notes:
        - Image uses bilinear interpolation.
        - Mask uses nearest-neighbor interpolation to preserve class IDs.
        - center is in pixel coordinates after previous transforms, e.g. after Resize.
          OpenCV expects center=(x, y), not (row, col).
    """
    
    def __init__(
        self,
        radius_bins: Optional[int] = None,
        angle_bins: Optional[int] = None,
        center: Optional[Sequence[float]] = None,
        max_radius: Optional[float] = None,
        pt_type = "corner",
        mode: str = "linear",
    ):
        super().__init__(p=1.0)
        
        if mode not in {"linear", "log"}:
            raise ValueError(f"PolarTransform mode must be 'linear' or 'log', got {mode!r}")

        self.radius_bins = radius_bins
        self.angle_bins = angle_bins
        self.center = tuple(center) if center is not None else None
        self.max_radius = max_radius
        self.mode = mode
        self.pt_type = pt_type
        
    def _resolve_geometry(self, img: np.ndarray) -> Tuple[Tuple[int, int], Tuple[float, float], float]:
        h, w = img.shape[:2]
        
        radius_bins = int(self.radius_bins or w)
        angle_bins = int(self.angle_bins or h)
        dsize = (radius_bins, angle_bins)
        
        if self.center is None:
            cx, cy  = resolve_center(h=h, w=w)
        else:
            cx, cy = float(self.center[0]), float(self.center[1])
        
        if self.max_radius is None:
            radius = resolve_radius(h=h, w=w, center_xy=(cx, cy), pt_type=self.pt_type)
        else:
            radius = self.max_radius
        
        return dsize, (cx, cy), radius
    
    def _flags(self, interpolation: int)->int:
        polar_flag = cv2.WARP_POLAR_LOG if self.mode=='log' else cv2.WARP_POLAR_LINEAR
        return interpolation | polar_flag | cv2.WARP_FILL_OUTLIERS
        
    def _warp(self, img: np.ndarray, interpolation: int) -> np.ndarray:
        dsize, center, radius = self._resolve_geometry(img)
        
        return cv2.warpPolar(
            img,
            dsize=dsize,
            center=center,
            maxRadius=radius,
            flags=self._flags(interpolation),
        )
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self._warp(img, interpolation=cv2.INTER_LINEAR)

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        warped = self._warp(mask, interpolation=cv2.INTER_NEAREST)
        return warped.astype(mask.dtype, copy=False)

    def get_transform_init_args_names(self):
        return ("radius_bins", "angle_bins", "center", "max_radius", "mode")

def resolve_center(h: int, w: int)-> tuple[float, float]:
    return (w-1)/2.0, (h-1)/2.0

def resolve_radius(
    h:int,
    w:int,
    center_xy: tuple[float, float],
    pt_type: Any,
)->float:
    cx, cy = center_xy
    name = pt_type.lower()
    
    if name == "corner":
        corners = np.array(
            [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]],
            dtype=np.float32
        )
        
        dist = np.sqrt((corners[:, 0] - cx)**2 + (corners[:, 1] - cy)**2)
        return float(dist.max())
    
    if name == "min_edge":
        return float(min(cx, cy, (w-1)-cx, (h-1)-cy))

def cart_to_polar_mask(
    mask: np.array,
    center: tuple[float, float],
    max_radius: float,
    radius_bins: int,
    angle_bins: int,
):
    flags = (
        cv2.WARP_POLAR_LINEAR
        | cv2.WARP_FILL_OUTLIERS
        | cv2.INTER_NEAREST
    )
    
    polar = cv2.warpPolar(
        mask.astype(np.float32, copy=False),
        dsize= (int(radius_bins), int(angle_bins)),
        center=center,
        maxRaidus=float(max_radius),
        flags=flags,
    )
    
    return np.rint(polar).astype(np.int64)

def polar_mask_to_cart(
    mask_polar: np.ndarray,
    out_h: int,
    out_w: int,
    center: tuple[float, float],
    max_radius: float,
    mode: str = "linear",
    fill_value: int = 0,
) -> np.ndarray:
    flags = (
        polar_flag_from_mode(mode)
        | cv2.WARP_INVERSE_MAP
        | cv2.WARP_FILL_OUTLIERS
        | cv2.INTER_NEAREST
    )

    cart = cv2.warpPolar(
        mask_polar.astype(np.float32, copy=False),
        dsize=(int(out_w), int(out_h)),
        center=(float(center[0]), float(center[1])),
        maxRadius=float(max_radius),
        flags=flags,
    )

    cart = np.nan_to_num(cart, nan=float(fill_value))
    return np.rint(cart).astype(np.int64)