"""RGB backgrounds and class contours for Cartesian fundus overlays.

This module reads the dataset image/ROI exactly as stored, without EXIF
reorientation or a full-image crop reconstruction. Numerical measurements and
polar geometry belong to the caller; this helper never evaluates a readout.
"""

from __future__ import annotations

import math
from os import PathLike
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FUNDUS_CONTOUR_COLORS = {"od": (96, 165, 250), "ppa": (232, 121, 249)}


def normalize_fundus_mode(value) -> str:
    """Select prediction and/or ground-truth RGB overlays independently."""
    if value is None:
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return "both" if value else "none"
    if isinstance(value, str):
        mode = value.strip().lower()
        mode = {"false": "none", "true": "both"}.get(mode, mode)
        if mode in {"none", "prediction", "groundtruth", "both"}:
            return mode
    raise ValueError(
        "eval.save_fundus_overlays must be none/prediction/groundtruth/both "
        "or a boolean"
    )


def validate_fundus_alpha(value) -> float:
    """Validate foreground fill opacity; contours remain opaque."""
    try:
        alpha = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("eval.fundus_overlay_alpha must be a finite number in [0, 1]") from exc
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("eval.fundus_overlay_alpha must be a finite number in [0, 1]")
    return alpha


def load_fundus_rgb(image_path) -> np.ndarray:
    """Read a Cartesian source image as RGB; do not silently orient or crop it."""
    if not isinstance(image_path, (str, PathLike)) or not str(image_path).strip():
        raise ValueError("A readable image_path is required when fundus overlays are enabled")
    try:
        with Image.open(Path(image_path)) as source:
            # Match the dataset's stored-pixel orientation. An EXIF transpose
            # here would move features independently of the prediction mask.
            return np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not read fundus image_path {str(image_path)!r}: {exc}") from exc


def resize_fundus_rgb(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize RGB with the pixel-centre convention used by vector endpoints."""
    if rgb.shape[:2] == shape:
        return rgb.copy()
    return cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def fundus_background(rgb: np.ndarray, mask: np.ndarray, alpha: float, *,
                      od_class: int, ppa_class: int) -> np.ndarray:
    """Keep the photograph unchanged outside foreground labels."""
    if rgb.shape[:2] != mask.shape:
        raise ValueError("Fundus RGB and Cartesian mask shapes must agree")
    canvas = rgb.copy()
    if alpha == 0.0:
        return canvas
    for label, key in ((od_class, "od"), (ppa_class, "ppa")):
        selected = mask == label
        color = np.asarray(FUNDUS_CONTOUR_COLORS[key], dtype=np.float64)
        values = (1.0 - alpha) * rgb[selected].astype(np.float64) + alpha * color
        canvas[selected] = np.rint(values).astype(np.uint8)
    return canvas


def draw_fundus_contours(canvas: np.ndarray, mask: np.ndarray, *,
                         od_class: int, ppa_class: int, thickness: int) -> None:
    """Draw all OD/PPA components and holes in RGB channel order in place."""
    # Give OD the visible stroke where the two classes share a boundary.
    for label, key in ((ppa_class, "ppa"), (od_class, "od")):
        binary = np.ascontiguousarray(mask == label, dtype=np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(canvas, contours, -1, FUNDUS_CONTOUR_COLORS[key],
                             thickness=thickness, lineType=cv2.LINE_AA)
