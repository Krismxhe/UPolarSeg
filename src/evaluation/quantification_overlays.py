"""Visualize existing polar measurements without changing their mathematics.

Hard lines describe radial boundaries. Soft lines describe equivalent radial
occupation lengths, not probability contours. Annotations are represented as
floating-point geometry and redrawn in each coordinate system; text and line
rasters are never passed through a polar image warp.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.evaluation.fundus_overlays import (
    FUNDUS_CONTOUR_COLORS,
    draw_fundus_contours,
    fundus_background,
    load_fundus_rgb,
    normalize_fundus_mode,
    resize_fundus_rgb,
    validate_fundus_alpha,
)
from src.evaluation.prediction_masks import (
    _as_hw,
    _atomic_save_png,
    _case_stem,
    _cfg_get,
    _input_shape,
    _label_mask,
    _resize_mask,
    _resolve_geometry,
    _warp_mask,
    normalize_save_mode,
)


_PALETTE = np.asarray(((20, 25, 34), (61, 91, 135), (113, 76, 129)), dtype=np.uint8)
_COLORS = {
    "hard": {"pmw": (250, 204, 21), "dv": (52, 211, 153), "dh": (251, 146, 60)},
    "soft": {"pmw": (251, 113, 133), "dv": (56, 189, 248), "dh": (216, 180, 254)},
}


def normalize_overlay_mode(value) -> str:
    """Use the same four modes and legacy boolean semantics as mask saving."""
    try:
        return normalize_save_mode(value)
    except ValueError as exc:
        raise ValueError(
            "eval.save_quantification_overlays must be none/polar/cartesian/both "
            "or a legacy boolean"
        ) from exc


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            # Supports floating tensor dtypes that NumPy cannot represent.
            if hasattr(value, "float"):
                return value.float().numpy()
            raise
    return np.asarray(value)


def _number(value, name):
    array = np.asarray(_numpy(value))
    if array.size != 1:
        raise ValueError(f"{name} must be a single-case scalar")
    try:
        return float(array.reshape(-1)[0])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _flag(value, name):
    scalar = _number(value, name)
    if scalar not in (0.0, 1.0):
        raise ValueError(f"{name} must be boolean")
    return bool(scalar)


def _annotation_complete(value):
    """Use the annotation-status conventions of quantification case records."""
    if value is None or (isinstance(value, str) and value == ""):
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "t"}:
            return True
        if normalized in {"0", "false", "no", "n", "f"}:
            return False
        raise ValueError(f"Invalid reference_complete flag: {value!r}")
    return _flag(value, "reference_complete")


def _profile(value, angles, name):
    array = np.asarray(_numpy(value))
    if array.shape != (angles,) or array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a single-case numeric profile with shape ({angles},)")
    # Preserve float32 arithmetic for the same argmax/tie convention as torch.
    return array if array.dtype.kind == "f" else array.astype(np.float64)


def _periodic(profile, row):
    position = float(row) % len(profile)
    left = math.floor(position)
    weight = position - left
    return (1.0 - weight) * float(profile[left]) + weight * float(profile[(left + 1) % len(profile)])


def _cart_point(center, radius, row, angles):
    theta = float(row) * (2.0 * math.pi / angles)
    return [center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)]


def _output_points(points, input_shape, output_shape):
    if output_shape is None:
        return None
    scale_x = output_shape[1] / input_shape[1]
    scale_y = output_shape[0] / input_shape[0]
    return [[(x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5] for x, y in points]


def _check_close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{name}={actual!r} is inconsistent with its supplied profiles ({expected!r})")


def _measurement(data, kind, *, readout, center, input_shape, output_shape):
    if data is None:
        raise ValueError(f"The selected {kind} readout was not supplied")
    required = ("pmw", "dv", "pvdr", "od_profile", "width_profile")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"The {kind} readout is missing fields: {missing}")
    angles = int(readout.angle_bins)
    radial_step = float(readout.max_radius) / int(readout.radius_bins)
    floor = float(readout.denominator_floor_px)
    raw = {name: _number(data[name], f"{kind}.{name}") for name in ("pmw", "dv", "pvdr")}
    od = _profile(data["od_profile"], angles, f"{kind}.od_profile")
    width = _profile(data["width_profile"], angles, f"{kind}.width_profile")
    flags = {}
    if kind == "hard":
        if "reference_valid" not in data:
            raise ValueError("The hard readout must include reference_valid")
        flags = {name: _flag(data[name], f"hard.{name}")
                 for name in ("reference_valid", "integral_compatible", "aux_valid") if name in data}
    finite = all(math.isfinite(value) for value in raw.values()) and bool(
        np.isfinite(od).all() and np.isfinite(width).all()
    )
    valid = finite and (flags["reference_valid"] if kind == "hard" else True)
    result = {
        "status": "nonfinite" if not finite else "valid" if valid else "invalid",
        "valid": bool(valid),
        "interpretation": "hard_radial_boundary" if kind == "hard" else "equivalent_radial_occupation",
        "flags": flags,
        "raw": raw,
        "denominator_floor_px": floor,
        "dv_floor_applied": bool(math.isfinite(raw["dv"]) and raw["dv"] < floor),
        "pmw": None,
        "dv": None,
        "dh": None,
    }
    if not finite:
        return result
    if np.any(od < 0) or np.any(width < 0):
        raise ValueError(f"The {kind} profiles must contain nonnegative radii/widths")

    pmw_row = int(np.argmax(width))  # Deterministic first-index tie break.
    inner = float(od[pmw_row])
    extent = float(width[pmw_row])
    outer = inner + extent
    _check_close(raw["pmw"], extent, f"{kind}.pmw")
    pmw_cart = [_cart_point(center, radius, pmw_row, angles) for radius in (inner, outer)]
    result["pmw"] = {
        "value": raw["pmw"], "angle_deg": pmw_row * 360.0 / angles,
        "angle_row": float(pmw_row), "radii": [inner, outer],
        "polar_endpoints": [[inner / radial_step, float(pmw_row)], [outer / radial_step, float(pmw_row)]],
        "cartesian_endpoints": pmw_cart,
        "cartesian_output_endpoints": _output_points(pmw_cart, input_shape, output_shape),
    }

    dv_mode = str(getattr(readout, "dv_mode", "fixed_vertical")).strip().lower()
    if dv_mode == "fixed_vertical":
        dv_row = (float(readout.vertical_angle_deg) % 360.0) * angles / 360.0
    else:
        half = angles // 2
        opposite = np.roll(od, -half)
        if angles % 2:
            opposite = 0.5 * (opposite + np.roll(od, -(half + 1)))
        dv_row = float(np.argmax(od + opposite))
    opposite_row = (dv_row + angles / 2.0) % angles
    first_radius = _periodic(od, dv_row)
    second_radius = _periodic(od, opposite_row)
    _check_close(raw["dv"], first_radius + second_radius, f"{kind}.dv")
    _check_close(raw["pvdr"], raw["pmw"] / max(raw["dv"], floor), f"{kind}.pvdr")
    dv_cart = [_cart_point(center, first_radius, dv_row, angles),
               _cart_point(center, second_radius, opposite_row, angles)]
    result["dv"] = {
        "value": raw["dv"], "angle_deg": dv_row * 360.0 / angles,
        "opposite_angle_deg": opposite_row * 360.0 / angles,
        "angle_rows": [dv_row, opposite_row], "radii": [first_radius, second_radius],
        "polar_segments": [
            [[0.0, dv_row], [first_radius / radial_step, dv_row]],
            [[0.0, opposite_row], [second_radius / radial_step, opposite_row]],
        ],
        "cartesian_endpoints": dv_cart,
        "cartesian_output_endpoints": _output_points(dv_cart, input_shape, output_shape),
    }
    # Dh is the chord perpendicular to this readout's chosen Dv, not another
    # independently maximized diameter. Canonicalize the undirected axis only
    # after selecting Dv; sample both new directions with periodic interpolation.
    dh_row = (dv_row + angles / 4.0) % (angles / 2.0)
    dh_opposite_row = (dh_row + angles / 2.0) % angles
    dh_first = _periodic(od, dh_row)
    dh_second = _periodic(od, dh_opposite_row)
    dh_cart = [_cart_point(center, dh_first, dh_row, angles),
               _cart_point(center, dh_second, dh_opposite_row, angles)]
    result["dh"] = {
        "value": dh_first + dh_second, "angle_deg": dh_row * 360.0 / angles,
        "opposite_angle_deg": dh_opposite_row * 360.0 / angles,
        "angle_rows": [dh_row, dh_opposite_row], "radii": [dh_first, dh_second],
        "polar_segments": [
            [[0.0, dh_row], [dh_first / radial_step, dh_row]],
            [[0.0, dh_opposite_row], [dh_second / radial_step, dh_opposite_row]],
        ],
        "cartesian_endpoints": dh_cart,
        "cartesian_output_endpoints": _output_points(dh_cart, input_shape, output_shape),
        "interpretation": "chord_perpendicular_to_selected_dv",
    }
    return result


def _validate_geometry(cfg, readout, mask):
    dataset = _cfg_get(cfg, "dataset", {}) or {}
    if int(_cfg_get(dataset, "num_classes", 3)) != 3 or _cfg_get(dataset, "mask_mode", "index") != "index":
        raise ValueError("Quantification overlays require 3-class index masks")
    if not bool(_cfg_get(_cfg_get(cfg, "polar", {}), "enabled", False)):
        raise ValueError("Quantification overlays require polar.enabled=true")
    cart_shape, polar_shape, center, radius, mode = _resolve_geometry(cfg)
    if mode != "linear":
        raise ValueError("Quantification overlays support linear polar geometry only")
    if not (0 <= center[0] < cart_shape[1] and 0 <= center[1] < cart_shape[0]):
        raise ValueError("The quantification polar center must lie inside the Cartesian input grid")
    if readout is None:
        raise ValueError("Quantification overlays require an existing readout")
    actual_shape = (int(readout.angle_bins), int(readout.radius_bins))
    if mask.shape != actual_shape or actual_shape != polar_shape:
        raise ValueError("Prediction mask, readout and configured polar grid shapes must agree")
    if {int(readout.od_class), int(readout.ppa_class)} != {1, 2} or np.any(mask > 2):
        raise ValueError("Quantification overlays require background=0 and OD/PPA=1/2")
    actual_radius = float(readout.max_radius)
    if not math.isfinite(actual_radius) or actual_radius <= 0:
        raise ValueError("Readout max_radius must be finite and positive")
    _check_close(actual_radius, radius, "readout.max_radius versus configured geometry")
    grid = np.asarray(_numpy(readout.valid_grid))
    if grid.shape != actual_shape:
        raise ValueError("Readout valid_grid shape differs from its polar grid")
    metadata = getattr(readout, "geometry_metadata", {}) or {}
    if metadata.get("center") is not None:
        actual_center = tuple(float(value) for value in metadata["center"])
        if len(actual_center) != 2 or not np.allclose(actual_center, center, rtol=0, atol=1e-5):
            raise ValueError("Readout center differs from configured polar geometry")
    if metadata.get("img_size") is not None and _as_hw(metadata["img_size"], "readout img_size") != cart_shape:
        raise ValueError("Readout Cartesian input size differs from configured geometry")
    if metadata.get("mode", "linear") != "linear":
        raise ValueError("Readout metadata must describe linear polar coordinates")
    dv_mode = str(getattr(readout, "dv_mode", "fixed_vertical")).strip().lower()
    if dv_mode not in {"fixed_vertical", "max_center_chord"}:
        raise ValueError("Readout dv_mode must be fixed_vertical or max_center_chord")
    floor = float(readout.denominator_floor_px)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("Readout denominator_floor_px must be finite and positive")
    if dv_mode == "fixed_vertical" and not math.isfinite(float(readout.vertical_angle_deg)):
        raise ValueError("The fixed vertical axis must be finite")
    quantification = _cfg_get(cfg, "quantification", {}) or {}
    absent = object()
    configured_mode = _cfg_get(quantification, "dv_mode", absent)
    if configured_mode is not absent and str(configured_mode).strip().lower() != dv_mode:
        raise ValueError("Readout dv_mode differs from explicit quantification configuration")
    for key in ("od_class", "ppa_class", "denominator_floor_px"):
        configured = _cfg_get(quantification, key, absent)
        if configured is not absent:
            _check_close(float(configured), float(getattr(readout, key)), f"configured {key} versus readout")
    configured_axis = _cfg_get(quantification, "vertical_angle_deg", absent)
    if dv_mode == "fixed_vertical" and configured_axis is not absent:
        _check_close(float(configured_axis) % 180.0, float(readout.vertical_angle_deg) % 180.0,
                     "configured fixed axis versus readout")
    return cart_shape, polar_shape, center, actual_radius, dv_mode


def _draw_line(canvas, points, color, *, dashed, thickness):
    start, end = (np.asarray(point, dtype=float) for point in points)
    length = float(np.linalg.norm(end - start))
    if length <= 1e-8:
        return
    intervals = [(0.0, length)] if not dashed else [
        (offset, min(offset + 7.0, length)) for offset in np.arange(0.0, length, 11.0)
    ]
    for left, right in intervals:
        p1 = tuple(np.rint((start + (end - start) * left / length) * 256).astype(int))
        p2 = tuple(np.rint((start + (end - start) * right / length) * 256).astype(int))
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA, shift=8)


def _segments(measurements, space):
    for kind, data in measurements.items():
        if not data["valid"]:
            continue
        for metric in ("pmw", "dv", "dh"):
            item = data[metric]
            if not item.get("annotation_valid", True):
                continue
            if metric == "pmw" and item["value"] <= 0:
                continue
            if space == "polar":
                pairs = [item["polar_endpoints"]] if metric == "pmw" else item["polar_segments"]
            else:
                pairs = [item["cartesian_output_endpoints"]]
            for pair in pairs:
                yield kind, metric, pair


def _legend(canvas, metadata, space):
    h, w = canvas.shape[:2]
    is_groundtruth = metadata.get("panel_kind") == "groundtruth"
    is_fundus = metadata.get("background_kind") == "fundus"
    if is_fundus:
        heading = "Fundus RGB | hard boundaries | ROI px"
    elif is_groundtruth:
        heading = "ROI px | ground truth boundaries"
    elif "soft" in metadata["readouts"]:
        heading = "ROI px | soft: equivalent occupation"
    else:
        heading = "ROI px | hard radial boundaries"
    rows = [(None, heading)]
    source = "Ground truth hard" if is_groundtruth else metadata["prediction_source"]
    rows.append((None, f"{source} | {metadata['dv_mode']}"))
    if is_fundus:
        rows.extend((
            (FUNDUS_CONTOUR_COLORS["od"], "OD contour"),
            (FUNDUS_CONTOUR_COLORS["ppa"], "PPA contour"),
            ((245, 245, 245), "Center: configured polar origin"),
        ))
    for kind, data in metadata["readouts"].items():
        prefix = "Ground truth hard" if is_groundtruth else kind.capitalize()
        if not data["valid"]:
            rows.append((None, f"{prefix}: {data['status'].upper()} (see JSON)"))
            if not data.get("pmw_annotation_valid", True):
                rows.append((None, f"{prefix} PMW: UNANNOTATED"))
            continue
        for metric in ("pmw", "dv", "dh"):
            label = {"pmw": "PMW", "dv": "Dv", "dh": "Dh"}[metric]
            if not data[metric].get("annotation_valid", True):
                rows.append((None, f"{prefix} {label}: UNANNOTATED"))
                continue
            suffix = " [denom floor]" if metric == "dv" and data["dv_floor_applied"] else ""
            style = "dashed" if kind == "soft" else "solid"
            value = data[metric]["value"]
            rows.append((_COLORS[kind][metric], f"{prefix} {label}: {value:.2f} {style}{suffix}"))
    font_size = min(14, max(10, min(h, w) // 40))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    line_height = font_size + 4
    text_width = max(font.getbbox(text)[2] for _, text in rows)
    box_width, box_height = int(text_width + 18), len(rows) * line_height + 12
    # Tiny test/thumbnail grids cannot contain legible text without hiding the
    # mask. Geometry and all values still remain in the adjacent JSON record.
    if box_width + 8 > w or box_height + 8 > h or box_height > h * 0.45:
        return canvas
    candidates = [(4, 4), (w - box_width - 4, 4),
                  (4, h - box_height - 4), (w - box_width - 4, h - box_height - 4)]
    segments = list(_segments(metadata["readouts"], space))

    def score(anchor):
        x, y = anchor
        avoid_mask = metadata.get("_legend_foreground")
        if avoid_mask is not None:
            foreground = float(avoid_mask[y:y + box_height, x:x + box_width].mean())
        else:
            region = canvas[y:y + box_height, x:x + box_width]
            foreground = float(np.any(region != _PALETTE[0], axis=-1).mean())
        overlaps = 0
        for _, _, pair in segments:
            xs, ys = zip(*pair)
            overlaps += not (max(xs) < x or min(xs) > x + box_width
                             or max(ys) < y or min(ys) > y + box_height)
        return overlaps * 10 + foreground

    x, y = min(candidates, key=score)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((x, y, x + box_width, y + box_height), radius=4, fill=(10, 14, 20))
    for index, (color, text) in enumerate(rows):
        draw.text((x + 8, y + 5 + index * line_height), text, font=font, fill=color or (225, 231, 239))
    return np.array(image)


def _render(mask, metadata, space, show_legend):
    canvas = _PALETTE[mask].copy()
    if show_legend:
        canvas = _legend(canvas, metadata, space)
    thickness = max(1, min(3, round(min(canvas.shape[:2]) / 256)))
    for kind, metric, points in _segments(metadata["readouts"], space):
        offsets = (-canvas.shape[0], 0, canvas.shape[0]) if space == "polar" else (0,)
        for offset in offsets:
            pair = [[x, y + offset] for x, y in points]
            _draw_line(canvas, pair, _COLORS[kind][metric], dashed=kind == "soft", thickness=thickness)
    return canvas


def _render_fundus(rgb, mask, metadata, *, readout, alpha, show_legend, center):
    """Draw hard geometry directly over a Cartesian photograph and contours."""
    canvas = fundus_background(rgb, mask, alpha, od_class=readout.od_class,
                               ppa_class=readout.ppa_class)
    panel_metadata = {**metadata, "background_kind": "fundus", "_legend_foreground": mask != 0}
    if show_legend:
        canvas = _legend(canvas, panel_metadata, "cartesian")
    thickness = max(1, min(3, round(min(canvas.shape[:2]) / 256)))
    draw_fundus_contours(canvas, mask, od_class=readout.od_class,
                         ppa_class=readout.ppa_class, thickness=thickness)
    for kind, metric, points in _segments(metadata["readouts"], "cartesian"):
        color = _COLORS[kind][metric]
        _draw_line(canvas, points, color, dashed=False, thickness=thickness)
        marker_radius = max(2, thickness + 1)
        for point in points:
            fixed_point = tuple(np.rint(np.asarray(point) * 256).astype(int))
            cv2.circle(canvas, fixed_point, (marker_radius + 1) * 256, (12, 16, 22),
                       -1, cv2.LINE_AA, shift=8)
            cv2.circle(canvas, fixed_point, marker_radius * 256, color,
                       -1, cv2.LINE_AA, shift=8)
    # This marker identifies the configured coordinate origin even for invalid
    # hard masks; it is not a replacement for a suppressed measurement line.
    x, y = center
    arm = max(3, thickness + 3)
    for pair in ([[x - arm, y], [x + arm, y]], [[x, y - arm], [x, y + arm]]):
        _draw_line(canvas, pair, (12, 16, 22), dashed=False, thickness=thickness + 2)
        _draw_line(canvas, pair, (245, 245, 245), dashed=False, thickness=thickness)
    return canvas


def _groundtruth_measurement(data, *, reference_complete, **geometry):
    """Apply the same annotation-completeness rules to every GT background."""
    complete = _annotation_complete(reference_complete)
    measurement = _measurement(data, "hard", **geometry)
    measurement["pmw_annotation_valid"] = complete
    measurement["pvdr_annotation_valid"] = complete
    for metric in ("pmw", "dv", "dh"):
        item = measurement[metric]
        if item is not None:
            item["annotation_valid"] = complete if metric == "pmw" else True
            item["status"] = "unannotated" if metric == "pmw" and not complete else measurement["status"]
    return {"source": "ground_truth", "hard": measurement, "annotation_complete": complete}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_save_json(payload, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class QuantificationOverlayWriter:
    """Save prediction/GT overlays and a scalar/vector geometry audit record.

    Prediction readouts may be combined or written as separate hard/soft panels.
    Ground-truth panels are independent and always use the supplied GT hard
    measurement. Optional Cartesian fundus photographs always display hard
    measurements, independently of mask-overlay layout and readout selection.
    No probability conversion or numerical readout runs here.
    """

    def __init__(self, cfg, output_dir, mode=None, *, polar_dir=None, cartesian_dir=None,
                 metadata_dir=None, gt_polar_dir=None, gt_cartesian_dir=None,
                 fundus_dir=None, gt_fundus_dir=None):
        self.cfg = cfg
        evaluation = _cfg_get(cfg, "eval", {}) or {}
        self.mode = normalize_overlay_mode(_cfg_get(evaluation, "save_quantification_overlays", "none") if mode is None else mode)
        self.gt_mode = normalize_overlay_mode(_cfg_get(evaluation, "save_quantification_gt_overlays", "none"))
        self.fundus_mode = normalize_fundus_mode(_cfg_get(evaluation, "save_fundus_overlays", "none"))
        self.fundus_alpha = validate_fundus_alpha(_cfg_get(evaluation, "fundus_overlay_alpha", 0.0))
        self.layout = str(_cfg_get(evaluation, "quantification_overlay_layout", "combined")).strip().lower()
        if self.layout not in {"combined", "separate"}:
            raise ValueError("eval.quantification_overlay_layout must be combined or separate")
        self.cartesian_size = str(_cfg_get(evaluation, "prediction_cartesian_size", "original")).strip().lower()
        if self.cartesian_size not in {"original", "input"}:
            raise ValueError("eval.prediction_cartesian_size must be original or input")
        self.readouts = str(_cfg_get(evaluation, "quantification_overlay_readouts", "both")).strip().lower()
        if self.readouts not in {"hard", "soft", "both"}:
            raise ValueError("eval.quantification_overlay_readouts must be hard, soft, or both")
        self.legend = _flag(_cfg_get(evaluation, "quantification_overlay_legend", True), "quantification_overlay_legend")
        output_dir = Path(output_dir)
        self.polar_dir = Path(polar_dir) if polar_dir is not None else output_dir / "overlays_polar"
        self.cartesian_dir = Path(cartesian_dir) if cartesian_dir is not None else output_dir / "overlays_cartesian"
        self.metadata_dir = Path(metadata_dir) if metadata_dir is not None else output_dir / "overlay_metadata"
        self.gt_polar_dir = Path(gt_polar_dir) if gt_polar_dir is not None else output_dir / "overlays_gt_polar"
        self.gt_cartesian_dir = Path(gt_cartesian_dir) if gt_cartesian_dir is not None else output_dir / "overlays_gt_cartesian"
        self.fundus_dir = Path(fundus_dir) if fundus_dir is not None else output_dir / "overlays_fundus"
        self.gt_fundus_dir = Path(gt_fundus_dir) if gt_fundus_dir is not None else output_dir / "overlays_gt_fundus"

    @property
    def needs_original_size(self):
        return self.cartesian_size == "original" and (
            self.mode in {"cartesian", "both"} or self.gt_mode in {"cartesian", "both"}
            or self.fundus_mode != "none"
        )

    def save(self, mask, case_id, *, readout, hard, soft, original_size=None,
             prediction_source="seg_logits", groundtruth_mask=None, groundtruth_hard=None,
             reference_complete=True, image_path=None):
        paths = {
            "overlay_polar_path": "", "overlay_cartesian_path": "",
            "overlay_polar_hard_path": "", "overlay_cartesian_hard_path": "",
            "overlay_polar_soft_path": "", "overlay_cartesian_soft_path": "",
            "overlay_gt_polar_path": "", "overlay_gt_cartesian_path": "",
            "overlay_fundus_path": "", "overlay_gt_fundus_path": "",
            "overlay_metadata_path": "",
        }
        pred_overlay_active = self.mode != "none"
        gt_overlay_active = self.gt_mode != "none"
        want_fundus = self.fundus_mode in {"prediction", "both"}
        want_gt_fundus = self.fundus_mode in {"groundtruth", "both"}
        fundus_active = want_fundus or want_gt_fundus
        pred_active = pred_overlay_active or want_fundus
        gt_active = gt_overlay_active or want_gt_fundus
        if not pred_active and not gt_active:
            return paths
        if pred_active and prediction_source not in {"seg_logits", "quant_logits"}:
            raise ValueError("prediction_source must be seg_logits or quant_logits")
        stem = _case_stem(case_id)
        prediction = None
        groundtruth = None
        if pred_active:
            if mask is None:
                raise ValueError("A prediction mask is required when prediction overlays are selected")
            prediction = _label_mask(_numpy(mask))
            geometry = _validate_geometry(self.cfg, readout, prediction)
        if gt_active:
            if groundtruth_mask is None or groundtruth_hard is None:
                raise ValueError("Ground-truth overlays require groundtruth_mask and groundtruth_hard")
            groundtruth = _label_mask(_numpy(groundtruth_mask))
            geometry = _validate_geometry(self.cfg, readout, groundtruth)
        input_shape, polar_shape, center, radius, dv_mode = geometry
        want_polar = self.mode in {"polar", "both"}
        want_cart = self.mode in {"cartesian", "both"}
        want_gt_polar = self.gt_mode in {"polar", "both"}
        want_gt_cart = self.gt_mode in {"cartesian", "both"}
        rgb = None
        source_shape = None
        if fundus_active:
            rgb = load_fundus_rgb(image_path)
            source_shape = tuple(rgb.shape[:2])
            if original_size is not None:
                declared_shape = _as_hw(original_size, "original_size", allow_square=False)
                if declared_shape != source_shape:
                    raise ValueError(
                        f"original_size={declared_shape} does not match fundus image_path size {source_shape}"
                    )
            original_size = source_shape
        output_shape = None
        if want_cart or want_gt_cart or fundus_active:
            if self.cartesian_size == "original":
                if original_size is None:
                    raise ValueError("original_size=(H, W) is required for original Cartesian overlays")
                output_shape = _as_hw(original_size, "original_size", allow_square=False)
            else:
                output_shape = _input_shape(self.cfg)

        measurements = {}
        if pred_overlay_active:
            for kind, data in (("hard", hard), ("soft", soft)):
                if self.readouts in {kind, "both"}:
                    measurements[kind] = _measurement(
                        data, kind, readout=readout, center=center, input_shape=input_shape,
                        output_shape=output_shape if want_cart else None,
                    )
        groundtruth_record = {}
        if gt_overlay_active:
            groundtruth_record = _groundtruth_measurement(
                groundtruth_hard, reference_complete=reference_complete,
                readout=readout, center=center, input_shape=input_shape,
                output_shape=output_shape if want_gt_cart else None,
            )
        fundus_record = {}
        if fundus_active:
            fundus_record = {
                "mode": self.fundus_mode, "source_image": str(image_path),
                "original_shape": list(source_shape), "output_shape": list(output_shape),
                "output_center_xy": _output_points([center], input_shape, output_shape)[0],
                "alpha": self.fundus_alpha, "units": "resized_roi_px",
                "contour_colors_rgb": {key: list(value) for key, value in FUNDUS_CONTOUR_COLORS.items()},
                "prediction_source": prediction_source if want_fundus else None,
                "prediction": {}, "groundtruth": {},
            }
            if want_fundus:
                fundus_record["prediction"] = {"hard": _measurement(
                    hard, "hard", readout=readout, center=center,
                    input_shape=input_shape, output_shape=output_shape,
                )}
            if want_gt_fundus:
                fundus_record["groundtruth"] = _groundtruth_measurement(
                    groundtruth_hard, reference_complete=reference_complete,
                    readout=readout, center=center, input_shape=input_shape,
                    output_shape=output_shape,
                )
        metadata = {
            "schema_version": "quantification-overlay-v1",
            "case_id": stem, "prediction_source": prediction_source if pred_active else None,
            "layout": self.layout, "mode": self.mode, "gt_mode": self.gt_mode,
            "dv_mode": dv_mode, "units": "resized_roi_px",
            "input_shape": list(polar_shape), "polar_shape": list(polar_shape),
            "cartesian_input_shape": list(input_shape),
            "cartesian_shape": list(output_shape) if output_shape is not None else None,
            "output_shapes": {"polar": list(polar_shape) if want_polar else None,
                              "cartesian": list(output_shape) if want_cart else None},
            "gt_output_shapes": {"polar": list(polar_shape) if want_gt_polar else None,
                                 "cartesian": list(output_shape) if want_gt_cart else None},
            "center_xy": list(center), "max_radius": radius,
            "radial_step": radius / int(readout.radius_bins),
            "cartesian_resize_coordinate_rule": "(coordinate+0.5)*output_size/input_size-0.5",
            "readouts": measurements,
            "groundtruth": groundtruth_record,
            "fundus": fundus_record,
        }
        prepared = []
        cartesian_masks = {}

        def cartesian_mask(kind):
            if kind not in cartesian_masks:
                polar_mask = prediction if kind == "prediction" else groundtruth
                cartesian_masks[kind] = _resize_mask(
                    _warp_mask(polar_mask, self.cfg, inverse=True), output_shape,
                )
            return cartesian_masks[kind]

        if pred_overlay_active:
            prediction_panels = []
            if want_polar:
                prediction_panels.append(("polar", prediction, self.polar_dir))
            if want_cart:
                prediction_panels.append(("cartesian", cartesian_mask("prediction"), self.cartesian_dir))
            for space, base, directory in prediction_panels:
                if self.layout == "combined":
                    prepared.append((f"overlay_{space}_path", _render(base, metadata, space, self.legend),
                                     directory / f"{stem}.png"))
                else:
                    for kind, data in measurements.items():
                        panel_metadata = {**metadata, "readouts": {kind: data}}
                        prepared.append((
                            f"overlay_{space}_{kind}_path", _render(base, panel_metadata, space, self.legend),
                            directory / kind / f"{stem}.png",
                        ))
        if gt_overlay_active:
            panel_metadata = {**metadata, "panel_kind": "groundtruth",
                              "readouts": {"hard": groundtruth_record["hard"]}}
            if want_gt_polar:
                prepared.append(("overlay_gt_polar_path", _render(groundtruth, panel_metadata, "polar", self.legend),
                                 self.gt_polar_dir / f"{stem}.png"))
            if want_gt_cart:
                prepared.append(("overlay_gt_cartesian_path", _render(cartesian_mask("groundtruth"), panel_metadata, "cartesian", self.legend),
                                 self.gt_cartesian_dir / f"{stem}.png"))
        if fundus_active:
            fundus_rgb = resize_fundus_rgb(rgb, output_shape)
            for selected, kind, key, directory in (
                (want_fundus, "prediction", "overlay_fundus_path", self.fundus_dir),
                (want_gt_fundus, "groundtruth", "overlay_gt_fundus_path", self.gt_fundus_dir),
            ):
                if not selected:
                    continue
                panel_metadata = {**metadata, "panel_kind": kind,
                                  "readouts": {"hard": fundus_record[kind]["hard"]}}
                rendered = _render_fundus(
                    fundus_rgb, cartesian_mask(kind), panel_metadata, readout=readout,
                    alpha=self.fundus_alpha, show_legend=self.legend,
                    center=fundus_record["output_center_xy"],
                )
                prepared.append((key, rendered, directory / f"{stem}.png"))
        destinations = [destination.resolve() for _, _, destination in prepared]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Selected overlay output paths must be distinct")
        for key, image, destination in prepared:
            _atomic_save_png(image, destination)
            paths[key] = str(destination)
        metadata_path = self.metadata_dir / f"{stem}.json"
        _atomic_save_json(metadata, metadata_path)
        paths["overlay_metadata_path"] = str(metadata_path)
        return paths
