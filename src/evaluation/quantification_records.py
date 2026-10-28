"""Stable scalar records between polar readouts and statistical reporting.

This module deliberately has no model, Lightning or torch import. Tensor results
are converted at the boundary; CSV/statistical consumers only see Python scalars.
Lengths are measured on the resized ROI grid, not native-image or physical units.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping


QUANTIFICATION_SCHEMA_VERSION = "quantification-v1"
QUANTIFICATION_PROTOCOL = "polar-radial-v1"
MAX_CENTER_CHORD_PROTOCOL = "polar-radial-v2"

_METADATA_FIELDS = (
    "quant_schema_version", "quant_protocol", "quant_geometry_id",
    "quant_geometry_json", "quant_prediction_source", "quant_space",
    "quant_length_unit", "quant_status", "quant_reason",
)
_VALUE_FIELDS = tuple(
    f"quant_{source}_{name}"
    for source in ("reference", "hard", "soft")
    for name in ("pmw", "dv", "pvdr")
)
_GEOMETRY_VALIDITY_FIELDS = (
    "quant_reference_valid", "quant_integral_compatible", "quant_aux_valid",
    "quant_hard_reference_valid", "quant_hard_integral_compatible",
    "quant_hard_aux_valid",
)
_VALIDITY_FIELDS = (
    *_GEOMETRY_VALIDITY_FIELDS, "quant_annotation_complete",
    "quant_reference_pmw_valid", "quant_reference_dv_valid",
    "quant_reference_pvdr_valid",
)


def _metric_specs():
    specs = []
    for name, target, suffix, unit in (
        ("pmw", "PMW", "_px", "px"),
        ("dv", "Dv", "_px", "px"),
        ("pvdr", "PVDR", "", "ratio"),
    ):
        for diagnostic, metric, subgroup in (
            (f"{name}_soft_mae{suffix}", "soft_mae", "reference_valid"),
            (f"{name}_hard_mae{suffix}", "hard_mae", "reference_and_prediction_valid"),
            (f"{name}_soft_hard_gap{suffix}", "soft_hard_gap", "reference_and_prediction_valid"),
            (f"gt_{name}_proxy_gap{suffix}", "gt_proxy_gap", "reference_valid"),
            (f"{name}_soft_mae_paired{suffix}", "soft_mae_paired", "reference_and_prediction_valid"),
        ):
            key = f"quant_{diagnostic}"
            specs.append({
                "key": key, "valid_key": f"{key}_valid",
                "eligible_key": f"quant_reference_{name}_valid",
                "target": target, "subgroup": subgroup, "metric": metric,
                "unit": unit, "higher_is_better": False, "kind": "continuous",
            })
    for name, eligible_key, higher in (
        ("reference_valid_fraction", None, True),
        ("aux_valid_fraction", None, True),
        ("hard_invalid_fraction", "quant_reference_valid", False),
        ("soft_dv_floor_fraction", "quant_reference_valid", False),
    ):
        key = f"quant_{name}"
        specs.append({
            "key": key, "valid_key": f"{key}_valid", "eligible_key": eligible_key,
            "target": "quantification", "subgroup": "all" if eligible_key is None else "reference_valid",
            "metric": name, "unit": "fraction", "higher_is_better": higher,
            "kind": "binary",
        })
    return tuple(specs)


QUANTIFICATION_METRIC_SPECS = _metric_specs()
QUANTIFICATION_FIELDS = (
    *_METADATA_FIELDS, *_VALUE_FIELDS, *_VALIDITY_FIELDS,
    *(field for spec in QUANTIFICATION_METRIC_SPECS
      for field in (spec["key"], spec["valid_key"])),
)


def _python_values(value):
    """Detach once per column, instead of synchronising a GPU for every case."""
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def quantification_protocol(readout) -> dict:
    """Identify the sampling grid and the diameter definition actually used.

    Fixed-axis readouts retain the original v1 geometry encoding so their CSVs
    remain comparable with existing runs. The maximum centre chord is a separate
    v2 protocol; its geometry excludes the unused fixed-axis angle, including any
    stale copy in ``geometry_metadata``.
    """
    dv_mode = str(getattr(readout, "dv_mode", "fixed_vertical")).strip().lower()
    if dv_mode not in {"fixed_vertical", "max_center_chord"}:
        raise ValueError(f"Unsupported quantification dv_mode: {dv_mode!r}")
    geometry = dict(getattr(readout, "geometry_metadata", {}))
    # These descriptors did not exist in v1. Resolve them from the actual mode,
    # rather than trusting copied metadata or changing a fixed-axis run's hash.
    for key in ("dv_mode", "diameter_definition", "opposite_interpolation", "opposite_angle_deg"):
        geometry.pop(key, None)
    geometry.update({
        "angle_bins": int(readout.angle_bins),
        "radius_bins": int(readout.radius_bins),
        "max_radius": float(readout.max_radius),
        "od_class": int(readout.od_class),
        "ppa_class": int(readout.ppa_class),
        "denominator_floor_px": float(readout.denominator_floor_px),
        "mode": "linear", "radial_step": "max_radius/radius_bins",
        "origin_weight": 0, "length_unit": "resized_roi_px",
    })
    if dv_mode == "max_center_chord":
        geometry.pop("vertical_angle_deg", None)
        geometry.pop("vertical_angle", None)
        geometry.update({
            "dv_mode": dv_mode,
            "diameter_definition": "max_theta[r(theta)+r(theta+pi)]",
            "opposite_angle_deg": 180.0,
            "opposite_interpolation": "periodic_linear",
        })
        protocol_name = MAX_CENTER_CHORD_PROTOCOL
    else:
        geometry["vertical_angle_deg"] = float(readout.vertical_angle_deg)
        protocol_name = QUANTIFICATION_PROTOCOL
    grid = _python_values(readout.valid_grid)
    geometry["valid_grid_sha256"] = hashlib.sha256(
        bytes(int(value) for row in grid for value in row)
    ).hexdigest()
    encoded = json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "quant_schema_version": QUANTIFICATION_SCHEMA_VERSION,
        "quant_protocol": protocol_name,
        "quant_geometry_id": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "quant_geometry_json": encoded,
        "quant_space": "resized_roi_polar",
        "quant_length_unit": "resized_roi_px",
    }


def unavailable_quantification_record(reason: str) -> dict:
    """Make unavailable measurements explicit instead of reporting zero errors."""
    row = {key: "" for key in _METADATA_FIELDS}
    row.update({key: float("nan") for key in _VALUE_FIELDS})
    row.update({key: False for key in _VALIDITY_FIELDS})
    for spec in QUANTIFICATION_METRIC_SPECS:
        row[spec["key"]] = float("nan")
        row[spec["valid_key"]] = False
    row.update({
        "quant_schema_version": QUANTIFICATION_SCHEMA_VERSION,
        "quant_status": "unavailable", "quant_reason": str(reason),
    })
    return row


def apply_annotation_completeness(record: Mapping, complete=True) -> dict:
    """Respect unknown PPA annotation without changing readout mathematics.

    OD/PPA metadata already uses reference_complete to distinguish an absent
    PPA from an unannotated PPA. Dv only requires the readable OD reference;
    PMW/PVDR reference-dependent statistics additionally require complete labels.
    """
    if complete is None or complete == "":
        complete = True
    if isinstance(complete, str):
        normalized = complete.strip().lower()
        if normalized in {"1", "true", "yes", "y", "t"}:
            complete = True
        elif normalized in {"0", "false", "no", "n", "f"}:
            complete = False
        else:
            raise ValueError(f"Invalid reference_complete flag: {complete!r}")
    if complete not in (True, False, 0, 1):
        raise ValueError(f"Invalid reference_complete flag: {complete!r}")
    row = dict(record)
    row["quant_annotation_complete"] = bool(complete)
    readable = bool(row.get("quant_reference_valid", False))
    for name, target in (("pmw", "PMW"), ("dv", "Dv"), ("pvdr", "PVDR")):
        eligible = readable and (bool(complete) or name == "dv")
        row[f"quant_reference_{name}_valid"] = eligible
        if not eligible:
            row[f"quant_reference_{name}"] = float("nan")
            for spec in QUANTIFICATION_METRIC_SPECS:
                if spec["target"] == target:
                    row[spec["key"]] = float("nan")
                    row[spec["valid_key"]] = False
    return row


def build_quantification_case_records(
    diagnostics: Mapping, *, protocol: Mapping, prediction_source: str,
) -> list[dict]:
    """Convert one diagnostic batch into one flat, CSV-safe row per case.

    Masked diagnostic values are NaN, never a protected zero. Original raw hard
    measurements are exported only when their own reference-valid flag is true.
    Soft estimates remain estimators, without inheriting hard validity claims.
    """
    required = ("quant_schema_version", "quant_protocol", "quant_geometry_id", "quant_space")
    if any(not protocol.get(key) for key in required):
        raise ValueError("Quantification records require explicit protocol and geometry metadata.")
    if prediction_source not in {"seg_logits", "quant_logits"}:
        raise ValueError(f"Unsupported quantification prediction source: {prediction_source!r}")

    columns = {}
    for source, result_key in (("reference", "target"), ("hard", "hard"), ("soft", "soft")):
        result = diagnostics[result_key]
        for name in ("pmw", "dv", "pvdr"):
            columns[f"quant_{source}_{name}"] = _python_values(result[name])
    for source, result_key in (("", "target"), ("hard_", "hard")):
        for name in ("reference_valid", "integral_compatible", "aux_valid"):
            columns[f"quant_{source}{name}"] = _python_values(diagnostics[result_key][name])

    metrics = dict(diagnostics["metrics"])
    # Same estimators as the original soft MAE, explicitly restricted to the
    # hard pair's cohort so hard/soft comparisons do not silently change cases.
    for name, suffix in (("pmw", "_px"), ("dv", "_px"), ("pvdr", "")):
        values, _ = metrics[f"{name}_soft_mae{suffix}"]
        _, pair = metrics[f"{name}_hard_mae{suffix}"]
        metrics[f"{name}_soft_mae_paired{suffix}"] = (values, pair)
    for spec in QUANTIFICATION_METRIC_SPECS:
        diagnostic_name = spec["key"][len("quant_"):]
        values, valid = metrics[diagnostic_name]
        columns[spec["key"]] = _python_values(values)
        columns[spec["valid_key"]] = _python_values(valid)

    size = len(columns["quant_reference_pmw"])
    if any(len(values) != size for values in columns.values()):
        raise ValueError("Quantification diagnostic columns have different batch sizes.")
    rows = []
    for index in range(size):
        row = {**protocol, "quant_prediction_source": prediction_source,
               "quant_status": "ok", "quant_reason": ""}
        row.update({key: bool(columns[key][index]) for key in _GEOMETRY_VALIDITY_FIELDS})
        for key in _VALUE_FIELDS:
            value = float(columns[key][index])
            allowed = (row["quant_reference_valid"] if key.startswith("quant_reference_")
                       else row["quant_hard_reference_valid"] if key.startswith("quant_hard_")
                       else True)
            row[key] = value if allowed and math.isfinite(value) else float("nan")
        for spec in QUANTIFICATION_METRIC_SPECS:
            value = float(columns[spec["key"]][index])
            valid = bool(columns[spec["valid_key"]][index]) and math.isfinite(value)
            row[spec["key"]] = value if valid else float("nan")
            row[spec["valid_key"]] = valid
        rows.append(apply_annotation_completeness(row))
    return rows
