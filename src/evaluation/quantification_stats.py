"""Scalar-only polar-quantification statistics and reference CSV parsing.

Segmentation statistics remain in odppa_stats. The quantification reports reuse
its cluster resampling routines, with explicit measurement eligibility, validity,
and protocol checks before any aggregation or paired comparison.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.evaluation.odppa_stats import (
    _cluster_bootstrap_mean,
    _cluster_bootstrap_proportion,
    _groups,
    _paired_cluster_bootstrap_delta,
    _paired_cluster_signflip_p,
    wilson_interval,
)
from src.evaluation.quantification_records import (
    QUANTIFICATION_FIELDS,
    QUANTIFICATION_METRIC_SPECS,
)


PROTOCOL_FIELDS = (
    "quant_schema_version", "quant_protocol", "quant_geometry_id", "quant_space",
    "quant_length_unit",
)
COHORT_FIELDS = (*PROTOCOL_FIELDS, "quant_prediction_source")
_REFERENCE_FLAGS = tuple(key for key in QUANTIFICATION_FIELDS if key in {
    "quant_reference_valid", "quant_aux_valid", "quant_integral_compatible",
    "quant_annotation_complete",
} or (key.startswith("quant_reference_") and key.endswith("_valid")))
_QUANT_BOOL_FIELDS = {
    key for key in QUANTIFICATION_FIELDS
    if key.endswith("_valid") or key.endswith("_compatible") or key == "quant_annotation_complete"
}
_QUANT_NUMBER_FIELDS = {
    *(f"quant_{source}_{name}" for source in ("reference", "hard", "soft")
      for name in ("pmw", "dv", "pvdr")),
    *(spec["key"] for spec in QUANTIFICATION_METRIC_SPECS),
}
_SEG_BOOL_FIELDS = {
    "reference_complete", "od_reference_valid", "od_pred_empty", "od_hd95_valid",
    "ppa_pred_empty", "ppa_hd95_valid",
}
_TEXT_FIELDS = {
    "case_id", "patient_id", "eye", "run_name", "checkpoint", "dataset", "split",
    "model", "encoder", "protocol_version", "ppa_status", "image_path", "mask_path",
    "pred_path", "pred_polar_path", "pred_cartesian_path",
    "overlay_polar_path", "overlay_cartesian_path", "overlay_metadata_path",
    "overlay_polar_hard_path", "overlay_cartesian_hard_path",
    "overlay_polar_soft_path", "overlay_cartesian_soft_path",
    "overlay_gt_polar_path", "overlay_gt_cartesian_path",
    "overlay_fundus_path", "overlay_gt_fundus_path",
    *(set(QUANTIFICATION_FIELDS) - _QUANT_BOOL_FIELDS - _QUANT_NUMBER_FIELDS),
}

QUANTIFICATION_METRIC_FIELDS = (
    *COHORT_FIELDS, "target", "subgroup", "metric", "unit", "higher_is_better", "kind",
    "estimate", "sd", "ci_lower", "ci_upper", "ci_method", "ci_status",
    "n_cases", "n_available_cases", "n_unavailable_cases", "availability_status",
    "quant_status_counts", "quant_reason_counts", "n_eligible_cases", "n_valid_cases",
    "n_invalid_cases", "invalid_fraction", "n_eligible_patients", "n_patients",
    "cluster_key", "numerator", "denominator",
)
QUANTIFICATION_COMPARISON_FIELDS = (
    *COHORT_FIELDS, "reference_quant_prediction_source",
    "target", "subgroup", "metric", "unit", "higher_is_better", "kind",
    "comparison_status", "n_current_cases", "n_reference_cases",
    "n_current_available_cases", "n_reference_available_cases",
    "n_current_eligible_cases", "n_reference_eligible_cases",
    "n_current_valid_cases", "n_reference_valid_cases", "n_eligible_pairs",
    "n_paired_cases", "n_paired_patients", "cluster_key",
    "estimate_current", "estimate_reference", "delta_current_minus_reference",
    "delta_ci_lower", "delta_ci_upper", "ci_method", "ci_status", "test_method",
    "test_status", "p_raw", "p_adjusted", "adjustment_status",
)


def has_quantification_records(records: Sequence[Mapping]) -> bool:
    return any(any(str(key).startswith("quant_") for key in row) for row in records)


def _strict_bool(value, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    raise ValueError(f"{field} requires an explicit boolean (true/false or 1/0), got {value!r}")


def _number(value, field: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} requires a numeric value, got {value!r}") from exc


def _with_identities(records: Sequence[Mapping], cluster_key: str) -> list[dict]:
    if not isinstance(cluster_key, str) or not cluster_key.strip():
        raise ValueError("cluster_key must be a nonempty column name.")
    result, seen = [], set()
    for source in records:
        row = dict(source)
        value = row.get("case_id")
        case_id = "" if value is None else str(value)
        if not case_id.strip():
            raise ValueError("Evaluation records require a nonempty case_id.")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id!r}")
        seen.add(case_id)
        row["case_id"] = case_id
        for key in {"patient_id", cluster_key}:
            value = row.get(key)
            row[key] = case_id if value is None or value == "" else str(value)
        result.append(row)
    return result


def read_reference_csv(path, *, cluster_key: str = "patient_id") -> list[dict]:
    """Parse typed columns without converting numeric-looking identity strings.

    Both ordinary per_case.csv and quantification_per_case.csv are accepted.
    Present boolean columns must contain an explicit true/false or 1/0 value.
    """
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Reference CSV contains duplicate column names.")
        if "case_id" not in fields:
            raise ValueError("Reference CSV requires a case_id column.")
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key is None:
                    raise ValueError("Reference CSV row has more values than its header.")
                if key in _TEXT_FIELDS or key == cluster_key:
                    row[key] = "" if value is None else str(value)
                elif key in _QUANT_BOOL_FIELDS or key in _SEG_BOOL_FIELDS:
                    row[key] = _strict_bool(value, key)
                elif key in _QUANT_NUMBER_FIELDS:
                    row[key] = _number(value, key)
                elif value is None or value == "":
                    row[key] = float("nan")
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return _with_identities(rows, cluster_key)


def _cohort_metadata(records: Sequence[Mapping]) -> dict:
    available = [row for row in records if row.get("quant_status") == "ok"]
    metadata = {}
    for key in COHORT_FIELDS:
        values = {str(row.get(key) or "") for row in available}
        if available and "" in values:
            raise ValueError(f"Available quantification records require {key}.")
        if len(values) > 1:
            raise ValueError(f"Mixed {key} values cannot be aggregated in one run.")
        fallback = {str(row.get(key) or "") for row in records} - {""}
        metadata[key] = next(iter(values)) if values else (
            next(iter(fallback)) if len(fallback) == 1 else ""
        )
    return metadata


def validate_quantification_records(
    records: Sequence[Mapping], *, cluster_key: str = "patient_id",
) -> list[dict]:
    """Return scalar copies; reject identity/protocol errors before CSV export."""
    rows = _with_identities(records, cluster_key)
    for row in rows:
        for key in QUANTIFICATION_FIELDS:
            if key in _QUANT_BOOL_FIELDS:
                row[key] = _strict_bool(row.get(key, False), key)
            elif key in _QUANT_NUMBER_FIELDS:
                value = _number(row.get(key), key)
                row[key] = value if math.isfinite(value) else float("nan")
            else:
                row[key] = str(row.get(key) or "")
        if not row["quant_status"]:
            row["quant_status"] = "unavailable"
            row["quant_reason"] = "missing_quantification_record"
        if row["quant_status"] != "ok":
            for key in _QUANT_BOOL_FIELDS:
                row[key] = False
            for key in _QUANT_NUMBER_FIELDS:
                row[key] = float("nan")
        for source, valid_key in (("reference", "quant_reference_valid"),
                                  ("hard", "quant_hard_reference_valid")):
            for name in ("pmw", "dv", "pvdr"):
                allowed = row.get(f"quant_reference_{name}_valid", row[valid_key]) if source == "reference" else row[valid_key]
                if not allowed:
                    row[f"quant_{source}_{name}"] = float("nan")
        for spec in QUANTIFICATION_METRIC_SPECS:
            key, valid_key = spec["key"], spec["valid_key"]
            if (not math.isfinite(row[key]) or
                    (spec["eligible_key"] is not None and not row[spec["eligible_key"]])):
                row[valid_key] = False
            if not row[valid_key]:
                row[key] = float("nan")
            elif spec["kind"] == "binary" and row[key] not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1 for a valid binary observation.")
    _cohort_metadata(rows)
    return rows


def _availability(records: Sequence[Mapping]) -> dict:
    available = sum(row.get("quant_status") == "ok" for row in records)
    return {
        "n_cases": len(records), "n_available_cases": available,
        "n_unavailable_cases": len(records) - available,
        "availability_status": ("no_cases" if not records else "no_available_cases"
                                if not available else "all_available"
                                if available == len(records) else "partially_available"),
        "quant_status_counts": json.dumps(dict(sorted(Counter(
            str(row.get("quant_status", "unavailable")) for row in records
        ).items())), separators=(",", ":")),
        "quant_reason_counts": json.dumps(dict(sorted(Counter(
            str(row["quant_reason"]) for row in records if row.get("quant_reason")
        ).items())), separators=(",", ":")),
    }


def _indices(records: Sequence[Mapping], spec: Mapping) -> tuple[list[int], list[int]]:
    eligible = [
        i for i, row in enumerate(records)
        if row.get("quant_status") == "ok"
        and (spec["eligible_key"] is None or row[spec["eligible_key"]])
    ]
    valid = [i for i in eligible if records[i][spec["valid_key"]]]
    return eligible, valid


def _sampling_count(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def summarize_quantification_records(
    records: Sequence[Mapping], *, n_boot=10000, seed=42, cluster_key="patient_id",
) -> list[dict]:
    n_boot = _sampling_count(n_boot, "n_boot")
    records = validate_quantification_records(records, cluster_key=cluster_key)
    common = {**_cohort_metadata(records), **_availability(records), "cluster_key": cluster_key}
    rows = []
    for index, spec in enumerate(QUANTIFICATION_METRIC_SPECS):
        eligible, valid = _indices(records, spec)
        values = np.asarray([records[i][spec["key"]] for i in valid], dtype=np.float64)
        patients = len(_groups(records, valid, cluster_key))
        estimate = float(values.mean()) if values.size else float("nan")
        sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        lo = hi = float("nan")
        method = "cluster_percentile_bootstrap"
        if not eligible:
            status = "no_eligible_cases"
        elif not valid:
            status = "no_valid_cases"
        elif patients < 2:
            status = "insufficient_clusters"
        elif spec["kind"] == "binary" and patients == len(valid):
            lo, hi = wilson_interval(int(values.sum()), len(valid))
            method, status = "wilson", "ok"
        elif spec["kind"] == "binary" and (np.all(values == 0) or np.all(values == 1)):
            status = "degenerate_binary_clusters"
        else:
            bootstrap = (_cluster_bootstrap_proportion if spec["kind"] == "binary"
                         else _cluster_bootstrap_mean)
            lo, hi = bootstrap(records, valid, spec["key"], cluster_key, n_boot, seed + index)
            status = "ok"
        rows.append({
            **common, **{key: spec[key] for key in (
                "target", "subgroup", "metric", "unit", "higher_is_better", "kind")},
            "estimate": estimate, "sd": sd, "ci_lower": lo, "ci_upper": hi,
            "ci_method": method, "ci_status": status,
            "n_eligible_cases": len(eligible), "n_valid_cases": len(valid),
            "n_invalid_cases": len(eligible) - len(valid),
            "invalid_fraction": (1.0 - len(valid) / len(eligible)) if eligible else float("nan"),
            "n_eligible_patients": len(_groups(records, eligible, cluster_key)),
            "n_patients": patients,
            "numerator": int(values.sum()) if spec["kind"] == "binary" else "",
            "denominator": len(valid) if spec["kind"] == "binary" else "",
        })
    return rows


def _check_pair_protocol(current: Sequence[Mapping], reference: Sequence[Mapping], cluster_key):
    reference_by_case = {row["case_id"]: (i, row) for i, row in enumerate(reference)}
    if {row["case_id"] for row in current} != set(reference_by_case):
        raise ValueError("Quantification reference case set differs from the current case set.")
    current_meta, reference_meta = _cohort_metadata(current), _cohort_metadata(reference)
    current_available = any(row["quant_status"] == "ok" for row in current)
    reference_available = any(row["quant_status"] == "ok" for row in reference)
    if current_available and reference_available:
        for key in PROTOCOL_FIELDS:
            if current_meta[key] != reference_meta[key]:
                raise ValueError(f"Quantification comparison {key} mismatch.")
    for row in current:
        _, other = reference_by_case[row["case_id"]]
        if row[cluster_key] != other[cluster_key]:
            raise ValueError(f"Cluster ID mismatch for case {row['case_id']!r}.")
        if row["quant_status"] != "ok" or other["quant_status"] != "ok":
            continue
        for key in _REFERENCE_FLAGS:
            if row[key] != other[key]:
                raise ValueError(f"Reference validity mismatch for case {row['case_id']!r}: {key}")
        for name in ("pmw", "dv", "pvdr"):
            key = f"quant_reference_{name}"
            left, right = row[key], other[key]
            if not ((math.isnan(left) and math.isnan(right)) or
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-8)):
                raise ValueError(f"Reference GT mismatch for case {row['case_id']!r}: {key}")
    return reference_by_case


def compare_quantification_records(
    current: Sequence[Mapping], reference: Sequence[Mapping], *, n_boot=10000,
    n_perm=100000, seed=42, cluster_key="patient_id",
) -> list[dict]:
    n_boot = _sampling_count(n_boot, "n_boot")
    n_perm = _sampling_count(n_perm, "n_perm")
    current = validate_quantification_records(current, cluster_key=cluster_key)
    reference = _with_identities(reference, cluster_key)
    missing = not has_quantification_records(reference)
    if missing:
        unavailable = "empty_reference" if not reference else "missing_quantification_reference"
        reference_by_case = {}
        reference_meta = {}
    else:
        reference = validate_quantification_records(reference, cluster_key=cluster_key)
        reference_by_case = _check_pair_protocol(current, reference, cluster_key)
        reference_meta = _cohort_metadata(reference)
        unavailable = ""
    common = {
        **_cohort_metadata(current),
        "reference_quant_prediction_source": reference_meta.get("quant_prediction_source", ""),
        "n_current_cases": len(current), "n_reference_cases": len(reference),
        "n_current_available_cases": sum(row.get("quant_status") == "ok" for row in current),
        "n_reference_available_cases": sum(row.get("quant_status") == "ok" for row in reference),
        "cluster_key": cluster_key,
    }
    rows = []
    for index, spec in enumerate(QUANTIFICATION_METRIC_SPECS):
        current_eligible, current_valid = _indices(current, spec)
        reference_eligible, reference_valid = ([], []) if missing else _indices(reference, spec)
        ref_eligible_set, ref_valid_set = set(reference_eligible), set(reference_valid)
        eligible_pairs = [
            (i, reference_by_case[current[i]["case_id"]][0]) for i in current_eligible
            if not missing and reference_by_case[current[i]["case_id"]][0] in ref_eligible_set
        ]
        current_valid_set = set(current_valid)
        pairs = [(i, j) for i, j in eligible_pairs if i in current_valid_set and j in ref_valid_set]
        patients = len(_groups(current, [i for i, _ in pairs], cluster_key))
        cur_values = np.asarray([current[i][spec["key"]] for i, _ in pairs])
        ref_values = np.asarray([reference[j][spec["key"]] for _, j in pairs])
        delta = float((cur_values - ref_values).mean()) if pairs else float("nan")
        lo = hi = p_value = float("nan")
        if unavailable:
            status = unavailable
        elif not common["n_current_available_cases"]:
            status = "no_available_current_cases"
        elif not common["n_reference_available_cases"]:
            status = "no_available_reference_cases"
        elif not eligible_pairs:
            status = "no_eligible_pairs"
        elif not pairs:
            status = "no_valid_pairs"
        elif patients < 2:
            status = "insufficient_clusters"
        else:
            lo, hi = _paired_cluster_bootstrap_delta(
                current, reference, pairs, spec["key"], cluster_key, n_boot, seed + index
            )
            p_value = _paired_cluster_signflip_p(
                current, reference, pairs, spec["key"], cluster_key, n_perm, seed + 1000 + index
            )
            status = "ok"
        rows.append({
            **common, **{key: spec[key] for key in (
                "target", "subgroup", "metric", "unit", "higher_is_better", "kind")},
            "comparison_status": status,
            "n_current_eligible_cases": len(current_eligible),
            "n_reference_eligible_cases": len(reference_eligible),
            "n_current_valid_cases": len(current_valid), "n_reference_valid_cases": len(reference_valid),
            "n_eligible_pairs": len(eligible_pairs), "n_paired_cases": len(pairs),
            "n_paired_patients": patients,
            "estimate_current": float(cur_values.mean()) if pairs else float("nan"),
            "estimate_reference": float(ref_values.mean()) if pairs else float("nan"),
            "delta_current_minus_reference": delta, "delta_ci_lower": lo, "delta_ci_upper": hi,
            "ci_method": "paired_cluster_percentile_bootstrap", "ci_status": status,
            "test_method": "paired_cluster_signflip_permutation", "test_status": status,
            "p_raw": p_value, "p_adjusted": float("nan"),
            "adjustment_status": "not_applied_single_run",
        })
    return rows
