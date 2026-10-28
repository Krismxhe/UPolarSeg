"""Diameter provenance must prevent comparisons across measurement definitions."""

from __future__ import annotations

import copy
import csv
import json
from types import SimpleNamespace

import pytest

from src.evaluation.quantification_records import (
    QUANTIFICATION_SCHEMA_VERSION,
    quantification_protocol,
    unavailable_quantification_record,
)
from src.evaluation.quantification_stats import (
    compare_quantification_records,
    read_reference_csv,
    validate_quantification_records,
)


def _readout(mode="fixed_vertical", axis=90.0):
    """Scalar/list stand-in: provenance does not require torch or a model."""
    return SimpleNamespace(
        dv_mode=mode,
        angle_bins=8,
        radius_bins=6,
        max_radius=12.0,
        od_class=1,
        ppa_class=2,
        vertical_angle_deg=axis,
        denominator_floor_px=1.0,
        valid_grid=[[True] * 6 for _ in range(8)],
        geometry_metadata={
            "img_size": 32,
            "center": (15.5, 15.5),
            "mode": "linear",
            "vertical_angle_deg": axis,
        },
    )


def _record(case_id, readout):
    return {
        **unavailable_quantification_record(""),
        **quantification_protocol(readout),
        "case_id": case_id,
        "patient_id": "patient-a",
        "quant_status": "ok",
        "quant_reason": "",
        "quant_prediction_source": "seg_logits",
    }


def test_fixed_vertical_preserves_original_geometry_hash():
    # Captured from the pre-change implementation for this exact geometry.
    legacy_hash = "172d26e0f61bd0cfa3bbcb24e7953b91c8eb41d5708437a166df13535606335e"
    legacy = _readout()
    del legacy.dv_mode
    fixed = _readout()
    fixed.geometry_metadata.update({
        "dv_mode": "fixed_vertical",
        "diameter_definition": "fixed axis",
        "opposite_interpolation": "periodic_linear",
        "opposite_angle_deg": 180.0,
    })
    before, after = quantification_protocol(legacy), quantification_protocol(fixed)
    assert before == after
    assert after["quant_protocol"] == "polar-radial-v1"
    assert after["quant_geometry_id"] == legacy_hash
    geometry = json.loads(after["quant_geometry_json"])
    assert geometry["vertical_angle_deg"] == 90.0
    assert "dv_mode" not in geometry


def test_max_center_chord_has_explicit_v2_diameter_provenance():
    protocol = quantification_protocol(_readout("max_center_chord"))
    assert protocol["quant_protocol"] == "polar-radial-v2"
    assert protocol["quant_schema_version"] == QUANTIFICATION_SCHEMA_VERSION == "quantification-v1"
    geometry = json.loads(protocol["quant_geometry_json"])
    assert geometry["dv_mode"] == "max_center_chord"
    assert geometry["diameter_definition"] == "max_theta[r(theta)+r(theta+pi)]"
    assert geometry["opposite_angle_deg"] == 180.0
    assert geometry["opposite_interpolation"] == "periodic_linear"
    assert "vertical_angle_deg" not in geometry


@pytest.mark.parametrize("axis", [0.0, 41.5, 90.0, 179.0, float("nan")])
def test_max_mode_ignores_fixed_axis_in_attributes_and_metadata(axis):
    baseline = quantification_protocol(_readout("max_center_chord"))
    changed = _readout("max_center_chord", axis)
    changed.geometry_metadata["vertical_angle"] = axis
    before = copy.deepcopy(changed.geometry_metadata)
    result = quantification_protocol(changed)
    assert result == baseline
    assert "vertical_angle" not in json.loads(result["quant_geometry_json"])
    # Removing unused provenance must not mutate the source readout metadata.
    assert set(changed.geometry_metadata) == set(before)
    assert "vertical_angle_deg" in changed.geometry_metadata


def test_fixed_axis_changes_and_actual_grid_changes_remain_distinguishable():
    assert quantification_protocol(_readout(axis=30.0))["quant_geometry_id"] != (
        quantification_protocol(_readout(axis=90.0))["quant_geometry_id"]
    )
    baseline = _readout("max_center_chord")
    changed = copy.deepcopy(baseline)
    changed.valid_grid[0][-1] = False
    assert quantification_protocol(baseline)["quant_geometry_id"] != (
        quantification_protocol(changed)["quant_geometry_id"]
    )


def test_unknown_diameter_mode_cannot_be_mislabelled_as_v1():
    with pytest.raises(ValueError, match="dv_mode"):
        quantification_protocol(_readout("unknown"))


def test_existing_cohort_validation_rejects_mixed_diameter_protocols():
    rows = [
        _record("001", _readout()),
        _record("002", _readout("max_center_chord")),
    ]
    with pytest.raises(ValueError, match="Mixed quant_protocol"):
        validate_quantification_records(rows)


def test_reference_csv_cannot_mix_fixed_and_max_chord_measurements(tmp_path):
    reference = _record("001", _readout())
    path = tmp_path / "fixed-axis-reference.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference))
        writer.writeheader()
        writer.writerow(reference)
    current = _record("001", _readout("max_center_chord"))
    with pytest.raises(ValueError, match="quant_protocol mismatch"):
        compare_quantification_records(
            [current], read_reference_csv(path), n_boot=4, n_perm=4,
        )
