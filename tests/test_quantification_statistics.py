"""Regression tests for quantified case cohorts, CSVs, and clustered inference."""

from __future__ import annotations

import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path

from src.evaluation.quantification_records import (
    QUANTIFICATION_FIELDS,
    QUANTIFICATION_METRIC_SPECS,
    apply_annotation_completeness,
    unavailable_quantification_record,
)
from src.evaluation.quantification_stats import (
    compare_quantification_records,
    read_reference_csv,
    summarize_quantification_records,
    validate_quantification_records,
)


def quant_record(case_id, patient_id, error=1.0, *, source="seg_logits"):
    row = unavailable_quantification_record("")
    row.update({
        "case_id": str(case_id), "patient_id": str(patient_id),
        "quant_schema_version": "quantification-v1", "quant_protocol": "polar-radial-v1",
        "quant_geometry_id": "geometry-a", "quant_geometry_json": "{}",
        "quant_space": "resized_roi_polar", "quant_length_unit": "resized_roi_px",
        "quant_prediction_source": source, "quant_status": "ok", "quant_reason": "",
        "quant_reference_valid": True, "quant_aux_valid": True,
        "quant_integral_compatible": True, "quant_hard_reference_valid": True,
        "quant_hard_aux_valid": True, "quant_hard_integral_compatible": True,
    })
    for name, value in (("pmw", 5.0), ("dv", 18.0), ("pvdr", 5.0 / 18.0)):
        for kind in ("reference", "hard", "soft"):
            row[f"quant_{kind}_{name}"] = value
    for spec in QUANTIFICATION_METRIC_SPECS:
        value = error if spec["kind"] == "continuous" else float(
            spec["metric"] in {"reference_valid_fraction", "aux_valid_fraction"}
        )
        row[spec["key"]] = value
        row[spec["valid_key"]] = True
    return apply_annotation_completeness(row)


def get_metric(rows, metric="hard_mae", target="PMW"):
    return next(row for row in rows if row["metric"] == metric and row["target"] == target)


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class QuantificationStatisticsTests(unittest.TestCase):
    def test_patient_cluster_bootstrap_keeps_case_weighted_mean(self):
        records = [quant_record(f"00{i}", "patient-a", 0.0) for i in range(3)]
        records.append(quant_record("003", "patient-b", 4.0))
        result = get_metric(summarize_quantification_records(records, n_boot=256, seed=17))
        self.assertEqual(result["estimate"], 1.0)
        self.assertEqual(result["n_patients"], 2)
        self.assertEqual((result["ci_lower"], result["ci_upper"]), (0.0, 4.0))
        self.assertEqual(result["ci_status"], "ok")

    def test_eligible_and_valid_counts_and_paired_soft_cohort(self):
        first = quant_record("001", "a", 1.0)
        second = quant_record("002", "b", 3.0)
        second["quant_hard_reference_valid"] = False
        for spec in QUANTIFICATION_METRIC_SPECS:
            if spec["subgroup"] == "reference_and_prediction_valid":
                second[spec["valid_key"]] = False
        second["quant_hard_invalid_fraction"] = 1.0
        third = quant_record("003", "c", 100.0)
        third["quant_reference_valid"] = False
        third["quant_aux_valid"] = False
        third = apply_annotation_completeness(third)
        third["quant_reference_valid_fraction"] = 0.0
        third["quant_aux_valid_fraction"] = 0.0
        fourth = {"case_id": "004", "patient_id": "d",
                  **unavailable_quantification_record("cartesian_input")}
        rows = summarize_quantification_records([first, second, third, fourth], n_boot=32)
        hard = get_metric(rows)
        self.assertEqual((hard["n_cases"], hard["n_available_cases"]), (4, 3))
        self.assertEqual((hard["n_eligible_cases"], hard["n_valid_cases"]), (2, 1))
        self.assertEqual(hard["n_invalid_cases"], 1)
        self.assertEqual(hard["invalid_fraction"], 0.5)
        self.assertEqual(hard["estimate"], 1.0)
        self.assertEqual(get_metric(rows, "soft_mae")["estimate"], 2.0)
        self.assertEqual(get_metric(rows, "soft_mae_paired")["estimate"], 1.0)
        failures = get_metric(rows, "hard_invalid_fraction", "quantification")
        self.assertEqual(failures["estimate"], 0.5)
        self.assertEqual(failures["denominator"], 2)

    def test_unknown_ppa_excludes_ppa_targets_but_keeps_dv(self):
        records = [apply_annotation_completeness(quant_record("001", "a"), False),
                   quant_record("002", "b")]
        rows = summarize_quantification_records(records, n_boot=32)
        self.assertEqual(get_metric(rows, target="PMW")["n_eligible_cases"], 1)
        self.assertEqual(get_metric(rows, target="PVDR")["n_eligible_cases"], 1)
        self.assertEqual(get_metric(rows, target="Dv")["n_eligible_cases"], 2)

    def test_one_patient_never_gets_a_confidence_interval_or_p_value(self):
        records = [quant_record("001", "shared"), quant_record("002", "shared")]
        for row in summarize_quantification_records(records, n_boot=32):
            self.assertEqual(row["ci_status"], "insufficient_clusters")
            self.assertTrue(math.isnan(row["ci_lower"]))
        for row in compare_quantification_records(records, records, n_boot=32, n_perm=32):
            self.assertEqual(row["ci_status"], "insufficient_clusters")
            self.assertTrue(math.isnan(row["p_raw"]))

    def test_unavailable_cases_are_not_zero_error_measurements(self):
        records = [{"case_id": "001", "patient_id": "a",
                    **unavailable_quantification_record("polar_disabled")}]
        for row in summarize_quantification_records(records, n_boot=16):
            self.assertEqual(row["availability_status"], "no_available_cases")
            self.assertEqual(row["n_available_cases"], 0)
            self.assertEqual(row["n_valid_cases"], 0)
            self.assertIn("polar_disabled", row["quant_reason_counts"])
            self.assertTrue(math.isnan(row["estimate"]))

    def test_nonfinite_or_masked_values_are_nan_in_formal_records(self):
        row = quant_record("001", "a")
        row["quant_hard_reference_valid"] = False
        row["quant_pmw_hard_mae_px_valid"] = False
        row["quant_dv_soft_mae_px"] = float("inf")
        normal = validate_quantification_records([row])[0]
        self.assertTrue(math.isnan(normal["quant_hard_pmw"]))
        self.assertTrue(math.isnan(normal["quant_pmw_hard_mae_px"]))
        self.assertFalse(normal["quant_dv_soft_mae_px_valid"])
        self.assertTrue(math.isnan(normal["quant_dv_soft_mae_px"]))
        self.assertEqual(row["quant_hard_pmw"], 5.0)  # caller's record is untouched

    def test_csv_preserves_identity_strings_and_parses_false(self):
        row = quant_record("0001", "0007")
        row["quant_pmw_hard_mae_px_valid"] = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.csv"
            write_rows(path, [row])
            loaded = read_reference_csv(path)[0]
            self.assertEqual(loaded["case_id"], "0001")
            self.assertEqual(loaded["patient_id"], "0007")
            self.assertIs(loaded["quant_pmw_hard_mae_px_valid"], False)
            row["quant_pmw_hard_mae_px_valid"] = "not_false"
            write_rows(path, [row])
            with self.assertRaisesRegex(ValueError, "explicit boolean"):
                read_reference_csv(path)

    def test_duplicate_cases_are_rejected_on_both_sides(self):
        row = quant_record("001", "a")
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            summarize_quantification_records([row, row], n_boot=16)
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            compare_quantification_records([row], [row, row], n_boot=16, n_perm=16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.csv"
            write_rows(path, [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
                read_reference_csv(path)

    def test_mixed_geometry_protocol_or_output_source_cannot_be_aggregated(self):
        for field in ("quant_protocol", "quant_geometry_id", "quant_schema_version",
                      "quant_space", "quant_prediction_source"):
            with self.subTest(field=field):
                records = [quant_record("001", "a"), quant_record("002", "b")]
                records[1][field] = "different"
                with self.assertRaisesRegex(ValueError, "Mixed"):
                    summarize_quantification_records(records, n_boot=16)

    def test_comparison_matches_case_ids_and_delta_direction(self):
        current = [quant_record("001", "a", 1.0, source="quant_logits"),
                   quant_record("002", "b", 2.0, source="quant_logits")]
        reference = [quant_record("002", "b", 3.0), quant_record("001", "a", 2.0)]
        result = get_metric(compare_quantification_records(
            current, reference, n_boot=64, n_perm=64,
        ))
        self.assertEqual(result["delta_current_minus_reference"], -1.0)
        self.assertEqual((result["delta_ci_lower"], result["delta_ci_upper"]), (-1.0, -1.0))
        self.assertEqual(result["n_paired_cases"], 2)
        self.assertEqual(result["n_paired_patients"], 2)
        self.assertEqual(result["quant_prediction_source"], "quant_logits")
        self.assertEqual(result["reference_quant_prediction_source"], "seg_logits")

    def test_comparison_checks_geometry_gt_annotation_and_cluster_identity(self):
        current = [quant_record("001", "a"), quant_record("002", "b")]
        changes = {
            "quant_geometry_id": "other-grid", "quant_protocol": "other-protocol",
            "quant_reference_pmw": 99.0, "quant_annotation_complete": False,
            "patient_id": "other-person",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                reference = copy.deepcopy(current)
                for row in reference:
                    row[field] = value
                with self.assertRaises(ValueError):
                    compare_quantification_records(current, reference, n_boot=16, n_perm=16)
        with self.assertRaisesRegex(ValueError, "case set differs"):
            compare_quantification_records(current, current[:1], n_boot=16, n_perm=16)

    def test_legacy_reference_yields_explicit_missing_quantification_status(self):
        current = [quant_record("001", "a")]
        legacy = [{"case_id": "001", "patient_id": "a", "od_dsc": 1.0}]
        for row in compare_quantification_records(current, legacy, n_boot=16, n_perm=16):
            self.assertEqual(row["comparison_status"], "missing_quantification_reference")
            self.assertEqual(row["n_paired_cases"], 0)
            self.assertTrue(math.isnan(row["delta_current_minus_reference"]))


def segmentation_record(case_id, patient_id):
    return {
        "case_id": case_id, "patient_id": patient_id, "eye": "R",
        "image_path": f"/images/{case_id}.png", "mask_path": f"/masks/{case_id}.png",
        "pred_path": "", "per_class": {
            c: {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0, "support": 10}
            for c in range(3)
        },
        "protocol_version": "odppa-v2", "reference_complete": True,
        "eval_height": 32, "eval_width": 32, "gt_od_area_px": 100,
        "pred_od_area_px": 100, "gt_ppa_area_px": 20, "pred_ppa_area_px": 20,
        "gt_od_equiv_diameter_px": 10.0, "nsd_tolerance_px": 0.1,
        "od_reference_valid": True, "od_pred_empty": False, "od_dsc": 1.0, "od_nsd": 1.0,
        "od_hd95_px": 0.0, "od_hd95_norm": 0.0, "od_hd95_valid": True,
        "ppa_status": "positive", "ppa_pred_empty": False, "ppa_dsc": 1.0, "ppa_nsd": 1.0,
        "ppa_hd95_px": 0.0, "ppa_hd95_norm": 0.0, "ppa_hd95_valid": True,
        "ppa_negative_correct": float("nan"), "ppa_fp_norm": float("nan"),
    }


class QuantificationReportTests(unittest.TestCase):
    def setUp(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf is required for the evaluator integration test")
        self.cfg = OmegaConf.create({
            "dataset": {"num_classes": 3, "class_names": ["BG", "OD", "PPA"],
                        "foreground_classes": [1, 2], "name": "test"},
            "model": {"arch": "Unet", "encoder": "test"},
            "eval": {"per_case": True, "per_class": True, "odppa": {
                "enabled": False, "bootstrap_samples": 32, "permutation_samples": 32,
                "statistics_seed": 17, "cluster_key": "patient_id", "reference_csv": None,
            }},
        })

    def _write(self, records, folder):
        from src.evaluation.evaluator import SegEvaluator
        return SegEvaluator(self.cfg, "test", "unit", "checkpoint").write(records, folder)

    def test_both_report_paths_append_quantification_without_changing_segmentation(self):
        plain = [segmentation_record("001", "0010"), segmentation_record("002", "0020")]
        quantified = [{**row, **quant_record(row["case_id"], row["patient_id"])} for row in plain]
        with tempfile.TemporaryDirectory() as folder:
            for odppa in (False, True):
                with self.subTest(odppa=odppa):
                    self.cfg.eval.odppa.enabled = odppa
                    before = self._write(plain, Path(folder) / f"before-{odppa}")
                    after = self._write(quantified, Path(folder) / f"after-{odppa}")
                    for report in ("summary", "metrics" if odppa else "per_class"):
                        self.assertEqual(Path(before[report]).read_text(), Path(after[report]).read_text())
                    with Path(after["per_case"]).open() as stream:
                        cases = list(csv.DictReader(stream))
                    self.assertEqual(cases[0]["case_id"], "001")
                    self.assertEqual(cases[0]["patient_id"], "0010")
                    self.assertTrue(set(QUANTIFICATION_FIELDS).issubset(cases[0]))
                    self.assertIn("quantification_per_case", after)
                    self.assertIn("quantification_metrics", after)
                    self.assertNotIn("quantification_metrics", before)

    def test_main_or_quantification_csv_can_be_a_reference_on_either_path(self):
        with tempfile.TemporaryDirectory() as folder:
            for odppa in (False, True):
                for reference_report in ("per_case", "quantification_per_case"):
                    with self.subTest(odppa=odppa, reference_report=reference_report):
                        self.cfg.eval.odppa.enabled = odppa
                        self.cfg.eval.odppa.reference_csv = None
                        baseline = [{**segmentation_record(case, patient),
                                     **quant_record(case, patient, 2.0)}
                                    for case, patient in (("001", "0010"), ("002", "0020"))]
                        ref = self._write(baseline, Path(folder) / f"ref-{odppa}-{reference_report}")
                        self.cfg.eval.odppa.reference_csv = ref[reference_report]
                        current = [{**row, **quant_record(row["case_id"], row["patient_id"],
                                                         1.0, source="quant_logits")}
                                   for row in baseline]
                        output = self._write(current, Path(folder) / f"cur-{odppa}-{reference_report}")
                        with Path(output["quantification_comparison"]).open() as stream:
                            result = get_metric(list(csv.DictReader(stream)))
                        self.assertEqual(float(result["delta_current_minus_reference"]), -1.0)
                        self.assertEqual(result["comparison_status"], "ok")
                        if odppa and reference_report == "per_case":
                            self.assertIn("comparison", output)
                        if odppa and reference_report == "quantification_per_case":
                            self.assertNotIn("comparison", output)

    def test_unavailable_quantification_is_exported_even_without_main_per_case(self):
        row = {**segmentation_record("001", "0010"),
               **unavailable_quantification_record("polar_disabled")}
        self.cfg.eval.per_case = False
        with tempfile.TemporaryDirectory() as folder:
            output = self._write([row], folder)
            self.assertNotIn("per_case", output)
            with Path(output["quantification_per_case"]).open() as stream:
                case = next(csv.DictReader(stream))
            self.assertEqual(case["quant_status"], "unavailable")
            self.assertTrue(math.isnan(float(case["quant_reference_pmw"])))
            with Path(output["quantification_metrics"]).open() as stream:
                result = next(csv.DictReader(stream))
            self.assertEqual(result["availability_status"], "no_available_cases")
            self.assertEqual(result["n_valid_cases"], "0")

    def test_reference_cannot_be_overwritten_before_it_is_compared(self):
        row = {**segmentation_record("001", "0010"), **quant_record("001", "0010")}
        with tempfile.TemporaryDirectory() as folder:
            output = self._write([row], folder)
            path = Path(output["per_case"])
            original = path.read_bytes()
            self.cfg.eval.odppa.reference_csv = str(path)
            with self.assertRaisesRegex(ValueError, "would be overwritten"):
                self._write([row], folder)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
