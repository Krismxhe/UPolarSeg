"""
Segmentation evaluator: aggregate per-case records into research CSVs.

Consumes the per-case records collected by ``SegModule.test_step`` and writes:

    <output_dir>/summary.csv      (one row for the run)
    <output_dir>/per_class.csv    (one row per class)
    <output_dir>/per_case.csv     (one row per case)
    <output_dir>/config.yaml      (resolved config)

A per-case record is::

    {
        "case_id": str,
        "image_path": str,
        "mask_path": str,
        "pred_path": str,                       # "" unless predictions are saved
        "pred_polar_path": str,                 # "" unless polar masks are saved
        "pred_cartesian_path": str,             # "" unless Cartesian masks are saved
        "overlay_polar_path": str,              # separate RGB visualization files
        "overlay_cartesian_path": str,
        "overlay_polar_hard_path": str,         # separate-layout prediction files
        "overlay_cartesian_hard_path": str,
        "overlay_polar_soft_path": str,
        "overlay_cartesian_soft_path": str,
        "overlay_gt_polar_path": str,           # independent hard ground truth
        "overlay_gt_cartesian_path": str,
        "overlay_metadata_path": str,           # measurement geometry/source JSON
        "per_class": {class_value: {dice, iou, precision, recall, support}},
    }

Aggregation is macro: per-case metrics (with the empty-mask conventions in
``src.metrics.functional``) are averaged across cases. ``mean_*`` excludes
background unless ``eval.include_background`` is true.

Extensibility (Phase 9b): register extra writers via ``register_writer`` to emit
additional CSVs (e.g. boundary_metrics.csv, clinical_metrics.csv) from the same
records without modifying this class.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from src.evaluation.quantification_records import QUANTIFICATION_FIELDS
from src.evaluation.quantification_stats import (
    QUANTIFICATION_COMPARISON_FIELDS,
    QUANTIFICATION_METRIC_FIELDS,
    compare_quantification_records,
    has_quantification_records,
    read_reference_csv,
    summarize_quantification_records,
    validate_quantification_records,
)
from src.utils.io import ensure_dir, write_csv, write_resolved_config

# Schemas (stable column order).
SUMMARY_FIELDS = [
    "run_name", "checkpoint", "dataset", "split", "model", "encoder",
    "num_cases", "mean_dice", "mean_iou", "mean_precision", "mean_recall",
]
PER_CLASS_FIELDS = [
    "run_name", "split", "class_id", "class_name",
    "dice", "iou", "precision", "recall", "support_pixels",
]
PER_CASE_BASE_FIELDS = [
    "run_name", "split", "case_id", "image_path", "mask_path",
    "dice_mean", "iou_mean", "precision_mean", "recall_mean", "pred_path",
]
# Appended after all existing per-case columns to preserve their order.
PREDICTION_PATH_FIELDS = ["pred_polar_path", "pred_cartesian_path"]
OVERLAY_PATH_FIELDS = [
    "overlay_polar_path", "overlay_cartesian_path",
    "overlay_polar_hard_path", "overlay_cartesian_hard_path",
    "overlay_polar_soft_path", "overlay_cartesian_soft_path",
    "overlay_gt_polar_path", "overlay_gt_cartesian_path", "overlay_metadata_path",
]
EXPORT_PATH_FIELDS = PREDICTION_PATH_FIELDS + OVERLAY_PATH_FIELDS

_METRICS = ("dice", "iou", "precision", "recall")


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class SegEvaluator:
    def __init__(self, cfg, split: str, run_name: str, checkpoint: str = ""):
        ds = cfg.dataset
        self.cfg = cfg
        self.split = split
        self.run_name = run_name
        self.checkpoint = checkpoint or ""
        self.num_classes = int(ds.num_classes)
        self.is_binary = self.num_classes == 1
        self.class_names = list(ds.class_names)
        eval_cfg = cfg.get("eval", {}) or {}
        self.include_background = bool(eval_cfg.get("include_background", False))
        self.write_per_case = bool(eval_cfg.get("per_case", True))
        self.write_per_class = bool(eval_cfg.get("per_class", True))
        self.dataset_name = str(ds.get("name", ""))
        self.model_name = str(cfg.model.get("arch", cfg.model.get("name", "")))
        self.encoder = str(cfg.model.get("encoder", ""))
        self.class_specs = self._build_class_specs(ds)
        self._extra_writers: List[Callable[[List[dict], object], None]] = []

        # Optional multi-task CSVs (Phase 9b); default OFF.
        from src.tasks import get_output_cfg, is_task_enabled
        self.clinical_enabled = is_task_enabled(cfg, "clinical")
        self.boundary_enabled = is_task_enabled(cfg, "boundary")
        self.clinical_metrics = list(get_output_cfg(cfg, "clinical").get(
            "metrics", ["area", "perimeter", "area_ratio"]))

    # ── Extension point for Phase 9b ──────────────────────────────────────────
    def register_writer(self, writer: Callable[[List[dict], object], None]) -> None:
        """Register a callable ``writer(records, output_dir)`` invoked by ``write``."""
        self._extra_writers.append(writer)

    # ── Class specification ───────────────────────────────────────────────────
    def _build_class_specs(self, ds) -> List[dict]:
        """List of {class_id, class_name, foreground} describing evaluated classes.

        Binary: pseudo-classes background(value 0) and foreground(value 1, named
        after class_names[0]). Multiclass: class i named class_names[i], marked
        foreground when i is in dataset.foreground_classes.
        """
        if self.is_binary:
            return [
                {"class_id": 0, "class_name": "background", "foreground": False},
                {"class_id": 1, "class_name": self.class_names[0], "foreground": True},
            ]
        fg = {int(c) for c in ds.foreground_classes}
        return [
            {"class_id": i, "class_name": self.class_names[i], "foreground": (i in fg)}
            for i in range(self.num_classes)
        ]

    @property
    def _mean_class_ids(self) -> List[int]:
        if self.include_background:
            return [c["class_id"] for c in self.class_specs]
        return [c["class_id"] for c in self.class_specs if c["foreground"]]

    @property
    def _foreground_specs(self) -> List[dict]:
        return [c for c in self.class_specs if c["foreground"]]

    # ── Row builders ───────────────────────────────────────────────────────────
    def _per_case_row(self, rec: dict, *, include_quant=False, cluster_key="patient_id") -> dict:
        pc = rec["per_class"]
        mean_ids = self._mean_class_ids
        row = {
            "run_name": self.run_name,
            "split": self.split,
            "case_id": rec.get("case_id", ""),
            "image_path": rec.get("image_path", ""),
            "mask_path": rec.get("mask_path", ""),
            "dice_mean": round(_mean([pc[c]["dice"] for c in mean_ids]), 6),
            "iou_mean": round(_mean([pc[c]["iou"] for c in mean_ids]), 6),
            "precision_mean": round(_mean([pc[c]["precision"] for c in mean_ids]), 6),
            "recall_mean": round(_mean([pc[c]["recall"] for c in mean_ids]), 6),
            "pred_path": rec.get("pred_path", ""),
        }
        for spec in self.class_specs:
            cid, name = spec["class_id"], spec["class_name"]
            for m in _METRICS:
                row[f"{m}_{name}"] = round(pc[cid][m], 6)
        if include_quant:
            for key in self._quantification_case_fields(cluster_key):
                row[key] = rec.get(key, "")
        for key in EXPORT_PATH_FIELDS:
            row[key] = rec.get(key, "")
        return row

    @staticmethod
    def _quantification_case_fields(cluster_key="patient_id") -> List[str]:
        return list(dict.fromkeys(("patient_id", cluster_key, *QUANTIFICATION_FIELDS)))

    def _per_case_fields(self, *, include_quant=False, cluster_key="patient_id") -> List[str]:
        extra = [f"{m}_{spec['class_name']}" for spec in self.class_specs for m in _METRICS]
        fields = PER_CASE_BASE_FIELDS + extra
        if include_quant:
            fields += [key for key in self._quantification_case_fields(cluster_key) if key not in fields]
        fields += [key for key in EXPORT_PATH_FIELDS if key not in fields]
        return fields

    def _per_class_rows(self, records: List[dict]) -> List[dict]:
        rows = []
        for spec in self.class_specs:
            cid = spec["class_id"]
            rows.append({
                "run_name": self.run_name,
                "split": self.split,
                "class_id": cid,
                "class_name": spec["class_name"],
                "dice": round(_mean([r["per_class"][cid]["dice"] for r in records]), 6),
                "iou": round(_mean([r["per_class"][cid]["iou"] for r in records]), 6),
                "precision": round(_mean([r["per_class"][cid]["precision"] for r in records]), 6),
                "recall": round(_mean([r["per_class"][cid]["recall"] for r in records]), 6),
                "support_pixels": int(sum(r["per_class"][cid]["support"] for r in records)),
            })
        return rows

    def _summary_row(self, per_class_rows: List[dict], num_cases: int) -> dict:
        mean_ids = set(self._mean_class_ids)
        fg = [r for r in per_class_rows if r["class_id"] in mean_ids]
        return {
            "run_name": self.run_name,
            "checkpoint": self.checkpoint,
            "dataset": self.dataset_name,
            "split": self.split,
            "model": self.model_name,
            "encoder": self.encoder,
            "num_cases": num_cases,
            "mean_dice": round(_mean([r["dice"] for r in fg]), 6),
            "mean_iou": round(_mean([r["iou"] for r in fg]), 6),
            "mean_precision": round(_mean([r["precision"] for r in fg]), 6),
            "mean_recall": round(_mean([r["recall"] for r in fg]), 6),
        }

    # ── Write ──────────────────────────────────────────────────────────────────
    def write(self, records: List[dict], output_dir) -> Dict[str, str]:
        out = ensure_dir(output_dir)

        eval_cfg = self.cfg.get("eval", {}) or {}
        odppa_cfg = eval_cfg.get("odppa", {}) or {}
        cluster_key = str(odppa_cfg.get("cluster_key", "patient_id"))
        include_quant = has_quantification_records(records)
        if include_quant:
            records = validate_quantification_records(records, cluster_key=cluster_key)
            reference_csv = odppa_cfg.get("reference_csv", None)
            if reference_csv:
                from pathlib import Path

                from hydra.utils import to_absolute_path

                reference_path = Path(to_absolute_path(str(reference_csv))).resolve()
                output_paths = {(out / name).resolve() for name in (
                    "per_case.csv", "quantification_per_case.csv",
                )}
                if reference_path in output_paths:
                    raise ValueError(
                        "Reference CSV would be overwritten by this evaluation. "
                        "Use a separate output directory for the current run."
                    )

        if bool(
            odppa_cfg.get(
                "enabled",
                False,
            )
        ):
            written = self._write_odppa(
                records,
                out,
                odppa_cfg,
            )
            if include_quant:
                written.update(self._write_quantification(records, out, odppa_cfg))
            return written

        per_class_rows = self._per_class_rows(records) if records else []
        summary_row = self._summary_row(per_class_rows, len(records))

        written = {}
        written["summary"] = str(write_csv(out / "summary.csv", [summary_row], SUMMARY_FIELDS))
        if self.write_per_class:
            written["per_class"] = str(write_csv(out / "per_class.csv", per_class_rows, PER_CLASS_FIELDS))
        if self.write_per_case:
            per_case_rows = [self._per_case_row(
                r, include_quant=include_quant, cluster_key=cluster_key,
            ) for r in records]
            written["per_case"] = str(
                write_csv(out / "per_case.csv", per_case_rows, self._per_case_fields(
                    include_quant=include_quant, cluster_key=cluster_key,
                ))
            )
        written["config"] = str(write_resolved_config(out / "config.yaml", self.cfg))

        # Optional multi-task CSVs (Phase 9b) — only when enabled AND the records
        # carry the corresponding fields. The core CSVs above are never affected.
        if self.clinical_enabled and records and "clinical" in records[0]:
            written["clinical"] = str(self._write_clinical(records, out))
        if self.boundary_enabled and records and "boundary" in records[0]:
            written["boundary"] = str(self._write_boundary(records, out))

        for writer in self._extra_writers:
            writer(records, out)
        if include_quant:
            written.update(self._write_quantification(records, out, odppa_cfg))
        return written

    def _write_odppa(self, records: List[dict], out, odppa_cfg) -> Dict[str, str]:
        """Write the OD/PPA paper-evaluation CSVs from one scalar record per case."""
        from src.evaluation.odppa_stats import compare_records, summarize_records

        n_boot = int(odppa_cfg.get("bootstrap_samples", 10000))
        n_perm = int(odppa_cfg.get("permutation_samples", 100000))
        seed = int(odppa_cfg.get("statistics_seed", 42))
        cluster_key = str(odppa_cfg.get("cluster_key", "patient_id"))

        # Reject accidental duplicate case IDs before any aggregation.
        ids = [str(r.get("case_id", "")) for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate case_id values remain after test collection; refusing CSV export.")

        common = {
            "run_name": self.run_name, "checkpoint": self.checkpoint,
            "dataset": self.dataset_name, "split": self.split,
            "model": self.model_name, "encoder": self.encoder,
        }

        scalar_fields = [
            "protocol_version", "case_id", "patient_id", "eye", "image_path", "mask_path",
            "reference_complete", "eval_height", "eval_width",
            "gt_od_area_px", "pred_od_area_px", "gt_ppa_area_px", "pred_ppa_area_px",
            "gt_od_equiv_diameter_px", "nsd_tolerance_px",
            "od_reference_valid", "od_pred_empty", "od_dsc", "od_nsd",
            "od_hd95_px", "od_hd95_norm", "od_hd95_valid",
            "ppa_status", "ppa_pred_empty", "ppa_dsc", "ppa_nsd",
            "ppa_hd95_px", "ppa_hd95_norm", "ppa_hd95_valid",
            "ppa_negative_correct", "ppa_fp_norm", "pred_path",
        ]
        if has_quantification_records(records):
            scalar_fields += [key for key in self._quantification_case_fields(cluster_key)
                              if key not in scalar_fields]
        scalar_fields += [key for key in EXPORT_PATH_FIELDS if key not in scalar_fields]
        per_case_rows = []
        for r in records:
            row = {**common}
            row.update({k: r.get(k, "") for k in scalar_fields})
            per_case_rows.append(row)
        per_case_fields = list(common) + scalar_fields

        metrics = summarize_records(
            records, n_boot=n_boot, seed=seed, cluster_key=cluster_key
        )
        metric_rows = [{**common, **m} for m in metrics]
        metric_fields = list(common) + [
            "target", "subgroup", "metric", "unit", "higher_is_better",
            "estimate", "sd", "ci_lower", "ci_upper", "ci_method", "ci_status",
            "n_eligible_cases", "n_valid_cases", "n_patients", "numerator", "denominator",
        ]

        summary = dict(common)
        summary.update({
            "num_cases": len(records),
            "num_patients": len({str(r.get(cluster_key) or r.get("case_id")) for r in records}),
            "num_od_valid": sum(bool(r.get("od_reference_valid")) for r in records),
            "num_ppa_positive": sum(r.get("ppa_status") == "positive" for r in records),
            "num_ppa_negative": sum(r.get("ppa_status") == "negative" for r in records),
            "num_ppa_unknown": sum(r.get("ppa_status") == "unknown" for r in records),
            "cluster_key": cluster_key,
            "bootstrap_samples": n_boot, "statistics_seed": seed,
        })
        for m in metrics:
            prefix = f"{str(m['target']).lower()}_{m['subgroup']}_{m['metric']}".replace("-", "_")
            summary[f"{prefix}_mean"] = m["estimate"]
            summary[f"{prefix}_sd"] = m["sd"]
            summary[f"{prefix}_ci_lower"] = m["ci_lower"]
            summary[f"{prefix}_ci_upper"] = m["ci_upper"]
            summary[f"{prefix}_n"] = m["n_valid_cases"]

        written = {}
        written["per_case"] = str(write_csv(out / "per_case.csv", per_case_rows, per_case_fields))
        written["metrics"] = str(write_csv(out / "metrics.csv", metric_rows, metric_fields))
        written["summary"] = str(write_csv(out / "summary.csv", [summary], list(summary.keys())))
        written["config"] = str(write_resolved_config(out / "config.yaml", self.cfg))

        reference_csv = odppa_cfg.get("reference_csv", None)
        if reference_csv:
            from hydra.utils import to_absolute_path
            ref_path = to_absolute_path(str(reference_csv))
            reference = read_reference_csv(ref_path, cluster_key=cluster_key)
            # A quantification-only reference is a valid input for the additional
            # report; it does not contain an OD/PPA segmentation comparison.
            quant_only = has_quantification_records(reference) and not any(
                "protocol_version" in row for row in reference
            )
            if not quant_only:
                comparison = compare_records(
                    records, reference, n_boot=n_boot, n_perm=n_perm, seed=seed, cluster_key=cluster_key
                )
                comp_rows = [{**common, "reference_csv": ref_path, **r} for r in comparison]
                comp_fields = list(common) + ["reference_csv", "target", "subgroup", "metric",
                    "higher_is_better", "n_paired_cases", "n_paired_patients",
                    "estimate_current", "estimate_reference", "delta_current_minus_reference",
                    "delta_ci_lower", "delta_ci_upper", "test_method", "p_raw",
                    "p_adjusted", "adjustment_status"]
                written["comparison"] = str(write_csv(out / "comparison.csv", comp_rows, comp_fields))

        return written

    def _write_quantification(self, records: List[dict], out, odppa_cfg) -> Dict[str, str]:
        """Additional readout reports shared by the ordinary and OD/PPA paths."""
        n_boot = int(odppa_cfg.get("bootstrap_samples", 10000))
        n_perm = int(odppa_cfg.get("permutation_samples", 100000))
        seed = int(odppa_cfg.get("statistics_seed", 42))
        cluster_key = str(odppa_cfg.get("cluster_key", "patient_id"))
        common = {
            "run_name": self.run_name, "checkpoint": self.checkpoint,
            "dataset": self.dataset_name, "split": self.split,
            "model": self.model_name, "encoder": self.encoder,
        }
        case_fields = list(dict.fromkeys((
            "case_id", "patient_id", cluster_key, "image_path", "mask_path", "pred_path",
            *QUANTIFICATION_FIELDS,
            *EXPORT_PATH_FIELDS,
        )))
        per_case = [{**common, **{key: row.get(key, "") for key in case_fields}} for row in records]
        metrics = summarize_quantification_records(
            records, n_boot=n_boot, seed=seed, cluster_key=cluster_key,
        )
        comparison = None
        reference_csv = odppa_cfg.get("reference_csv", None)
        if reference_csv:
            from hydra.utils import to_absolute_path
            ref_path = to_absolute_path(str(reference_csv))
            reference = read_reference_csv(ref_path, cluster_key=cluster_key)
            comparison = compare_quantification_records(
                records, reference, n_boot=n_boot, n_perm=n_perm,
                seed=seed, cluster_key=cluster_key,
            )
        written = {
            "quantification_per_case": str(write_csv(
                out / "quantification_per_case.csv", per_case, [*common, *case_fields],
            )),
            "quantification_metrics": str(write_csv(
                out / "quantification_metrics.csv", [{**common, **row} for row in metrics],
                [*common, *QUANTIFICATION_METRIC_FIELDS],
            )),
        }
        if comparison is not None:
            # Missing quantification in a legacy reference yields explicit
            # status rows with NaN estimates, not a fabricated paired result.
            written["quantification_comparison"] = str(write_csv(
                out / "quantification_comparison.csv",
                [{**common, "reference_csv": ref_path, **row} for row in comparison],
                [*common, "reference_csv", *QUANTIFICATION_COMPARISON_FIELDS],
            ))
        return written

    # ── Optional multi-task writers (Phase 9b) ────────────────────────────────
    def _write_clinical(self, records: List[dict], out):
        fg = self._foreground_specs
        fields = ["run_name", "split", "case_id", "image_path", "mask_path", "unit"]
        fields += [f"{m}_{s['class_name']}" for s in fg for m in self.clinical_metrics]
        rows = []
        for r in records:
            clinical = r["clinical"]
            row = {
                "run_name": self.run_name, "split": self.split,
                "case_id": r.get("case_id", ""), "image_path": r.get("image_path", ""),
                "mask_path": r.get("mask_path", ""), "unit": clinical.get("unit", "pixel"),
            }
            for s in fg:
                vals = clinical["per_class"][s["class_id"]]
                for m in self.clinical_metrics:
                    row[f"{m}_{s['class_name']}"] = round(float(vals[m]), 6)
            rows.append(row)
        return write_csv(out / "clinical_metrics.csv", rows, fields)

    def _write_boundary(self, records: List[dict], out):
        fg = self._foreground_specs
        fields = ["run_name", "split", "case_id", "image_path", "mask_path", "boundary_width_px"]
        fields += [f"boundary_{m}_{s['class_name']}" for s in fg for m in ("dice", "iou")]
        fields += ["boundary_dice_mean", "boundary_iou_mean"]
        rows = []
        for r in records:
            boundary = r["boundary"]
            row = {
                "run_name": self.run_name, "split": self.split,
                "case_id": r.get("case_id", ""), "image_path": r.get("image_path", ""),
                "mask_path": r.get("mask_path", ""),
                "boundary_width_px": boundary.get("width_px", ""),
            }
            dices, ious = [], []
            for s in fg:
                vals = boundary["per_class"][s["class_id"]]
                row[f"boundary_dice_{s['class_name']}"] = round(float(vals["boundary_dice"]), 6)
                row[f"boundary_iou_{s['class_name']}"] = round(float(vals["boundary_iou"]), 6)
                dices.append(vals["boundary_dice"])
                ious.append(vals["boundary_iou"])
            row["boundary_dice_mean"] = round(_mean(dices), 6)
            row["boundary_iou_mean"] = round(_mean(ious), 6)
            rows.append(row)
        return write_csv(out / "boundary_metrics.csv", rows, fields)
