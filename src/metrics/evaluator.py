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
    def _per_case_row(self, rec: dict) -> dict:
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
        return row

    def _per_case_fields(self) -> List[str]:
        extra = [f"{m}_{spec['class_name']}" for spec in self.class_specs for m in _METRICS]
        return PER_CASE_BASE_FIELDS + extra

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
        per_class_rows = self._per_class_rows(records) if records else []
        summary_row = self._summary_row(per_class_rows, len(records))

        written = {}
        written["summary"] = str(write_csv(out / "summary.csv", [summary_row], SUMMARY_FIELDS))
        if self.write_per_class:
            written["per_class"] = str(write_csv(out / "per_class.csv", per_class_rows, PER_CLASS_FIELDS))
        if self.write_per_case:
            per_case_rows = [self._per_case_row(r) for r in records]
            written["per_case"] = str(
                write_csv(out / "per_case.csv", per_case_rows, self._per_case_fields())
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
