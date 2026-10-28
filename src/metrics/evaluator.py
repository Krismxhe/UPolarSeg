"""Compatibility exports for the evaluation reporting API."""

from src.evaluation.evaluator import (
    PER_CASE_BASE_FIELDS,
    PER_CLASS_FIELDS,
    SUMMARY_FIELDS,
    SegEvaluator,
)

__all__ = [
    "SegEvaluator",
    "SUMMARY_FIELDS",
    "PER_CLASS_FIELDS",
    "PER_CASE_BASE_FIELDS",
]