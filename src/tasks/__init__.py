"""
Task layer (Phase 9b).

Lightweight helpers describing which optional outputs are active. The task
config lives under ``cfg.task.outputs.<name>`` and every optional task defaults
to ``enabled: false`` so the segmentation baseline is unaffected.
"""

from __future__ import annotations


def get_task_outputs(cfg) -> dict:
    """Return cfg.task.outputs as a plain dict ({} when absent)."""
    task = cfg.get("task", {}) or {}
    return task.get("outputs", {}) or {}


def get_output_cfg(cfg, name: str) -> dict:
    return get_task_outputs(cfg).get(name, {}) or {}


def is_task_enabled(cfg, name: str) -> bool:
    """True if optional task ``name`` is enabled. Segmentation is always on."""
    if name == "segmentation":
        return True
    return bool(get_output_cfg(cfg, name).get("enabled", False))
