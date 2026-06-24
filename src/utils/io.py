"""Small filesystem helpers for writing evaluation artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from omegaconf import OmegaConf


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path, rows: List[Dict], fieldnames: Optional[Sequence[str]] = None) -> Path:
    """Write a list of dict rows to a CSV. Header is always written.

    If fieldnames is None it is inferred from the first row (empty file with no
    header when there are no rows).
    """
    path = Path(path)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_resolved_config(path, cfg) -> Path:
    """Dump a fully-resolved (interpolations expanded) config to YAML."""
    path = Path(path)
    OmegaConf.save(config=cfg, f=str(path), resolve=True)
    return path
