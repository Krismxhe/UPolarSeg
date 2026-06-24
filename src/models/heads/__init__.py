"""Optional task heads (Phase 9b skeletons).

These are building blocks for future multi-task models. They are NOT attached to
the SMP baselines (which return a bare logits tensor), so importing them has no
effect on the segmentation baseline.
"""

from src.models.heads.boundary_head import BoundaryHead
from src.models.heads.clinical_head import ClinicalHead

__all__ = ["BoundaryHead", "ClinicalHead"]
