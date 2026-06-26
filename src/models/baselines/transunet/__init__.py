"""TransUNet: hybrid CNN + ViT segmentation baseline.

Lives in the baseline layer (``provider=baseline``). Built via
:func:`src.models.baselines.registry.build_baseline_model`.
"""

from src.models.baselines.transunet.model import build_transunet

__all__ = ["TransUNet", "build_transunet"]
