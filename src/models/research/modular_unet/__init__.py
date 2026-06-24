"""ModularUNet: a controllable UNet for skip-connection module research.

The first research model (``provider=research``). Built via
:func:`src.models.research.registry.build_research_model`.
"""

from src.models.research.modular_unet.model import ModularUNet, build_modular_unet
from src.models.research.modular_unet.skip_modules import SKIP_MODULES, IdentitySkip, build_skip_module

__all__ = [
    "ModularUNet",
    "build_modular_unet",
    "SKIP_MODULES",
    "IdentitySkip",
    "build_skip_module",
]
