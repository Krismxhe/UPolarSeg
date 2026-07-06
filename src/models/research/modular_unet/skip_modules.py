"""
Skip-connection modules + registry (Phase 6).

A SkipModule transforms a skip feature before it is fused into the decoder. It
may use the decoder feature and the decoder level index. Contract:

    forward(skip, decoder_feature=None, level: int = 0) -> skip_out

v1 constraints (kept simple for clean ablation):
  - spatial size is preserved;
  - channel count is preserved (C_out == C_skip), so downstream conv channel
    bookkeeping in the decoder stays valid.

Add a new module by subclassing SkipModule and registering it in SKIP_MODULES;
then select it via ``model.skip.name=<name>``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipModule(nn.Module):
    """Base skip module (no-op). Subclasses override forward."""
    def out_channels_by_level(self, input_channels_by_level):
        """Return skip output channels for each skip level.

        Channel-preserving modules return the input list. Channel-expanding
        modules should override this method.
        """
        return list(input_channels_by_level or [])
        
    def forward(self, skip, decoder_feature=None, level: int = 0):
        return skip


class IdentitySkip(SkipModule):
    """Pass-through skip — the baseline / ablation control."""

    def __init__(self, channels_by_level=None, **params):
        super().__init__()

    def forward(self, skip, decoder_feature=None, level: int = 0):
        return skip

class _PMSRExpandBlock(nn.Module):
    """
    Per-level PMSR expansion block.
    
    Input:
        S_l: B x C_l x H_theta x W_radius
    
    Output:
        concat(S_l, radial_extra, angular_extra)
    
    The original skip channels are preserved and radial/angular polar evidence is appended
    as additional channels. This return PMSR into a feature augmentation module rather than
    a channel-preserving residual correction.
    """
    def __init__(
        self,
        channels: int,
        k_radial: int = 7,
        k_angular: int = 7,
        reduction: int = 4,
        min_hidden_channels: int = 32,
        extra_channels: Optional[int] = None,
        extra_ratio: float = 0.25,
        min_extra_channels: int = 16,
        max_extra_channels: Optional[int] = 128,
        axis_conv_type: str = "depthwise",
        axis_groups: int = 8,
        use_radius_coord: bool = True,
        use_radius_gate: bool = False,
        radial_padding: str = "replicate",
        angular_padding: str = "circular",
        norm: str = "bn",
        activation: str = "relu",
        init_gamma: float = 1e-3,
        axis_mode: str = "both",
    ):

        super().__init__()

        # initialize parameters
        self.channels = int(channels)
        self.k_radial = _positive_odd(k_radial, "k_radial")
        self.k_angular = _positive_odd(k_angular, "k_angular")
        self.use_radius_coord = bool(use_radius_coord)
        self.use_radius_gate = bool(use_radius_gate)
        self.radial_padding = str(radial_padding).lower()
        self.angular_padding = str(angular_padding).lower()

        # branch_mode -> axis_mode
        axis_mode = str(axis_mode).lower()
        if axis_mode not in {"both", "radial", "angular"}:
            raise ValueError(
                f"axis_mode must be 'both', 'radial', or 'angular', got {axis_mode!r}"
            )
        self.axis_mode = axis_mode

        # ensure the input parameters are legal
        reduction = max(int(reduction), 1)
        hidden = max(self.channels // reduction, int(min_hidden_channels))
        hidden = min(hidden, self.channels)
        
        self.extra_channels = _compute_extra_channels(
            self.channels,
            extra_channels = extra_channels,
            extra_ratio = extra_ratio,
            min_extra_channels = min_extra_channels,
            max_extra_channels = max_extra_channels,
        )
        
        pre_in_channels = self.channels + (1 if self.use_radius_coord else 0)
        self.pre = nn.Sequential(
            nn.Conv2d(pre_in_channels, hidden, kernel_size=1, bias=False),
            _make_norm(norm, hidden),
            _make_activation(activation),
        )
        
        groups = _axis_groups(hidden, axis_conv_type, axis_groups)
        
        self.radial_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, self.k_radial),
            padding=0,
            groups=groups,
            bias=False,
        )
        self.radial_post = nn.Sequential(
            _make_norm(norm, hidden),
            _make_activation(activation),
        )
        self.radial_out = nn.Conv2d(hidden, self.extra_channels, kernel_size=1, bias=False)
        
        self.angular_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(self.k_angular, 1),
            padding=0,
            groups=groups,
            bias=False
        )
        self.angular_post = nn.Sequential(
            _make_norm(norm, hidden),
            _make_activation(activation),
        )
        self.angular_out = nn.Conv2d(hidden, self.extra_channels, kernel_size=1, bias=False)

        if self.use_radius_gate:
            self.radius_gate = nn.Sequential(
                nn.Conv2d(1, hidden, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.radius_gate = None
            
        # Extra channels start near zero, so the new pathway is initially a small
        # supplement. Note: decoder conv fan-in still changes because channels are
        # expanded; use channel-matched controls for rigorous ablation.
        self.gamma_radial = nn.Parameter(torch.tensor(float(init_gamma)))
        self.gamma_angular = nn.Parameter(torch.tensor(float(init_gamma)))
    
    @property
    def out_channels(self) -> int:
        n_extra = 0
        if self.axis_mode in {"both", "radial"}:
            n_extra += self.extra_channels
        if self.axis_mode in {"both", "angular"}:
            n_extra += self.extra_channels
        return self.channels + n_extra
    
    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        r_coord = None
        if self.use_radius_coord or self.radius_gate is not None:
            r_coord = _radius_coord_like(skip)
        
        if self.use_radius_coord:
            z = torch.cat([skip, r_coord], dim=1)
        else:
            z = skip
        z = self.pre(z)
        
        parts = [skip]
        
        if self.axis_mode in {"both", "radial"}:
            pr = self.k_radial // 2
            radial = self.radial_conv(_pad_radius(z, pr, mode = self.radial_padding))
            radial = self.radial_post(radial)
            # why add the gate control
            radial_extra = self.gamma_radial * self.radial_out(radial)
            parts.append(radial_extra)
        
        if self.axis_mode in {"both", "angular"}:
            pt = self.k_angular // 2
            angular = self.angular_conv(_pad_theta(z, pt, mode=self.angular_padding))
            if self.radius_gate is not None:
                angular = angular * self.radius_gate(r_coord)
            angular = self.angular_post(angular)
            angular_extra = self.gamma_angular * self.angular_out(angular)
            parts.append(angular_extra)
            
        return torch.cat(parts, dim=1)


class PolarMetricSkipExpansion(SkipModule):
    """
    PMSR-Expand: channel-expanding pre-concat skip refinement.

    Instead of forcing radial/angular evidence back into the original skip
    channel basis, this module appends them as additional channels:

        S_l' = concat(S_l, Delta_r, Delta_theta)
        X_l  = concat(D_l, S_l')

    Spatial size is preserved, but channel count increases. The decoder must
    therefore use ``out_channels_by_level`` when constructing DecoderBlocks.
    """
    
    def __init__(
        self,
        channels_by_level=None,
        levels="all",
        k_radial: int = 7,
        k_angular: int = 7,
        reduction: int = 4,
        min_hidden_channels: int = 32,
        extra_channels: Optional[int] = None,
        extra_ratio: float = 0.25,
        min_extra_channels: int =16,
        max_extra_channels: Optional[int] = 128,
        axis_conv_type: str = "depthwise",
        axis_groups: int = 8,
        use_radius_coord: bool = True,
        use_radius_gate: bool = False,
        radial_padding: str = "replicate",
        angular_padding: str = "circular",
        norm: str = "bn",
        activation: str = "relu",
        init_gamma: float = 1e-3,
        axis_mode: str = "both",
        **unsed,
    ):
        super().__init__()
        self.active_levels = _parse_levels(levels)
        channels_by_level = list(channels_by_level or [])
        
        self.blocks = nn.ModuleList(
            [
                _PMSRExpandBlock(
                    channels=int(c),
                    k_radial=k_radial,
                    k_angular=k_angular,
                    reduction=reduction,
                    min_hidden_channels=min_hidden_channels,
                    extra_channels=extra_channels,
                    extra_ratio=extra_ratio,
                    min_extra_channels=min_extra_channels,
                    max_extra_channels=max_extra_channels,
                    axis_conv_type=axis_conv_type,
                    axis_groups=axis_groups,
                    use_radius_coord=use_radius_coord,
                    use_radius_gate=use_radius_gate,
                    radial_padding=radial_padding,
                    angular_padding=angular_padding,
                    norm=norm,
                    activation=activation,
                    init_gamma=init_gamma,
                    axis_mode=axis_mode,
                )
                for c in channels_by_level
            ]
        )
        
    def out_channels_by_level(self, input_channels_by_level):
        out = []
        for level, c in enumerate(list(input_channels_by_level or [])):
            if level >= len(self.blocks):
                out.append(int(c))
            elif self.active_levels is not None and level not in self.active_levels:
                out.append(int(c))
            else:
                out.append(int(self.blocks[level].out_channels))
        return out
    
    def forward(self, skip, decoder_feature=None, level: int = 0):
        if skip is None:
            return skip
        level = int(level)
        if level >= len(self.blocks):
            return skip
        if self.active_levels is not None and level not in self.active_levels:
            return skip
        return self.blocks[level](skip)
    
class ZeroSkipExpansion(SkipModule):
    """
    Channel-expanded identity control.

    It appends zero-valued extra channels with the same width rule as
    PMSR-Expand. This controls for the widened decoder input channel count, but
    does not add polar-axis evidence. Use it as a strict channel-count control.
    """

    def __init__(
        self,
        channels_by_level=None,
        levels="all",
        extra_channels: Optional[int] = None,
        extra_ratio: float = 0.25,
        min_extra_channels: int = 16,
        max_extra_channels: Optional[int] = 128,
        axis_mode: str = "both",
        **unused,
    ):
        super().__init__()
        self.active_levels = _parse_levels(levels)
        self.channels_by_level = [int(c) for c in list(channels_by_level or [])]
        axis_mode = str(axis_mode).lower()
        if axis_mode not in {"both", "radial", "angular"}:
            raise ValueError(
                f"axis_mode must be 'both', 'radial', or 'angular', got {axis_mode!r}"
            )
        self.axis_mode = axis_mode
        self.extra_channels_by_level = [
            _compute_extra_channels(
                c,
                extra_channels=extra_channels,
                extra_ratio=extra_ratio,
                min_extra_channels=min_extra_channels,
                max_extra_channels=max_extra_channels,
            )
            for c in self.channels_by_level
        ]

    def _extra_count(self, level: int) -> int:
        e = self.extra_channels_by_level[level]
        n = 0
        if self.axis_mode in {"both", "radial"}:
            n += e
        if self.axis_mode in {"both", "angular"}:
            n += e
        return n

    def out_channels_by_level(self, input_channels_by_level):
        out = []
        for level, c in enumerate(list(input_channels_by_level or [])):
            if level >= len(self.extra_channels_by_level):
                out.append(int(c))
            elif self.active_levels is not None and level not in self.active_levels:
                out.append(int(c))
            else:
                out.append(int(c) + self._extra_count(level))
        return out

    def forward(self, skip, decoder_feature=None, level: int = 0):
        if skip is None:
            return skip
        level = int(level)
        if level >= len(self.extra_channels_by_level):
            return skip
        if self.active_levels is not None and level not in self.active_levels:
            return skip
        n_extra = self._extra_count(level)
        zeros = skip.new_zeros(skip.shape[0], n_extra, skip.shape[-2], skip.shape[-1])
        return torch.cat([skip, zeros], dim=1)

def _positive_odd(k:int, name: str) -> int:
    k = int(k)
    if k<1 or k%2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got{k}")
    
    return k

def _make_norm(norm: str, channels: int) -> nn.Module:
    
    if norm in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(channels)
    if norm in {"gn", "groupnorm", "group_norm"}:
        groups = math.gcd(int(channels), 32)
        return nn.GroupNorm(num_groups=max(groups, 1), num_channels=channels)
    if norm in {"none", "identity", "id"}:
        return nn.Identity()
    
    raise ValueError(f"Unknown norm={norm!r}; use 'bn', 'gn', or 'none'.")

def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()

    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"none", "identity", "id"}:
        return nn.Identity()

    raise ValueError(f"Unknown activation={name!r}; use 'relu', 'silu', 'gelu', or 'none'.")

def _parse_levels(levels):
    """Return None for all levels, otherwise a set of active level indices"""
    if levels is None:
        return None
    if isinstance(levels, str):
        value = levels.strip().lower()
        if value == "all":
            return None
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
    if isinstance(levels, int):
        return {int(levels)}
    if isinstance(levels, Iterable):
        return {int(v) for v in levels}
    raise ValueError(f"Unsupported levels={levels!r}")

def _radius_coord_like(x: torch.Tensor) -> torch.Tensor:
    """Return normalized radius coordinate with shape B x 1 x H_theta x W_radius."""
    b, _, h, w = x.shape
    if w <= 1:
        r = torch.zeros(w, device=x.device, dtype=x.dtype)
    else:
        r = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype)
    return r.view(1, 1, 1, w).expand(b, 1, h, w)

def _circular_pad_theta(x: torch.Tensor, pad:int) -> torch.Tensor:
    """Circular padding on H/theta axis. Robust even when pad >=H."""
    if pad==0:
        return x
    h = x.shape[-2]
    idx = torch.torch.arange(-pad, h+pad, device=x.device).remainder(h)
    return x.index_select(dim=-2, index=idx)

def _pad_theta(x: torch.Tensor, pad: int, mode: str = "circular") -> torch.Tensor:
    """Pad H/theta axis. Circular is the geometry-aware default"""
    if pad == 0:
        return x
    mode = str(mode).lower()
    if mode == "circular":
        return _circular_pad_theta(x, pad)
    if mode in {"zero", "zeros", "constant"}:
        return F.pad(x, (0, 0, pad, pad), mode="constant", value=0.0)
    if mode in {"replicate", reflect}:
        if mode == "reflect" and pad>=x.shape[-2]:
            mode = "replicate"
        return F.pad(x, (0, 0, pad, pad), mode=mode)
    raise ValueError(f"Unknown angular padding mode={mode!r}")

def _pad_radius(x: torch.Tensor, pad: int, mode: str = "replicate") -> torch.Tensor:
    """Pad W/radius axis. Radius is not periodic, so circular padding is invalid."""
    if pad == 0:
        return x
    mode = str(mode).lower()
    if mode in {"zero", "zeros", "constant"}:
        return F.pad(x, (pad, pad, 0, 0), mode="constant", value=0.0)
    if mode in {"replicate", "reflect"}:
        if mode == "reflect" and pad >= x.shape[-1]:
            mode = "replicate"
        return F.pad(x, (pad, pad, 0, 0), mode=mode)
    if mode == "circular":
        raise ValueError("radial_padding='circular' is invalid: radius is not periodic.")
    raise ValueError(f"Unknown radial padding mode={mode!r}")

def _axis_groups(hidden_channels: int, axis_conv_type: str, axis_groups: int) -> int:
    """Resolve group count for radial/angular axis convolutions"""
    axis_conv_type = str(axis_conv_type).lower()
    hidden_channels = int(hidden_channels)
    if axis_conv_type in {"depthwise", "dw"}:
        return hidden_channels
    if axis_conv_type in {"full", "dense", "standard"}:
        axis_groups = max(int(axis_groups), 1)
        return max(math.gcd(hidden_channels, axis_groups), 1)
    raise ValueError(
        f"Unknown axis_conv_type={axis_conv_type!r}; use 'depthwise', 'group', or 'full'."
    )
    
def _compute_extra_channels(
    in_channels: int,
    extra_channels: Optional[int],
    extra_ratio: float,
    min_extra_channels: int,
    max_extra_channels: Optionalp[int],
) -> int:
    """Compute per-axis extra channel width"""
    if extra_channels is not None and int(extra_channels) > 0:
        out = int(extra_channels)
    else:
        out = int(round(float(in_channels) * float(extra_ratio)))
    
    out = max(out, int(min_extra_channels), 1)
    if max_extra_channels is not None and int(max_extra_channels) > 0:
        out = min(out, int(max_extra_channels))
    return out

# name -> class. New skip modules register here.
SKIP_MODULES = {
    "identity": IdentitySkip,
    "zero_expand": ZeroSkipExpansion,
    "identity_expand": ZeroSkipExpansion,
    "pmsr_expand": PolarMetricSkipExpansion,
    "pmsr_expansion": PolarMetricSkipExpansion,
    "polar_metric_expand": PolarMetricSkipExpansion,
    "polar_axis_expand": PolarMetricSkipExpansion,
}


def build_skip_module(skip_cfg, channels_by_level):
    """Instantiate the skip module selected by ``skip_cfg.name``."""
    cfg = skip_cfg or {}
    name = str(cfg.get("name", "identity")).lower()
    if name not in SKIP_MODULES:
        raise ValueError(f"Unknown skip module '{name}'. Available: {sorted(SKIP_MODULES)}")

    params = dict(cfg.get("params", {}) or {})
    # Make top-level skip.levels effective.
    params.setdefault("levels", cfg.get("levels", "all"))

    return SKIP_MODULES[name](channels_by_level=channels_by_level, **params)