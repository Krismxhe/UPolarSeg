"""Identity-initialized logit correction for polar quantification.

Inputs:
    features:   [B, C, angle, radius]
    seg_logits: [B, 3, angle, radius]

Output:
    quant_logits: [B, 3, angle, radius]

The original quantification loss is unchanged.
Both segmentation inputs are detached.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ResidualQuantLogitsAdapter(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        valid_grid: Tensor,
        hidden_channels: int = 8,
        max_logit_shift: float = 1.0,
        od_class: int = 1,
        ppa_class: int = 2,
    ) -> None:
        super().__init__()

        if feature_channels < 1 or hidden_channels < 1:
            raise ValueError(
                "Channel counts must be positive."
            )

        if (
            valid_grid.ndim != 2
            or min(valid_grid.shape) < 2
        ):
            raise ValueError(
                "valid_grid must be [angle, radius], both >= 2."
            )

        if {od_class, ppa_class} != {1, 2}:
            raise ValueError(
                "Requires background=0 and OD/PPA=1/2."
            )

        if (
            not math.isfinite(max_logit_shift)
            or max_logit_shift <= 0
        ):
            raise ValueError(
                "max_logit_shift must be finite and positive."
            )

        self.feature_channels = int(feature_channels)
        self.od_class = int(od_class)
        self.ppa_class = int(ppa_class)
        self.max_logit_shift = float(max_logit_shift)

        self.register_buffer(
            "valid_grid",
            valid_grid.detach().bool().clone(),
            persistent=False,
        )

        self.project = nn.Conv2d(
            feature_channels,
            hidden_channels,
            kernel_size=1,
        )

        # 角度维手动循环填充。
        # 半径维有物理端点，使用零填充。
        self.local = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=(0, 1),
            groups=hidden_channels,
        )

        self.out = nn.Conv2d(
            hidden_channels,
            2,
            kernel_size=1,
        )

        self.act = nn.ReLU()

        # 只将最后一层置零，不要把整个网络全部置零。
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        features: Tensor,
        seg_logits: Tensor,
    ) -> Tensor:
        a, r = self.valid_grid.shape

        if (
            features.ndim != 4
            or tuple(features.shape[1:]) != (
                self.feature_channels, a, r
            )
        ):
            raise ValueError(
                "Expected features "
                "[B, feature_channels, angle, radius]."
            )

        if tuple(seg_logits.shape) != (
            features.shape[0], 3, a, r
        ):
            raise ValueError(
                "seg_logits must align with features: "
                "[B, 3, angle, radius]."
            )

        if (
            features.device != seg_logits.device
            or features.device != self.valid_grid.device
        ):
            raise ValueError(
                "Features, logits and adapter "
                "must be on the same device."
            )

        if (
            not features.is_floating_point()
            or not seg_logits.is_floating_point()
        ):
            raise TypeError(
                "Features and logits must be floating-point tensors."
            )

        with torch.autocast(
            device_type=features.device.type,
            enabled=False,
        ):
            # 两条上游连接都要断开。
            z = seg_logits.detach().float()
            f = features.detach().float()

            h = self.act(self.project(f))

            # 相邻角度的局部信息；不压缩分辨率。
            h = torch.cat(
                (
                    h[:, :, -1:],
                    h,
                    h[:, :, :1],
                ),
                dim=2,
            )

            h = self.act(self.local(h))

            d = (
                self.max_logit_shift
                * self.out(h).tanh()
            )

            # 只在有效区域学习修正。
            # 不顺便清理原始 logits 的图像外预测。
            d = d * self.valid_grid[None, None]

            channels = [
                torch.zeros_like(d[:, 0]),
                None,
                None,
            ]
            channels[self.od_class] = d[:, 0]
            channels[self.ppa_class] = d[:, 1]

            delta = torch.stack(
                channels,
                dim=1,
            )

            # 新建输出，不原地修改原分割 logits。
            return z + delta