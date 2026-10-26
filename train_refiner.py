"""Stage B: train only the detached profile head.

Segmentation metrics are inherited from SegModule.
The old quantification training loss is never called.
"""
from __future__ import annotations

import torch
from torchmetrics import Metric
from omegaconf import OmegaConf

from src.datasets.batch import unpack_batch
from src.models.seg_module import SegModule
from src.losses.polar_quantification import (
    build_polar_quantification,
    _valid_mean,
)
from mengxian.medseg_research.src.models.research.ordered_radial_logits import (
    DetachedProfileRefiner,
)


KEYS = (
    "profile_mae",
    "head_pmw_mae_px",
    "head_dv_mae_px",
    "head_pvdr_mae",
    "paired_base_pmw_mae_px",
    "paired_head_pmw_mae_px",
    "paired_base_dv_mae_px",
    "paired_head_dv_mae_px",
    "paired_base_pvdr_mae",
    "paired_head_pvdr_mae",
    "reference_valid_fraction",
    "base_invalid_fraction",
    "head_invalid_fraction",
    "paired_valid_fraction",
)


class _MeasurementStats(Metric):
    """Explicit sums/counts: zero valid cases are NA, not zero MAE."""

    def __init__(self):
        super().__init__()

        self.add_state(
            "sums",
            torch.zeros(len(KEYS), dtype=torch.float64),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "counts",
            torch.zeros(len(KEYS), dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    def update(self, values):
        for i, key in enumerate(KEYS):
            value, valid = values[key]

            self.sums[i] += torch.where(
                valid,
                value.detach().double(),
                0,
            ).sum()

            self.counts[i] += valid.sum()

    def compute(self):
        return self.sums.clone(), self.counts.clone()


class DetachedQuantModule(SegModule):
    def __init__(self, cfg):
        cfg = OmegaConf.create(
            OmegaConf.to_container(
                OmegaConf.create(cfg),
                resolve=True,
            )
        )

        if (cfg.get("quantification") or {}).get(
            "enabled", False
        ):
            raise ValueError(
                "Stage B requires "
                "quantification.enabled=false."
            )

        if cfg.model.name != "modular_unet":
            raise ValueError(
                "This integration targets ModularUNet."
            )

        # 后面会完整加载检查点，不需要重新下载 ImageNet 权重。
        build_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True)
        )
        build_cfg.model.encoder_weights = None

        super().__init__(build_cfg)

        self.cfg = cfg
        self.save_hyperparameters({
            "cfg": OmegaConf.to_container(
                cfg,
                resolve=True,
            )
        })

        # 仅复用旧模块的几何配置和测量函数。
        # 不调用它的 forward() 或旧 quantification loss。
        geometry_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True)
        )

        geometry_cfg.quantification = OmegaConf.merge(
            geometry_cfg.get("quantification") or {},
            {"enabled": True},
        )

        self.readout = build_polar_quantification(
            geometry_cfg
        )

        q = cfg.profile_refiner

        self.quant_head = DetachedProfileRefiner(
            feature_channels=int(
                cfg.model.decoder_channels[-1]
            ),
            radial_bins=int(q.radial_bins),
            max_correction_fraction=float(
                q.max_correction_fraction
            ),
        )

        self.model.requires_grad_(False)
        self.model.eval()

        # 随新检查点一起保存，避免误用随机初始化的分割模型。
        self.register_buffer(
            "backbone_loaded",
            torch.tensor(False),
        )

        self.val_refiner_stats = _MeasurementStats()
        self.test_refiner_stats = _MeasurementStats()

    def load_segmentation_state(self, state_dict):
        # 只读取原分割模型，但严格匹配其所有参数和 BN buffers。
        state = {
            k[len("model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

        if not state:
            raise ValueError(
                "Expected a SegModule checkpoint "
                "with model.* keys."
            )

        self.model.load_state_dict(
            state,
            strict=True,
        )

        self.backbone_loaded.fill_(True)
        self.model.eval()

    def on_fit_start(self):
        if any(
            p.requires_grad
            for p in self.model.parameters()
        ):
            raise RuntimeError(
                "Segmentation parameters must stay frozen."
            )

        self.print(
            "[Stage B] segmentation frozen/eval; "
            "optimizer trains quant_head only."
        )

    def train(self, mode=True):
        super().train(mode)

        # Lightning 会调用父模块的 train()。
        # 原分割模型必须重新保持 eval()。
        if hasattr(self, "model"):
            self.model.eval()

        return self

    def forward(self, images):
        if not bool(self.backbone_loaded):
            raise RuntimeError(
                "Load the trained segmentation checkpoint first."
            )

        self.model.eval()

        with torch.no_grad():
            out = self.model(
                images,
                return_features=True,
            )

            logits = out["seg_logits"]
            probabilities = logits.float().softmax(dim=1)

            base = self.readout.measure_mask(
                logits.argmax(dim=1)
            )

        limit = self.readout.radii[
            self.readout.last_valid
        ]

        # 新头必须位于 no_grad() 外部。
        with torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            profiles = self.quant_head(
                out["features"]["decoder_final"],
                probabilities,
                base["od_profile"],
                base["width_profile"],
                limit,
                self.readout.max_radius,
            )

            # 同一个测量函数：PMW、DV、PVDR 计算不变。
            refined = self.readout._summarize(
                profiles["od_profile"],
                profiles["width_profile"],
            )

        return {
            "seg_logits": logits,
            "base_quant": base,
            "refined_quant": refined,
        }

    def _profile_error(self, prediction, target):
        od_error = (
            prediction["od_profile"]
            - target["od_profile"]
        ).abs().mean(dim=-1)

        width_error = (
            prediction["width_profile"]
            - target["width_profile"]
        ).abs().mean(dim=-1)

        return (
            od_error + width_error
        ) / self.readout.max_radius

    def training_step(self, batch, batch_idx):
        images, masks, _ = unpack_batch(batch)

        out = self(images)
        target = self.readout.measure_mask(masks)

        error = self._profile_error(
            out["refined_quant"],
            target,
        )

        loss = _valid_mean(
            error,
            target["reference_valid"],
        )

        self.log(
            "train/refiner_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        return loss

    @torch.no_grad()
    def _update_refiner_stats(self, stage, out, target):
        base = out["base_quant"]
        head = out["refined_quant"]
        ref = target["reference_valid"]

        limit = self.readout.radii[
            self.readout.last_valid
        ]

        head_ok = (
            torch.isfinite(
                head["od_profile"]
            ).all(dim=-1)
            & torch.isfinite(
                head["width_profile"]
            ).all(dim=-1)
            & (
                head["dv"]
                >= self.readout.denominator_floor_px
            )
            & (
                (
                    head["od_profile"]
                    + head["width_profile"]
                ) < limit
            ).all(dim=-1)
        )

        # 公平的配对比较：两种测量都有效的同一批病例。
        common = (
            ref
            & base["reference_valid"]
            & head_ok
        )

        values = {
            "profile_mae": (
                self._profile_error(head, target),
                ref,
            )
        }

        for key, suffix in (
            ("pmw", "_px"),
            ("dv", "_px"),
            ("pvdr", ""),
        ):
            head_error = (
                head[key] - target[key]
            ).abs()

            base_error = (
                base[key] - target[key]
            ).abs()

            values[f"head_{key}_mae{suffix}"] = (
                head_error,
                ref & head_ok,
            )

            values[f"paired_base_{key}_mae{suffix}"] = (
                base_error,
                common,
            )

            values[f"paired_head_{key}_mae{suffix}"] = (
                head_error,
                common,
            )

        values.update({
            "reference_valid_fraction": (
                ref.float(),
                torch.ones_like(ref),
            ),
            "base_invalid_fraction": (
                (~base["reference_valid"]).float(),
                ref,
            ),
            "head_invalid_fraction": (
                (~head_ok).float(),
                ref,
            ),
            "paired_valid_fraction": (
                common.float(),
                ref,
            ),
        })

        getattr(
            self,
            f"{stage}_refiner_stats",
        ).update(values)

    def validation_step(self, batch, batch_idx):
        images, masks, _ = unpack_batch(batch)

        out = self(images)
        preds = out["seg_logits"].argmax(dim=1)

        # 保留原分割评估流程。
        self.val_dice.update(preds, masks)
        self.val_iou.update(preds, masks)

        self._update_refiner_stats(
            "val",
            out,
            self.readout.measure_mask(masks),
        )

    def test_step(self, batch, batch_idx):
        images, masks, meta = unpack_batch(batch)

        out = self(images)
        logits = out["seg_logits"]
        preds = logits.argmax(dim=1)

        self.test_dice.update(preds, masks)
        self.test_iou.update(preds, masks)

        self._collect_records(
            logits,
            preds,
            masks,
            meta,
            batch_idx,
        )

        self._update_refiner_stats(
            "test",
            out,
            self.readout.measure_mask(masks),
        )

    def on_validation_epoch_start(self):
        self.val_refiner_stats.reset()

    def on_test_epoch_start(self):
        self.test_records = []
        self.test_refiner_stats.reset()

    def _finish_refiner_stats(self, stage, prefix):
        meter = getattr(
            self,
            f"{stage}_refiner_stats",
        )

        # 所有进程参与统计同步，不只在主进程调用。
        sums, counts = meter.compute()
        meter.reset()

        if stage == "val" and counts[0] == 0:
            raise RuntimeError(
                "No valid GT measurements in validation. "
                "Check centre, labels and readout validity rules."
            )

        printed = []

        for i, key in enumerate(KEYS):
            name = f"{prefix}/refiner_{key}"

            self.log(
                name + "_n",
                counts[i],
                sync_dist=False,
            )

            if counts[i] > 0:
                value = sums[i] / counts[i]

                self.log(
                    name,
                    value,
                    sync_dist=False,
                    prog_bar=(key == "profile_mae"),
                )

                printed.append(
                    f"{key}={value.item():.5f} "
                    f"(n={int(counts[i])})"
                )
            else:
                printed.append(
                    f"{key}=NA (n=0)"
                )

        if self.trainer.is_global_zero:
            self.print(
                f"\n[{prefix} refiner]\n  "
                + "\n  ".join(printed)
            )

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_dice.reset()
            self.val_iou.reset()
            self.val_refiner_stats.reset()
            return

        self._log_metrics(
            "val",
            self.val_dice,
            self.val_iou,
        )

        self._finish_refiner_stats(
            "val",
            "val",
        )

    def on_test_epoch_end(self):
        self._log_metrics(
            self.eval_split,
            self.test_dice,
            self.test_iou,
        )

        self._finish_refiner_stats(
            "test",
            self.eval_split,
        )

    def configure_optimizers(self):
        q = self.cfg.profile_refiner

        # 原分割参数不进入 optimizer，包括 weight decay。
        opt = torch.optim.AdamW(
            self.quant_head.parameters(),
            lr=float(q.lr),
            weight_decay=float(q.weight_decay),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(self.cfg.train.epochs),
            eta_min=1e-6,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }