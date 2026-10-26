"""
PyTorch Lightning module for 2D semantic segmentation.

Supports:
  - Binary segmentation  (num_classes = 1, mask_mode = "binary")
  - Multi-class segmentation (num_classes > 1, mask_mode = "index")

Loss:
  - Built via src.losses.factory.build_loss(cfg.loss, cfg.dataset).
  - Default: Dice + CrossEntropy (multiclass) / Dice + BCE (binary).
  - SegModule itself has no direct segmentation-models-pytorch dependency.

Metrics (per-class and mean, foreground only):
  - Dice (F1)
  - IoU (Jaccard)
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import (
    BinaryF1Score, MulticlassF1Score,
    BinaryJaccardIndex, MulticlassJaccardIndex,
)
from torchmetrics import MeanMetric
import numpy as np
import pandas as pd

from src.datasets.batch import unpack_batch
from src.losses.factory import build_loss
from src.models.factory import build_model
from src.models.outputs import normalize_model_output

from src.losses.polar_quantification import (
    QUANT_VAL_KEYS,
    build_polar_quantification,
)

from src.models.research.residual_quant_logits import (
    ResidualQuantLogitsAdapter,
)

class SegModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        # Accept OmegaConf DictConfig or plain dict (the latter happens when
        # loading from a checkpoint, since Lightning serialises hparams as dict)
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg

        # Save as plain dict so Lightning can serialise it to the checkpoint
        self.save_hyperparameters({'cfg': OmegaConf.to_container(cfg, resolve=True)})

        self.num_classes      = cfg.dataset.num_classes
        self.foreground_ids   = list(cfg.dataset.foreground_classes)
        self.class_names      = list(cfg.dataset.class_names)
        self._is_binary       = (self.num_classes == 1)
            
        # ── Model ─────────────────────────────────────────────────────────────
        # Built via the factory so SegModule stays agnostic to the provider
        # (SMP today; custom TransUNet / ModularUNet in later phases).
        self.model = build_model(cfg.model, cfg.dataset)

        # ── Loss ──────────────────────────────────────────────────────────────
        # Built via the loss factory; SegModule no longer depends on smp directly.
        # cfg.get('loss', None) → None falls back to the previous default loss,
        # keeping checkpoints saved before the loss config was introduced working.
        self.loss = build_loss(cfg.get('loss', None), cfg.dataset)
        self.quantification = build_polar_quantification(cfg)

        self.quant_cfg = cfg.get('quantification') or {}
        self.quant_weight = float(
            self.quant_cfg.get('weight', 1.0)
        )

        self.quant_val_metrics = torch.nn.ModuleDict()
        self.quant_test_metrics = torch.nn.ModuleDict()

        if self.quantification is not None:

            self.quant_val_metrics = torch.nn.ModuleDict({
                name: MeanMetric()
                for name in QUANT_VAL_KEYS
            })

            self.quant_test_metrics = torch.nn.ModuleDict({
                name: MeanMetric()
                for name in QUANT_VAL_KEYS
            })

            if self.quant_cfg.adapter_enabled:
                self.adapter_cfg = self.quant_cfg.get("adapter") or {}

                self.quant_head = ResidualQuantLogitsAdapter(
                    feature_channels=int(
                        cfg.model.decoder_channels[-1]
                    ),
                    valid_grid=self.quantification.valid_grid,
                    hidden_channels=int(
                        self.adapter_cfg.get("hidden_channels", 8)
                    ),
                    max_logit_shift=float(
                        self.adapter_cfg.get("max_logit_shift", 1.0)
                    ),
                    od_class=self.quantification.od_class,
                    ppa_class=self.quantification.ppa_class,
                )

                weights = (
                    self.quantification.valid_grid.float()
                    * (
                        self.quantification.max_radius
                        / self.quantification.radius_bins
                    )
                )
                weights[:, 0] = 0.0

        # ── Metrics ───────────────────────────────────────────────────────────
        # torchmetrics >= 1.0 API
        if self._is_binary:
            metric_cls_dice = BinaryF1Score
            metric_cls_iou  = BinaryJaccardIndex
            metric_kwargs   = {}
        else:
            metric_cls_dice = MulticlassF1Score
            metric_cls_iou  = MulticlassJaccardIndex
            metric_kwargs   = {'num_classes': self.num_classes, 'average': 'none'}

        self.val_dice  = metric_cls_dice(**metric_kwargs)
        self.val_iou   = metric_cls_iou(**metric_kwargs)
        self.test_dice = metric_cls_dice(**metric_kwargs)
        self.test_iou  = metric_cls_iou(**metric_kwargs)

        # Prefix used when logging test-phase metrics; can be overridden by
        # evaluate.py to reflect the actual split being evaluated.
        self.eval_split = 'test'

        # Per-case records collected during test_step → consumed by evaluate.py
        # to write explicit summary/per_class/per_case CSV (Phase 4).
        self.test_records = []

        # Optional multi-task eval outputs (Phase 9b); default OFF so the
        # segmentation baseline is unaffected.
        from src.tasks import get_output_cfg, is_task_enabled
        self._clinical_enabled = is_task_enabled(cfg, 'clinical')
        self._boundary_enabled = is_task_enabled(cfg, 'boundary')
        self._clinical_metrics = list(get_output_cfg(cfg, 'clinical').get(
            'metrics', ['area', 'perimeter', 'area_ratio']))
        self._boundary_width = int(get_output_cfg(cfg, 'boundary').get('boundary_width_px', 3))
        self._boundary_loss_weight = float(
            get_output_cfg(cfg, "boundary").get("loss_weight", 0.0)
        )
        self._boundary_train_enabled = (
            self._boundary_enabled and self._boundary_loss_weight > 0.0
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        return self.model(x)

    # ── Loss helper ───────────────────────────────────────────────────────────

    def _loss(self, logits, masks):
        """
        logits : B × C × H × W  (C=1 for binary)
        masks  : B × H × W  (long, class indices)

        Returns (total, dice, aux); the binary/multiclass handling lives inside
        the loss module built by build_loss(...).
        """
        return self.loss(logits, masks)

    # ── Prediction helper ─────────────────────────────────────────────────────

    def _predict(self, logits):
        """Returns hard predictions: B × H × W (long)."""
        if self._is_binary:
            return (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        return logits.argmax(dim=1)

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        images, masks, _meta = unpack_batch(batch)
        loss = None

        if self.quantification is not None and not self.quant_cfg.adapter_enabled:
            outputs = normalize_model_output(
                images,
            )

            logits = outputs["seg_logits"]

            loss_seg, loss_dice, loss_aux = self._loss(
                logits,
                masks,
            )

            loss_quant, stats = self.quantification(
                logits,
                masks,
            )

            loss = (
                loss_seg
                + self.quant_weight * loss_quant
            )

            values = {
                'loss_seg': loss_seg,
                'loss_quant': loss_quant,
                'quant_aux_valid_fraction': (
                    stats['aux_valid_fraction']
                ),
            }

            for name, value in values.items():
                self.log(
                    f'train/{name}',
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=images.shape[0],
                )

        elif self.quantification is not None and self.quant_cfg.adapter_enabled:
            outputs = normalize_model_output(
                self.model(
                    images,
                    return_features = True,
                )
            )

            logits = outputs["seg_logits"]

            quant_logits = self.quant_head(
                outputs["features"]["decoder_final"],
                outputs["seg_logits"],
            )

            loss_seg, loss_dice, loss_aux = self._loss(
                logits,
                masks,
            )

            loss_quant, stats = self.quantification(
                quant_logits,
                masks,
            )

            loss = (
                loss_seg
                + self.quant_weight * loss_quant
            )

            values = {
                'loss_seg': loss_seg,
                'loss_quant': loss_quant,
                'quant_aux_valid_fraction': (
                    stats['aux_valid_fraction']
                ),
            }

            for name, value in values.items():
                self.log(
                    f'train/{name}',
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=images.shape[0],
                )

        else:
            outputs = normalize_model_output(
                self(images)
            )
            logits = outputs['seg_logits']

            loss_seg, loss_dice, loss_aux = self._loss(
                logits,
                masks,
            )

            loss = loss_seg

        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        self.log(
            'train/loss_dice',
            loss_dice,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        self.log(
            'train/loss_ce',
            loss_aux,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, _meta = unpack_batch(batch)

        if self.quantification is not None and self.quant_cfg.adapter_enabled:
            outputs = normalize_model_output(
                self.model(
                    images,
                    return_features = True,
                )
            )

            logits = outputs["seg_logits"]

            quant_logits = self.quant_head(
                outputs["features"]["decoder_final"],
                outputs["seg_logits"],
            )

            loss, _, _ = self._loss(logits, masks)
            preds = self._predict(logits)

            self.val_dice.update(preds, masks)
            self.val_iou.update(preds, masks)

            self._update_quant_metrics(
                'val',
                quant_logits,
                masks,
            )            

        else:
            outputs = normalize_model_output(
                self(images)
            )
            logits = outputs['seg_logits']

            loss, _, _ = self._loss(logits, masks)
            preds = self._predict(logits)

            self.val_dice.update(preds, masks)
            self.val_iou.update(preds, masks)

            self._update_quant_metrics(
                'val',
                logits,
                masks,
            )

        # Preserve the original meaning:
        # val/loss remains segmentation loss.
        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

    def on_validation_epoch_end(self):
        self._log_metrics(
            'val',
            self.val_dice,
            self.val_iou,
        )

        self._log_quant_metrics('val', 'val')

    def on_test_epoch_start(self):
        self.test_records = []

        for metric in self.quant_test_metrics.values():
            metric.reset()

    def test_step(self, batch, batch_idx):
        images, masks, meta = unpack_batch(batch)

        outputs = normalize_model_output(
            self(images)
        )
        logits = outputs['seg_logits']
        preds = self._predict(logits)

        self.test_dice.update(preds, masks)
        self.test_iou.update(preds, masks)

        self._collect_records(
            logits,
            preds,
            masks,
            meta,
            batch_idx,
        )

        self._update_quant_metrics(
            'test',
            logits,
            masks,
        )

    def on_test_epoch_end(self):
        # 原有全局指标
        self._log_metrics(
            self.eval_split,
            self.test_dice,
            self.test_iou,
        )

        # ── 1. 把 test_records 展平成 DataFrame ────────────────────────────
        seg_metrics  = ('dice', 'iou', 'precision', 'recall')
        clin_metrics = list(self._clinical_metrics) if self._clinical_enabled else []

        rows = []
        for r in self.test_records:
            for c in range(self.num_classes):
                pc = r['per_class'][c]
                row = {
                    'case_id':  r['case_id'],
                    'class_id': c,
                    'support':  int(pc.get('support', 0)),
                }
                for m in seg_metrics:
                    row[m] = float(pc[m])
                if self._clinical_enabled and 'clinical' in r:
                    clin_c = r['clinical']['per_class'].get(c, {})
                    for cm in clin_metrics:
                        row[f'clin_{cm}'] = float(clin_c.get(cm, float('nan')))
                rows.append(row)

        if not rows:
            self._log_quant_metrics('test', self.eval_split)
            return

        df = pd.DataFrame(rows)
        all_cols = list(seg_metrics) + [f'clin_{m}' for m in clin_metrics]

        # ── 2. all 口径：所有图 ────────────────────────────────────────────
        agg_all = df.groupby('class_id')[all_cols].mean()

        # ── 3. pos 口径：只保留该类 support>0 的图 ─────────────────────────
        agg_pos = (df[df['support'] > 0]
                   .groupby('class_id')[all_cols].mean())

        # ── 4. 逐类日志 ────────────────────────────────────────────────────
        for c in range(self.num_classes):
            name = self.class_names[c]
            for m in seg_metrics:
                self.log(f'{self.eval_split}/{m}_all_{name}',
                         float(agg_all.loc[c, m]), sync_dist=True)
                self.log(f'{self.eval_split}/{m}_pos_{name}',
                         float(agg_pos.loc[c, m]) if c in agg_pos.index else 0.0,
                         sync_dist=True)
            for cm in clin_metrics:
                key = f'clin_{cm}'
                self.log(f'{self.eval_split}/{key}_all_{name}',
                         float(agg_all.loc[c, key]), sync_dist=True)
                self.log(f'{self.eval_split}/{key}_pos_{name}',
                         float(agg_pos.loc[c, key]) if c in agg_pos.index else 0.0,
                         sync_dist=True)

        # ── 5. 前景 macro ──────────────────────────────────────────────────
        fg = self.foreground_ids
        for m in seg_metrics:
            macro_all = float(agg_all.loc[fg, m].mean())
            macro_pos = float(agg_pos.loc[fg, m].mean()) if len(agg_pos) else 0.0
            self.log(f'{self.eval_split}/{m}_macro_all', macro_all,
                     prog_bar=(m == 'dice'), sync_dist=True)
            self.log(f'{self.eval_split}/{m}_macro_pos', macro_pos,
                     prog_bar=(m == 'dice'), sync_dist=True)
        for cm in clin_metrics:
            key = f'clin_{cm}'
            macro_all = float(agg_all.loc[fg, key].mean())
            macro_pos = float(agg_pos.loc[fg, key].mean()) if len(agg_pos) else 0.0
            self.log(f'{self.eval_split}/{key}_macro_all', macro_all, sync_dist=True)
            self.log(f'{self.eval_split}/{key}_macro_pos', macro_pos, sync_dist=True)

        # ── 6. 空图 FPR ──────────────────────
        ppa_c = self.foreground_ids[-1]
        empty = df[(df['class_id'] == ppa_c) & (df['support'] == 0)]
        if len(empty) > 0:
            fpr = float((empty['dice'] < 1.0).sum() / len(empty))
            self.log(f'{self.eval_split}/ppa_empty_total',
                     float(len(empty)), sync_dist=True)
            self.log(f'{self.eval_split}/ppa_empty_fpr',
                     fpr, prog_bar=True, sync_dist=True)

        # ── 7. 原有 quant 指标 ─────────────────────────────────────────────
        self._log_quant_metrics('test', self.eval_split)

    # ── Per-case record collection (Phase 4) ──────────────────────────────────

    @staticmethod
    def _meta_get(meta, key, idx, default):
        """Fetch meta[key][idx] from a (possibly collated) batch metadata dict."""
        if isinstance(meta, dict) and key in meta:
            value = meta[key]
            try:
                return value[idx]
            except (TypeError, IndexError, KeyError):
                return value
        return default

    def _collect_records(self, logits, preds, masks, meta, batch_idx):
        from src.metrics.functional import case_class_metrics, predict_from_logits

        # Use the configured threshold for the per-case binary prediction (the
        # torchmetrics aggregate above keeps its own 0.5 default).
        if self._is_binary:
            threshold = float(self.cfg.get('eval', {}).get('threshold', 0.5))
            rec_preds = predict_from_logits(logits, is_binary=True, threshold=threshold)
            class_values = [0, 1]
        else:
            rec_preds = preds
            class_values = list(range(self.num_classes))

        # Foreground (class_value, class_name) pairs for optional task metrics.
        if self._is_binary:
            fg_classes = [(1, self.class_names[0])]
        else:
            fg_classes = [(int(c), self.class_names[int(c)]) for c in self.foreground_ids]

        for b in range(rec_preds.shape[0]):
            pred_b = rec_preds[b].cpu()
            mask_b = masks[b].cpu()
            per_class = case_class_metrics(pred_b, mask_b, class_values)
            record = {
                'case_id':    self._meta_get(meta, 'case_id',    b, f'case_{batch_idx}_{b}'),
                'image_path': self._meta_get(meta, 'image_path', b, ''),
                'mask_path':  self._meta_get(meta, 'mask_path',  b, ''),
                'pred_path':  '',
                'per_class':  per_class,
            }
            if self._clinical_enabled:
                record['clinical'] = self._case_clinical(pred_b, fg_classes)
            if self._boundary_enabled:
                record['boundary'] = self._case_boundary(pred_b, mask_b, fg_classes)
            self.test_records.append(record)

    def _case_clinical(self, pred_b, fg_classes):
        from src.metrics.clinical_metrics import PIXEL_UNIT, compute_class_clinical
        total_pixels = int(pred_b.numel())
        per_class = {
            value: compute_class_clinical(pred_b == value, self._clinical_metrics, total_pixels)
            for value, _ in fg_classes
        }
        return {'unit': PIXEL_UNIT, 'per_class': per_class}

    def _case_boundary(self, pred_b, mask_b, fg_classes):
        from src.metrics.boundary_metrics import boundary_scores
        per_class = {
            value: boundary_scores(pred_b == value, mask_b == value, width_px=self._boundary_width)
            for value, _ in fg_classes
        }
        return {'width_px': self._boundary_width, 'per_class': per_class}

    # ── Metric logging helper ─────────────────────────────────────────────────

    def _log_metrics(self, prefix, dice_metric, iou_metric):
        dice_scores = dice_metric.compute()
        iou_scores  = iou_metric.compute()
        dice_metric.reset()
        iou_metric.reset()

        if self._is_binary:
            # Binary: both are scalars
            self.log(f'{prefix}/dice_mean', dice_scores, prog_bar=True, sync_dist=True)
            self.log(f'{prefix}/iou_mean',  iou_scores,  prog_bar=True, sync_dist=True)
        else:
            # Multi-class: both are 1-D tensors of length num_classes
            for i, name in enumerate(self.class_names):
                self.log(f'{prefix}/dice_{name}', dice_scores[i], sync_dist=True)
                self.log(f'{prefix}/iou_{name}',  iou_scores[i],  sync_dist=True)

            # Mean over foreground classes only (exclude background)
            fg = self.foreground_ids
            dice_mean = dice_scores[fg].mean()
            iou_mean  = iou_scores[fg].mean()
            self.log(f'{prefix}/dice_mean', dice_mean, prog_bar=True, sync_dist=True)
            self.log(f'{prefix}/iou_mean',  iou_mean,  prog_bar=True, sync_dist=True)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        opt_cfg = self.cfg.optimizer
        sch_cfg = self.cfg.scheduler

        # Optimiser
        name = opt_cfg.name.lower()
        # params = self.model.parameters()
        params = [
            {"params": self.model.parameters()}
        ]

        if self.quant_cfg != {} and self.quant_cfg.adapter_enabled:
            adapter_cfg = (
                self.cfg.quantification.get("adapter") or {}
            )

            params.append({
                "params": self.quant_head.parameters(),
                "lr": float(
                    adapter_cfg.get("lr", opt_cfg.lr)
                ),
            })

        wd = opt_cfg.get('weight_decay', 0.0)

        if name == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=opt_cfg.lr, weight_decay=wd)
        elif name == 'adam':
            optimizer = torch.optim.Adam(params, lr=opt_cfg.lr, weight_decay=wd)
        elif name == 'sgd':
            optimizer = torch.optim.SGD(
                params, lr=opt_cfg.lr,
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=wd,
            )
        else:
            raise ValueError(f"Unknown optimizer '{name}'. Choose: adam, adamw, sgd")

        # Scheduler
        sch_name = sch_cfg.name.lower()

        if sch_name == 'none':
            return optimizer

        if sch_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.train.epochs,
                eta_min=sch_cfg.eta_min,
            )
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

        if sch_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sch_cfg.step_size,
                gamma=sch_cfg.gamma,
            )
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

        if sch_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=sch_cfg.patience,
                factor=sch_cfg.factor,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.cfg.checkpoint.monitor,
                    'interval': 'epoch',
                },
            }

        raise ValueError(f"Unknown scheduler '{sch_name}'. Choose: cosine, step, plateau, none")
    
    def _make_boundary_target(self, masks, width_px: int = 3):
        """
        masks: B×H×W, long class-index mask.
        returns: B×1×H×W float boundary target.

        class-agnostic boundary:
        - marks pixels around any class transition
        - also marks foreground touching image border
        """
        m = masks
        b = torch.zeros_like(m, dtype=torch.bool)

        # vertical class transitions
        diff_h = m[:, 1:, :] != m[:, :-1, :]
        b[:, 1:, :] |= diff_h
        b[:, :-1, :] |= diff_h

        # horizontal class transitions
        diff_w = m[:, :, 1:] != m[:, :, :-1]
        b[:, :, 1:] |= diff_w
        b[:, :, :-1] |= diff_w

        # foreground touching image border should count as boundary
        # assumes background id = 0, consistent with usual multiclass masks
        fg = m > 0
        b[:, 0, :] |= fg[:, 0, :]
        b[:, -1, :] |= fg[:, -1, :]
        b[:, :, 0] |= fg[:, :, 0]
        b[:, :, -1] |= fg[:, :, -1]

        b = b.float().unsqueeze(1)

        # thicken the boundary band; width_px=3 is a good first value for 512×512
        width_px = max(int(width_px), 1)
        if width_px > 1:
            k = 2 * width_px - 1
            b = F.max_pool2d(
                b,
                kernel_size=k,
                stride=1,
                padding=width_px - 1,
            )
            b = (b > 0).float()

        return b


    def _boundary_loss(self, boundary_logits, masks):
        target = self._make_boundary_target(
            masks,
            width_px=self._boundary_width,
        ).to(device=boundary_logits.device, dtype=boundary_logits.dtype)

        if target.shape[-2:] != boundary_logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=boundary_logits.shape[-2:],
                mode="nearest",
            )

        # Boundary pixels are sparse, so BCE needs positive reweighting.
        pos = target.sum()
        neg = target.numel() - pos
        pos_weight = (neg / (pos + 1.0)).clamp(1.0, 20.0).detach()

        bce = F.binary_cross_entropy_with_logits(
            boundary_logits,
            target,
            pos_weight=pos_weight.view(1),
        )

        probs = torch.sigmoid(boundary_logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()

        return 0.5 * bce + 0.5 * dice, bce, dice

    @torch.no_grad()
    def _update_quant_metrics(self, stage, logits, masks):
        if self.quantification is None:
            return

        metrics = getattr(
            self,
            f'quant_{stage}_metrics',
        )

        stats = self.quantification.diagnostics(
            logits,
            masks,
        )

        for name, (values, valid) in stats.items():
            metrics[name].update(
                values,
                weight=valid.float(),
            )

    def _log_quant_metrics(self, stage, prefix):
        if self.quantification is None:
            return

        metrics = getattr(
            self,
            f'quant_{stage}_metrics',
        )

        for name, metric in metrics.items():
            # MeanMetric synchronizes its sums/counts at compute().
            self.log(
                f'{prefix}/quant_{name}',
                metric.compute(),
                prog_bar=True,
                sync_dist=True,
            )

            metric.reset()