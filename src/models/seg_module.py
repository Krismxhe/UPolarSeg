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
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import (
    BinaryF1Score, MulticlassF1Score,
    BinaryJaccardIndex, MulticlassJaccardIndex,
)

from src.datasets.batch import unpack_batch
from src.losses.factory import build_loss
from src.models.factory import build_model
from src.models.outputs import normalize_model_output


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
        outputs = normalize_model_output(self(images))
        logits = outputs['seg_logits']
        loss, loss_dice, loss_aux = self._loss(logits, masks)

        self.log('train/loss',      loss,      on_step=True,  on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/loss_dice', loss_dice, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_ce',   loss_aux,  on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, _meta = unpack_batch(batch)
        outputs = normalize_model_output(self(images))
        logits = outputs['seg_logits']
        loss, _, _ = self._loss(logits, masks)
        preds = self._predict(logits)

        self.val_dice.update(preds, masks)
        self.val_iou.update(preds, masks)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self._log_metrics('val', self.val_dice, self.val_iou)

    def test_step(self, batch, batch_idx):
        images, masks, _meta = unpack_batch(batch)
        outputs = normalize_model_output(self(images))
        logits = outputs['seg_logits']
        preds = self._predict(logits)
        self.test_dice.update(preds, masks)
        self.test_iou.update(preds, masks)

    def on_test_epoch_end(self):
        self._log_metrics(self.eval_split, self.test_dice, self.test_iou)

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
        params = self.model.parameters()
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
