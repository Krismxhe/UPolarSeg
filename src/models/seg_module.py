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

import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import (
    BinaryF1Score, MulticlassF1Score,
    BinaryJaccardIndex, MulticlassJaccardIndex,
)
from torchmetrics import MeanMetric

from src.datasets.batch import original_size_from_metadata, unpack_batch
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
from src.quantification.polar_readout import build_polar_readout
from src.quantification.diagnostics import compute_quantification_diagnostics
from src.evaluation.quantification_records import (
    apply_annotation_completeness,
    build_quantification_case_records,
    quantification_protocol,
    unavailable_quantification_record,
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

            if bool(self.quant_cfg.get("adapter_enabled", False)):
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
        self._prediction_writer = None
        self._quantification_overlay_writer = None

        from src.evaluation.quantification_overlays import normalize_overlay_mode

        eval_cfg = cfg.get('eval', {}) or {}
        self._quantification_overlay_mode = normalize_overlay_mode(
            eval_cfg.get('save_quantification_overlays', 'none')
        )
        self._quantification_gt_overlay_mode = normalize_overlay_mode(
            eval_cfg.get('save_quantification_gt_overlays', 'none')
        )
        self._quantification_overlay_active = (
            self._quantification_overlay_mode != 'none' or self._quantification_gt_overlay_mode != 'none'
        )
        self._quantification_overlay_source = str(
            eval_cfg.get('quantification_overlay_source', 'seg_logits')
        ).strip().lower()
        self._quantification_overlay_readouts = str(
            eval_cfg.get('quantification_overlay_readouts', 'both')
        ).strip().lower()
        if self._quantification_overlay_mode != 'none':
            if self._quantification_overlay_source not in {'seg_logits', 'quantification'}:
                raise ValueError(
                    "eval.quantification_overlay_source must be 'seg_logits' or 'quantification'."
                )
            if self._quantification_overlay_readouts not in {'hard', 'soft', 'both'}:
                raise ValueError(
                    "eval.quantification_overlay_readouts must be 'hard', 'soft', or 'both'."
                )

        # Optional multi-task eval outputs (Phase 9b); default OFF so the
        # segmentation baseline is unaffected.
        from src.tasks import get_output_cfg, is_task_enabled
        self._clinical_enabled = is_task_enabled(cfg, 'clinical')
        self._boundary_enabled = is_task_enabled(cfg, 'boundary')
        self._clinical_metrics = list(get_output_cfg(cfg, 'clinical').get(
            'metrics', ['area', 'perimeter', 'area_ratio']))
        self._boundary_width = int(get_output_cfg(cfg, 'boundary').get('boundary_width_px', 3))


        # ─ OD/PPA research evaluation ───────────────────────────────────────
        # Evaluation only: does not affect training / validation / loss.
        from src.metrics.odppa_eval import (
            enabled as odppa_eval_enabled,
            load_metadata_csv,
        )

        self._odppa_eval_enabled = odppa_eval_enabled(cfg)

        odppa_cfg = (
            (cfg.get("eval", {}) or {}).get("odppa", {}) or {}
        )

        self._odppa_metadata = (
            load_metadata_csv(odppa_cfg.get("metadata_csv", None))
            if self._odppa_eval_enabled or self._quantification_gt_overlay_mode != 'none'
            else {}
        )

        # Measurement availability is independent of auxiliary-loss training.
        # Reuse the loss-owned geometry when present; otherwise register one
        # evaluation-only readout so Lightning moves its buffers with the model.
        self._quant_eval_readout = None
        self._quant_protocol = None
        self._quant_unavailable_reason = "not_requested"
        if self.quantification is not None:
            self._quant_protocol = quantification_protocol(self.quantification)
            self._quant_unavailable_reason = ""
        elif self._odppa_eval_enabled or self._quantification_overlay_active:
            polar = cfg.get("polar", {}) or {}
            geometry = polar.get("geometry", {}) or {}
            if self.num_classes != 3 or cfg.dataset.mask_mode != "index":
                self._quant_unavailable_reason = "requires_three_class_index_masks"
            elif not bool(polar.get("enabled", False)):
                self._quant_unavailable_reason = "requires_polar_coordinates"
            elif geometry.get("mode", "linear") != "linear":
                self._quant_unavailable_reason = "requires_linear_radial_grid"
            else:
                self._quant_eval_readout = build_polar_readout(cfg)
                self._quant_protocol = quantification_protocol(self._quant_eval_readout)
                self._quant_unavailable_reason = ""
                if self._odppa_eval_enabled or self._quantification_overlay_mode != 'none':
                    self.quant_test_metrics = torch.nn.ModuleDict({
                        name: MeanMetric() for name in QUANT_VAL_KEYS
                    })

        if self._quantification_overlay_active and self._quantification_readout() is None:
            raise ValueError(
                'Quantification overlays require three-class index masks on an enabled '
                'linear polar grid: ' + self._quant_unavailable_reason
            )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        return self.model(x)

    def _quantification_readout(self):
        """Return the shared measurement capability independently of loss enablement."""
        if self.quantification is not None:
            return self.quantification
        return self._quant_eval_readout

    def _evaluation_outputs(self, images):
        """Select the same explicit segmentation/quantification paths in val and test."""
        if self.quantification is not None and bool(self.quant_cfg.get("adapter_enabled", False)):
            outputs = normalize_model_output(self.model(images, return_features=True))
            quant_logits = self.quant_head(
                outputs["features"]["decoder_final"], outputs["seg_logits"],
            )
            return outputs, quant_logits, "quant_logits"
        outputs = normalize_model_output(self(images))
        return outputs, outputs["seg_logits"], "seg_logits"

    @torch.no_grad()
    def _quantification_overlay_batch(
        self, seg_logits, quant_logits, prediction_source, *, diagnostics=None, target_masks=None,
    ):
        """Prepare one source-consistent batch, reusing evaluation readouts when possible.

        This helper also serves checkpoint export without GT diagnostics. The
        CPU arrays are transient rendering inputs and never enter DDP records.
        """
        if not self._quantification_overlay_active:
            return None
        readout = self._quantification_readout()
        if readout is None:
            raise ValueError('Quantification overlays require an available polar readout.')
        if self._quantification_gt_overlay_mode != 'none' and target_masks is None:
            raise ValueError('Ground-truth quantification overlays require target masks.')

        mask = hard = soft = groundtruth_mask = groundtruth_hard = None
        source = 'seg_logits'
        if self._quantification_overlay_mode != 'none':
            use_quantification = self._quantification_overlay_source == 'quantification'
            logits = quant_logits if use_quantification else seg_logits
            source = prediction_source if use_quantification else 'seg_logits'
            mask = logits.argmax(dim=1)
            reuse = diagnostics is not None and source == prediction_source
            if self._quantification_overlay_readouts in {'hard', 'both'}:
                hard = diagnostics['hard'] if reuse else readout.measure_mask(mask)
            if self._quantification_overlay_readouts in {'soft', 'both'}:
                soft = diagnostics['soft'] if reuse else readout.read_probabilities(
                    logits.float().softmax(dim=1)
                )
        if self._quantification_gt_overlay_mode != 'none':
            groundtruth_mask = target_masks.to(device=readout.valid_grid.device)
            groundtruth_hard = (
                diagnostics['target'] if diagnostics is not None
                else readout.measure_mask(groundtruth_mask)
            )

        def cpu_readout(values):
            if values is None:
                return None
            return {key: value.detach().cpu().numpy() for key, value in values.items()}

        return {
            'mask': None if mask is None else mask.detach().cpu().numpy(),
            'hard': cpu_readout(hard),
            'soft': cpu_readout(soft),
            'groundtruth_mask': None if groundtruth_mask is None else groundtruth_mask.detach().cpu().numpy(),
            'groundtruth_hard': cpu_readout(groundtruth_hard),
            'prediction_source': source,
        }

    def _save_quantification_overlay(
        self, writer, payload, index, case_id, original_size=None, reference_complete=True,
    ):
        """Save one case using the same payload contract in evaluation and the CLI."""
        def case_readout(values):
            return None if values is None else {key: value[index] for key, value in values.items()}

        return writer.save(
            None if payload['mask'] is None else payload['mask'][index], case_id,
            readout=self._quantification_readout(),
            hard=case_readout(payload['hard']),
            soft=case_readout(payload['soft']),
            original_size=original_size,
            prediction_source=payload['prediction_source'],
            groundtruth_mask=(
                None if payload['groundtruth_mask'] is None else payload['groundtruth_mask'][index]
            ),
            groundtruth_hard=case_readout(payload['groundtruth_hard']),
            reference_complete=reference_complete,
        )

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

    def _predict(self, logits, threshold=0.5):
        """Returns hard predictions: B × H × W (long)."""
        if self._is_binary:
            return (torch.sigmoid(logits).squeeze(1) > threshold).long()
        return logits.argmax(dim=1)

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        images, masks, _meta = unpack_batch(batch)
        loss = None

        if self.quantification is not None and not bool(self.quant_cfg.get("adapter_enabled", False)):
            outputs = normalize_model_output(
                self(images),
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

        elif self.quantification is not None and bool(self.quant_cfg.get("adapter_enabled", False)):
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
        outputs, quant_logits, _source = self._evaluation_outputs(images)
        logits = outputs["seg_logits"]
        loss, _, _ = self._loss(logits, masks)
        preds = self._predict(logits)
        self.val_dice.update(preds, masks)
        self.val_iou.update(preds, masks)
        self._update_quant_metrics("val", quant_logits, masks)

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
        self._prediction_writer = None
        self._quantification_overlay_writer = None

        # Each DDP rank saves its own samples. CSV export and rank-zero report
        # generation happen separately after all records have been gathered.
        from hydra.utils import to_absolute_path
        from src.evaluation.prediction_masks import PredictionMaskWriter, normalize_save_mode
        from src.evaluation.quantification_overlays import QuantificationOverlayWriter

        eval_cfg = self.cfg.get('eval', {}) or {}
        mode = normalize_save_mode(eval_cfg.get('save_predictions', 'none'))
        if mode != 'none' or self._quantification_overlay_active:
            default_dir = os.path.join(
                self.cfg.logging.save_dir, self.cfg.logging.name, 'eval', self.eval_split,
            )
            output_dir = to_absolute_path(str(eval_cfg.get('output_dir') or default_dir))
            if mode != 'none':
                self._prediction_writer = PredictionMaskWriter(self.cfg, output_dir, mode=mode)
            if self._quantification_overlay_active:
                self._quantification_overlay_writer = QuantificationOverlayWriter(
                    self.cfg, output_dir, mode=self._quantification_overlay_mode,
                )

        for metric in self.quant_test_metrics.values():
            metric.reset()

    def test_step(self, batch, batch_idx):
        images, masks, meta = unpack_batch(batch)
        outputs, quant_logits, prediction_source = self._evaluation_outputs(images)
        logits = outputs['seg_logits']
        threshold = float((self.cfg.get('eval', {}) or {}).get('threshold', 0.5))
        preds = self._predict(logits, threshold=threshold)

        self.test_dice.update(preds, masks)
        self.test_iou.update(preds, masks)

        quant_records = None
        quant_stats = None
        diagnostics = None
        readout = self._quantification_readout()
        if readout is not None and (
            self.quantification is not None or self._odppa_eval_enabled
            or self._quantification_overlay_mode != 'none'
        ):
            diagnostics = compute_quantification_diagnostics(readout, quant_logits, masks)
            quant_records = build_quantification_case_records(
                diagnostics, protocol=self._quant_protocol,
                prediction_source=prediction_source,
            )
            quant_stats = diagnostics["metrics"]
        elif self._odppa_eval_enabled:
            quant_records = [
                unavailable_quantification_record(self._quant_unavailable_reason)
                for _ in range(masks.shape[0])
            ]

        overlay_batch = self._quantification_overlay_batch(
            logits, quant_logits, prediction_source, diagnostics=diagnostics, target_masks=masks,
        )

        self._collect_records(
            logits,
            preds,
            masks,
            meta,
            batch_idx,
            quant_records=quant_records,
            overlay_batch=overlay_batch,
        )

        if quant_stats is not None:
            # Collection applies case metadata (including incomplete PPA
            # annotations). Test logs and exported records use the same cohort.
            quant_stats = {
                name: (values, valid & valid.new_tensor([
                    row[f"quant_{name}_valid"] for row in quant_records
                ]))
                for name, (values, valid) in quant_stats.items()
            }

        self._update_quant_metrics(
            'test',
            quant_logits,
            masks,
            stats=quant_stats,
        )

    def on_test_epoch_end(self):
        self._log_metrics(
            self.eval_split,
            self.test_dice,
            self.test_iou,
        )

        self._log_quant_metrics(
            'test',
            self.eval_split,
        )

        # Gather per-case records across GPUs before evaluate.py writes CSV.
        self._gather_test_records()

    def _gather_test_records(self):
        """Gather scalar per-case records across DDP ranks."""

        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return

        world_size = dist.get_world_size()

        gathered = [None for _ in range(world_size)]

        dist.all_gather_object(
            gathered,
            self.test_records,
        )

        if dist.get_rank() == 0:
            merged = []
            seen = set()

            for shard in gathered:
                for record in shard or []:

                    case_id = str(
                        record.get("case_id", "")
                    )

                    # DistributedSampler can pad the dataset with
                    # duplicated cases.
                    if case_id in seen:
                        continue

                    seen.add(case_id)
                    merged.append(record)

            self.test_records = merged

        else:
            self.test_records = []

    # ── Per-case record collection (Phase 4) ──────────────────────────────────
    @staticmethod
    def _meta_get(meta, key, idx, default):
        """Fetch meta[key][idx] from a (possibly collated) batch metadata dict."""
        if isinstance(meta, dict) and key in meta:
            value = meta[key]
            if isinstance(value, (str, bytes)):
                return value
            try:
                value = value[idx]
            except (TypeError, IndexError, KeyError):
                pass
            # default_collate turns numeric/bool metadata into scalar tensors.
            # Normalize before both native-mask and quantification policies.
            if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
                value = value.item()
            return value
        return default

    def _collect_records(
        self, logits, preds, masks, meta, batch_idx, quant_records=None, overlay_batch=None,
    ):
        from src.metrics.functional import case_class_metrics, predict_from_logits

        if quant_records is not None and len(quant_records) != masks.shape[0]:
            raise ValueError("Segmentation and quantification batches must have the same case count.")
        if overlay_batch is not None:
            for key in ('mask', 'groundtruth_mask'):
                if overlay_batch[key] is not None and len(overlay_batch[key]) != masks.shape[0]:
                    raise ValueError('Segmentation and overlay batches must have the same case count.')

        # Keep CSVs, saved masks, and aggregate test metrics on one threshold.
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

            per_class = case_class_metrics(
                pred_b,
                mask_b,
                class_values,
            )

            case_id = str(
                self._meta_get(
                    meta,
                    'case_id',
                    b,
                    f'case_rank{self.global_rank}_{batch_idx}_{b}',
                )
            )

            image_path = str(
                self._meta_get(
                    meta,
                    'image_path',
                    b,
                    '',
                )
            )

            mask_path = str(
                self._meta_get(
                    meta,
                    'mask_path',
                    b,
                    '',
                )
            )

            record = {
                'case_id': case_id,
                'image_path': image_path,
                'mask_path': mask_path,
                'pred_path': '',
                'pred_polar_path': '',
                'pred_cartesian_path': '',
                'overlay_polar_path': '',
                'overlay_cartesian_path': '',
                'overlay_polar_hard_path': '',
                'overlay_cartesian_hard_path': '',
                'overlay_polar_soft_path': '',
                'overlay_cartesian_soft_path': '',
                'overlay_gt_polar_path': '',
                'overlay_gt_cartesian_path': '',
                'overlay_metadata_path': '',
                'per_class': per_class,
            }

            case_metadata = dict(self._odppa_metadata.get(case_id, {}))
            for field in ("patient_id", "eye", "reference_complete"):
                if case_metadata.get(field) in (None, ""):
                    value = self._meta_get(meta, field, b, None)
                    if value not in (None, ""):
                        case_metadata[field] = value

            # ── OD/PPA research evaluation ─────────────────────────────────
            if self._odppa_eval_enabled:

                from src.metrics.odppa_eval import build_case_record

                odppa_record = build_case_record(
                    pred_native=rec_preds[b]
                        .detach()
                        .cpu()
                        .numpy(),

                    mask_path=mask_path,

                    case_id=case_id,

                    image_path=image_path,

                    cfg=self.cfg,

                    metadata=case_metadata,
                )

                record.update(
                    odppa_record
                )

            # Preserve lexical IDs (including leading zeros) and use the same
            # patient grouping for segmentation and quantification statistics.
            patient_id = case_metadata.get("patient_id")
            record["patient_id"] = str(case_id if patient_id in (None, "") else patient_id)
            complete = case_metadata.get(
                "reference_complete",
                self._meta_get(meta, "reference_complete", b,
                               record.get("reference_complete", True)),
            )
            if quant_records is not None:
                quant_records[b] = apply_annotation_completeness(quant_records[b], complete)
                record.update(quant_records[b])
                complete = quant_records[b]['quant_annotation_complete']

            if self._clinical_enabled:
                record['clinical'] = self._case_clinical(pred_b, fg_classes)
            if self._boundary_enabled:
                record['boundary'] = self._case_boundary(pred_b, mask_b, fg_classes)

            writer = self._prediction_writer
            overlay_writer = self._quantification_overlay_writer
            original_size = None
            mask_needs_original = (
                writer is not None and writer.mode in {'cartesian', 'both'}
                and writer.cartesian_size == 'original'
            )
            if mask_needs_original or (overlay_writer is not None and overlay_writer.needs_original_size):
                original_size = original_size_from_metadata(meta, b)
                if original_size is None and image_path:
                    from PIL import Image
                    with Image.open(image_path) as image:
                        original_size = image.height, image.width
            if writer is not None:
                record.update(writer.save(
                    pred_b.detach().numpy(), case_id, original_size=original_size,
                ))
            if overlay_writer is not None and overlay_batch is not None:
                record.update(self._save_quantification_overlay(
                    overlay_writer, overlay_batch, b, case_id, original_size=original_size,
                    reference_complete=complete,
                ))
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

        if self.quantification is not None and bool(self.quant_cfg.get("adapter_enabled", False)):
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

    @torch.no_grad()
    def _update_quant_metrics(self, stage, logits, masks, stats=None):
        metrics = getattr(
            self,
            f'quant_{stage}_metrics',
        )
        if not metrics:
            return
        if stats is None:
            readout = self._quantification_readout()
            if readout is None:
                return
            stats = compute_quantification_diagnostics(readout, logits, masks)["metrics"]

        for name, (values, valid) in stats.items():
            valid = valid & torch.isfinite(values)
            metrics[name].update(
                torch.where(valid, values, torch.zeros_like(values)),
                weight=valid.float(),
            )

    def _log_quant_metrics(self, stage, prefix):
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
