"""
Evaluate a saved checkpoint on any dataset split and print a metric table.

Usage:
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt split=val
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt split=train
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt dataset=optic split=test
"""

import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.datasets.seg_dataset import SegDataModule
from src.evaluation.prediction_masks import normalize_save_mode
from src.evaluation.quantification_overlays import normalize_fundus_mode, normalize_overlay_mode
from src.metrics.evaluator import SegEvaluator
from src.models.seg_module import SegModule
from src.utils.io import ensure_dir, write_resolved_config


@hydra.main(version_base=None, config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.get('checkpoint', None)
    if ckpt_path is None:
        raise ValueError(
            'Please provide a checkpoint path:\n'
            '  python evaluate.py checkpoint=outputs/.../best.ckpt'
        )

    split = cfg.split
    eval_cfg = cfg.get('eval', {}) or {}
    save_mode = normalize_save_mode(eval_cfg.get('save_predictions', 'none'))
    overlay_mode = normalize_overlay_mode(eval_cfg.get('save_quantification_overlays', 'none'))
    gt_overlay_mode = normalize_overlay_mode(eval_cfg.get('save_quantification_gt_overlays', 'none'))
    fundus_mode = normalize_fundus_mode(eval_cfg.get('save_fundus_overlays', 'none'))
    # Resolve before testing, independently of CSV export, so every rank's
    # writers and the final reports use the same destination.
    default_dir = os.path.join(cfg.logging.save_dir, cfg.logging.name, 'eval', split)
    output_dir = to_absolute_path(str(eval_cfg.get('output_dir', None) or default_dir))
    OmegaConf.update(cfg, 'eval.output_dir', output_dir, force_add=True)
    datamodule = SegDataModule(cfg, eval_split=split)

    odppa_cfg = (
        (cfg.get("eval", {}) or {}).get(
            "odppa",
            {},
        ) or {}
    )

    if odppa_cfg.get("enabled", False):

        if (
            int(cfg.dataset.num_classes) < 3
            or str(cfg.dataset.mask_mode) != "index"
        ):
            raise ValueError(
                "OD/PPA evaluation requires "
                "multiclass index masks."
            )

        if (
            bool(cfg.polar.enabled)
            and str(
                cfg.polar.eval.metric_space
            ).lower() != "cart"
        ):
            raise ValueError(
                "OD/PPA evaluation requires "
                "polar.eval.metric_space=cart."
            )

    # Load model weights from checkpoint; architecture is defined by cfg
    model = SegModule.load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
    model.eval_split = split  # ensure metric keys reflect the actual split

    run_name = f"{cfg.logging.name}_eval_{split}"
    loggers = [
        TensorBoardLogger(save_dir=cfg.logging.save_dir, name=run_name),
        CSVLogger(save_dir=cfg.logging.save_dir, name=run_name),
    ]

    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        num_nodes=cfg.hardware.num_nodes,
        strategy=cfg.hardware.strategy,
        logger=loggers,
    )
    results = trainer.test(model, datamodule=datamodule)

    if not trainer.is_global_zero:
        return

    print(f'\n── Results ({split}) ──────────────────────')
    for k, v in results[0].items():
        print(f'  {k:<30} {v:.4f}')

    # ── Explicit research CSV (summary / per_class / per_case) ─────────────────
    if eval_cfg.get('save_csv', True):
        evaluator = SegEvaluator(cfg, split=split, run_name=cfg.logging.name,
                                 checkpoint=str(ckpt_path))
        written = evaluator.write(model.test_records, output_dir)
        print(f'\n[INFO] Evaluation CSV written to: {output_dir}')
        for name, path in written.items():
            print(f'  {name:<10} {path}')
    elif any(mode != 'none' for mode in (save_mode, overlay_mode, gt_overlay_mode, fundus_mode)):
        # Image-only exports still retain their resolved geometry, prediction
        # source, readout selection and threshold alongside the output files.
        config_path = write_resolved_config(ensure_dir(output_dir) / 'config.yaml', cfg)
        print(f'\n[INFO] Evaluation config written to: {config_path}')

    if save_mode != 'none':
        print('\n[INFO] Prediction masks written from seg_logits:')
        if save_mode in {'polar', 'both'}:
            print(f"  polar      {os.path.join(output_dir, 'masks_polar')}")
        if save_mode in {'cartesian', 'both'}:
            print(f"  cartesian  {os.path.join(output_dir, 'masks_cartesian')}")

    if overlay_mode != 'none':
        print('\n[INFO] Quantification overlays written:')
        layout = str(eval_cfg.get('quantification_overlay_layout', 'combined')).strip().lower()
        readouts = str(eval_cfg.get('quantification_overlay_readouts', 'both')).strip().lower()
        for space in ('polar', 'cartesian'):
            if overlay_mode not in {space, 'both'}:
                continue
            directory = os.path.join(output_dir, f'overlays_{space}')
            if layout == 'separate':
                for kind in ('hard', 'soft'):
                    if readouts in {kind, 'both'}:
                        print(f"  {space} {kind:<5} {os.path.join(directory, kind)}")
            else:
                print(f'  {space:<10} {directory}')
    if gt_overlay_mode != 'none':
        print('\n[INFO] Ground-truth hard quantification overlays written:')
        if gt_overlay_mode in {'polar', 'both'}:
            print(f"  polar      {os.path.join(output_dir, 'overlays_gt_polar')}")
        if gt_overlay_mode in {'cartesian', 'both'}:
            print(f"  cartesian  {os.path.join(output_dir, 'overlays_gt_cartesian')}")
    if fundus_mode != 'none':
        print('\n[INFO] Hard quantification overlays on fundus images written:')
        if fundus_mode in {'prediction', 'both'}:
            print(f"  prediction   {os.path.join(output_dir, 'overlays_fundus')}")
        if fundus_mode in {'groundtruth', 'both'}:
            print(f"  ground truth {os.path.join(output_dir, 'overlays_gt_fundus')}")
    if overlay_mode != 'none' or gt_overlay_mode != 'none' or fundus_mode != 'none':
        print(f"  metadata   {os.path.join(output_dir, 'overlay_metadata')}")


if __name__ == '__main__':
    main()
