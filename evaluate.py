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
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.datasets.seg_dataset import SegDataModule
from src.metrics.evaluator import SegEvaluator
from src.models.seg_module import SegModule


@hydra.main(version_base=None, config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.get('checkpoint', None)
    if ckpt_path is None:
        raise ValueError(
            'Please provide a checkpoint path:\n'
            '  python evaluate.py checkpoint=outputs/.../best.ckpt'
        )

    split = cfg.split
    datamodule = SegDataModule(cfg, eval_split=split)

    # Load model weights from checkpoint; architecture is defined by cfg
    model = SegModule.load_from_checkpoint(ckpt_path, cfg=cfg)
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

    print(f'\n── Results ({split}) ──────────────────────')
    for k, v in results[0].items():
        print(f'  {k:<30} {v:.4f}')

    # ── Explicit research CSV (summary / per_class / per_case) ─────────────────
    eval_cfg = cfg.get('eval', {}) or {}
    if eval_cfg.get('save_csv', True) and trainer.is_global_zero:
        if trainer.world_size > 1:
            print('[WARN] CSV export is single-process for now; with DDP only the '
                  'rank-0 shard is recorded (TODO: rank-safe per-case merge).')

        # Resolve output dir to an absolute path (Hydra may not chdir, so a
        # relative path would be ambiguous).
        default_dir = os.path.join(cfg.logging.save_dir, cfg.logging.name, 'eval', split)
        output_dir = to_absolute_path(eval_cfg.get('output_dir', None) or default_dir)

        evaluator = SegEvaluator(cfg, split=split, run_name=cfg.logging.name,
                                 checkpoint=str(ckpt_path))
        written = evaluator.write(model.test_records, output_dir)
        print(f'\n[INFO] Evaluation CSV written to: {output_dir}')
        for name, path in written.items():
            print(f'  {name:<10} {path}')


if __name__ == '__main__':
    main()
