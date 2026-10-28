"""
Training entry point.

Basic usage:
    python train.py

Override anything from the command line:
    python train.py model=deeplabv3plus model.encoder=resnet50
    python train.py train.batch_size=4 train.epochs=50
    python train.py augmentation=heavy optimizer=sgd

Sweep multiple values (Hydra multirun):
    python train.py --multirun model=unet,unetplusplus train.lr=1e-3,1e-4
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.datasets.seg_dataset import SegDataModule
from src.models.seg_module import SegModule


@hydra.main(version_base=None, config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed, workers=True)

    print('\n' + '=' * 60)
    print(OmegaConf.to_yaml(cfg))
    print('=' * 60 + '\n')

    # ── Data ──────────────────────────────────────────────────────────────────
    datamodule = SegDataModule(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SegModule(cfg)

    # ── Logger ────────────────────────────────────────────────────────────────
    if cfg.logging.logger == 'wandb':
        logger = WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            save_dir=cfg.logging.save_dir,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=cfg.logging.save_dir,
            name=cfg.logging.name,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            filename=cfg.checkpoint.filename,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar(),
    ]

    if cfg.train.early_stopping:
        callbacks.append(EarlyStopping(
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            patience=cfg.train.early_stopping_patience,
            verbose=True,
        ))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        num_nodes=cfg.hardware.num_nodes,
        strategy=cfg.hardware.strategy,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.train.val_interval,
        precision=cfg.train.precision,

        # Reproducibility
        deterministic=True,
        benchmark=False,
    )

    trainer.fit(model, datamodule=datamodule)

    # ── Test with best checkpoint ──────────────────────────────────────────────
    print('\n[INFO] Running test set evaluation with best checkpoint...')
    trainer.test(model, datamodule=datamodule, ckpt_path='best')


if __name__ == '__main__':
    main()
