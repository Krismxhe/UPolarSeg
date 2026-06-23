"""
Loss factory.

``SegModule`` builds its loss via ``build_loss(cfg.loss, cfg.dataset)`` and never
imports segmentation-models-pytorch directly. The task type (binary vs
multiclass) is derived from ``dataset_cfg.num_classes`` — the single source of
truth — so it never has to be repeated in the loss config.
"""

from src.losses.segmentation_losses import DiceWithAuxLoss


def _task_mode(dataset_cfg) -> str:
    """binary when num_classes == 1, else multiclass."""
    return "binary" if int(dataset_cfg.num_classes) == 1 else "multiclass"


def build_loss(loss_cfg, dataset_cfg):
    """Return a loss module whose ``forward`` yields ``(total, dice, aux)``.

    Backward compatibility: if ``loss_cfg`` is None (e.g. an old checkpoint whose
    config predates the loss group), fall back to the previous default —
    Dice + BCE for binary / Dice + CE for multiclass, both weighted 1.0.
    """
    mode = _task_mode(dataset_cfg)

    if loss_cfg is None:
        return DiceWithAuxLoss(mode=mode, dice_weight=1.0, aux_weight=1.0, from_logits=True)

    name = str(loss_cfg.get("name", "dice_bce" if mode == "binary" else "dice_ce")).lower()
    from_logits = bool(loss_cfg.get("from_logits", True))

    if name == "dice_ce":
        if mode != "multiclass":
            raise ValueError(
                "loss 'dice_ce' is for multiclass segmentation (num_classes > 1); "
                "use loss=dice_bce for binary."
            )
        return DiceWithAuxLoss(
            mode="multiclass",
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            aux_weight=loss_cfg.get("ce_weight", 1.0),
            from_logits=from_logits,
        )

    if name == "dice_bce":
        if mode != "binary":
            raise ValueError(
                "loss 'dice_bce' is for binary segmentation (num_classes == 1); "
                "use loss=dice_ce for multiclass."
            )
        return DiceWithAuxLoss(
            mode="binary",
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            aux_weight=loss_cfg.get("bce_weight", 1.0),
            from_logits=from_logits,
        )

    raise ValueError(
        f"Unknown loss name '{name}'. Available: 'dice_ce' (multiclass), 'dice_bce' (binary)."
    )
