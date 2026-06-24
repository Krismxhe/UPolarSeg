# Migration guide — `custom` → `smp` / `baseline` / `research`

The model layer moved from a two-provider layout (`smp` + `custom`) to three
explicit layers. **Nothing breaks**: old configs and commands still run, because
`provider=custom` is kept as a deprecated compatibility alias and the old flat
experiment names are kept as forwarding wrappers. This guide shows the preferred
new form.

## Why

- `TransUNet` is an established **baseline**, not a user research model.
- `ModularUNet` is the first **research** model, not a generic "custom" model.
- Separating baseline vs research implementations keeps experiment tables clear,
  while a single shared pipeline (dataset / augmentation / loss / optimizer /
  trainer / evaluator / CSV) keeps comparisons fair.

## Provider layers

| provider | for | configs | built by |
|---|---|---|---|
| `smp` | SMP architectures (Unet, Unet++, DeepLabV3+, FPN, MAnet, …) | `configs/model/smp/*` | `src/models/smp_models.py` |
| `baseline` | established non-SMP baselines (TransUNet; UNETR/SwinUNETR *planned*) | `configs/model/baseline/*` | `src/models/baselines/registry.py` |
| `research` | your own methods (ModularUNet; PolarSeg/… *planned*) | `configs/model/research/*` | `src/models/research/registry.py` |
| `custom` | **deprecated** alias for old configs | `configs/model/custom/*` | `factory.py` → baseline/research registry |

`factory.py` is the only provider router; `SegModule` imports no concrete model.

## Command migration

| Old (deprecated, still works) | New (preferred) |
|---|---|
| `python train.py model=custom/transunet train.img_size=224` | `python train.py model=baseline/transunet train.img_size=224` |
| `python train.py model=custom/modular_unet` | `python train.py model=research/modular_unet_identity` (or `model=research/modular_unet`) |
| `python train.py +experiment=smp_unet_resnet34` | `python train.py +experiment=baselines/smp_unet_resnet34` |
| `python train.py +experiment=transunet_r50_vit_b16` | `python train.py +experiment=baselines/transunet_r50_vit_b16` |
| `python train.py +experiment=modular_unet_identity` | `python train.py +experiment=research/modular_unet_identity` |

Evaluation is unchanged and provider-agnostic:

```bash
python evaluate.py model=baseline/transunet \
    checkpoint=outputs/<name>/checkpoints/best.ckpt split=test eval.save_csv=true
```

Legacy `model=unet` (no `provider` field) still resolves to `smp` and is unchanged.

## Checkpoints

Old checkpoints keep loading — PyTorch-Lightning checkpoints store the
`state_dict` + hyperparameters, not Python module import paths, so relocating the
TransUNet / ModularUNet packages does not invalidate them as long as the same
config rebuilds the same architecture.

## Deprecation timeline

`provider=custom` and the flat `+experiment=<name>` wrappers are kept for
backward compatibility and may be removed in a future release. Prefer the
`smp` / `baseline` / `research` providers and the `baselines/` / `research/`
experiment paths in new work.
