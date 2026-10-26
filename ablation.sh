python train.py \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=radial \
    train.img_size=512 \
    dataset=beijing_1024_polar \
    train.batch_size=8 \
    hardware.devices=[1] \
    polar=enabled \
    augmentation=medium_polar \
    logging.name=lite_radial \
    train.early_stopping_patience=30 

python train.py \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=angular \
    train.img_size=512 \
    dataset=beijing_1024_polar \
    hardware.strategy=ddp_find_unused_parameters_true \
    train.batch_size=8 \
    hardware.devices=1 \
    polar=enabled \
    augmentation=medium_polar \
    logging.name=lite_both

python train.py \
    model=unet \
    train.img_size=512 \
    dataset=beijing_1024_polar \
    hardware.devices=[3] \
    train.batch_size=8 \
    polar=enabled \
    logging.name=unet_polar




python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_radial/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=beijing_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=radial \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_radial_beijing


python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_radial/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=tibet_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=radial \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_radial_tibet