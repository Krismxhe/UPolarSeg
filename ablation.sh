python train.py \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=angular \
    train.img_size=512 \
    dataset=beijing_1024_polar \
    hardware.devices=[0,1] \
    hardware.strategy=ddp_find_unused_parameters_true \
    train.batch_size=4 \
    polar=enabled \
    augmentation=medium_polar \
    logging.name=lite_angular





python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_angular/version_0/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=beijing_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=angular \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_angular_beiijing


python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_angular/version_0/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=tibet_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=angular \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_angular_tibet