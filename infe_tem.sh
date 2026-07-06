'''python infe_tem.py \
    checkpoint=outputs/TransUNet_resnet50_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/transunet_beijing \
    model=baseline/transunet polar=enabled dataset=beijing_1024_polar train.img_size=512 


python infe_tem.py \
    checkpoint=outputs/TransUNet_resnet50_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/transunet_tibet \
    model=baseline/transunet polar=enabled dataset=tibet_1024_polar train.img_size=512 '''

python infe_tem.py \
    checkpoint=outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_both_beijing \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=both polar=enabled dataset=beijing_1024_polar train.img_size=512

python infe_tem.py \
    checkpoint=outputs/lite_angular/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_angular_beijing \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=angular polar=enabled dataset=beijing_1024_polar train.img_size=512

python infe_tem.py \
    checkpoint=outputs/lite_radial/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_radial_beijing \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=radial polar=enabled dataset=beijing_1024_polar train.img_size=512

#tibet
python infe_tem.py \
    checkpoint=outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_both_tibet \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=both polar=enabled dataset=tibet_1024_polar train.img_size=512

# PMSR Lite Radial — Tibet
python infe_tem.py \
    checkpoint=outputs/lite_radial/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_radial_tibet \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=radial polar=enabled dataset=tibet_1024_polar train.img_size=512 
  

# PMSR Lite Angular — Tibet
python infe_tem.py \
    checkpoint=outputs/lite_angular/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_angular_tibet \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=angular polar=enabled dataset=tibet_1024_polar train.img_size=512 



