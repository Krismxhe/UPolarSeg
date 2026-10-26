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
    checkpoint=outputs/lite_radial/version_1/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_radial_beijing \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=radial polar=enabled dataset=beijing_1024_polar train.img_size=512

#tibet
python infe_tem.py \
    checkpoint=outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_both_tibet \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=both polar=enabled dataset=tibet_1024_polar train.img_size=512

# PMSR Lite Radial — Tibet
python infe_tem.py \
    checkpoint=outputs/lite_radial/version_1/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_radial_tibet \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=radial polar=enabled dataset=tibet_1024_polar train.img_size=512 
  

# PMSR Lite Angular — Tibet
python infe_tem.py \
    checkpoint=outputs/lite_angular/version_0/checkpoints/best.ckpt \
    ++output_dir=predictions/ablation_angular_tibet \
    model=research/modular_unet_pmsr_lite model.skip.params.axis_mode=angular polar=enabled dataset=tibet_1024_polar train.img_size=512 


cd ~/ziyue/ziyue/mengxian/medseg_research

# Beijing
python infe_tem.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/ModularUNet_resnet34_beijing_1024_polar/version_3/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=beijing_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_identity \
    polar=enabled \
    model.heads.boundary.enabled=true \
    task.outputs.boundary.enabled=true \
    ++output_dir=predictions/ablation_bh_beijing

# Tibet
python infe_tem.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/ModularUNet_resnet34_beijing_1024_polar/version_3/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=tibet_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_identity \
    polar=enabled \
    model.heads.boundary.enabled=true \
    task.outputs.boundary.enabled=true \
    ++output_dir=predictions/ablation_bh_tibet



python eval.py \
    checkpoint=/home/ziyue/ziyue/mengxian/eval_model_weights/UPolarSeg_best_weights/unet_pmsr_lite/version_0/checkpoints/best.ckpt \
    dataset=beijing_1024_polar \
    model=research/modular_unet_pmsr_lite \
    polar=enabled \
    +output_csv=/home/ziyue/ziyue/mengxian/eval_results/beijing_upolarseg.csv \
    +model_name=UPolarSeg

python bootstrap_paired_test.py \
    --csvs /home/ziyue/ziyue/mengxian/eval_results/per_case_beijing_all.csv \
           /home/ziyue/ziyue/mengxian/eval_results/per_case_tibet_all.csv \
    --target UPolarSeg \
    --n_boot 1000 \
    --output /home/ziyue/ziyue/mengxian/eval_results/paired_test_results.csv