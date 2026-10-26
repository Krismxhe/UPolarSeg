#!/bin/bash

cd /home/ziyue/ziyue/mengxian/medseg_research
'''
echo "=== 1. Unet-pmsr-lite (UPolarSeg) ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_10/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_upolarsq

echo "=== 2. TransUNet ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/TransUNet_resnet50_beijing_1024_cart/version_0/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_cart \
    split=test \
    train.img_size=512 \
    model=baseline/transunet \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_transunet_cart

echo "=== 3. TransUNet-polar ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/TransUNet_resnet50_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=baseline/transunet \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_transunet_polar

echo "=== 4. Unet-resnet34 ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/Unet_resnet34_beijing_1024_cart/version_0/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_cart \
    split=test \
    train.img_size=512 \
    model=unet \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_unet_cart

echo "=== 5. Unet-resnet34-polar ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/Unet_resnet34_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=unet \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_unet_polar

echo "=== 6. Ablation: lite_radial ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_radial/version_1/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=radial \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_lite_radial

echo "=== 7. Ablation: lite_angular ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/lite_angular/version_0/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    model.skip.params.axis_mode=angular \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_lite_angular

echo "=== All evaluations done ==="


echo "=== 1. deeplabv3+_cart ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/DeepLabV3Plus_resnet50_beijing_1024_cart/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=beijing_1024_cart \
    split=test \
    train.img_size=512 \
    model=deeplabv3plus \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/beijing_deeplabv3+_resnet50_cart

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/DeepLabV3Plus_resnet50_beijing_1024_cart/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=tibet_1024_cart \
    split=test \
    train.img_size=512 \
    model=deeplabv3plus \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/tibet_deeplabv3+_resnet50_cart

  python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/DeepLabV3Plus_resnet50_beijing_1024_cart/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=zhengzhou_1024_cart \
    split=test \
    train.img_size=512 \
    model=deeplabv3plus \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_deeplabv3+_resnet50_cart  
'''

echo "=== 2. deeplabv3+_polar ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/DeepLabV3Plus_resnet50_beijing_1024_polar/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=beijing_1024_polar \
    split=test \
    train.img_size=512 \
    model=deeplabv3plus \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/beijing_deeplabv3+_resnet50_polar

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/DeepLabV3Plus_resnet50_beijing_1024_polar/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=tibet_1024_polar \
    split=test \
    train.img_size=512 \
    model=deeplabv3plus \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/tibet_deeplabv3+_resnet50_polar

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/DeepLabV3Plus_resnet50_beijing_1024_polar/version_1/checkpoints/best.ckpt \
    hardware.devices=[4] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=deeplabv3plus \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_deeplabv3+_resnet50_polar

'''
echo "=== 3. segformer_cart ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/SegFormer_mit_b2_beijing_1024_cart/version_1/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=beijing_1024_cart \
    split=test \
    train.img_size=512 \
    model=segformer \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/beijing_segformer_cart

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/SegFormer_mit_b2_beijing_1024_cart/version_1/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=tibet_1024_cart \
    split=test \
    train.img_size=512 \
    model=segformer \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/tibet_segformer_cart

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/SegFormer_mit_b2_beijing_1024_cart/version_1/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_cart \
    split=test \
    train.img_size=512 \
    model=segformer \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_segformer_cart

echo "=== 4. segformer_polar ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/SegFormer_mit_b2_beijing_1024_polar/version_1/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=beijing_1024_polar \
    split=test \
    train.img_size=512 \
    model=segformer \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/beijing_segformer_polar

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/SegFormer_mit_b2_beijing_1024_polar/version_1/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=tibet_1024_polar \
    split=test \
    train.img_size=512 \
    model=segformer \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/tibet_segformer_polar

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/SegFormer_mit_b2_beijing_1024_polar/version_1/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=segformer \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_segformer_polar

echo "=== 5. Unet-pmsr-lite (UPolarSeg) version0 ==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_0/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_upolarsq

echo "=== 6. Unet-pmsr-lite (UPolarSeg) version 26==="
python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_26/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=beijing_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/beijing_upolarsq_v26

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_26/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=tibet_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/tibet_upolarsq_v26

python evaluate.py \
    checkpoint=/home/ziyue/ziyue/mengxian/medseg_research/outputs/ModularUNet_PMSR_lite_resnet34_beijing_1024_polar/version_26/checkpoints/best.ckpt \
    hardware.devices=[0] \
    dataset=zhengzhou_1024_polar \
    split=test \
    train.img_size=512 \
    model=research/modular_unet_pmsr_lite \
    polar=enabled \
    eval.output_dir=/home/ziyue/ziyue/mengxian/medseg_research/outputs/zhengzhou_upolarsq_v26
    '''