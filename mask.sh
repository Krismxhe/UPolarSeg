OUTPUT_ROOT=/home/ziyue/ziyue/mengxian/pred_mask
IMG_SIZE=512
MAX_RADIUS=362
SPLIT=test


MODELS=(
"/home/ziyue/ziyue/mengxian/eval_model_weights/DeepLabv3+_resnet34_cart_best_weights/version_1/checkpoints/best.ckpt|deeplabv3plus_resnet34|DeepLabv3+_resnet34_cart|cart"
"/home/ziyue/ziyue/mengxian/eval_model_weights/DeepLabv3+_resnet34_polar_best_weights/version_4/checkpoints/best.ckpt|deeplabv3plus_resnet34|DeepLabv3+_resnet34_polar|polar"

"/home/ziyue/ziyue/mengxian/eval_model_weights/SegFormer_polar_best_weights/version_1/checkpoints/best.ckpt|segformer|SegFormer_polar|polar"
)

DATASETS_CART=(
"beijing_1024_cart|beijing"
"tibet_1024_cart|tibet"
"zhengzhou_1024_cart|zhengzhou"
)

DATASETS_POLAR=(
"beijing_1024_polar|beijing"
"tibet_1024_polar|tibet"
"zhengzhou_1024_polar|zhengzhou"
)


for entry in "${MODELS[@]}"; do
    IFS='|' read -r CKPT MODEL_CFG MODEL_NAME SPACE <<< "$entry"

    if [ "$SPACE" == "cart" ]; then
        DATASETS=("${DATASETS_CART[@]}")
    else
        DATASETS=("${DATASETS_POLAR[@]}")
    fi

    for ds_entry in "${DATASETS[@]}"; do
        IFS='|' read -r DATASET SHORT <<< "$ds_entry"
        echo ""
        echo "======================================================"
        echo "  Model  : ${MODEL_NAME}"
        echo "  Dataset: ${DATASET} (${SHORT})"
        echo "======================================================"

        python predict_masks.py \
            --ckpt "$CKPT" \
            --model_config "$MODEL_CFG" \
            --dataset "$DATASET" \
            --dataset_short "$SHORT" \
            --split "$SPLIT" \
            --model_name "$MODEL_NAME" \
            --output_root "$OUTPUT_ROOT" \
            --img_size "$IMG_SIZE" \
            --max_radius "$MAX_RADIUS"
    done
done

echo ""
echo "All done. Masks saved to $OUTPUT_ROOT"