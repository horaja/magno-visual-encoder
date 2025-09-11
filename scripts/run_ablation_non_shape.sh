#!/bin/bash
#SBATCH --job-name=NonShapeAblation
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/ablation_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/ablation_%j.err"

source configs/default_config.sh

# Ablation study parameters
NON_SHAPE_PERCENTAGES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
PATCH_PERCENTAGE=0.4  # Keep this fixed

echo "=========================================="
echo "NON-SHAPE TOKEN ABLATION STUDY"
echo "=========================================="
echo "Patch percentage (fixed): ${PATCH_PERCENTAGE}"
echo "Non-shape percentages: ${NON_SHAPE_PERCENTAGES[@]}"
echo "=========================================="

eval "$(mamba shell hook --shell bash)"
mamba activate ${CONDA_ENV_NAME}

for NSP in "${NON_SHAPE_PERCENTAGES[@]}"; do
    echo ""
    echo "Training with non_shape_percentage=${NSP}"
    echo "-----------------------------------------"
    
    # Update output paths with non-shape percentage
    MODEL_DIR="${MODEL_CHECKPOINT_DIR}/nsp_${NSP}"
    mkdir -p ${MODEL_DIR}
    
    python src/train.py \
        --magno_dir "${MAGNO_DATA_DIR}" \
        --lines_dir "${LINES_DATA_DIR}" \
        --output_dir "${MODEL_DIR}" \
        --tensorboard_dir "${TENSORBOARD_LOG_DIR}/nsp_${NSP}" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --patience ${PATIENCE} \
        --patch_percentage ${PATCH_PERCENTAGE} \
        --non_shape_percentage ${NSP} \
        --num_workers ${NUM_WORKERS} \
        --img_size ${IMG_SIZE} \
        --patch_size ${PATCH_SIZE} \
        --vit_model_name "${VIT_MODEL_NAME}"
    
    # Evaluate immediately after training
    echo "Evaluating model with non_shape_percentage=${NSP}"
    
    python src/evaluate.py \
        --magno_dir "${MAGNO_DATA_DIR}" \
        --lines_dir "${LINES_DATA_DIR}" \
        --model_dir "${MODEL_DIR}" \
        --plots_dir "${RESULTS_PLOTS_DIR}/nsp_${NSP}" \
        --results_file "${RESULTS_LOGS_DIR}/eval_nsp_${NSP}.json" \
        --patch_percentage ${PATCH_PERCENTAGE} \
        --non_shape_percentage ${NSP} \
        --img_size ${IMG_SIZE} \
        --patch_size ${PATCH_SIZE} \
        --num_workers ${NUM_WORKERS}
done

echo "Generating ablation summary plots..."
python src/plot_ablation_results.py \
    --results_dir "${RESULTS_LOGS_DIR}" \
    --output_path "${RESULTS_PLOTS_DIR}/ablation_non_shape_tokens.png"

echo ""
echo "=========================================="
echo "Ablation study completed!"
echo "=========================================="