#!/bin/bash
#SBATCH --job-name=Preprocess_Data
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/tools/logs/preprocess_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/tools/logs/preprocess_%j.err"

# ============================================
# LOAD CONFIGURATION
# ============================================
source configs/default_config.sh

# Setup environment
eval "$(mamba shell hook --shell bash)"
mamba activate ${CONDA_ENV_NAME}

# Change to line drawer directory
cd tools/informative-drawings

# Process both train and validation sets
for SPLIT in train val; do
    echo ""
    echo "Processing split: ${SPLIT}"
    echo "-----------------------------------------"
    
    python test_magno.py \
        --name "${LINE_DRAWING_STYLE}" \
        --dataroot "${PROJECT_ROOT}/${RAW_DATA_ROOT}/${SPLIT}/" \
        --color_output_dir "${PROJECT_ROOT}/${COLOR_DATA_DIR}/${SPLIT}/" \
        --line_drawing_output_dir "${PROJECT_ROOT}/${LINES_DATA_DIR}/${SPLIT}/" \
        --ld_size ${LD_IMG_SIZE} \
        --color_img_size ${IMG_SIZE} \
        --cuda
    
    echo "Completed ${SPLIT} split"
done

cd ${PROJECT_ROOT}
echo ""
echo "=========================================="
echo "Preprocessing completed at: $(date)"
echo "=========================================="