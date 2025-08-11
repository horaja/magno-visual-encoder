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
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/preprocess_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/preprocess_%j.err"

# ============================================
# LOAD CONFIGURATION
# ============================================
source configs/default_config.sh

echo "=========================================="
echo "DATA PREPROCESSING PIPELINE"
echo "=========================================="
echo "Start Time: $(date)"
echo ""
echo "Configuration:"
echo "  Raw Data: ${RAW_DATA_ROOT}"
echo "  Output Magno: ${MAGNO_DATA_DIR}"
echo "  Output Lines: ${LINES_DATA_DIR}"
echo "  Processing Size: ${PREPROCESSING_SIZE}"
echo "  Style: ${LINE_DRAWING_STYLE}"
echo "=========================================="

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
        --dataroot "${RAW_DATA_ROOT}/${SPLIT}/" \
        --magno_output_dir "${MAGNO_DATA_DIR}/${SPLIT}/" \
        --line_drawing_output_dir "${LINES_DATA_DIR}/${SPLIT}/" \
        --size ${PREPROCESSING_SIZE}
    
    echo "Completed ${SPLIT} split"
done

cd ${PROJECT_ROOT}
echo ""
echo "=========================================="
echo "Preprocessing completed at: $(date)"
echo "=========================================="