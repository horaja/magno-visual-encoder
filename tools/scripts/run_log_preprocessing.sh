#!/bin/bash
#SBATCH --job-name=LoG_Preprocess
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH -p gpu
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/tools/logs/log_preprocess_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/tools/logs/log_preprocess_%j.err"

# ============================================
# LOAD CONFIGURATION
# ============================================
source configs/default_config.sh

# Define LoG specific parameters
export LOG_DATA_DIR="${PREPROCESSED_DATA_ROOT}/l_o_g"
export LOG_SIGMA=1.0  # Standard deviation for LoG filter
export LOG_THRESHOLD=75  # Threshold (percentile for simplified method)
export LOG_PREPROCESSING_SIZE=224  # Target image size
export LOG_METHOD="simplified"  # Method: simplified, zero_cross, or marr_hildreth

# Store test mode flag
TEST_MODE=""

# Parse command line arguments (optional overrides)
while [[ $# -gt 0 ]]; do
    case $1 in
        --sigma)
            export LOG_SIGMA="$2"
            shift 2
            ;;
        --threshold)
            export LOG_THRESHOLD="$2"
            shift 2
            ;;
        --size)
            export LOG_PREPROCESSING_SIZE="$2"
            shift 2
            ;;
        --method)
            export LOG_METHOD="$2"
            shift 2
            ;;
        --test)
            TEST_MODE="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "LAPLACIAN OF GAUSSIAN PREPROCESSING"
echo "=========================================="
echo "Start Time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo ""
echo "Configuration:"
echo "  Raw Data Root: ${RAW_DATA_ROOT}"
echo "  Output Directory: ${LOG_DATA_DIR}"
echo "  Target Size: ${LOG_PREPROCESSING_SIZE}x${LOG_PREPROCESSING_SIZE}"
echo "  LoG Sigma: ${LOG_SIGMA}"
echo "  Threshold: ${LOG_THRESHOLD}"
echo "  Method: ${LOG_METHOD}"
echo "  CPUs: ${SLURM_CPUS}"
echo "  Memory: ${SLURM_MEM}"
echo "=========================================="

# Create output directories
mkdir -p ${LOG_DATA_DIR}/train
mkdir -p ${LOG_DATA_DIR}/val
mkdir -p tools/logs

# Setup environment
echo ""
echo "Setting up environment: ${CONDA_ENV_NAME}"
eval "$(mamba shell hook --shell bash)"
mamba env update -f environment.yml || mamba env create -f environment.yml -y
mamba activate ${CONDA_ENV_NAME}

# NOW check if we're in test mode (AFTER environment is activated!)
if [ "$TEST_MODE" = "true" ]; then
    echo "Running parameter test mode..."
    python tools/src/preprocess_log.py \
        --test "${RAW_DATA_ROOT}/train/n01440764/n01440764_10026.JPEG" \
        --output_dir "${LOG_DATA_DIR}/parameter_tests"
    exit 0
fi

# Process both train and validation sets
TOTAL_START=$(date +%s)

for SPLIT in train val; do
    echo ""
    echo "Processing ${SPLIT} split..."
    echo "-----------------------------------------"
    
    SPLIT_START=$(date +%s)
    
    python tools/src/preprocess_log.py \
        --input_dir "${RAW_DATA_ROOT}/${SPLIT}" \
        --output_dir "${LOG_DATA_DIR}/${SPLIT}" \
        --size ${LOG_PREPROCESSING_SIZE} \
        --sigma ${LOG_SIGMA} \
        --threshold ${LOG_THRESHOLD} \
        --method ${LOG_METHOD} \
        --verify
    
    if [ $? -eq 0 ]; then
        SPLIT_END=$(date +%s)
        SPLIT_TIME=$((SPLIT_END - SPLIT_START))
        echo "Successfully completed ${SPLIT} split in ${SPLIT_TIME} seconds"
    else
        echo "ERROR: Failed to process ${SPLIT} split"
        exit 1
    fi
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "=========================================="
echo "LoG PREPROCESSING COMPLETE"
echo "=========================================="
echo "Total processing time: ${TOTAL_TIME} seconds"
echo "Completed at: $(date)"
echo ""

# Print final summary
echo "Output Summary:"
echo "--------------"
for SPLIT in train val; do
    if [ -d "${LOG_DATA_DIR}/${SPLIT}" ]; then
        NUM_IMAGES=$(find "${LOG_DATA_DIR}/${SPLIT}" -name "*_log.png" | wc -l)
        NUM_CLASSES=$(find "${LOG_DATA_DIR}/${SPLIT}" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "${SPLIT}: ${NUM_IMAGES} images across ${NUM_CLASSES} classes"
        
        # Check a sample image
        SAMPLE_IMG=$(find "${LOG_DATA_DIR}/${SPLIT}" -name "*_log.png" | head -1)
        if [ -f "$SAMPLE_IMG" ]; then
            IMG_INFO=$(python -c "
from PIL import Image
import numpy as np
img = Image.open('$SAMPLE_IMG')
arr = np.array(img)
white_ratio = np.sum(arr > 0) / arr.size * 100
print(f'Size: {img.size}, Mode: {img.mode}, White pixels: {white_ratio:.2f}%')
")
            echo "  Sample image: ${IMG_INFO}"
        fi
    fi
done

echo "=========================================="