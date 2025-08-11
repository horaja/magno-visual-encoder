#!/bin/bash
#SBATCH --job-name=SelectiveMagnoViT_Eval
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH -p gpu
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/eval_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/eval_%j.err"

# ============================================
# LOAD CONFIGURATION
# ============================================
source configs/default_config.sh

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --patch-percentage)
            export PATCH_PERCENTAGE="$2"
            shift 2
            ;;
        --visualize)
            export VISUALIZE="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================
# SETUP
# ============================================
echo "=========================================="
echo "EVALUATION CONFIGURATION"
echo "=========================================="
echo "Model: pp=${PATCH_PERCENTAGE}"
echo "Checkpoint: ${MODEL_CHECKPOINT_DIR}/best_model_pp${PATCH_PERCENTAGE}.pth"
echo "Results will be saved to: ${RESULTS_ROOT_DIR}"
echo "=========================================="

# Create output directories
mkdir -p ${RESULTS_PLOTS_DIR}
mkdir -p ${RESULTS_LOGS_DIR}

# Setup environment
eval "$(mamba shell hook --shell bash)"
mamba activate ${CONDA_ENV_NAME}

# ============================================
# RUN EVALUATION
# ============================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${RESULTS_LOGS_DIR}/eval_pp${PATCH_PERCENTAGE}_${TIMESTAMP}.txt"

echo "Running evaluation..."
python src/evaluate.py \
    --magno_dir "${MAGNO_DATA_DIR}" \
    --lines_dir "${LINES_DATA_DIR}" \
    --model_dir "${MODEL_CHECKPOINT_DIR}" \
    --plots_dir "${RESULTS_PLOTS_DIR}" \
    --results_file "${RESULTS_FILE}" \
    --patch_percentage ${PATCH_PERCENTAGE} \
    --img_size ${IMG_SIZE} \
    --patch_size ${PATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --visualize ${VISUALIZE:-false}

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo "Plots saved to: ${RESULTS_PLOTS_DIR}"
echo "=========================================="