#!/bin/bash

## Job name
#SBATCH --job-name=SelectiveMagnoViT_Train
## Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@andrew.cmu.edu

## Resource Allocation
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00-08:00:00

# Queue Selection
#SBATCH -p gpu

#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/training_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/training_%j.err"

# ============================================
# LOAD CONFIGURATION
# ============================================
source configs/default_config.sh

# Override any settings if needed via command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --patch-percentage)
            export PATCH_PERCENTAGE="$2"
            shift 2
            ;;
        --batch-size)
            export BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            export LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            export EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================
# JOB INFORMATION
# ============================================

# for debug only!
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi 

echo "=========================================="
echo "SLURM JOB CONFIGURATION"
echo "=========================================="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $HOSTNAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo ""

echo "=========================================="
echo "EXPERIMENT CONFIGURATION"
echo "=========================================="
echo "Experiment: ${EXPERIMENT_NAME} ${EXPERIMENT_VERSION}"
echo ""
echo "DATA CONFIGURATION:"
echo "  Magno Dir: ${MAGNO_DATA_DIR}"
echo "  Lines Dir: ${LINES_DATA_DIR}"
echo "  Image Size: ${IMG_SIZE}x${IMG_SIZE}"
echo ""
echo "MODEL CONFIGURATION:"
echo "  ViT Model: ${VIT_MODEL_NAME}"
echo "  Patch Size: ${PATCH_SIZE}"
echo "  Patch Percentage: ${PATCH_PERCENTAGE}"
echo "  Num Classes: ${NUM_CLASSES}"
echo ""
echo "TRAINING CONFIGURATION:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "  Early Stopping Patience: ${PATIENCE}"
echo "  Num Workers: ${NUM_WORKERS}"
echo ""
echo "OUTPUT CONFIGURATION:"
echo "  Checkpoint Dir: ${MODEL_CHECKPOINT_DIR}"
echo "  Results Dir: ${RESULTS_ROOT_DIR}"
echo "=========================================="

# ============================================
# CREATE DIRECTORIES
# ============================================
mkdir -p ${MODEL_CHECKPOINT_DIR}
mkdir -p ${RESULTS_LOGS_DIR}
mkdir -p ${RESULTS_PLOTS_DIR}
mkdir -p ${TENSORBOARD_LOG_DIR}

# ============================================
# ENVIRONMENT SETUP
# ============================================
echo ""
echo "Setting up environment: ${CONDA_ENV_NAME}"
eval "$(mamba shell hook --shell bash)"
mamba env update -f environment.yml || mamba env create -f environment.yml -y
mamba activate ${CONDA_ENV_NAME}
echo "Python: $(which python)"
echo "=========================================="

# ============================================
# RUN TRAINING
# ============================================
echo ""
echo "Starting training..."
python src/train.py \
    --magno_dir "${MAGNO_DATA_DIR}" \
    --lines_dir "${LINES_DATA_DIR}" \
    --output_dir "${MODEL_CHECKPOINT_DIR}" \
    --tensorboard_dir "${TENSORBOARD_LOG_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --patience ${PATIENCE} \
    --patch_percentage ${PATCH_PERCENTAGE} \
    --num_workers ${NUM_WORKERS} \
    --img_size ${IMG_SIZE} \
    --patch_size ${PATCH_SIZE} \
    --vit_model_name "${VIT_MODEL_NAME}"

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="