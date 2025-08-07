#!/bin/bash

#SBATCH --job-name=SelectiveMagnoViT_Train
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu

# --- Resource Allocation ---
# Requesting a GPU, as the final model will require it.
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00 # Set a reasonable time for a full training run

# --- GPU Queue Selection ---
#SBATCH -p gpu

# --- Logging ---
# Create a dedicated logs directory for training runs
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/training_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/training_%j.err"

echo "--- Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) ---"
echo "Running on host: $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job started at: $(date)"

# --- Configuration & Hyperparameters ---
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="drawings"
TRAIN_SCRIPT_PATH="src/train.py"

# --- Create Log Directory ---
mkdir -p logs

# --- Environment Setup ---
echo "Loading modules..."
module load anaconda3-2023.03 cuda-12.4
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}
echo "Python executable: $(which python)"
echo "----------------------------------------"


# --- SCRIPT EXECUTION ---

# --- Step 1 & 2: Verification (COMMENTED OUT) ---
# These steps have been successfully validated.
#
# echo "--- Running model verification script... ---"
# python src/model.py
# echo "--- Model verification finished. ---"
# echo "----------------------------------------"


# --- Step 3: Full Training Run (ACTIVE) ---
# This block is now active to run the full training and validation pipeline.
echo "--- Starting full training run... ---"
python ${TRAIN_SCRIPT_PATH} \
  --magno_dir "data/preprocessed/magno_images" \
  --lines_dir "data/preprocessed/line_drawings" \
  --output_dir "models/checkpoints" \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --patch_percentage 0.25 \
  --num_workers 4
echo "--- Training script finished. ---"


# --- Cleanup ---
conda deactivate
echo "Job finished at: $(date)"
echo "--- Slurm Job Finished ---"