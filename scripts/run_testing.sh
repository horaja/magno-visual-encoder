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
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/testing_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/logs/testing_%j.err"

echo "--- Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) ---"
echo "Running on host: $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job started at: $(date)"

# --- Configuration & Hyperparameters ---
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="drawings"
TEST_SCRIPT_PATH="src/test.py"
PATCH_PERCENTAGE=0.30 # The patch percentage of the model you want to test

# --- Create Log and Results Directory ---
mkdir -p logs
mkdir -p results

# --- Environment Setup ---
echo "Loading modules..."
module load anaconda3-2023.03 cuda-12.4
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}
echo "Python executable: $(which python)"
echo "----------------------------------------"


# --- SCRIPT EXECUTION ---

# --- Full Testing Run (ACTIVE) ---
# This block runs the evaluation script on your best saved checkpoint.
echo "--- Starting model evaluation... ---"
python ${TEST_SCRIPT_PATH} \
  --magno_dir "data/preprocessed/magno_images" \
  --lines_dir "data/preprocessed/line_drawings" \
  --model_dir "models/checkpoints" \
  --results_dir "results" \
  --patch_percentage ${PATCH_PERCENTAGE} \
  --img_size 256 \
  --num_workers 4
echo "--- Testing script finished. ---"


# --- Cleanup ---
conda deactivate
echo "Job finished at: $(date)"
echo "--- Slurm Job Finished ---"