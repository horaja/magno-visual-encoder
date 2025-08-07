#!/bin/bash

#SBATCH --job-name=Magno_LineGen
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu

# --- Resource Allocation ---
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00

#SBATCH -p gpu

# --- Logging ---
# Save logs to the tools/logs directory.
# The %j placeholder is replaced with the job ID.
#SBATCH --output="/user_data/horaja/workspace/dual-stream-vlm/encoder/tools/logs/line_gen_%j.out"
#SBATCH --error="/user_data/horaja/workspace/dual-stream-vlm/encoder/tools/logs/line_gen_%j.err"

echo "--- Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) ---"
echo "Running on host: $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job started at: $(date)"

# --- Configuration ---
# Define key paths and parameters here for easy editing.
# These paths are relative to the project root (where you run sbatch).
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="drawings"
LINE_DRAWER_DIR="tools/informative-drawings"

# Input directory containing the pre-processed Magno images
INPUT_DATA_DIR="${PROJECT_ROOT}/data/preprocessed/magno_images/"

# Output directory where the final line drawings will be saved
OUTPUT_RESULTS_DIR="${PROJECT_ROOT}/data/preprocessed/line_drawings/"

# Name of the pre-trained style model to use (e.g., opensketch_style)
# This corresponds to the checkpoint folder name.
STYLE_NAME="opensketch_style"

# --- Environment Setup ---
echo "Loading modules..."
module load anaconda3 cuda

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}
echo "Python executable: $(which python)"

# --- Run Line Drawing Generation ---
echo "Changing directory to: ${LINE_DRAWER_DIR}"
cd ${LINE_DRAWER_DIR}

echo "Starting line drawing generation..."
python test.py \
  --name "${STYLE_NAME}" \
  --dataroot "${INPUT_DATA_DIR}" \
  --results_dir "${OUTPUT_RESULTS_DIR}" \
  --checkpoints_dir "checkpoints" \
  --size 256 \
  --input_nc 1 \
  --output_nc 1 \
  --how_many 10000 # Set to a high number to process all images

echo "--- Python script completed ---"

# --- Cleanup ---
conda deactivate
cd ${PROJECT_ROOT}
echo "Current directory restored to: $(pwd)"
echo "Job finished at: $(date)"
echo "--- Slurm Job Finished ---"