#!/bin/bash

#SBATCH --job-name=Full_Preprocessing
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

echo "--- Starting Slurm Job: Full Preprocessing Pipeline ---"

# --- Configuration ---
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="drawings"
LINE_DRAWER_DIR="${PROJECT_ROOT}/tools/informative-drawings"
SCRIPT_NAME="test_magno.py"
STYLE_NAME="opensketch_style"

# --- Create Log Directory ---
mkdir -p logs

# --- Environment Setup ---
eval "$(mamba shell hook --shell bash)"

mamba env update -f environment.yml || mamba env create -f environment.yml -b

mamba activate ${CONDA_ENV_NAME}

# --- Change to the line drawer directory ---
cd ${LINE_DRAWER_DIR}
echo "Changed directory to: $(pwd)"

# --- Main Processing Loop ---
for SUB_DIR in train val; do
  echo "----------------------------------------"
  echo "Processing subdirectory: ${SUB_DIR}"
  echo "----------------------------------------"

  # Define all paths dynamically
  RAW_DATA_INPUT_DIR="${PROJECT_ROOT}/data/raw_dataset/${SUB_DIR}/"
  MAGNO_OUTPUT_DIR="${PROJECT_ROOT}/data/preprocessed/magno_images/${SUB_DIR}/"
  LINE_DRAWING_OUTPUT_DIR="${PROJECT_ROOT}/data/preprocessed/line_drawings/${SUB_DIR}/"

  python ${SCRIPT_NAME} \
    --name "${STYLE_NAME}" \
    --dataroot "${RAW_DATA_INPUT_DIR}" \
    --magno_output_dir "${MAGNO_OUTPUT_DIR}" \
    --line_drawing_output_dir "${LINE_DRAWING_OUTPUT_DIR}" \
    --size 32 # Process at the model's expected size

done

echo "--- All Python scripts completed ---"

# --- Cleanup ---
deactivate
cd ${PROJECT_ROOT}
echo "Current directory restored to: $(pwd)"
echo "--- Slurm Job Finished ---"