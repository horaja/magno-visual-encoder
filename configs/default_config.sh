#!/bin/bash
# Default configuration for SelectiveMagnoViT experiments

# ============================================
# EXPERIMENT SETTINGS
# ============================================
export EXPERIMENT_NAME="selective_magno_vit"
export EXPERIMENT_VERSION="v1.0"

# ============================================
# DATA PATHS
# ============================================
export RAW_DATA_ROOT="data/raw_dataset"
export PREPROCESSED_DATA_ROOT="data/preprocessed"
export MAGNO_DATA_DIR="${PREPROCESSED_DATA_ROOT}/magno_images"
export LINES_DATA_DIR="${PREPROCESSED_DATA_ROOT}/line_drawings"

# ============================================
# MODEL HYPERPARAMETERS
# ============================================
export IMG_SIZE=64
export PATCH_SIZE=4
export PATCH_PERCENTAGE=0.4
export NUM_CLASSES=10
export VIT_MODEL_NAME="vit_tiny_patch16_224.augreg_in21k"

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
export EPOCHS=100
export BATCH_SIZE=32
export LEARNING_RATE=1e-5
export WEIGHT_DECAY=0.1
export PATIENCE=20
export NUM_WORKERS=4

# ============================================
# PREPROCESSING SETTINGS
# ============================================
export LINE_DRAWING_STYLE="opensketch_style"
export PREPROCESSING_SIZE=64

# ============================================
# OUTPUT DIRECTORIES
# ============================================
export MODEL_CHECKPOINT_DIR="models/checkpoints"
export RESULTS_ROOT_DIR="results"
export RESULTS_PLOTS_DIR="${RESULTS_ROOT_DIR}/plots"
export RESULTS_LOGS_DIR="${RESULTS_ROOT_DIR}/logs"
export TENSORBOARD_LOG_DIR="logs/tensorboard/job_${SLURM_JOB_ID}"

# ============================================
# SLURM SETTINGS
# ============================================
export SLURM_TIME="08:00:00"
export SLURM_MEM="32gb"
export SLURM_CPUS=8
export SLURM_GPU_COUNT=1
export SLURM_PARTITION="gpu"
export SLURM_USER_EMAIL="horaja@cs.cmu.edu"

# ============================================
# ENVIRONMENT SETTINGS
# ============================================
export CONDA_ENV_NAME="drawings"
export PROJECT_ROOT=$(pwd)