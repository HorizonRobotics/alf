#!/bin/bash
# Training script for FPO with Flow Matching MLP Actor Network on Bullet Humanoid
# This uses DiffusionFPOAlgorithm with FlowMatchingMLPActorNetwork as the actor network

# turn off gin config
export ALF_USE_GIN=0

# Set the output directory.
# NOTE: if the directory already exists, the training will resume from the last checkpoint.
# DiT-based FPO
# OUTPUT_DIR="/mnt/cwai/hpfs0/qiang.liu/e2e-rl/fpo_humanoid"

# MLP-based FPO
OUTPUT_DIR="/mnt/cwai/hpfs0/qiang.liu/e2e-rl/fpo_humanoid_mlp"

# Set the config file path (relative to ALF root)
CONF_FILE="alf/examples/fpo_bullet_humanoid_conf.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Optional: Remove old checkpoint if it exists (uncomment if you want to reuse the same directory)
# rm -rf "${OUTPUT_DIR}/train/algorithm"

# Run training
python -m alf.bin.train \
    --root_dir="${OUTPUT_DIR}" \
    --conf="${CONF_FILE}" \
    --alsologtostderr

