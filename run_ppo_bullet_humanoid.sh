#!/bin/bash
# Training script for PPO on Bullet Humanoid using gin configuration
export ALF_USE_GIN=1

# Set the output directory for logs, checkpoints, and summaries
OUTPUT_DIR="${HOME}/tmp/ppo_bullet_humanoid"

# Set the gin file path (relative to ALF root)
GIN_FILE="alf/examples/ppo_bullet_humanoid.gin"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run training
python -m alf.bin.train \
    --root_dir="${OUTPUT_DIR}" \
    --gin_file="${GIN_FILE}" \
    --alsologtostderr

