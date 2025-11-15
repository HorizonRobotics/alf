#!/bin/bash
# Training script for PPO with Flow Matching Actor Network on Bullet Humanoid
# This uses PPOAlgorithm with FlowMatchingActorNetwork as the actor network

# turn off gin config
export ALF_USE_GIN=0

# Set the output directory
OUTPUT_DIR="/mnt/nas26/qiang.liu/experiments/fm_humanoid"

# Set the config file path (relative to ALF root)
CONF_FILE="alf/examples/fpo_bullet_humanoid_conf.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run training
python -m alf.bin.train \
    --root_dir="${OUTPUT_DIR}" \
    --conf="${CONF_FILE}" \
    --alsologtostderr

