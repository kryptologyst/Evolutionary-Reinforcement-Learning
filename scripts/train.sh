#!/bin/bash

# Training script for Evolutionary RL
# Usage: ./scripts/train.sh [config_file] [additional_args]

set -e

# Default configuration
CONFIG_FILE=${1:-"configs/default.yaml"}
shift || true

# Create logs directory
mkdir -p logs

# Run training
python -m src.train.train \
    --config "$CONFIG_FILE" \
    "$@"

echo "Training completed successfully!"
