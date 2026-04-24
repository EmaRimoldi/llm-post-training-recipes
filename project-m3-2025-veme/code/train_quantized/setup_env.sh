#!/bin/bash
# filepath: /home/nevali/mnlp/quantization/setup_env.sh

# This script sets up the environment for running the QLoRA script
# Make it executable with: chmod +x setup_env.sh

# Set Python path
export PYTHONPATH=/home/nevali/mnlp:$PYTHONPATH

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0

# Uncomment and modify these lines based on your setup:
# Load conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_environment_name

# Or activate virtual environment
# source /path/to/your/venv/bin/activate

echo "Environment setup complete"
echo "Python path: $PYTHONPATH"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"