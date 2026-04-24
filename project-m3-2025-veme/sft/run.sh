#!/bin/bash

# Script to set Weights & Biases environment variables and run the training script.

# --- Set Environment Variables ---

export WANDB_PROJECT="mnlp_m2"
export WANDB_RUN_NAME="0.6B_sft_next_token_masked_loss_run_1"

# Optional: Uncomment and set if you need to specify your W&B API key
# export WANDB_API_KEY="YOUR_API_KEY_HERE"

# Optional: Uncomment to ensure W&B runs in online mode (it's usually the default)
# export WANDB_MODE="online"

# --- Echo Variables (for verification) ---
echo "Weights & Biases Project: $WANDB_PROJECT"
echo "Weights & Biases Run Name: $WANDB_RUN_NAME"

# --- Run the Training Script ---
echo "Starting train.py..."
python train.py

# --- Script End ---
echo "Training script finished."

# To make this script executable:
# chmod +x run_training.sh
#
# To run this script:
# ./run_training.sh
# On gnoto use bash
