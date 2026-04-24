#!/bin/bash

# --- Set Environment Variables ---
export ENV_NAME="dpo_env"
export WANDB_PROJECT="qwen_dpo_sft"
export WANDB_RUN_NAME="0.6B_dpo_run_logs_1ep_fd"


# --- Create and Activate Conda Environment ---
echo "Creating Conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y  # change python version if needed
source activate $ENV_NAME  # or `conda activate $ENV_NAME` if conda init setup is already done

# --- Install Pip Requirements from train_dpo ---
echo "Installing pip requirements..."
pip install -r train_dpo/pip_requirements.txt  # Path to requirements.txt in the subdir


# Optional: Uncomment if needed
# export WANDB_API_KEY="your_api_key"
# export WANDB_MODE="online"

echo "Weights & Biases Project: $WANDB_PROJECT"
echo "Weights & Biases Run Name: $WANDB_RUN_NAME"

# --- Check GPU Availability ---
echo "Checking GPU availability..."
nvidia-smi  # Lists available GPUs (optional, for verification)

# --- Run the DPO Training Script ---
echo "Starting train_DPO.py..."

#example configuration, change if needed
CUDA_VISIBLE_DEVICES=0,1 python train_dpo/train_DPO.py \
    --epochs 1 \
    --batch_size 2 \
    --max_length 900 \
    --lr 2e-6 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "MoroM02/MoroM02" \
    --dataset_name "MoroM02/MNLP_M3_dpo_dataset" \
    --wandb_project "$WANDB_PROJECT" \
    --loss "LogS" \
    --grad_steps 12 \
    --gamma_hinge 0.5 \
    --run_name "0.6B_dpo_run_logs_1ep_fd"

echo "DPO training script finished."