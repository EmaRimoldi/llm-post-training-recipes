#!/bin/bash

set -euo pipefail

export ENV_NAME="${ENV_NAME:-dpo_env}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen_dpo_sft}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-0.6B_dpo_run_logs_1ep_fd}"

echo "Creating Conda environment '${ENV_NAME}'..."
conda create -n "$ENV_NAME" python=3.10 -y
source activate "$ENV_NAME"

echo "Installing requirements..."
pip install -r pip_requirements.txt

echo "Weights & Biases Project: $WANDB_PROJECT"
echo "Weights & Biases Run Name: $WANDB_RUN_NAME"

echo "Checking GPU availability..."
nvidia-smi || true

echo "Starting train_DPO.py..."
CUDA_VISIBLE_DEVICES=0,1 python train_DPO.py \
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
    --run_name "$WANDB_RUN_NAME"

echo "DPO training script finished."
