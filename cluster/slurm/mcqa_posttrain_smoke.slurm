#!/bin/bash

#SBATCH --job-name=stemtune-posttrain
#SBATCH --output=logs/stemtune-posttrain-%j.out
#SBATCH --error=logs/stemtune-posttrain-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_LIMIT="${TRAIN_LIMIT:-32}"
EVAL_LIMIT="${EVAL_LIMIT:-24}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/mcqa_posttrain_smoke_gpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

if [ -n "${ACTIVATE_SCRIPT}" ]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Python: $(${PYTHON_BIN} --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

${PYTHON_BIN} -m stemtune posttrain-mcqa \
  --device cuda \
  --model-id "${MODEL_ID}" \
  --train-limit "${TRAIN_LIMIT}" \
  --eval-limit "${EVAL_LIMIT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${OUTPUT_DIR}"

echo "Job finished at: $(date)"
