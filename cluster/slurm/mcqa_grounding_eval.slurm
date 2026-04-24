#!/bin/bash

#SBATCH --job-name=stemtune-mcqa
#SBATCH --output=logs/stemtune-mcqa-%j.out
#SBATCH --error=logs/stemtune-mcqa-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
SEEDS="${SEEDS:-7,11,13,17,23}"
LIMIT="${LIMIT:-24}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/mcqa_grounding_qwen25_0p5b_gpu}"
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

${PYTHON_BIN} -m stemtune benchmark-mcqa \
  --model-id "${MODEL_ID}" \
  --seeds "${SEEDS}" \
  --limit "${LIMIT}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR}"

echo "Job finished at: $(date)"
