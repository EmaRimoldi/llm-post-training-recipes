#!/bin/bash

set -euo pipefail

ACTIVATE_SCRIPT="${1:-}"
REPO_DIR="${REPO_DIR:-$PWD}"
PARTITION="${PARTITION:-mit_normal_gpu}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY_GB="${MEMORY_GB:-32}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

mkdir -p "${REPO_DIR}/logs"

sbatch \
  --partition="${PARTITION}" \
  --gres="${GRES}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY_GB}G" \
  --time="${TIME_LIMIT}" \
  --export=ALL,REPO_DIR="${REPO_DIR}",ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT}" \
  cluster/slurm/mcqa_grounding_eval.slurm
