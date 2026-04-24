#!/bin/bash
# filepath: /users/nevali/projects/mnlp/quantization/run_qlora_contrastive.slurm

#SBATCH --job-name=mnlp
#SBATCH --output=qlora_contrastive%j.out    # Standard output log, %j is JobID
#SBATCH --error=qlora_contrastive%j.err     # Standard error log
#SBATCH --nodes=1                            # Run all srun steps on a single node
#SBATCH --ntasks=1                           # Run the run_sweep.sh script as one task
#SBATCH --cpus-per-task=8                    # Request CPUs for the script (adjust as needed, e.g., 2 per GPU)
#SBATCH --mem=64G                            # Total memory for the job (adjust as needed)
#SBATCH --gres=gpu:1                         # Request 4 GPUs for the entire job
#SBATCH --time=9:00:00                      # Maximum job wall time
#SBATCH -A a-infra01-1                       # Your account
#SBATCH --begin=now                 # Start job in 12 hours


# Optional: Email notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emanuele.nevali@epfl.ch


# Create logs directory if it doesn't exist
mkdir -p logs

# Load any necessary modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/nevali/mnlp:$PYTHONPATH

# Change to the quantization directory
cd /users/nevali/projects/mnlp/quantization

# Activate your conda/virtual environment if needed
# source activate your_env_name
# or
# conda activate your_env_name

# Run the script with timestamp logging
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python qlora_contrastive_full.py
# python qlora_next_token_correct_full.py

echo "Job finished at: $(date)"