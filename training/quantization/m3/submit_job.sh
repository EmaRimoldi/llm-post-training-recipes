#!/bin/bash
# filepath: /users/nevali/projects/mnlp/quantization/submit_job.sh

# Make logs directory
mkdir -p logs

# Make scripts executable
chmod +x setup_env.sh
chmod +x run_qlora_contrastive.slurm

# Submit the job
echo "Submitting QLoRA contrastive training job..."
JOB_ID=$(sbatch run_qlora_contrastive.slurm | awk '{print $4}')

echo "Job submitted with ID: $JOB_ID"
echo "Monitor with: squeue -u $USER"
echo "Check logs with:"
echo "  tail -f logs/qlora_contrastive_${JOB_ID}.out"
echo "  tail -f logs/qlora_contrastive_${JOB_ID}.err"
echo "Cancel with: scancel $JOB_ID"