#!/bin/bash
# filepath: /home/nevali/mnlp/quantization/monitor_job.sh

# Script to monitor the running job
if [ $# -eq 0 ]; then
    echo "Usage: $0 <job_id>"
    echo "Or run without arguments to see all your jobs"
    squeue -u $USER
    exit 1
fi

JOB_ID=$1

echo "Monitoring job $JOB_ID"
echo "Job status:"
squeue -j $JOB_ID

echo -e "\n=== Recent output ==="
if [ -f "logs/qlora_contrastive_${JOB_ID}.out" ]; then
    tail -20 logs/qlora_contrastive_${JOB_ID}.out
else
    echo "Output file not found yet"
fi

echo -e "\n=== Recent errors ==="
if [ -f "logs/qlora_contrastive_${JOB_ID}.err" ]; then
    tail -20 logs/qlora_contrastive_${JOB_ID}.err
else
    echo "Error file not found yet"
fi