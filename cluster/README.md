# Cluster Execution

The repository includes a minimal SLURM path for running GPU-backed MCQA evaluations.

Start with the public grounding benchmark:

```bash
bash cluster/submit_mcqa_grounding.sh /path/to/venv/bin/activate
```

Useful overrides:

```bash
PARTITION=mit_preemptable \
GRES=gpu:1 \
CPUS_PER_TASK=8 \
MEMORY_GB=32 \
TIME_LIMIT=02:00:00 \
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
SEEDS=7,11,13,17,23 \
LIMIT=24 \
OUTPUT_DIR=docs/results/mcqa_grounding_qwen25_0p5b_gpu \
bash cluster/submit_mcqa_grounding.sh /path/to/venv/bin/activate
```

The job writes:

- `summary.json`
- `report.md`
- `benchmark.png`
- `predictions.csv`
- `seed_summary.csv`

For the tiny post-training smoke test:

```bash
bash cluster/submit_mcqa_posttrain.sh /path/to/venv/bin/activate
```

Useful overrides:

```bash
PARTITION=mit_normal_gpu \
GRES=gpu:1 \
CPUS_PER_TASK=8 \
MEMORY_GB=32 \
TIME_LIMIT=02:00:00 \
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
TRAIN_LIMIT=32 \
EVAL_LIMIT=24 \
EPOCHS=4 \
OUTPUT_DIR=docs/results/mcqa_posttrain_smoke_gpu \
bash cluster/submit_mcqa_posttrain.sh /path/to/venv/bin/activate
```

Recommended partitions from the current cluster snapshot:

- `mit_normal_gpu`: first choice for a short, non-preemptable GPU run.
- `mit_preemptable`: fallback when queue pressure is lower there.

The login node does not expose `nvidia-smi`, so verify GPU visibility inside the SLURM allocation or job logs.
