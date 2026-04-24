# MCQA Grounding Benchmark

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `allenai/sciq`
- Device: `mps`
- Seeds: `7, 11, 13, 17, 23`
- Examples per seed: `24`

## Aggregate Results

| Condition | Mean Accuracy | Mean Valid Rate | Mean Latency (s) |
|---|---:|---:|---:|
| Plain question-only | 0.733 | 1.000 | 0.085 |
| Grounded with support passage | 0.958 | 1.000 | 0.140 |

- Mean accuracy gain: `+0.225`
- Accuracy gain std: `0.100`
- Total evaluated examples: `120`

## Per-Seed Accuracy

| Seed | Plain | Grounded | Gain |
|---:|---:|---:|---:|
| 7 | 0.667 | 1.000 | +0.333 |
| 11 | 0.708 | 1.000 | +0.292 |
| 13 | 0.708 | 0.958 | +0.250 |
| 17 | 0.833 | 0.917 | +0.083 |
| 23 | 0.750 | 0.917 | +0.167 |
