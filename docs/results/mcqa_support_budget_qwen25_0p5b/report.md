# MCQA Support Budget Study

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `allenai/sciq`
- Device: `mps`
- Seeds: `7, 11, 13, 17, 23`
- Examples per seed: `24`

## Aggregate Results

| Condition | Mean Accuracy | Mean Latency (s) | Gain vs Plain |
|---|---:|---:|---:|
| Question only | 0.733 | 0.079 | +0.000 |
| 24-word support | 0.900 | 0.083 | +0.167 |
| 48-word support | 0.958 | 0.091 | +0.225 |
| Full support | 0.958 | 0.107 | +0.225 |
