# MCQA Quantization Retention Study

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `allenai/sciq`
- Device: `cpu`
- Condition: `plain`
- Seeds: `7, 11, 13`
- Examples per seed: `24`

| Variant | Mean Accuracy | Mean Valid Rate | Mean Latency (s) |
|---|---:|---:|---:|
| Full precision | 0.694 | 1.000 | 0.271 |
| Dynamic int8 | 0.208 | 0.514 | 1.050 |

- Accuracy delta: `-0.486`
- Latency delta: `+0.779` seconds
