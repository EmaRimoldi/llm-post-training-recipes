# MCQA Post-Training Smoke Test

This smoke test measures contract-compliant task behavior, not just raw answer accuracy.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `allenai/sciq`
- Device: `cpu`
- Train examples: `32`
- Eval examples: `24`

| Phase | Strict Accuracy | Contract Valid Rate | Avg Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.000 | 0.000 | 0.314 |
| After tiny LoRA post-training | 0.708 | 1.000 | 1.740 |

- Strict accuracy gain: `+0.708`
- Contract-valid gain: `+1.000`
