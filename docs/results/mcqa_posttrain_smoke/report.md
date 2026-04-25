# MCQA Post-Training Smoke Test

This smoke test measures contract-compliant task behavior, not just raw answer accuracy.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `allenai/sciq`
- Device: `cpu`
- Train examples: `32`
- Eval examples: `24`

| Phase | Letter Accuracy | Contract Valid Rate | Strict Accuracy | Weighted Score | Avg Latency (s) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.667 | 0.000 | 0.000 | 0.567 | 0.319 |
| After tiny LoRA post-training | 0.750 | 1.000 | 0.750 | 0.787 | 2.011 |

- Letter accuracy gain: `+0.083`
- Contract-valid gain: `+1.000`
- Strict accuracy gain: `+0.750`
- Weighted score gain: `+0.221`
