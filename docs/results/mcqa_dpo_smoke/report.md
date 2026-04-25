# MCQA DPO Smoke Test

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `allenai/sciq`
- Device: `cpu`
- Train examples: `32`
- Eval examples: `24`

| Phase | Letter Accuracy | Contract Valid Rate | Strict Accuracy | Weighted Score |
|---|---:|---:|---:|---:|
| Baseline | 0.667 | 0.000 | 0.000 | 0.567 |
| After tiny DPO | 0.583 | 1.000 | 0.583 | 0.646 |

- Letter accuracy gain: `-0.083`
- Contract-valid gain: `+1.000`
- Weighted score gain: `+0.079`
