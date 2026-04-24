# MCQA Evidence Study

- Dataset: `allenai/sciq`
- Device: `mps`
- Models: `Qwen/Qwen2.5-0.5B-Instruct`
- Seeds: `7, 11, 13, 17, 23`
- Examples per seed: `24`

## Model Summary

| Model | Plain | Mismatched Support | Correct Support | Correct Gain | Mismatched Gain |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.733 | 0.733 | 0.958 | +0.225 | -0.000 |

## Interpretation

- Correct support should lift accuracy materially above the question-only baseline.
- Mismatched support should not produce the same gain; otherwise the effect could be explained by prompt length alone.
- The latency frontier helps decide whether the accuracy gain is worth the added context cost for deployment.
