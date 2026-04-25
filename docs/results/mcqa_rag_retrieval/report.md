# MCQA Retrieval Study

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `allenai/sciq`
- Device: `mps`
- Corpus size: `524`
- Retrieval hit@1: `0.847`

| Condition | Mean Accuracy | Mean Latency (s) |
|---|---:|---:|
| Question only | 0.694 | 0.174 |
| Retrieved support | 0.958 | 0.269 |
| Oracle support | 0.986 | 0.297 |

- Retrieved gain vs plain: `+0.264`
- Oracle gain vs plain: `+0.292`
