# Open-Source Alignment Playbook

This is the shortest operational guide for using this repository as a practical framework rather than as a code archive.

## 1. Choose A Starting Model

Use the selector:

```bash
python -m stemtune list-tasks
python -m stemtune show-task sft
python -m stemtune --task sft --gpu-memory-gb 16
python -m stemtune recommend --task dpo --gpu-memory-gb 24 --prefer-multilingual
python -m stemtune list-models --task rag --gpu-memory-gb 48 --prefer-long-context --prefer-tool-use
```

The current selector recommends among a small set of open-source model families that are strong defaults for the workflows in this repository:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-8B`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `mistralai/Mistral-Small-3.1-24B-Instruct-2503`

## 2. Match The Task To The Recipe

### SFT

Use when:

- you have supervised question-answer pairs;
- behavior is still too weak to justify preference optimization;
- you want the simplest adaptation baseline.

Start in:

- [training/sft/m2](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/sft/m2)
- [training/sft/m3](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/sft/m3)

### MCQA

Use when:

- the target output is one correct choice out of a small candidate set;
- evaluation is answer-accuracy driven;
- you want a compact route from domain data to a useful STEM benchmark model.

Start in:

- [datasets/builders/mcqa](/Users/emanuelerimoldi/Documents/GitHub/MNLP/datasets/builders/mcqa)
- [training/mcqa](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/mcqa)

### DPO

Use when:

- you already have a decent SFT or instruct baseline;
- you can build `chosen` / `rejected` pairs;
- you want to improve ranking-style preferences rather than raw next-token learning.

Start in:

- [datasets/builders/dpo](/Users/emanuelerimoldi/Documents/GitHub/MNLP/datasets/builders/dpo)
- [training/dpo](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/dpo)

### Quantization

Use when:

- the aligned model is good enough but too expensive to serve;
- you want local deployment or smaller hardware requirements;
- you need a calibrated compression path after training.

Start in:

- [datasets/calibration](/Users/emanuelerimoldi/Documents/GitHub/MNLP/datasets/calibration)
- [training/quantization/m2](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/quantization/m2)
- [training/quantization/m3](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/quantization/m3)

### RAG / Retrieval-Aware Training

Use when:

- the task depends on external documents;
- knowledge freshness matters;
- you want better grounding rather than only more parametric memorization.

Start in:

- [retrieval/knowledge_base](/Users/emanuelerimoldi/Documents/GitHub/MNLP/retrieval/knowledge_base)
- [training/rag](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/rag)

## 3. Place Data In The Right Local Paths

Tracked data in git is intentionally lightweight. For new work, do not copy your project into historical or repository-specific folders.

Bootstrap a neutral workspace first:

```bash
python -m stemtune init-project \
  --name "Your Project Name" \
  --task mcqa \
  --base-model Qwen/Qwen3-8B \
  --hf-namespace your-name \
  --output-dir ./workspaces
```

Then put your large local corpora under the generated project directories, for example:

```text
workspaces/your-project/
├── data/raw/
├── data/processed/
└── knowledge_base/raw/
```

## 4. Practical Default Choices

Use these defaults unless you have a strong reason not to:

- `Qwen/Qwen3-0.6B`: smoke tests, cheap experiments, low-risk recipe debugging.
- `Qwen/Qwen3-8B`: default open-source post-training starting point for SFT, MCQA, and DPO on a strong single GPU.
- `Meta-Llama-3.1-8B-Instruct`: good if you want a large ecosystem, 128k context, and assistant-style downstream behavior.
- `Mistral-Small-3.1-24B-Instruct-2503`: good if you care about long context, tool use, local serving, and stronger reasoning at a higher hardware budget.

## 5. Fast Routing Logic

If you are unsure where to begin, use this routing:

- Start with `Qwen/Qwen3-0.6B` for smoke tests, prompt formatting validation, and cheap debugging.
- Start with `Qwen/Qwen3-8B` for most SFT, MCQA, and DPO work on a strong single GPU.
- Start with `meta-llama/Meta-Llama-3.1-8B-Instruct` if you care more about ecosystem maturity and long context than about using a smaller experimental base.
- Start with `mistralai/Mistral-Small-3.1-24B-Instruct-2503` if retrieval, tool use, or stronger local reasoning matters enough to justify a larger memory budget.

## 6. One Reasonable End-To-End Flow

For a new STEM MCQA project:

1. Run `python -m stemtune show-task mcqa`.
2. Run `python -m stemtune --task mcqa --gpu-memory-gb 24`.
3. Normalize the dataset in [datasets/builders/mcqa](/Users/emanuelerimoldi/Documents/GitHub/MNLP/datasets/builders/mcqa).
4. Fine-tune in [training/mcqa](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/mcqa).
5. If answer accuracy is good but style/ranking is weak, move to [training/dpo](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/dpo).
6. If serving cost is too high, continue in [training/quantization](/Users/emanuelerimoldi/Documents/GitHub/MNLP/training/quantization).

## Model References

These recommendations are anchored in official model documentation and model cards:

- [Qwen3](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
