# STEMTune

`STEMTune` is the operator-facing layer of this repository.

It does not replace the training code. It helps you decide:

- which open-source model family is a sensible starting point;
- which alignment recipe to use for your task;
- how to move from data to training with the folder layout in this repo.

## Quickstart

```bash
python -m stemtune list-tasks
python -m stemtune show-task mcqa
python -m stemtune --task sft --gpu-memory-gb 16
python -m stemtune --task mcqa --gpu-memory-gb 24 --prefer-multilingual
python -m stemtune recommend --task dpo --gpu-memory-gb 24
python -m stemtune list-models --task rag --gpu-memory-gb 48 --prefer-long-context --prefer-tool-use
python -m stemtune init-project --name "Biomedical MCQA" --task mcqa --base-model Qwen/Qwen3-8B --hf-namespace your-name --output-dir ./workspaces
python -m stemtune smoke-mcqa --limit 12 --output-dir artifacts/evals/smoke_mcqa
python -m stemtune posttrain-mcqa --train-limit 32 --eval-limit 24 --epochs 2 --batch-size 4 --learning-rate 5e-5 --max-new-tokens 64
python -m stemtune benchmark-mcqa --model-id Qwen/Qwen2.5-0.5B-Instruct --limit 24 --seeds 7,11,13,17,23
python -m stemtune study-mcqa --models Qwen/Qwen2.5-0.5B-Instruct --limit 24 --seeds 7,11,13,17,23
python -m stemtune study-support-budget --model-id Qwen/Qwen2.5-0.5B-Instruct --limit 24 --seeds 7,11,13,17,23
```

## What The Selector Returns

- a recommended starting model;
- the reason it fits your constraints;
- the recipe folders to inspect first;
- concrete next steps inside this repository.

## Commands

- `python -m stemtune list-tasks`: list the supported alignment tasks.
- `python -m stemtune show-task <task>`: explain when to use a recipe and where to start.
- `python -m stemtune recommend --task <task> --gpu-memory-gb <n>`: recommend a model and repo path.
- `python -m stemtune list-models --task <task> --gpu-memory-gb <n>`: rank the curated model catalog for your constraints.
- `python -m stemtune init-project ...`: generate a neutral workspace with your own manifests, configs, and Hub targets.
- `python -m stemtune smoke-mcqa ...`: run a small public-dataset smoke test and emit metrics plus a comparison plot.
- `python -m stemtune posttrain-mcqa ...`: run a tiny LoRA post-training smoke test and compare baseline vs adapted behavior.
- `python -m stemtune benchmark-mcqa ...`: run a multi-seed MCQA benchmark and emit aggregate metrics plus a benchmark plot.
- `python -m stemtune study-mcqa ...`: run the support relevance ablation with correct vs mismatched evidence.
- `python -m stemtune study-support-budget ...`: measure how much support text is actually needed.

For compatibility, the original entrypoint still works:

```bash
python stemtune/select_stack.py --task mcqa --gpu-memory-gb 24
```

## Smoke Test

`smoke-mcqa` is a runnable check that compares two conditions on the public `allenai/sciq` dataset:

- `plain`: question and answer choices only;
- `grounded`: question, answer choices, and the support passage.

It writes:

- `predictions.csv`
- `summary.json`
- `report.md`
- `comparison.png`

This is a fast way to verify model loading, dataset handling, evaluation, and artifact generation without launching a full training job.

## Post-Training Smoke Test

`posttrain-mcqa` is the smallest end-to-end post-training demo in the repo.

It fine-tunes a tiny LoRA adapter on `allenai/sciq` and evaluates whether the adapted model can satisfy a strict machine-readable MCQA contract. The tracked run is here:

- [docs/results/mcqa_posttrain_smoke](../docs/results/mcqa_posttrain_smoke/report.md)

In the current tracked result:

- letter accuracy moves from `0.667` to `0.750`
- strict accuracy moves from `0.000` to `0.750`
- contract-valid rate moves from `0.000` to `1.000`
- weighted score moves from `0.567` to `0.788`

## Multi-Seed Benchmark

`benchmark-mcqa` extends the same idea across multiple seeds so you can decide whether grounding is actually helping instead of trusting a single lucky run.

It writes:

- `predictions.csv`
- `seed_summary.csv`
- `summary.json`
- `report.md`
- `benchmark.png`

The default benchmark in the repository uses `Qwen/Qwen2.5-0.5B-Instruct` on `allenai/sciq` and tracks the results under [docs/results/mcqa_grounding_qwen25_0p5b](../docs/results/mcqa_grounding_qwen25_0p5b/report.md).

## Evidence Studies

Two deeper studies are bundled as first-class commands:

- `study-mcqa`: checks whether the gain comes from relevant evidence rather than from prompt length alone;
- `study-support-budget`: checks how much support text is needed before returns flatten out.

The tracked results live in:

- [docs/results/mcqa_evidence_study](../docs/results/mcqa_evidence_study/report.md)
- [docs/results/mcqa_support_budget_qwen25_0p5b](../docs/results/mcqa_support_budget_qwen25_0p5b/report.md)

## Project Bootstrap

`init-project` is the command that makes STEMTune usable beyond the original repository.

It generates:

- `stemtune.project.json`: top-level project metadata;
- `configs/dataset.json`: your input and normalization contract;
- `configs/knowledge_base.json`: your corpus and chunking contract;
- `configs/training.json`: your base model and training settings;
- `configs/evaluation.json`: your metrics and promotion gates;
- `configs/publish.json`: your own Hub namespace and target repos;
- `.env.example`: environment variables with your own identifiers;
- `runbook.md`: the end-to-end automation outline.

This keeps the framework decoupled from repository-specific datasets and from any hardcoded Hugging Face profile.

## Design Philosophy

The selector intentionally stays simple.

It is not trying to be a benchmark oracle. It is a practical routing layer for:

- small-model experimentation;
- single-GPU or prosumer hardware;
- open-source post-training workflows;
- the specific task families covered by this repository.

For the full playbook, see [open-source-alignment-playbook.md](../docs/open-source-alignment-playbook.md).
