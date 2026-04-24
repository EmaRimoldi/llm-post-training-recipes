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

For compatibility, the original entrypoint still works:

```bash
python stemtune/select_stack.py --task mcqa --gpu-memory-gb 24
```

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

For the full playbook, see [open-source-alignment-playbook.md](/Users/emanuelerimoldi/Documents/GitHub/MNLP/docs/open-source-alignment-playbook.md).
