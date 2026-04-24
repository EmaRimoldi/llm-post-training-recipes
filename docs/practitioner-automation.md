# Practitioner Automation

This document explains how to use `STEMTune` as a reusable framework rather than as a frozen coursework repository.

## Core Principle

The training recipes in this repository are still useful, but they should no longer dictate:

- your Hugging Face namespace;
- your dataset names;
- your knowledge-base corpus;
- your local folder layout;
- your base model choice.

The new `init-project` command creates a practitioner-owned workspace that isolates those decisions.

## Bootstrap A New Project

```bash
python -m stemtune init-project \
  --name "Clinical Retrieval QA" \
  --task rag \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --hf-namespace your-name \
  --output-dir ./workspaces
```

This creates a new folder with:

- `README.md`
- `stemtune.project.json`
- `.env.example`
- `runbook.md`
- `configs/dataset.json`
- `configs/knowledge_base.json`
- `configs/training.json`
- `configs/publish.json`
- `data/`, `knowledge_base/`, `artifacts/`, and `scripts/`

## What Each Config Owns

### `configs/dataset.json`

Defines:

- where your raw training data comes from;
- what record schema it follows;
- where normalized output should be written.

This is where you adapt proprietary data, internal JSONL corpora, or public Hub datasets to the format your chosen recipe needs.

### `configs/knowledge_base.json`

Defines:

- where your source documents live;
- how they should be chunked;
- where the retrieval index should be stored;
- which Hub dataset repository, if any, should receive the processed corpus.

This is the contract for building your own RAG-ready corpus.

### `configs/training.json`

Defines:

- the base model you chose;
- the processed dataset location;
- training hyperparameters;
- output locations for model artifacts;
- the recipe folders to inspect first.

This keeps model selection and training settings under your control instead of under course-specific defaults.

### `configs/publish.json`

Defines:

- your own Hub namespace;
- target model, dataset, and KB repository IDs;
- whether those repositories should be private.

This is the separation line between the framework and any original author account.

## Practical Workflow

1. Use `python -m stemtune show-task <task>` to confirm the right alignment method.
2. Use `python -m stemtune --task <task> --gpu-memory-gb <budget>` to choose a starting model.
3. Use `python -m stemtune init-project ...` to generate your workspace.
4. Put your own raw assets inside the generated `data/` and `knowledge_base/` folders.
5. Write the smallest possible project-specific converters in `scripts/`.
6. Reuse the training and retrieval recipes from the main repository.
7. Publish to your own Hub repos using the generated config files.

## Why This Matters

Without this separation, the repository reads like a cleaned-up student project.

With this separation, it reads like a reusable framework:

- model selection is explicit;
- automation is scaffolded;
- data ownership is practitioner-controlled;
- Hub publishing is namespace-agnostic;
- the original repository becomes a recipe library rather than a one-off experiment dump.
