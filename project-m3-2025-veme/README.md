# Milestone 3: DPO, RAG, And Advanced Post-Training

This is the most advanced milestone in the repository and the strongest section from a portfolio perspective.

## Scope

The work expands beyond basic fine-tuning into:

- DPO-based preference optimization;
- MCQA data generation and adaptation;
- RAFT/RAG-style data generation;
- STEM knowledge-base preparation from external corpora;
- model upload and deployment-oriented workflows around Hugging Face.

## Main Components

- `code/train_dpo/`: DPO training and evaluation utilities.
- `code/train_rag/`: RAG/RAFT training variants and dataset generation.
- `code/train_quantized/`: follow-up quantization experiments.
- `RAG/KB/`: knowledge-base preparation for ArXiv and Wikipedia STEM content.
- `RAG/raft/`: RAFT-oriented fine-tuning and dataset utilities.
- `mcqa/`: MCQA-specific training scripts.
- `data/`: dataset generation utilities and preprocessing scripts.
- `model_configs/`: model configuration metadata.

## Portfolio Value

This milestone is where the repository starts to look like applied LLM systems work rather than course submission code:

- preference-data usage is tied to optimization, not just collection;
- retrieval and document-preparation workflows appear explicitly;
- multiple training strategies are compared in code rather than discussed abstractly.

## Practical Notes

Several scripts in this folder now rely on environment variables instead of hardcoded credentials.

Common examples:

- `HF_TOKEN`
- `HF_USERNAME`
- `HF_MODEL_REPO_ID`
- `HF_MODEL_REPO_NAME`
- `HF_DATASET_REPO_ID`
- `HF_DATASET_REPO_NAME`
- `MNLP_GPT_WRAPPER_API_KEY`

That makes the code safer to publish and easier to reuse on another machine.
